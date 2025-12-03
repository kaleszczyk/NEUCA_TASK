# Panel Summarizer – README

## Cel aplikacji
- Podsumowanie długich debat/paneli oraz zadawanie pytań na podstawie transkryptu (RAG).
- Wejście: YouTube URL lub istniejące dane. Wyjście: strukturalne streszczenie i odpowiedzi oparte na cytowanych kontekstach.

## Architektura (warstwy i komponenty)
- UI: Gradio (`panel_summarizer_ui/app.py`)
  - Ekrany: Process YouTube, Summarize, Ask.
  - Streaming odpowiedzi z API: `/summarize_stream` i `/ask_stream`.
  - Podgląd wykorzystanych kontekstów: `GET /used_contexts`.
- API: FastAPI (`panel_summarizer_ai_app/main.py`)
  - ETL: `/process_youtube` → pobranie audio, podział na 3‑min, transkrypcja (Whisper), diarizacja (pyannote), scalanie segmentów, chunking tekstu, embedding i indeks do wektorów.
  - Summarize: `/summarize_stream` (kontrolowany budżet tokenów, map‑reduce).
  - Ask: `/ask_stream` (retrieval + cross‑encoder reranking → LLM odpowiedź; obsługa wielu pytań w jednej wiadomości przez LLM parser).
  - Diagnostyka: `/health` (ścieżki, katalogi, próbki transkryptów).
- RAG repozytorium: ChromaDB + OpenAIEmbeddings (`text-embedding-3-small`).
- Reranking: `sentence-transformers` CrossEncoder (MS MARCO MiniLM).
- LLM: OpenAI chat.completions (gpt‑4.1/gpt‑5.1) z `max_completion_tokens`.
- Dane: `data/transcripts` (txt/json/_chunks.json) + kolekcja “panel”.

## Transkrypcja i diarizacja (ETL szczegóły)
- Pobranie audio: `yt-dlp` + `ffmpeg`.
- Podział na krótsze pliki: audio dzielone na odcinki ok. 3 minut (≈180 s) w celu stabilnej i szybkiej obróbki.
- Transkrypcja: Whisper (`openai-whisper`)
  - Model Whisper (konfigurowalny); generuje tekst oraz znaczniki czasu (per fragment).
  - Wynik: pliki `.json` (z segmentami, timestampami) i `.txt` (ciągły tekst).
- Diarizacja: `pyannote-audio`
  - Wykrywa mówców i przypisuje segmenty wypowiedzi do identyfikatorów `speaker`.
  - Metadane segmentów: `speaker`, `start`, `end`, `text`.
- Scalanie
  - Po przetworzeniu 3‑minutowych odcinków segmenty są scalane w jeden spójny zbiór z zachowaniem metadanych (mówca i czasy).
- Usuwanie wypełniaczy typu: "yyy" "mmm"
- Uwaga na kolejność
  - Indeksowanie wektorów nie jest wykonywane przed chunkingiem: najpierw transkrypcja + diarizacja (na 3‑min plikach), następnie tekstowy chunking, a dopiero potem embedding i zapis do Chroma.

## Chunking 
- Cel: uzyskać semantyczne “chunki” bez mieszania mówców i z precyzyjnymi znacznikami czasu.
- Krok 1: Grupowanie segmentów w “turny”
  - `_group_contiguous_turns`: łączy kolejne segmenty tego samego mówcy w jedną wypowiedź (`turn`) i buduje mapę znaków do czasu (`char_spans`).
- Krok 2: Rozbijanie turnów na chunki
  - `_build_splitter`: używa `RecursiveCharacterTextSplitter` (z `langchain_text_splitters`), preferencyjnie z `tiktoken` (jeśli dostępny). Fallback na splitter znakowy z funkcją długości zbliżoną do tokenów.
  - Parametry (domyślne):
    - `chunk_min_tokens` ≈ 400, `chunk_max_tokens` ≈ 1000,
    - `overlap_ratio` ≈ 0.15,
    - encoding: `cl100k_base` (dla zgodności z modelami OpenAI).
  - `_split_turn_into_chunks`: dla każdej części oblicza pozycje znakowe i interpoluje czas (`_chars_to_time`) by nadać dokładne `start`/`end`.
  - Filtr: bardzo małe części są odrzucane, jeśli powstało wiele sensownych chunków.
- Krok 3: Sortowanie i identyfikatory
  - `chunk_segments`: scala chunki z wszystkich turnów, sortuje po czasie i nadaje sekwencyjne `id`.
- Wyjście:
  - `chunk_transcript_json`: zapisuje wynik do `*_chunks.json` ze strukturą:
    - `speaker`, `text`, `start`, `end`, `tokens`, `turn_start`, `turn_end`, `id`.

## Przepływ danych end‑to‑end
- Process YouTube
  - UI: `POST /process_youtube` z URL → status i `video_id`.
  - API: yt‑dlp + ffmpeg → podział audio na 3‑min odcinki; `transcribe_api` (Whisper) → JSON/TXT; diarizacja (pyannote); scalanie; `chunk_transcript_json` → chunki z metadanymi (speaker, start/end).
  - `vectors_repository.store_chunks` → `embed_documents` → upsert do Chroma.
- Summarize
  - UI: `POST /summarize_stream` z `video_id` lub `override_text`.
  - API: wybór tekstu (api_utils), `summarizer.summarize`:
    - Jeśli wejście mieści się w budżecie: pojedyncze zapytanie z `SUMMARY_PROMPT_SYSTEM/TEMPLATE`.
    - Jeśli nie: dzielenie na minimalną liczbę części (token‑budget), streszczenia cząstkowe, łączenie i finalne podsumowanie.
- Ask
  - UI: `POST /ask_stream` z `question` i `top_k` → stream odpowiedzi.
  - API: `build_contexts_for_ask` → do 20 kandydatów; cross‑encoder reranking; wybór `top_k`; LLM generuje odpowiedź; zapamiętanie `LAST_USED_CONTEXTS`.
  - Obsługa wielu pytań: wejście może zawierać wiele pytań; LLM parser wydziela listę pytań, odpowiedzi generowane są sekwencyjnie i scalane w jeden stream.
  - UI: po streamie `GET /used_contexts` → lista użytych fragmentów (speaker, przedziały czasu, excerpt).

## Kluczowe pliki (rola)
- `panel_summarizer_ui/app.py`: interfejs, streaming i podgląd kontekstów.
- `panel_summarizer_ai_app/main.py`: endpointy, globalne bufory (`LAST_USED_CONTEXTS`), `health`.
- `panel_summarizer_ai_app/summarizer.py`: prompty, `summarize` (map‑reduce, `max_completion_tokens`), `answer` (RAG + wielokrotne pytania).
- `panel_summarizer_ai_app/vectors_repository.py`: embeddingi i zapytania do Chroma, CrossEncoder (reranking), logi/fallbacki.
- `panel_summarizer_ai_app/chunking.py`: implementacja chunkingu (turny, splitter, interpolacja czasu).
- `panel_summarizer_ai_app/yt_utils.py`: `extract_video_id`, ścieżki do danych.
- `transcribe.py`, `yt_download.py`: pobranie/konwersja audio; `pyannote-audio` diarizacja; Whisper transkrypcja.
- `config.py`/`.env`: konfiguracja (API keys, model, ścieżki).

## Wyzwania i rozwiązania
- Limity API i zgodność parametrów:
  - `max_tokens` → `max_completion_tokens` (gpt‑4.1/gpt‑5.1).
  - Map‑reduce i budżety: `SUMMARIZE_INPUT_TOKEN_BUDGET`, `PARTIAL_SUMMARY_MAX_TOKENS`.
- Jakość retrievalu:
  - Cross‑encoder reranking (≤20 kandydatów) → lepsza trafność; przycinanie kontekstów.
- Wielokrotne odpytania w jednym query


## Konfiguracja i parametryzacja
- Env:
  - `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`,
  - `SUMMARIZE_INPUT_TOKEN_BUDGET`, `PARTIAL_SUMMARY_MAX_TOKENS`,
  - `CROSS_ENCODER_MODEL`,
  - `COLLECTION_NAME`, `FFMPEG_DIR`,
  - `HUGGINGFACE_TOKEN`.

- Requirements:
  - `fastapi`, `uvicorn`, `gradio`, `requests`, `yt-dlp`, `openai`, `tiktoken`, `python-dotenv`, `chromadb`, `langchain-openai`, `sentence-transformers`, `torch`, `pyannote-audio`, `openai-whisper`, `huggingface_hub`.

## Obsługa jakości i ewaluacja
- W aplikacji: podgląd użytych kontekstów (speaker, czas, fragment).
- Bez LLM: cross‑encoder/NLI scoring (relevancy i faithfulness per zdanie).
- Opcjonalnie: `/ask_eval_ce` i/lub LLM‑as‑judge.

## Demo – plan pokazania
- Process: URL → `video_id` → przetwarzanie 3‑min odcinków → transkrypcja/diaryzacja → chunking → indeks w Chroma.
- Summarize: stream z map‑reduce, ograniczenia tokenów.
- Ask: kilka pytań (np. z `notes.txt`) → stream odpowiedzi → źródła z `/used_contexts`.
- Health: `/health` (sprawdzenie ścieżek/danych).

## Roadmapa
- Cytowania [1]/[2] w odpowiedziach, linki do fragmentów.
- Persistent Chroma + wersjonowanie.
- Caching i optymalizacja kosztów.
- Monitoring (liczba zapytań, score rerankingu, czasy).