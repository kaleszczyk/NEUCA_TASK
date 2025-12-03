def documentation():
    """
    Prosta dokumentacja endpointów API (tekst/JSON).
    """
    return {
        "title": "Panel Summarizer API – Dokumentacja",
        "base_url": "/",
        "endpoints": [
            {
                "method": "GET",
                "path": "/health",
                "description": "Status serwera i ścieżki robocze.",
                "input": "brak",
                "output": {
                    "status": "ok",
                    "cwd": "ścieżka robocza",
                    "app_dir": "ścieżka katalogu aplikacji",
                    "data_dir": "ścieżka DATA_DIR",
                    "data_dir_exists": "bool",
                    "transcripts_dir": "ścieżka katalogu transkryptów",
                    "transcripts_dir_exists": "bool",
                    "transcripts_sample": "lista kilku plików (jeśli istnieją)"
                }
            },
            {
                "method": "POST",
                "path": "/process_youtube",
                "description": "Pobiera audio z YouTube, transkrybuje i buduje chanki. Jeśli istnieją kompletne pliki, używa ich.",
                "input": {
                    "json": {"url": "pełny URL filmu YouTube"}
                },
                "output": {
                    "mode": "reuse_existing | processed_new",
                    "video_id": "ID filmu",
                    "audio": "ścieżka do pobranego audio (gdy processed_new)",
                    "transcript_txt": "ścieżka do TXT transkryptu",
                    "transcript_json": "ścieżka do JSON transkryptu",
                    "chunks_json": "ścieżka do JSON z chunkami",
                    "indexed_chunks": "liczba zindeksowanych chunków"
                }
            },
            {
                "method": "POST",
                "path": "/summarize",
                "description": "Streamuje podsumowanie panelu (tekst), bazując na TXT/JSON/_chunks.json.",
                "input": {
                    "json": {
                        "override_text": "opcjonalny surowy tekst do streszczenia",
                        "video_id": "opcjonalne ID filmu do odnalezienia plików",
                        "max_tokens": "int, limit tokenów podsumowania (domyślnie 800)"
                    }
                },
                "output": "text/event-stream (StreamingResponse) – napływający tekst podsumowania"
            },
            {
                "method": "POST",
                "path": "/ask",
                "description": "Streamuje odpowiedź na pytanie. Zapamiętuje ostatnio użyte konteksty w pamięci.",
                "input": {
                    "json": {
                        "question": "pytanie użytkownika",
                        "top_k": "int, liczba kontekstów do pobrania (domyślnie 5)"
                    }
                },
                "output": "text/event-stream (StreamingResponse) – napływający tekst odpowiedzi",
                "notes": [
                    "Globalna zmienna LAST_USED_CONTEXTS przechowuje konteksty użyte w ostatnim wywołaniu."
                ]
            },
            {
                "method": "GET",
                "path": "/used_contexts",
                "description": "Zwraca ostatnio użyte konteksty (bez ponownego zapytania do bazy).",
                "input": "brak",
                "output": {
                    "count": "liczba kontekstów",
                    "contexts": "[{text, speaker, start, end}]"
                }
            },
            {
                "method": "GET",
                "path": "/",
                "description": "Informacyjny endpoint root.",
                "input": "brak",
                "output": {
                    "message": "opis dostępnych endpointów",
                    "data_dir": "ścieżka DATA_DIR",
                    "data_dir_exists": "bool"
                }
            }
        ],
        "notes": [
            "Endpointy /summarize oraz /ask zwracają StreamingResponse (tekst napływa fragmentami).",
            "Użyj /used_contexts po zakończeniu streamu /ask, aby podejrzeć konteksty.",
            "Proces /process_youtube indeksuje chanki do kolekcji wektorowej 'panel'."
        ]
    }