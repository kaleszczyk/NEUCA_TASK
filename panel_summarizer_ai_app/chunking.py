from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

# LangChain splitter -langchain_text_splitters poczytac dokumentacje
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Opcjonalnie: tiktoken do liczenia tokenów (dokładniej niż znaki)
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

def _token_len_fn(model_name: str = "cl100k_base"):
    if _HAS_TIKTOKEN:
        try:
            enc = tiktoken.get_encoding(model_name)
        except KeyError:
            # fallback do najczęściej dostępnego encodera
            enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    avg_chars_per_token = 4.0
    return lambda s: max(1, int(len(s) / avg_chars_per_token))

def _build_splitter(
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    model_name: str = "cl100k_base"
) -> RecursiveCharacterTextSplitter:
    if _HAS_TIKTOKEN:
        try:
            # Spróbuj utworzyć splitter z tiktoken encoderem
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=model_name,            # jawnie przekazujemy nazwę encodera
                chunk_size=chunk_size_tokens,
                chunk_overlap=chunk_overlap_tokens,
            )
        except Exception:
            # Fallback na znakowy splitter z funkcją długości opartą o tiktoken
            pass
    # fallback: char-based z length_function liczącą "tokeny"
    length_fn = _token_len_fn(model_name)
    approx_chars = int(chunk_size_tokens * 4)
    approx_overlap = int(chunk_overlap_tokens * 4)
    return RecursiveCharacterTextSplitter(
        chunk_size=approx_chars,
        chunk_overlap=approx_overlap,
        length_function=length_fn,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],  # semantyczniej
    )

def _group_contiguous_turns(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Grupowanie kolejnych segmentów tego samego mówcy w "turny"
    turns: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for seg in segments:
        text = (seg.get("text", "") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker", "UNKNOWN")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))

        if cur and cur["speaker"] == speaker:
            cur["segments"].append({"start": start, "end": end, "text": text})
            cur["end"] = end
        else:
            if cur:
                turns.append(cur)
            cur = {
                "speaker": speaker,
                "start": start,
                "end": end,
                "segments": [{"start": start, "end": end, "text": text}],
            }

    if cur:
        turns.append(cur)

    # scal tekst i zbuduj mapę znaków do czasu
    for t in turns:
        parts = []
        char_spans = []  # [(seg_start, seg_end, char_start, char_end)]
        char_cursor = 0
        for s in t["segments"]:
            seg_text = s["text"].strip()
            if not seg_text:
                continue
            if parts:
                # zachowaj pojedynczą spację między segmentami
                parts.append(" ")
                char_cursor += 1
            start_c = char_cursor
            parts.append(seg_text)
            char_cursor += len(seg_text)
            end_c = char_cursor
            char_spans.append((s["start"], s["end"], start_c, end_c))
        t["text"] = "".join(parts)
        t["char_spans"] = char_spans  # do wyliczenia timestampów chunków
    return turns

def _chars_to_time(char_spans: List[Tuple[float, float, int, int]], char_pos: int) -> float:
    # Znajdź segment tekstowy obejmujący char_pos i zinterpoluj czas
    for (s_start, s_end, c_start, c_end) in char_spans:
        if c_start <= char_pos <= c_end:
            if c_end == c_start:
                return s_start
            ratio = (char_pos - c_start) / max(1, (c_end - c_start))
            return s_start + ratio * (s_end - s_start)
    # poza zakresem – przytnij do najbliższego
    if char_spans:
        if char_pos < char_spans[0][2]:
            return char_spans[0][0]
        if char_pos > char_spans[-1][3]:
            return char_spans[-1][1]
    return 0.0

def _split_turn_into_chunks(
    turn: Dict[str, Any],
    chunk_min_tokens: int = 400,
    chunk_max_tokens: int = 1000,
    overlap_ratio: float = 0.15,
    model_name: str = "cl100k_base"
) -> List[Dict[str, Any]]:
    text = turn["text"]
    if not text:
        return []

    target_tokens = max(chunk_min_tokens, min(chunk_max_tokens, 800))
    overlap_tokens = max(0, int(overlap_ratio * target_tokens))
    splitter = _build_splitter(target_tokens, overlap_tokens, model_name=model_name)

    # Rozbij na treści (bez metadanych) i ręcznie wyznacz pozycje start/end
    parts = splitter.split_text(text)

    # Wyznacz start/end znakowe przez szukanie kolejnych wystąpień
    chunks: List[Dict[str, Any]] = []
    cursor = 0
    for part in parts:
        if not part.strip():
            continue
        # znajdź indeks od bieżącego kursora (zapewnia poprawne pozycje przy powtórzeniach)
        idx = text.find(part, cursor)
        if idx == -1:
            # awaryjnie: spróbuj od początku
            idx = text.find(part)
        start_char = max(0, idx)
        end_char = start_char + len(part)
        cursor = end_char  # przesuwamy kursor naprzód

        # mapowanie znak->czas
        c_start = _chars_to_time(turn["char_spans"], start_char)
        c_end = _chars_to_time(turn["char_spans"], end_char)

        tok_len = _token_len_fn()(part)
        # odrzuć zbyt małe, jeśli mamy wiele
        if tok_len < chunk_min_tokens and len(parts) > 1:
            continue

        chunks.append({
            "speaker": turn["speaker"],
            "text": part.strip(),
            "start": round(c_start, 2),
            "end": round(c_end, 2),
            "tokens": tok_len,
            "turn_start": round(turn["start"], 2),
            "turn_end": round(turn["end"], 2),
        })

    if not chunks and text.strip():
        chunks.append({
            "speaker": turn["speaker"],
            "text": text.strip(),
            "start": round(turn["start"], 2),
            "end": round(turn["end"], 2),
            "tokens": _token_len_fn()(text),
            "turn_start": round(turn["start"], 2),
            "turn_end": round(turn["end"], 2),
        })

    return chunks

def chunk_segments(
    segments: List[Dict[str, Any]],
    chunk_min_tokens: int = 400,
    chunk_max_tokens: int = 1000,
    overlap_ratio: float = 0.15,
    model_name: str = "cl100k_base"
) -> List[Dict[str, Any]]:
    """
    Wejście: lista segmentów [{start,end,speaker,text}, ...]
    Wyjście: lista chunków z timestampami bez mieszania mówców.
    """
    turns = _group_contiguous_turns(segments)
    all_chunks: List[Dict[str, Any]] = []
    for t in turns:
        pieces = _split_turn_into_chunks(
            t,
            chunk_min_tokens=chunk_min_tokens,
            chunk_max_tokens=chunk_max_tokens,
            overlap_ratio=overlap_ratio,
            model_name=model_name,
        )
        all_chunks.extend(pieces)
    # Posortuj po czasie i nadaj id
    all_chunks.sort(key=lambda x: x["start"])
    for idx, ch in enumerate(all_chunks):
        ch["id"] = idx
    return all_chunks

def load_segments(json_path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    # Obsłuż format: albo lista segmentów, albo dict z kluczem "segments"
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    raise ValueError("Nieprawidłowy format pliku z segmentami.")

def chunk_transcript_json(
    json_path: str | Path,
    out_path: Optional[str | Path] = None,
    chunk_min_tokens: int = 400,
    chunk_max_tokens: int = 1000,
    overlap_ratio: float = 0.15,
    model_name: str = "cl100k_base"
) -> str:
    """
    Wczytuje segmenty (z diarization + timestamps), tworzy chunki bez mieszania mówców
    i zapisuje wynik do JSON.
    """
    segments = load_segments(json_path)
    chunks = chunk_segments(
        segments,
        chunk_min_tokens=chunk_min_tokens,
        chunk_max_tokens=chunk_max_tokens,
        overlap_ratio=overlap_ratio,
        model_name=model_name,
    )
    if out_path is None:
        p = Path(json_path)
        out_path = str(p.with_name(p.stem + "_chunks.json"))
    Path(out_path).write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)