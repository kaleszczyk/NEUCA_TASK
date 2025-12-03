import os
import json
from pathlib import Path
from typing import Any, List, Dict

from config import DATA_DIR
from vectors_repository import get_collection, query_and_rerank_crossencoder

COLLECTION_NAME_DEFAULT = os.getenv("COLLECTION_NAME", "panel")

def resolve_text_for_summarize(data: Any) -> str:
    """
    Buduje tekst do streszczenia (TXT -> JSON -> _chunks.json), albo rzuca wyjątek.
    data: obiekt z polami override_text, video_id
    """
    if getattr(data, "override_text", None):
        return data.override_text

    tdir = Path(DATA_DIR) / "transcripts"

    video_id = getattr(data, "video_id", None)
    if video_id:
        txt_path = tdir / f"{video_id}.txt"
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8")

        json_path = tdir / f"{video_id}.json"
        if json_path.exists():
            obj = json.loads(json_path.read_text(encoding="utf-8"))
            segs = obj["segments"] if isinstance(obj, dict) and "segments" in obj else obj
            return "\n".join(
                f"[{s.get('start',0):.2f}-{s.get('end',0):.2f}] {s.get('speaker','UNKNOWN')}: {s.get('text','')}"
                for s in segs
            )

        chunks_path = tdir / f"{video_id}_chunks.json"
        if chunks_path.exists():
            arr = json.loads(chunks_path.read_text(encoding="utf-8"))
            return "\n".join(
                f"[{c.get('start')}-{c.get('end')}] {c.get('speaker','UNKNOWN')}: {c.get('text','')}"
                for c in arr
            )

        raise FileNotFoundError(f"Brak plików dla video_id={video_id} w {tdir}")

    # fallback bez video_id
    txts = sorted(tdir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if txts:
        return txts[0].read_text(encoding="utf-8")

    jsons = sorted([p for p in tdir.glob("*.json") if not p.name.endswith("_chunks.json")],
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if jsons:
        obj = json.loads(jsons[0].read_text(encoding="utf-8"))
        segs = obj["segments"] if isinstance(obj, dict) and "segments" in obj else obj
        return "\n".join(
            f"[{s.get('start',0):.2f}-{s.get('end',0):.2f}] {s.get('speaker','UNKNOWN')}: {s.get('text','')}"
            for s in segs
        )

    chunk_files = sorted(tdir.glob("*_chunks.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if chunk_files:
        arr = json.loads(chunk_files[0].read_text(encoding="utf-8"))
        return "\n".join(
            f"[{c.get('start')}-{c.get('end')}] {c.get('speaker','UNKNOWN')}: {c.get('text','')}"
            for c in arr
        )

    raise FileNotFoundError(f"Brak transkryptów w {tdir} (TXT/JSON/_chunks.json)")


def build_contexts_for_ask(question: str, top_k: int, collection_name: str = COLLECTION_NAME_DEFAULT) -> List[Dict]:
    """
    Pobiera 20 kandydatów z wektorów i zwraca top_k (max 20) po rerankingu cross-encoderem.
    """
    col = get_collection(collection_name)
    n_candidates = 20
    k = min(int(top_k or 5), 20)
    return query_and_rerank_crossencoder(col, question, n_candidates=n_candidates, top_k=k)