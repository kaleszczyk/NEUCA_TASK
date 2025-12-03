from zipfile import Path
import re
from typing import Optional
from config import DATA_DIR

#ekstrakcja video_id z URL
def extract_video_id(url: str) -> Optional[str]:
    m = re.search(r"[?&]v=([A-Za-z0-9_\-]{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_\-]{6,})", url)
    if m:
        return m.group(1)
    return None

#metoda sciezki z transkrupcją i chunkami 
def existing_paths_for_id(video_id: str):
    transcripts_dir = Path(DATA_DIR) / "transcripts"
    txt = transcripts_dir / f"{video_id}.txt"
    jsn = transcripts_dir / f"{video_id}.json"
    chunks = transcripts_dir / f"{video_id}_chunks.json"
    return txt, jsn, chunks