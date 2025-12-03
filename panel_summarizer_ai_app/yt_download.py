from yt_dlp import YoutubeDL
from pathlib import Path
import os
from config import FFMPEG_DIR, DATA_DIR

DATA_DIR = Path(DATA_DIR) / "audio"
DATA_DIR.mkdir(parents=True, exist_ok=True)

#pobranie audio z YouTube w formacie WAV
def download_audio_from_youtube(url: str) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(DATA_DIR / "%(id)s.%(ext)s"), 
        "prefer_ffmpeg": True,
        "ffmpeg_location": FFMPEG_DIR,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}
        ],
        "postprocessor_args": ["-ac", "1", "-ar", "16000"],
        "keepvideo": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id")

    expected = DATA_DIR / f"{video_id}.wav"
    if expected.exists():
        return str(expected)

    matches = list(DATA_DIR.glob(f"{video_id}*.wav"))
    if matches:
        return str(matches[0])

    original = next(DATA_DIR.glob(f"{video_id}.*"), None)
    return str(original) if original else str(expected)