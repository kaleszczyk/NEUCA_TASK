import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
FFMPEG_DIR = os.getenv("FFMPEG_DIR")

env_data_dir = os.getenv("DATA_DIR")
DATA_DIR = str(Path(env_data_dir).resolve())



