import json
import os
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api_utils import build_contexts_for_ask
from evaluator import evaluate_answer_crossencoder
from yt_download import download_audio_from_youtube
from transcribe import transcribe_api
from chunking import chunk_transcript_json
from vectors_repository import get_collection, store_chunks, query_db
from summarizer import summarize, answer
from api_doc import documentation
from yt_utils import extract_video_id 
from api_utils import resolve_text_for_summarize, build_contexts_for_ask

os.environ.setdefault("PYANNOTE_AUDIO_DISABLE_TORCHCODEC", "1")

app = FastAPI()

# Bufory globalne
LAST_USED_CONTEXTS: List[Dict[str, Any]] = []
LAST_ASK_ANSWER: str = ""  # pełna odpowiedź z /ask_stream
LAST_ASK_EVAL_CE: dict = {}

APP_DIR = Path(__file__).resolve().parent 
from config import DATA_DIR
DATA_DIR_PATH = Path(DATA_DIR)
TRANSCRIPTS_DIR = DATA_DIR_PATH / "transcripts"
COLLECTION_NAME = "panel"

# budowanie modeli ewaluacyjnych cross-encoder
from evaluator import _get_ce
relevancy_model = os.getenv("RELEVANCY_CE_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
nli_model = os.getenv("NLI_CE_MODEL", "cross-encoder/nli-deberta-v3-base")
_get_ce(relevancy_model)
_get_ce(nli_model)

class YouTubeIn(BaseModel):
    url: str

class AskIn(BaseModel):
    question: str
    top_k: int = 5

class SummarizeIn(BaseModel):
    override_text: Optional[str] = None
    video_id: Optional[str] = None
    max_tokens: int = 800

def _existing_paths_for_id(video_id: str):
    transcripts_dir = Path(DATA_DIR) / "transcripts"
    txt = transcripts_dir / f"{video_id}.txt"
    jsn = transcripts_dir / f"{video_id}.json"
    chunks = transcripts_dir / f"{video_id}_chunks.json"
    return txt, jsn, chunks

@app.post("/process_youtube")
def process_youtube(data: YouTubeIn):
    url = data.url
    print(f"[PROCESS] /process_youtube called. url={url}")

    try:
        print("[PROCESS] Extracting video_id...")
        vid = extract_video_id(url)
        print(f"[PROCESS] video_id={vid}")
        if not vid:
            msg = "Nie można wyodrębnić video_id z URL. Podaj URL w formacie YouTube."
            print(f"[ERROR] {msg}")
            return {"error": msg}

        transcript_txt_path, transcript_json_path, chunks_json_path = _existing_paths_for_id(vid)
        print(f"[PROCESS] Paths -> TXT={transcript_txt_path} JSON={transcript_json_path} CHUNKS={chunks_json_path}")

        # Warunek użycia istniejących plików: transkrypcja JSON i chanki muszą istnieć
        if transcript_txt_path.exists() and transcript_json_path.exists() and chunks_json_path.exists():
            print("[PROCESS] Existing transcript JSON and chunks found. Reusing them...")
            print("[PROCESS] Loading chunks...")
            chunks = json.loads(Path(chunks_json_path).read_text(encoding="utf-8"))
            print(f"[PROCESS] Loaded {len(chunks)} chunks.")
            print(f"[PROCESS] Indexing chunks into collection '{COLLECTION_NAME}'...")
            col = get_collection(COLLECTION_NAME)
            print(f"[PROCESS] Storing chunks into collection '{COLLECTION_NAME}'...")
            store_chunks(col, chunks)
            print("[PROCESS] Indexing done.")

            return {
                "mode": "reuse_existing",
                "video_id": vid,
                "transcript_txt": str(transcript_txt_path) if transcript_txt_path.exists() else None,
                "transcript_json": str(transcript_json_path),
                "chunks_json": str(chunks_json_path),
                "indexed_chunks": len(chunks)
            }

        # Brak kompletu plików -> pełne przetworzenie od nowa
        print("[PROCESS] Missing transcript/chunks. Running full pipeline...")
        print("[PROCESS] Step 1: Download audio from YouTube...")
        audio_path = download_audio_from_youtube(url)
        print(f"[PROCESS] Audio downloaded: {audio_path}")

        print("[PROCESS] Step 2: Transcribe audio via API...")
        transcript_txt_path = transcribe_api(audio_path)  # zwraca ścieżkę txt
        print(f"[PROCESS] Transcription TXT: {transcript_txt_path}")

        transcript_json_path = str(Path(transcript_txt_path).with_suffix(".json"))
        print(f"[PROCESS] Expected transcription JSON: {transcript_json_path}")
        if not Path(transcript_json_path).exists():
            msg = "Brak pliku JSON transkryptu po transcribe_api"
            print(f"[ERROR] {msg}")
            return {"error": msg, "audio": audio_path, "txt": transcript_txt_path}

        print("[PROCESS] Step 3: Chunk transcript JSON...")
        chunks_json_path = chunk_transcript_json(
            transcript_json_path,
            chunk_min_tokens=400,
            chunk_max_tokens=1000,
            overlap_ratio=0.15
        )
        print(f"[PROCESS] Chunks JSON created: {chunks_json_path}")

        print("[PROCESS] Step 4: Load chunks and index into vector DB...")
        chunks = json.loads(Path(chunks_json_path).read_text(encoding="utf-8"))
        print(f"[PROCESS] Loaded {len(chunks)} chunks.")
        col = get_collection(COLLECTION_NAME)
        store_chunks(col, chunks)
        print(f"[PROCESS] Indexed chunks into collection '{COLLECTION_NAME}'.")

        return {
            "mode": "processed_new",
            "video_id": vid,
            "audio": audio_path,
            "transcript_txt": transcript_txt_path,
            "transcript_json": transcript_json_path,
            "chunks_json": chunks_json_path,
            "indexed_chunks": len(chunks)
        }
    except Exception as e:
        print("[ERROR] Exception in process_youtube:")
        print(str(e))
        print(traceback.format_exc())
        return {"error": f"process_youtube failed: {e}"}


@app.post("/summarize_stream")
def summarize_stream(data: "SummarizeIn"):

    print(f"[STREAM] /summarize video_id={data.video_id} override={bool(data.override_text)} max_tokens={data.max_tokens}")

    def generator():
        try:
            text = resolve_text_for_summarize(data)
            for delta in summarize(text, max_tokens=int(data.max_tokens or 2000)): #dla bezpieczenstwa
                if delta:
                    yield delta
        except Exception as e:
            print("[ERROR] summarize_stream failed:")
            print(traceback.format_exc())
            yield f"\n\n[ERROR] {e}"

    return StreamingResponse(generator(), media_type="text/plain; charset=utf-8")

@app.post("/ask_stream")
def ask_stream(data: AskIn):
    def generator():
        import traceback
        global LAST_USED_CONTEXTS, LAST_ASK_ANSWER
        LAST_USED_CONTEXTS = []
        LAST_ASK_ANSWER = ""
        try:
            contexts = build_contexts_for_ask(data.question, data.top_k)
            LAST_USED_CONTEXTS = contexts
            chunks: List[str] = []
            for delta in answer(contexts, data.question, top_k=len(contexts)):
                if delta:
                    chunks.append(delta)
                    yield delta
            # zapisz pełną odpowiedź po zakończeniu streamu
            LAST_ASK_ANSWER = "".join(chunks)
        except Exception as e:
            print("[ERROR] ask_stream failed:")
            print(traceback.format_exc())
            yield f"\n\n[ERROR] {e}"
    return StreamingResponse(generator(), media_type="text/plain; charset=utf-8")

@app.get("/used_contexts")
def used_contexts():
    try:
        return {"contexts": LAST_USED_CONTEXTS}
    except Exception as e:
        return {"error": str(e), "contexts": []}

@app.post("/ask_eval_ce")
def ask_eval_ce(data: AskIn):
    """
    Ewaluacja bez LLM (cross-encoder): relevancy i faithfulness dla zapisanej odpowiedzi i kontekstów.
    Najpierw używa LAST_ASK_ANSWER; jeśli pusty, generuje niestrumieniowo.
    """
    try:
        print(f"[EVAL_CE] /ask_eval_ce question={data.question}")
        contexts = LAST_USED_CONTEXTS
        print(f"contexts: {contexts}")
        answer_txt = LAST_ASK_ANSWER 
        print(f"LAST_ASK_ANSWER: {answer_txt}")
        metrics = evaluate_answer_crossencoder(data.question, answer_txt, contexts)
        print(f"[EVAL_CE] metrics: {metrics}")
        result = {
            "question": data.question,
            "answer": answer_txt,
            "metrics_ce": metrics,
            "used_contexts_count": len(contexts),
        }
        global LAST_ASK_EVAL_CE
        LAST_ASK_EVAL_CE = result
        return result
    except Exception as e:
        return {"error": str(e), "question": data.question}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cwd": os.getcwd(),
        "app_dir": str(APP_DIR),
        "data_dir": str(DATA_DIR_PATH),
        "data_dir_exists": DATA_DIR_PATH.exists(),
        "transcripts_dir": str(TRANSCRIPTS_DIR),
        "transcripts_dir_exists": TRANSCRIPTS_DIR.exists(),
        "transcripts_sample": [p.name for p in list(TRANSCRIPTS_DIR.glob('*'))[:5]] if TRANSCRIPTS_DIR.exists() else [],
    }

@app.get("/docs")
def docs():
    return {
        documentation()
    }
    

@app.get("/")
def root():
    return {
        "message": "API działa. Dostępne: /process_youtube, /summarize, /ask, /health, dokumentacja pod /docs",
    }

