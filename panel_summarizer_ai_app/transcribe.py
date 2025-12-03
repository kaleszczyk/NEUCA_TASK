from pathlib import Path
from openai import OpenAI, APIConnectionError
import time, json, os, math, subprocess, tempfile, re
import soundfile as sf
import shutil
import torch
from torch.serialization import add_safe_globals
add_safe_globals([torch.torch_version.TorchVersion])
from pyannote.audio import Pipeline
from config import HUGGINGFACE_TOKEN, DATA_DIR, FFMPEG_DIR, OPENAI_API_KEY

os.environ.setdefault("PYANNOTE_AUDIO_DISABLE_TORCHCODEC", "1")

TRANSCRIPT_DIR = Path(DATA_DIR) / "transcripts"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

SEGMENT_SECONDS =180  # 600 10 min; zmniejsz do 300/180 jeśli potrzeba
EXPECTED_SPEAKERS = 5  # 0 = nieznana liczba mówców
MODEL_NAME = "whisper-1"  # lub "gpt-4o-mini-transcribe"

OVERLAP_SECONDS = 3    # zakładka między chunkami

FILLER_PATTERN = re.compile(r"\b(uh|umm|er|yyy+|ee+|mmm+)\b", re.IGNORECASE)
def clean_fillers(text: str) -> str:
    return re.sub(r"\s+", " ", FILLER_PATTERN.sub("", text)).strip()

def _ff_bin(name: str) -> str:
    # spróbuj z FFMPEG_DIR, potem z PATH
    if FFMPEG_DIR:
        exe = f"{name}.exe" if os.name == "nt" else name
        cand = Path(FFMPEG_DIR) / exe
        if cand.exists():
            print(f"[FF] Using {name} from FFMPEG_DIR: {cand}")
            return str(cand)
    found = shutil.which(name)
    if found:
        print(f"[FF] Using {name} from PATH: {found}")
        return found
    raise FileNotFoundError(f"{name} not found. Set FFMPEG_DIR in .env or add it to PATH.")

FFPROBE_BIN = _ff_bin("ffprobe")
FFMPEG_BIN = _ff_bin("ffmpeg")

def _get_duration(audio_path: str) -> float:
    print(f"[DURATION] Probing duration via ffprobe for: {audio_path}")
    result = subprocess.run([
        FFPROBE_BIN, "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ], capture_output=True, text=True, check=True)
    dur = float(result.stdout.strip())
    print(f"[DURATION] {dur:.2f}s")
    return dur

def _split_audio(audio_path: str) -> list[str]:
    print(f"[SPLIT] Preparing chunks for: {audio_path}")
    duration = _get_duration(audio_path)
    parts = []
    total_segments = math.ceil(duration / SEGMENT_SECONDS)
    tmpdir = Path(tempfile.gettempdir()) / f"chunks_{Path(audio_path).stem}"
    tmpdir.mkdir(exist_ok=True)
    print(f"[SPLIT] Total chunks: {total_segments}, dir: {tmpdir}")
    for i in range(total_segments):
        # start z overlapem (oprócz pierwszego)
        start = max(0, i * SEGMENT_SECONDS - (OVERLAP_SECONDS if i > 0 else 0))
        # długość segmentu + overlap jeśli nie ostatni
        seg_len = SEGMENT_SECONDS + (OVERLAP_SECONDS if i < total_segments - 1 else 0)
        out = tmpdir / f"{Path(audio_path).stem}_chunk_{i:03d}.wav"
        if not out.exists():
            print(f"[SPLIT] Creating chunk {i}: start={start}s -> {out}")
            subprocess.run([
                FFMPEG_BIN, "-y", "-i", str(audio_path),
                "-ss", str(start),
                "-t", str(seg_len),
                "-ac", "1", "-ar", "16000",
                str(out)
            ], check=True)
        else:
            print(f"[SPLIT] Chunk {i} already exists: {out}")
        parts.append(str(out))
    return parts

def _transcribe_segment(path: str, base_offset: float, retries: int = 3, timeout: float = 600.0) -> dict:
    print(f"[TRANSCRIBE] Start segment: file={path}, offset={base_offset:.2f}")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with open(path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model=MODEL_NAME,
                    file=f,
                    response_format="verbose_json",
                    timeout=timeout
                )
            data = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp)
            segs = data.get("segments", [])
            for s in segs:
                s["start"] = (s.get("start") or 0.0) + base_offset
                s["end"] = (s.get("end") or s.get("start") or 0.0) + base_offset
            print(f"[TRANSCRIBE] OK segments={len(segs)}")
            return {"segments": segs, "text": data.get("text", "")}
        except APIConnectionError as e:
            print(f"[TRANSCRIBE] APIConnectionError attempt={attempt}: {e}")
            last_err = e
            time.sleep(2 * attempt)
        except Exception as e:
            print(f"[TRANSCRIBE] ERROR attempt={attempt}: {e}")
            last_err = e
            break
    print(f"[TRANSCRIBE] FAIL: {last_err}")
    raise last_err or RuntimeError(f"Transkrypcja segmentu nieudana: {path}")


def _diarize_segment(path: str, base_offset: float, pipeline: Pipeline) -> list:
    print(f"[DIAR] Start diarization: file={path}, offset={base_offset:.2f}")
    kwargs = {}
    if EXPECTED_SPEAKERS > 0:
        # wymuszenie liczby mówców (jeśli znana)
        kwargs["num_speakers"] = EXPECTED_SPEAKERS
    diarization = pipeline({"audio": path}, **kwargs)
    speaker_segments = []
    cnt = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start + base_offset,
            "end": turn.end + base_offset,
            "speaker": speaker
        })
        cnt += 1
    print(f"[DIAR] OK segments={cnt}")
    return speaker_segments

def assign_speakers(whisper_segments: list, speaker_segments: list) -> list:
    print(f"[ASSIGN] Assigning speakers: whisper={len(whisper_segments)}, diar={len(speaker_segments)}")
    def find_speaker(ts):
        for s in speaker_segments:
            if s["start"] <= ts <= s["end"]:
                return s["speaker"]
        return "UNKNOWN"
    enriched = []
    unknown = 0
    for seg in whisper_segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", start)
        speaker = find_speaker((start + end) / 2.0)
        if speaker == "UNKNOWN":
            unknown += 1
        enriched.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": seg.get("text", "")
        })
    print(f"[ASSIGN] Done. UNKNOWN={unknown}/{len(whisper_segments)}")
    return enriched

def save_transcript_outputs(audio_path: str, segments: list, pretty_txt: bool = True) -> tuple[str, str]:
    stem = Path(audio_path).stem
    json_out = TRANSCRIPT_DIR / f"{stem}.json"
    txt_out = TRANSCRIPT_DIR / f"{stem}.txt"

    print(f"[SAVE] Writing JSON: {json_out}")
    json_out.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SAVE] Writing TXT: {txt_out}")
    if pretty_txt:
        lines = [f"[{s['start']:.2f} - {s['end']:.2f}] {s['speaker']}: {clean_fillers(s['text'])}" for s in segments]
        txt_out.write_text("\n".join(lines), encoding="utf-8")
    else:
        txt_out.write_text("\n".join(clean_fillers(s["text"]) for s in segments), encoding="utf-8")

    print(f"[SAVE] Done.")
    return str(json_out), str(txt_out)

def _relabel_speakers_globally(speaker_segments: list) -> list:
    print(f"[RELABEL] Raw diar segments={len(speaker_segments)}")
    # proste mapowanie lokalnych etykiet na globalne
    mapping = {}
    next_id = 1
    def g(label):
        nonlocal next_id
        if label not in mapping:
            mapping[label] = f"SPEAKER_{next_id:02d}"
            next_id += 1
        return mapping[label]
    relabeled = [{**s, "speaker": g(s["speaker"])} for s in speaker_segments]
    # merge sąsiadujących
    merged = []
    for seg in sorted(relabeled, key=lambda x: (x["start"], x["end"])):
        if merged and merged[-1]["speaker"] == seg["speaker"] and seg["start"] <= merged[-1]["end"] + 0.3:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)
    print(f"[RELABEL] Merged segments={len(merged)} unique_speakers={len({m['speaker'] for m in merged})}")
    return merged

def transcribe_api(audio_path: str) -> str:
    print(f"[START] transcribe_api audio={audio_path}")
    # weryfikacja pliku audio
    try:
        _ = sf.info(audio_path)
        print(f"[CHECK] soundfile.info OK")
    except Exception as e:
        print(f"[CHECK] soundfile.info ERROR: {e}")
        raise RuntimeError(f"Nieprawidłowy plik audio: {audio_path}") from e

    # przygotuj pipeline diarization (raz)
    hf_token = HUGGINGFACE_TOKEN
    if not hf_token:
        print("[DIAR] Missing HUGGINGFACE_TOKEN in .env")
        raise RuntimeError("Brak HUGGINGFACE_TOKEN w .env.")

    print("[DIAR] Loading pyannote pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN  # <-- poprawiony parametr
        )
        print("[DIAR] Pipeline loaded.")
    except Exception as e:
        print(f"[DIAR] Pipeline load ERROR: {e}. Diarization will fallback to UNKNOWN.")
        pipeline = None

    # podział na chunki
    chunks = _split_audio(audio_path)
    print(f"[SPLIT] Chunks ready: {len(chunks)}")

    all_text = []
    all_whisper_segments = []
    all_speaker_segments = []
    offset = 0.0
    chunks = _split_audio(audio_path)
    for idx, chunk in enumerate(chunks):
        print(f"[LOOP] Processing chunk {idx}/{len(chunks)-1}: {chunk}, base_offset={offset:.2f}")

        # transkrypcja chunku
        try:
            res = _transcribe_segment(chunk, base_offset=offset)
            segs = res["segments"]
            for s in segs:
                all_whisper_segments.append(s)
        except Exception as e:
            print(f"[LOOP] Transcribe ERROR chunk={idx}: {e}")

        # diarizacja chunku
        if pipeline is not None:
            try:
                diar_segs = _diarize_segment(chunk, base_offset=offset, pipeline=pipeline)
                all_speaker_segments.extend(diar_segs)
            except Exception as e:
                print(f"[LOOP] Diar ERROR chunk={idx}: {e}")
        offset += SEGMENT_SECONDS

    if not all_speaker_segments:
        # fallback: jeden mówca
        total_dur = offset
        all_speaker_segments = [{"start": 0.0, "end": total_dur, "speaker": "UNKNOWN"}]

    # globalne relabel
    all_speaker_segments = _relabel_speakers_globally(all_speaker_segments)

    # przypisanie mówców do segmentów whisper
    enriched = assign_speakers(all_whisper_segments, all_speaker_segments)

    json_path, txt_path = save_transcript_outputs(audio_path, enriched, pretty_txt=True)
    return txt_path