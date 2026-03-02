"""
WhisperX FastAPI Service
========================
Wraps WhisperX (ASR + alignment + diarization) in a simple REST API.

Endpoints:
    POST /transcribe  — Upload audio file, get back transcribed segments
    GET  /health      — Health check
"""

import os
import gc
import time
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — loads HF_TOKEN from the root .env file
# ──────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN", "")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v2")

# ──────────────────────────────────────────────────────────────────────────────
# Compatibility patches — MUST run BEFORE importing whisperx / pyannote
# ──────────────────────────────────────────────────────────────────────────────
import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    import torchaudio
    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding
        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
except ImportError:
    pass

try:
    import huggingface_hub
    _original_hf_hub_download = huggingface_hub.hf_hub_download
    def _patched_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _original_hf_hub_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = _patched_hf_hub_download
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Now safe to import whisperx (patches are applied)
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import whisperx
from whisperx.diarize import DiarizationPipeline
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# ──────────────────────────────────────────────────────────────────────────────
# Overlap detection (from main_v5_final.py)
# ──────────────────────────────────────────────────────────────────────────────

def detect_overlaps(segments: List[Dict], threshold: float = 0.1) -> List[Dict]:
    segments = sorted(segments, key=lambda x: x["start"])
    for seg in segments:
        seg.setdefault("overlap", False)
    for i, curr in enumerate(segments):
        for j in range(i + 1, len(segments)):
            nxt = segments[j]
            if nxt["start"] >= curr["end"]:
                break
            curr["overlap"] = True
            nxt["overlap"] = True
    return segments

# ──────────────────────────────────────────────────────────────────────────────
# Model holder — loaded once at startup
# ──────────────────────────────────────────────────────────────────────────────

class Models:
    asr_model = None
    diarize_model = None

def load_models():
    print(f"Loading WhisperX model ({WHISPER_MODEL_SIZE}) on {DEVICE} ({COMPUTE_TYPE})...")
    Models.asr_model = whisperx.load_model(
        WHISPER_MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE
    )
    print("[OK] WhisperX ASR loaded")

    if not HF_TOKEN:
        print("⚠ WARNING: HF_TOKEN not set — diarization will fail")
    Models.diarize_model = DiarizationPipeline(
        use_auth_token=HF_TOKEN, device=DEVICE
    )
    print("[OK] Diarization pipeline loaded")

def unload_models():
    Models.asr_model = None
    Models.diarize_model = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    unload_models()

app = FastAPI(
    title="WhisperX Service",
    description="ASR + alignment + speaker diarization powered by WhisperX",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model": WHISPER_MODEL_SIZE,
        "models_loaded": Models.asr_model is not None,
    }


def _numpy_to_python(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_python(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
):
    """
    Upload an audio file and get back transcribed, aligned, and diarized segments.

    Returns JSON with:
      - language: detected language code
      - segments: list of {start, end, text, speaker, overlap}
      - processing_time_s: wall-clock time
    """
    if Models.asr_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Save uploaded file to a temp location
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        start_time = time.time()

        # Step 1 — Transcribe
        audio = whisperx.load_audio(tmp_path)
        result = Models.asr_model.transcribe(audio, batch_size=16, language=language)
        detected_language = result["language"]

        # Step 2 — Align
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language, device=DEVICE
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, DEVICE,
            return_char_alignments=False,
        )
        # Free alignment model right away
        del model_a, metadata
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Step 3 — Diarize
        diarize_segments = Models.diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Step 4 — Overlap detection
        result["segments"] = detect_overlaps(result["segments"])

        # Build clean response
        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text", "").strip(),
                "speaker": seg.get("speaker", "UNKNOWN"),
                "overlap": seg.get("overlap", False),
            })

        elapsed = time.time() - start_time

        return JSONResponse(content=_numpy_to_python({
            "language": detected_language,
            "segments": segments,
            "processing_time_s": round(elapsed, 2),
        }))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(tmp_path)
