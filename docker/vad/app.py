# VAD Microservice — Silero VAD + pydub audio slicing
#
# POST /split  →  accepts .wav, returns JSON with base64-encoded audio clips
# GET  /health →  liveness check

import base64
import io
import tempfile
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydub import AudioSegment
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

# ── Model Loading ───────────────────────────────────────────────────────

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    ml_models["vad_model"] = model
    ml_models["vad_utils"] = utils
    print("Silero VAD ready.")
    yield
    ml_models.clear()


app = FastAPI(title="VAD Preprocessing Service", lifespan=lifespan)


# ── Sync inference (runs in threadpool) ─────────────────────────────────

def _split_audio(tmp_path: str):
    """Run VAD + slice audio. Returns list of segment dicts."""
    model = ml_models["vad_model"]
    utils = ml_models["vad_utils"]
    get_speech_timestamps, _, read_audio, *_ = utils

    wav = read_audio(tmp_path, sampling_rate=16000)
    segments = get_speech_timestamps(wav, model, sampling_rate=16000)

    if not segments:
        return []

    audio = AudioSegment.from_wav(tmp_path)
    results = []

    for idx, seg in enumerate(segments):
        start_sec = seg["start"] / 16000
        end_sec = seg["end"] / 16000

        clip = audio[int(start_sec * 1000): int(end_sec * 1000)]

        buf = io.BytesIO()
        clip.export(buf, format="wav")
        clip_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        results.append({
            "index": idx,
            "start_time": round(start_sec, 3),
            "end_time": round(end_sec, 3),
            "audio_base64": clip_b64,
        })

    return results


# ── Endpoints ───────────────────────────────────────────────────────────

@app.post("/split")
async def split_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files supported.")

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments = await run_in_threadpool(_split_audio, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"total_segments": len(segments), "segments": segments}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "vad_model" in ml_models}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8002)
