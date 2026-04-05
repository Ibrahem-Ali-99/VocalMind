from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import os
import shutil
import tempfile
import subprocess
import numpy as np
import torch
from funasr import AutoModel
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import asyncio

# Global dictionary to hold the loaded model
ml_models = {}
inference_lock = asyncio.Lock()
MAX_INFERENCE_SECONDS = int(os.getenv("EMOTION_MAX_AUDIO_SECONDS", "30"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the model at startup so it's ready for requests
    print("Loading model iic/emotion2vec_plus_base inside worker...")
    ml_models["emotion2vec"] = AutoModel(model="iic/emotion2vec_plus_base", trust_remote_code=True, disable_update=True, device=device)
    print("Model loaded successfully.")
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(title="Emotion Recognition API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "emotion2vec" in ml_models}

def _run_inference(filepath):
    print(f"Starting inference on file: {filepath}")
    return ml_models["emotion2vec"].generate(input=filepath, extract_embedding=False)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not (filename.endswith('.wav') or filename.endswith('.mp3')):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported.")

    suffix = '.wav' if filename.endswith('.wav') else '.mp3'

    # Save the uploaded file temporarily
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    normalized_wav_path = None
    try:
        with os.fdopen(fd, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        # Normalize all inputs to a bounded mono 16k wav to avoid unstable inference on long/raw files.
        fd_wav, normalized_wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd_wav)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                temp_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-t",
                str(MAX_INFERENCE_SECONDS),
                normalized_wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        inference_path = normalized_wav_path

        # Run inference using the pre-loaded FunaSR model via threadpool
        async with inference_lock:
            results = await run_in_threadpool(_run_inference, inference_path)
        print(f"Inference finished. Results: {results}")

        # Parse result
        if results and len(results) > 0:
            res = results[0]
            highest_score_idx = np.argmax(res['scores'])
            raw_label = res['labels'][highest_score_idx]

            # The label comes as "中文/english" like "开心/happy"
            emotion = raw_label.split('/')[-1] if '/' in raw_label else raw_label
            confidence = float(res['scores'][highest_score_idx])

            return {
                "emotion": emotion,
                "confidence": confidence,
                "raw_result": {
                    "labels": [lbl.split('/')[-1] if '/' in lbl else lbl for lbl in res['labels']],
                    "scores": [float(s) for s in res['scores']]
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Model inference returned empty results.")

    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode(errors="ignore")
        raise HTTPException(status_code=400, detail=f"Failed to decode/convert audio: {stderr[:400]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if normalized_wav_path and os.path.exists(normalized_wav_path):
            os.remove(normalized_wav_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
