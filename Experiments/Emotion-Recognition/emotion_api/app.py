from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import os
import shutil
import tempfile
import numpy as np
import torch
from funasr import AutoModel
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

# Global dictionary to hold the loaded model
ml_models = {}

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

def _run_inference(filepath):
    print(f"Starting inference on file: {filepath}")
    return ml_models["emotion2vec"].generate(input=filepath, extract_embedding=False)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    # Save the uploaded file temporarily
    fd, temp_path = tempfile.mkstemp(suffix='.wav')
    try:
        with os.fdopen(fd, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        # Run inference using the pre-loaded FunaSR model via threadpool
        results = await run_in_threadpool(_run_inference, temp_path)
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
