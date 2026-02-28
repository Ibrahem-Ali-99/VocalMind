# Emotion router.
# Logic: Thin HTTP layer — validates input then delegates to service.py.

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.routes.emotion.service import emotion_client

router = APIRouter()


@router.post("/analyze", response_model=dict, summary="Analyze Audio Emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    """
    Accepts a .wav audio upload and returns the detected emotion.
    Proxies the file to the GPU-accelerated Emotion API container.
    """
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    return await emotion_client.analyze_audio(file)
