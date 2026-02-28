from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.emotion_client import emotion_client

router = APIRouter()

@router.post("/analyze", response_model=dict, summary="Analyze Audio Emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    """
    Analyzes an uploaded audio file (must be .wav) to detect the primary emotion.
    Proxies the request to the dedicated GPU-accelerated Emotion API container.
    """
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
        
    # Delegate to the httpx client service
    result = await emotion_client.analyze_audio(file)
    
    return result
