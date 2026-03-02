# Emotion router.
# /analyze  → single file emotion (existing)
# /process  → full pipeline: VAD split → emotion per segment → return utterance list

from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api.routes.emotion.pipeline import process_audio
from app.api.routes.emotion.service import emotion_client

router = APIRouter()


@router.post("/analyze", summary="Single-file Emotion Analysis")
async def analyze_emotion(file: UploadFile = File(...)):
    """Analyze one audio file and return the dominant emotion."""
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
    return await emotion_client.analyze_audio(file)


@router.post("/process", summary="Full Audio Processing Pipeline")
async def process_emotion(
    file: UploadFile = File(...),
    interaction_id: UUID = Query(..., description="Parent interaction UUID"),
):
    """
    Full pipeline:
      1. Split audio into speech segments (Silero VAD)
      2. Run emotion analysis on each segment (GPU Docker container)
      3. Return utterance-shaped results (ready for DB insertion)
    """
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    audio_bytes = await file.read()

    utterances = await process_audio(audio_bytes, file.filename, interaction_id)

    return {
        "interaction_id": str(interaction_id),
        "total_segments": len(utterances),
        "utterances": utterances,
    }
