# Emotion router.
# /analyze  → single-file emotion analysis
# /process  → full pipeline: VAD split → emotion per segment → return utterance list

from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.api.routes.emotion.pipeline import process_audio
from app.api.routes.emotion.service import emotion_client

router = APIRouter()


class AnalyzeRequest(BaseModel):
    file_path: str


@router.post("/analyze-local", summary="Local-file Emotion Analysis (Kaggle/Server)")
async def analyze_local_emotion(request: AnalyzeRequest):
    """Analyze one audio file from a local path and return the dominant emotion."""
    if not (request.file_path.endswith(".wav") or request.file_path.endswith(".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")
    return await emotion_client.analyze_local_file(request.file_path)


@router.post("/analyze", summary="Single-file Emotion Analysis (Upload)")
async def analyze_emotion(file: UploadFile = File(...)):
    """Upload an audio file and return the dominant emotion."""
    if not (file.filename.endswith(".wav") or file.filename.endswith(".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")
    return await emotion_client.analyze_audio(file)


@router.post("/process", summary="Full Audio Processing Pipeline")
async def process_emotion(
    file: UploadFile = File(...),
    interaction_id: UUID = Query(..., description="Parent interaction UUID"),
):
    """
    Full pipeline:
      1. Split audio into speech segments (Silero VAD)
      2. Run emotion analysis on each segment
      3. Return utterance-shaped results (ready for DB insertion)
    """
    if not (file.filename.endswith(".wav") or file.filename.endswith(".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")

    audio_bytes = await file.read()

    utterances = await process_audio(audio_bytes, file.filename, interaction_id)

    return {
        "interaction_id": str(interaction_id),
        "total_segments": len(utterances),
        "utterances": utterances,
    }
