# Emotion router.
# /analyze  → single-file emotion analysis
# /process  → full pipeline: VAD split → emotion per segment → return utterance list

from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.api.routes.emotion.pipeline import process_audio
from app.api.routes.emotion.service import emotion_client
from app.core.emotion_fusion import fuse_emotion_signals

router = APIRouter()


class AnalyzeRequest(BaseModel):
    file_path: str


class EmotionFusionRequest(BaseModel):
    text: str
    acoustic_emotion: str
    acoustic_confidence: float | None = None


class EmotionFusionResponse(BaseModel):
    emotion: str
    confidence: float
    text_emotion: str
    text_confidence: float
    acoustic_emotion: str
    acoustic_confidence: float
    model: str


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


@router.post("/fuse", response_model=EmotionFusionResponse, summary="Fuse Text and Acoustic Emotion")
async def fuse_emotion(payload: EmotionFusionRequest):
    fused = fuse_emotion_signals(
        text=payload.text,
        acoustic_emotion=payload.acoustic_emotion,
        acoustic_confidence=payload.acoustic_confidence,
    )
    return EmotionFusionResponse(
        emotion=fused.emotion,
        confidence=fused.confidence,
        text_emotion=fused.text_emotion,
        text_confidence=fused.text_confidence,
        acoustic_emotion=fused.acoustic_emotion,
        acoustic_confidence=fused.acoustic_confidence,
        model=fused.model,
    )


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
