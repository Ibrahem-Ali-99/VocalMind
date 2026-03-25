# Diarization router.
# /analyze       → upload-based speaker diarization
# /analyze-local → local-file speaker diarization

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.routes.diarization.service import diarization_client

router = APIRouter()


class AnalyzeLocalRequest(BaseModel):
    file_path: str


@router.post("/analyze", summary="Speaker Diarization (Upload)")
async def analyze_diarization(file: UploadFile = File(...)):
    """Upload an audio file and return speaker-timestamped segments."""
    if not (file.filename.endswith(".wav") or file.filename.endswith(".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")
    return await diarization_client.analyze_audio(file)


@router.post("/analyze-local", summary="Speaker Diarization (Local File)")
async def analyze_local_diarization(request: AnalyzeLocalRequest):
    """Analyze a local audio file and return speaker-timestamped segments."""
    return await diarization_client.analyze_local_file(request.file_path)
