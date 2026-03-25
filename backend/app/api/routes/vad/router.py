from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.routes.vad.service import vad_client
from app.core.inference_contracts import is_supported_audio_filename


router = APIRouter()


class AnalyzeLocalRequest(BaseModel):
    file_path: str


@router.post("/analyze", summary="Voice Activity Detection (Upload)")
async def analyze_vad(file: UploadFile = File(...)):
    if not is_supported_audio_filename(file.filename):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")
    return await vad_client.analyze_audio(file)


@router.post("/analyze-local", summary="Voice Activity Detection (Local File)")
async def analyze_local_vad(request: AnalyzeLocalRequest):
    if not is_supported_audio_filename(request.file_path):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")
    return await vad_client.analyze_local_file(request.file_path)
