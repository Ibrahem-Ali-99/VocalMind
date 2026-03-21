# Interactions endpoints — list and detail views for the Session Inspector.
# Performance: violation flags via LEFT JOIN subquery (no N+1 loops).

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import select, func
from uuid import UUID
import httpx
import io
import wave
import asyncio
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.database import engine

from app.api.deps import SessionDep
from app.core.config import settings
from app.models.interaction import Interaction
from app.models.interaction_score import InteractionScore
from app.models.utterance import Utterance
from app.models.emotion_event import EmotionEvent
from app.models.policy import CompanyPolicy, PolicyCompliance
from app.models.user import User as UserModel

router = APIRouter()


@router.get("")
async def list_interactions(session: SessionDep):
    """List all interactions with agent name and scores."""

    # Subquery: count violations per interaction (eliminates N+1)
    violation_subq = (
        select(
            PolicyCompliance.interaction_id,
            func.count(PolicyCompliance.id).label("viol_count"),
        )
        .where(PolicyCompliance.is_compliant == False)  # noqa: E712
        .group_by(PolicyCompliance.interaction_id)
        .subquery()
    )

    stmt = (
        select(
            Interaction.id,
            Interaction.agent_id,
            UserModel.name.label("agent_name"),
            Interaction.interaction_date,
            Interaction.duration_seconds,
            Interaction.language_detected,
            Interaction.has_overlap,
            Interaction.processing_status,
            Interaction.audio_file_path,
            InteractionScore.overall_score,
            InteractionScore.empathy_score,
            InteractionScore.policy_score,
            InteractionScore.resolution_score,
            InteractionScore.was_resolved,
            InteractionScore.avg_response_time_seconds,
            func.coalesce(violation_subq.c.viol_count, 0).label("viol_count"),
        )
        .join(UserModel, UserModel.id == Interaction.agent_id)
        .outerjoin(InteractionScore, InteractionScore.interaction_id == Interaction.id)
        .outerjoin(violation_subq, violation_subq.c.interaction_id == Interaction.id)
        .order_by(Interaction.interaction_date.desc())
    )
    result = await session.exec(stmt)
    rows = result.all()

    interactions = []
    for row in rows:
        mins = row.duration_seconds // 60
        secs = row.duration_seconds % 60

        interactions.append({
            "id": str(row.id),
            "agentName": row.agent_name,
            "agentId": str(row.agent_id),
            "date": row.interaction_date.strftime("%Y-%m-%d") if row.interaction_date else "",
            "time": row.interaction_date.strftime("%I:%M %p") if row.interaction_date else "",
            "duration": f"{mins}:{secs:02d}",
            "language": row.language_detected or "Unknown",
            "overallScore": round(row.overall_score * 10, 0) if row.overall_score else 0,
            "empathyScore": round(row.empathy_score * 10, 0) if row.empathy_score else 0,
            "policyScore": round(row.policy_score * 10, 0) if row.policy_score else 0,
            "resolutionScore": round(row.resolution_score * 10, 0) if row.resolution_score else 0,
            "resolved": row.was_resolved or False,
            "hasViolation": row.viol_count > 0,
            "hasOverlap": row.has_overlap,
            "responseTime": f"{row.avg_response_time_seconds:.1f}s" if row.avg_response_time_seconds else "N/A",
            "status": str(row.processing_status.value) if row.processing_status else "pending",
            "audioFilePath": row.audio_file_path or None,
        })

    return interactions


@router.get("/{interaction_id}")
async def get_interaction_detail(interaction_id: UUID):
    """Get a single interaction with utterances, emotion events, and policy violations concurrently."""

    async def _fetch_interaction():
        async with AsyncSession(engine) as s:
            stmt = (
                select(
                    Interaction.id,
                    Interaction.agent_id,
                    UserModel.name.label("agent_name"),
                    Interaction.interaction_date,
                    Interaction.duration_seconds,
                    Interaction.language_detected,
                    Interaction.has_overlap,
                    Interaction.processing_status,
                    Interaction.audio_file_path,
                    InteractionScore.overall_score,
                    InteractionScore.empathy_score,
                    InteractionScore.policy_score,
                    InteractionScore.resolution_score,
                    InteractionScore.was_resolved,
                    InteractionScore.avg_response_time_seconds,
                )
                .join(UserModel, UserModel.id == Interaction.agent_id)
                .outerjoin(InteractionScore, InteractionScore.interaction_id == Interaction.id)
                .where(Interaction.id == interaction_id)
            )
            return (await s.exec(stmt)).first()

    async def _fetch_utterances():
        async with AsyncSession(engine) as s:
            stmt = (
                select(Utterance)
                .where(Utterance.interaction_id == interaction_id)
                .order_by(Utterance.start_time_seconds)
            )
            return (await s.exec(stmt)).all()

    async def _fetch_events():
        async with AsyncSession(engine) as s:
            stmt = (
                select(EmotionEvent)
                .where(EmotionEvent.interaction_id == interaction_id)
                .order_by(EmotionEvent.jump_to_seconds)
            )
            return (await s.exec(stmt)).all()

    async def _fetch_violations():
        async with AsyncSession(engine) as s:
            stmt = (
                select(
                    PolicyCompliance.id,
                    PolicyCompliance.interaction_id,
                    CompanyPolicy.policy_title,
                    CompanyPolicy.policy_category,
                    PolicyCompliance.llm_reasoning,
                    PolicyCompliance.compliance_score,
                    PolicyCompliance.evidence_text,
                )
                .join(CompanyPolicy, CompanyPolicy.id == PolicyCompliance.policy_id)
                .where(
                    PolicyCompliance.interaction_id == interaction_id,
                    PolicyCompliance.is_compliant == False,  # noqa: E712
                )
            )
            return (await s.exec(stmt)).all()

    # Execute all 4 queries concurrently
    row, utterances_rows, events_rows, viol_rows = await asyncio.gather(
        _fetch_interaction(),
        _fetch_utterances(),
        _fetch_events(),
        _fetch_violations()
    )

    if not row:
        raise HTTPException(status_code=404, detail="Interaction not found")

    mins = row.duration_seconds // 60
    secs = row.duration_seconds % 60

    policy_violations = [
        {
            "id": str(v.id),
            "interactionId": str(v.interaction_id),
            "policyName": v.policy_title,
            "policyTitle": v.policy_title,
            "category": v.policy_category,
            "description": v.evidence_text or "",
            "reasoning": v.llm_reasoning or "",
            "severity": "high" if v.compliance_score < 0.3 else ("medium" if v.compliance_score < 0.6 else "low"),
            "score": round(v.compliance_score * 100, 0),
        }
        for v in viol_rows
    ]

    return {
        "interaction": {
            "id": str(row.id),
            "agentName": row.agent_name,
            "agentId": str(row.agent_id),
            "date": row.interaction_date.strftime("%Y-%m-%d") if row.interaction_date else "",
            "time": row.interaction_date.strftime("%I:%M %p") if row.interaction_date else "",
            "duration": f"{mins}:{secs:02d}",
            "language": row.language_detected or "Unknown",
            "overallScore": round(row.overall_score * 10, 0) if row.overall_score else 0,
            "empathyScore": round(row.empathy_score * 10, 0) if row.empathy_score else 0,
            "policyScore": round(row.policy_score * 10, 0) if row.policy_score else 0,
            "resolutionScore": round(row.resolution_score * 10, 0) if row.resolution_score else 0,
            "resolved": row.was_resolved or False,
            "hasViolation": len(policy_violations) > 0,
            "hasOverlap": row.has_overlap,
            "responseTime": f"{row.avg_response_time_seconds:.1f}s" if row.avg_response_time_seconds else "N/A",
            "status": str(row.processing_status.value) if row.processing_status else "pending",
            "audioFilePath": row.audio_file_path or None,
        },
        "utterances": utterances,
        "emotionEvents": emotion_events,
        "policyViolations": policy_violations,
    }


def generate_dummy_wav(duration_seconds: int) -> bytes:
    """Generate a dummy silent WAV file in memory."""
    buf = io.BytesIO()
    sample_rate = 8000
    n_samples = duration_seconds * sample_rate
    
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b'\x00\x00' * n_samples)
        
    return buf.getvalue()


@router.get("/{interaction_id}/audio")
async def get_interaction_audio(interaction_id: UUID, session: SessionDep):
    """Stream the audio file for an interaction from Supabase Storage."""

    # Get the audio path and duration
    result = await session.exec(
        select(Interaction.audio_file_path, Interaction.duration_seconds).where(Interaction.id == interaction_id)
    )
    row = result.first()

    if not row:
        raise HTTPException(status_code=404, detail="Interaction not found")

    audio_path, duration = row

    # Fallback to dummy generated audio if Supabase is not configured yet
    if not settings.SUPABASE_URL or not audio_path:
        dummy_wav = generate_dummy_wav(duration or 180)
        return StreamingResponse(
            iter([dummy_wav]),
            media_type="audio/wav",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(dummy_wav)),
            },
        )

    # Build Supabase Storage URL
    # audio_file_path is typically: "bucket_name/path/to/file.wav"
    # Supabase Storage URL: {SUPABASE_URL}/storage/v1/object/{path}
    storage_url = f"{settings.SUPABASE_URL}/storage/v1/object/{audio_path}"

    # Stream from Supabase Storage
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                storage_url,
                headers={
                    "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
                    "apikey": settings.SUPABASE_SERVICE_KEY,
                },
                timeout=30.0,
            )

        if response.status_code != 200:
            # Fallback to dummy if backend storage fails (e.g., file deleted)
            dummy_wav = generate_dummy_wav(duration or 180)
            return StreamingResponse(
                iter([dummy_wav]),
                media_type="audio/wav",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(dummy_wav)),
                },
            )

        # Determine content type from file extension
        ext = audio_path.rsplit(".", 1)[-1].lower() if "." in audio_path else "wav"
        content_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "m4a": "audio/mp4",
        }
        content_type = content_types.get(ext, "audio/wav")

        return StreamingResponse(
            iter([response.content]),
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(response.content)),
            },
        )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Audio fetch timed out")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Cannot reach storage")
