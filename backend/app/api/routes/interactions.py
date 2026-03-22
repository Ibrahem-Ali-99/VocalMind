# Interactions endpoints — list and detail views for the Session Inspector.
# Performance: violation flags via LEFT JOIN subquery (no N+1 loops).

from collections import Counter
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import select, func
from uuid import UUID
import httpx
import io
import wave

from app.api.deps import SessionDep
from app.core.config import settings
from app.core.emotion_fusion import fuse_emotion_signals
from app.llm_trigger.service import evaluate_interaction_triggers
from app.models.interaction import Interaction
from app.models.interaction_score import InteractionScore
from app.models.organization import Organization
from app.models.utterance import Utterance
from app.models.emotion_event import EmotionEvent
from app.models.policy import CompanyPolicy, PolicyCompliance
from app.models.user import User as UserModel

router = APIRouter()


def _compact_distribution(labels: list[str]) -> list[dict[str, float | int | str]]:
    if not labels:
        return []

    counts = Counter(labels)
    total = len(labels)
    rows = [
        {
            "emotion": emotion,
            "count": count,
            "pct": round((count / total) * 100, 2),
        }
        for emotion, count in counts.items()
    ]
    rows.sort(key=lambda item: (-int(item["count"]), str(item["emotion"])))
    return rows


def _build_emotion_comparison_payload(utterances_rows: list[Utterance]) -> dict:
    acoustic_labels: list[str] = []
    text_labels: list[str] = []
    fused_labels: list[str] = []

    acoustic_text_agreements = 0
    fused_acoustic_agreements = 0
    fused_text_agreements = 0

    for u in utterances_rows:
        acoustic_emotion = u.emotion or "neutral"
        acoustic_confidence = u.emotion_confidence or 0.0
        fused = fuse_emotion_signals(
            text=u.text or "",
            acoustic_emotion=acoustic_emotion,
            acoustic_confidence=acoustic_confidence,
        )

        acoustic_labels.append(acoustic_emotion)
        text_labels.append(fused.text_emotion)
        fused_labels.append(fused.emotion)

        if fused.text_emotion == acoustic_emotion:
            acoustic_text_agreements += 1
        if fused.emotion == acoustic_emotion:
            fused_acoustic_agreements += 1
        if fused.emotion == fused.text_emotion:
            fused_text_agreements += 1

    total = len(utterances_rows)
    if total == 0:
        return {
            "totalUtterances": 0,
            "distributions": {
                "acoustic": [],
                "text": [],
                "fused": [],
            },
            "quality": {
                "acousticTextAgreementRate": 0.0,
                "fusedMatchesAcousticRate": 0.0,
                "fusedMatchesTextRate": 0.0,
                "disagreementCount": 0,
            },
        }

    disagreement_count = total - acoustic_text_agreements
    return {
        "totalUtterances": total,
        "distributions": {
            "acoustic": _compact_distribution(acoustic_labels),
            "text": _compact_distribution(text_labels),
            "fused": _compact_distribution(fused_labels),
        },
        "quality": {
            "acousticTextAgreementRate": round((acoustic_text_agreements / total) * 100, 2),
            "fusedMatchesAcousticRate": round((fused_acoustic_agreements / total) * 100, 2),
            "fusedMatchesTextRate": round((fused_text_agreements / total) * 100, 2),
            "disagreementCount": disagreement_count,
        },
    }


def _trim_quote(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) <= 220:
        return text
    return text[:217].rstrip() + "..."


def _build_evidence_payload(
    utterances_rows: list[Utterance],
    events_rows: list[EmotionEvent],
    viol_rows: list,
) -> dict:
    """
    Deprecated: Verbatim quotes and raw citations are now replaced by
    narrative reasoning in the LLM Trigger evaluation cards.
    """
    return {
        "emotionShiftQuotes": [],
        "processAdherenceQuotes": [],
        "nliPolicyQuotes": [],
        "citations": [],
    }


def _map_llm_trigger_report(report) -> dict:
    def _citation_to_dict(citation) -> dict:
        return {
            "source": citation.source,
            "speaker": citation.speaker,
            "quote": citation.quote,
            "utteranceIndex": citation.utterance_index,
        }

    return {
        "available": True,
        "interactionId": str(report.interaction_id),
        "emotionShift": {
            "isDissonanceDetected": report.emotion_shift.is_dissonance_detected,
            "dissonanceType": report.emotion_shift.dissonance_type,
            "rootCause": report.emotion_shift.root_cause,
            "counterfactualCorrection": report.emotion_shift.counterfactual_correction,
            "evidenceQuotes": report.emotion_shift.evidence_quotes,
            "citations": [_citation_to_dict(c) for c in report.emotion_shift.citations],
        },
        "processAdherence": {
            "detectedTopic": report.process_adherence.detected_topic,
            "isResolved": report.process_adherence.is_resolved,
            "efficiencyScore": report.process_adherence.efficiency_score,
            "justification": report.process_adherence.justification,
            "missingSopSteps": report.process_adherence.missing_sop_steps,
            "evidenceQuotes": report.process_adherence.evidence_quotes,
            "citations": [_citation_to_dict(c) for c in report.process_adherence.citations],
        },
        "nliPolicy": {
            "nliCategory": report.nli_policy.nli_category,
            "justification": report.nli_policy.justification,
            "evidenceQuotes": report.nli_policy.evidence_quotes,
            "citations": [_citation_to_dict(c) for c in report.nli_policy.citations],
        },
        "derived": {
            "customerText": report.derived_customer_text,
            "acousticEmotion": report.derived_acoustic_emotion,
            "fusedEmotion": report.derived_fused_emotion,
            "agentStatement": report.derived_agent_statement,
        },
    }


async def _resolve_llm_org_filter(
    session: SessionDep,
    interaction_id: UUID,
    llm_org_filter: str | None,
) -> str | None:
    if llm_org_filter and llm_org_filter.strip():
        return llm_org_filter.strip()

    stmt = (
        select(Organization.slug)
        .join(Interaction, Interaction.organization_id == Organization.id)
        .where(Interaction.id == interaction_id)
    )
    result = await session.exec(stmt)
    org_slug = result.first()
    return org_slug.strip() if isinstance(org_slug, str) and org_slug.strip() else None


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
async def get_interaction_detail(
    interaction_id: UUID,
    session: SessionDep,
    include_llm_triggers: bool = False,
    llm_org_filter: str | None = None,
    llm_force_rerun: bool = False,
):
    """Get a single interaction with utterances, emotion events, and policy violations."""

    # ── Interaction + score ──
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
    result = await session.exec(stmt)
    row = result.first()

    if not row:
        raise HTTPException(status_code=404, detail="Interaction not found")

    mins = row.duration_seconds // 60
    secs = row.duration_seconds % 60

    # ── Utterances ──
    utt_result = await session.exec(
        select(Utterance)
        .where(Utterance.interaction_id == interaction_id)
        .order_by(Utterance.start_time_seconds)
    )
    utterances_rows = utt_result.all()

    utterances = []
    for u in utterances_rows:
        acoustic_emotion = u.emotion or "neutral"
        acoustic_confidence = u.emotion_confidence or 0.0
        fused = fuse_emotion_signals(
            text=u.text or "",
            acoustic_emotion=acoustic_emotion,
            acoustic_confidence=acoustic_confidence,
        )
        utterances.append(
            {
                "id": str(u.id),
                "interactionId": str(u.interaction_id),
                "speaker": u.speaker_role.value if u.speaker_role else "unknown",
                "text": u.text or "",
                "startTime": u.start_time_seconds,
                "endTime": u.end_time_seconds,
                "timestamp": f"{int(u.start_time_seconds) // 60:02d}:{int(u.start_time_seconds) % 60:02d}",
                "emotion": acoustic_emotion,
                "confidence": acoustic_confidence,
                "textEmotion": fused.text_emotion,
                "textConfidence": fused.text_confidence,
                "fusedEmotion": fused.emotion,
                "fusedConfidence": fused.confidence,
                "fusionModel": fused.model,
            }
        )

    # ── Emotion Events ──
    event_result = await session.exec(
        select(EmotionEvent)
        .where(EmotionEvent.interaction_id == interaction_id)
        .order_by(EmotionEvent.jump_to_seconds)
    )
    events_rows = event_result.all()

    emotion_events = [
        {
            "id": str(e.id),
            "interactionId": str(e.interaction_id),
            "previousEmotion": e.previous_emotion or "neutral",
            "newEmotion": e.new_emotion,
            "fromEmotion": e.previous_emotion or "neutral",
            "toEmotion": e.new_emotion,
            "jumpToSeconds": e.jump_to_seconds,
            "timestamp": f"{int(e.jump_to_seconds) // 60:02d}:{int(e.jump_to_seconds) % 60:02d}",
            "confidenceScore": e.confidence_score or 0,
            "delta": e.emotion_delta or 0,
            "speaker": e.speaker_role.value if e.speaker_role else "customer",
            "llmJustification": e.llm_justification or "",
            "justification": e.llm_justification or "",
        }
        for e in events_rows
    ]

    # ── Policy Violations ──
    viol_stmt = (
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
    viol_result = await session.exec(viol_stmt)
    viol_rows = viol_result.all()

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

    emotion_comparison = _build_emotion_comparison_payload(utterances_rows)
    emotion_comparison["evidence"] = _build_evidence_payload(
        utterances_rows=utterances_rows,
        events_rows=events_rows,
        viol_rows=viol_rows,
    )

    llm_triggers = None
    if include_llm_triggers:
        try:
            resolved_org_filter = await _resolve_llm_org_filter(
                session=session,
                interaction_id=interaction_id,
                llm_org_filter=llm_org_filter,
            )
            report = await evaluate_interaction_triggers(
                session=session,
                interaction_id=interaction_id,
                org_filter=resolved_org_filter,
            )
            llm_triggers = _map_llm_trigger_report(report)
            llm_triggers["orgFilter"] = resolved_org_filter
            llm_triggers["forcedRerun"] = llm_force_rerun
        except Exception as exc:
            llm_triggers = {
                "available": False,
                "error": str(exc),
            }

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
        "emotionComparison": emotion_comparison,
        "llmTriggers": llm_triggers,
        "emotionEvents": emotion_events,
        "policyViolations": policy_violations,
    }


@router.get("/{interaction_id}/emotion-comparison")
async def get_interaction_emotion_comparison(interaction_id: UUID, session: SessionDep):
    """Return compact acoustic vs text vs fused emotion comparison for manager panel."""
    interaction_result = await session.exec(
        select(Interaction.id).where(Interaction.id == interaction_id)
    )
    if not interaction_result.first():
        raise HTTPException(status_code=404, detail="Interaction not found")

    utt_result = await session.exec(
        select(Utterance)
        .where(Utterance.interaction_id == interaction_id)
        .order_by(Utterance.start_time_seconds)
    )
    utterances_rows = utt_result.all()

    payload = _build_emotion_comparison_payload(utterances_rows)
    payload["interactionId"] = str(interaction_id)
    return payload


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
