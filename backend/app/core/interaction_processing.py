from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.routes.full.service import full_client
from app.api.routes.transcription.service import transcription_client
from app.core.config import settings
from app.core.database import engine
from app.core.emotion_fusion import build_deterministic_emotion_analysis
from app.core.inference_contracts import (
    audio_content_type,
    build_local_full_response,
    is_supported_audio_filename,
)
from app.llm_trigger.service import evaluate_interaction_triggers
from app.models.emotion_event import EmotionEvent
from app.models.enums import JobStage, JobStatus, ProcessingStatus, SpeakerRole
from app.models.interaction import Interaction
from app.models.interaction_score import InteractionScore
from app.models.llm_trigger_cache import InteractionLLMTriggerCache
from app.models.organization import Organization
from app.models.policy import CompanyPolicy, OrganizationPolicy, PolicyCompliance
from app.models.processing import ProcessingJob
from app.models.transcript import Transcript
from app.models.utterance import Utterance


logger = logging.getLogger(__name__)

_processing_queue: asyncio.Queue[UUID | None] | None = None
_worker_task: asyncio.Task[None] | None = None

STAGE_ORDER: tuple[JobStage, ...] = (
    JobStage.diarization,
    JobStage.stt,
    JobStage.emotion,
    JobStage.reasoning,
    JobStage.scoring,
    JobStage.rag_eval,
)


def _deterministic_emotion_analysis(text: str) -> dict:
    return build_deterministic_emotion_analysis(text)


def _storage_root() -> Path:
    return Path(settings.LOCAL_AUDIO_STORAGE_DIR)


def _sanitize_filename(filename: str) -> str:
    cleaned = Path(filename or "").name.strip()
    return cleaned or f"audio_{datetime.now(timezone.utc).timestamp():.0f}.wav"


def _normalize_lookup_text(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


async def _resolve_policy_record_for_report(
    session: AsyncSession,
    organization_id: UUID,
    report,
) -> CompanyPolicy | None:
    policies_result = await session.exec(
        select(CompanyPolicy)
        .join(OrganizationPolicy, OrganizationPolicy.policy_id == CompanyPolicy.id)
        .where(
            OrganizationPolicy.organization_id == organization_id,
            OrganizationPolicy.is_active.is_(True),
            CompanyPolicy.is_active.is_(True),
        )
    )
    policies = list(policies_result.all())
    if not policies:
        return None

    policy_quotes = [
        _normalize_lookup_text(citation.quote)
        for citation in report.nli_policy.citations
        if citation.source == "policy" and (citation.quote or "").strip()
    ]
    reference_hints: list[str] = []
    for claim in report.explainability.claim_provenance:
        if not claim.retrieved_policy:
            continue
        reference_hints.append(_normalize_lookup_text(claim.retrieved_policy.reference))
        if claim.retrieved_policy.provenance:
            reference_hints.append(_normalize_lookup_text(claim.retrieved_policy.provenance))

    best_policy: CompanyPolicy | None = None
    best_score = -1
    for policy in policies:
        title_norm = _normalize_lookup_text(policy.policy_title)
        text_norm = _normalize_lookup_text(policy.policy_text)
        title_tokens = set(title_norm.split())
        score = 0

        for quote in policy_quotes:
            if quote and quote in text_norm:
                score += 8

        for hint in reference_hints:
            if not hint:
                continue
            hint_tokens = set(hint.split())
            score += len(hint_tokens.intersection(title_tokens)) * 2
            if hint in title_norm or title_norm in hint:
                score += 4

        if score > best_score:
            best_score = score
            best_policy = policy

    return best_policy or policies[0]


def build_audio_storage_path(organization_slug: str, interaction_id: UUID, filename: str) -> Path:
    return _storage_root() / organization_slug / str(interaction_id) / _sanitize_filename(filename)


async def save_audio_upload(organization_slug: str, interaction_id: UUID, filename: str, content: bytes) -> Path:
    target_path = build_audio_storage_path(organization_slug, interaction_id, filename)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(content)
    return target_path.resolve()


def _speaker_role_from_label(label: str | None, index: int) -> SpeakerRole:
    normalized = (label or "").strip().lower()
    if "agent" in normalized or normalized in {"speaker_1", "spk1", "s1"}:
        return SpeakerRole.agent
    if "customer" in normalized or normalized in {"speaker_0", "spk0", "s0"}:
        return SpeakerRole.customer
    if normalized.endswith("00") or normalized.endswith("_0"):
        return SpeakerRole.customer
    if normalized.endswith("01") or normalized.endswith("_1"):
        return SpeakerRole.agent
    return SpeakerRole.customer if index % 2 == 0 else SpeakerRole.agent


async def create_processing_jobs(session: AsyncSession, interaction_id: UUID) -> None:
    existing_result = await session.exec(
        select(ProcessingJob.stage).where(ProcessingJob.interaction_id == interaction_id)
    )
    existing_stages = set(existing_result.all())

    for stage in STAGE_ORDER:
        if stage in existing_stages:
            continue
        session.add(ProcessingJob(interaction_id=interaction_id, stage=stage, status=JobStatus.pending))


async def _set_job_status(
    session: AsyncSession,
    interaction_id: UUID,
    stage: JobStage,
    status: JobStatus,
    error_message: str | None = None,
) -> None:
    result = await session.exec(
        select(ProcessingJob).where(
            ProcessingJob.interaction_id == interaction_id,
            ProcessingJob.stage == stage,
        )
    )
    job = result.first()
    if not job:
        job = ProcessingJob(interaction_id=interaction_id, stage=stage)
        session.add(job)

    job.status = status
    if status == JobStatus.running and not job.started_at:
        job.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    if status == JobStatus.completed:
        job.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        job.error_message = None
    if status == JobStatus.failed:
        job.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        job.error_message = error_message

    await session.commit()


async def _set_interaction_status(session: AsyncSession, interaction_id: UUID, status: ProcessingStatus) -> None:
    interaction = await session.get(Interaction, interaction_id)
    if not interaction:
        return
    interaction.processing_status = status
    session.add(interaction)
    await session.commit()


async def enqueue_interaction_processing(interaction_id: UUID) -> None:
    if _processing_queue is None:
        raise RuntimeError("Processing worker has not been started")
    await _processing_queue.put(interaction_id)


async def start_processing_worker() -> None:
    global _worker_task, _processing_queue
    if _worker_task and not _worker_task.done():
        return
    _processing_queue = asyncio.Queue()
    _worker_task = asyncio.create_task(_worker_loop(), name="interaction-processing-worker")


async def stop_processing_worker() -> None:
    global _worker_task, _processing_queue
    if not _worker_task or _processing_queue is None:
        return
    await _processing_queue.put(None)
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    finally:
        _worker_task = None
        _processing_queue = None


async def _worker_loop() -> None:
    if _processing_queue is None:
        return
    while True:
        interaction_id = await _processing_queue.get()
        try:
            if interaction_id is None:
                return
            await process_interaction(interaction_id)
        except Exception:
            logger.exception("Interaction processing worker failed for %s", interaction_id)
            if interaction_id is not None:
                await mark_interaction_failed(interaction_id, "Interaction processing failed")
        finally:
            _processing_queue.task_done()


async def interaction_has_active_jobs(session: AsyncSession, interaction_id: UUID) -> bool:
    jobs_result = await session.exec(
        select(ProcessingJob).where(ProcessingJob.interaction_id == interaction_id)
    )
    for job in jobs_result.all():
        if job.status in {JobStatus.pending, JobStatus.running}:
            return True
    return False


async def reset_interaction_for_reprocess(session: AsyncSession, interaction_id: UUID) -> None:
    """Reset full processing artifacts and job stages for a clean re-run."""
    await session.exec(delete(EmotionEvent).where(EmotionEvent.interaction_id == interaction_id))
    await session.exec(delete(Utterance).where(Utterance.interaction_id == interaction_id))
    await session.exec(delete(PolicyCompliance).where(PolicyCompliance.interaction_id == interaction_id))
    await session.exec(delete(InteractionScore).where(InteractionScore.interaction_id == interaction_id))
    await session.exec(delete(InteractionLLMTriggerCache).where(InteractionLLMTriggerCache.interaction_id == interaction_id))

    transcript_result = await session.exec(
        select(Transcript).where(Transcript.interaction_id == interaction_id)
    )
    transcript = transcript_result.first()
    if transcript:
        transcript.full_text = ""
        transcript.overall_confidence = None
        session.add(transcript)

    jobs_result = await session.exec(
        select(ProcessingJob).where(ProcessingJob.interaction_id == interaction_id)
    )
    jobs = jobs_result.all()
    existing_stages = {job.stage for job in jobs}

    for stage in STAGE_ORDER:
        if stage in existing_stages:
            continue
        session.add(ProcessingJob(interaction_id=interaction_id, stage=stage, status=JobStatus.pending))

    for job in jobs:
        job.status = JobStatus.pending
        job.error_message = None
        job.started_at = None
        job.completed_at = None
        job.retry_count = 0
        session.add(job)

    interaction = await session.get(Interaction, interaction_id)
    if interaction:
        interaction.processing_status = ProcessingStatus.pending
        session.add(interaction)

    await session.commit()


async def process_interaction(interaction_id: UUID) -> None:
    async with AsyncSession(engine, expire_on_commit=False) as session:
        interaction = await session.get(Interaction, interaction_id)
        if not interaction:
            logger.warning("Skipping missing interaction %s", interaction_id)
            return

        await _set_job_status(session, interaction_id, JobStage.diarization, JobStatus.running)
        await _set_job_status(session, interaction_id, JobStage.stt, JobStatus.running)
        await _set_job_status(session, interaction_id, JobStage.emotion, JobStatus.running)

        audio_path = Path(interaction.audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {interaction.audio_file_path}")

        audio_bytes = audio_path.read_bytes()
        if not is_supported_audio_filename(audio_path.name):
            raise ValueError(f"Unsupported audio file: {audio_path.name}")

        try:
            analysis = await full_client.analyze_bytes(
                audio_bytes,
                audio_path.name,
                audio_content_type(audio_path.name),
            )
        except Exception:
            # Keep processing alive when local inference services are temporarily unavailable.
            if not settings.IS_LOCAL:
                raise
            logger.exception(
                "Full analysis failed for %s; using transcription-only fallback",
                interaction_id,
            )
            transcription = await transcription_client.analyze_bytes(
                audio_bytes,
                audio_path.name,
                audio_content_type(audio_path.name),
            )
            analysis = build_local_full_response(
                transcription,
                _deterministic_emotion_analysis(transcription.get("text") or ""),
            )

        await session.exec(delete(EmotionEvent).where(EmotionEvent.interaction_id == interaction_id))
        await session.exec(delete(Utterance).where(Utterance.interaction_id == interaction_id))
        await session.exec(delete(PolicyCompliance).where(PolicyCompliance.interaction_id == interaction_id))
        await session.exec(delete(InteractionScore).where(InteractionScore.interaction_id == interaction_id))
        await session.exec(delete(InteractionLLMTriggerCache).where(InteractionLLMTriggerCache.interaction_id == interaction_id))

        transcript_result = await session.exec(
            select(Transcript).where(Transcript.interaction_id == interaction_id)
        )
        transcript = transcript_result.first() or Transcript(interaction_id=interaction_id)

        segments = analysis.get("segments", [])
        transcript_text = (analysis.get("text") or "").strip()
        if not transcript_text:
            transcript_text = " ".join((segment.get("text") or "").strip() for segment in segments).strip()

        transcript.full_text = transcript_text
        transcript.overall_confidence = float(analysis.get("top_score") or 0.0)
        session.add(transcript)
        await session.flush()

        utterances: list[Utterance] = []
        for index, segment in enumerate(segments):
            speaker_role = _speaker_role_from_label(segment.get("speaker"), index)
            emotion_scores = segment.get("emotion_scores") or []
            emotion_confidence = 0.0
            if emotion_scores:
                emotion_confidence = float(emotion_scores[0].get("score") or 0.0)

            utterance = Utterance(
                interaction_id=interaction_id,
                transcript_id=transcript.id,
                speaker_role=speaker_role,
                user_id=interaction.agent_id if speaker_role == SpeakerRole.agent else None,
                sequence_index=index,
                start_time_seconds=float(segment.get("start") or 0.0),
                end_time_seconds=float(segment.get("end") or 0.0),
                text=(segment.get("text") or "").strip(),
                emotion=(segment.get("emotion") or "neutral").strip() or "neutral",
                emotion_confidence=emotion_confidence,
            )
            utterances.append(utterance)
            session.add(utterance)

        await session.flush()

        previous_emotion: str | None = None
        for utterance in utterances:
            current_emotion = utterance.emotion or "neutral"
            if previous_emotion is None:
                previous_emotion = current_emotion
                continue
            if current_emotion == previous_emotion:
                continue

            session.add(
                EmotionEvent(
                    interaction_id=interaction_id,
                    utterance_id=utterance.id,
                    previous_emotion=previous_emotion,
                    new_emotion=current_emotion,
                    emotion_delta=1.0,
                    speaker_role=utterance.speaker_role or SpeakerRole.customer,
                    llm_justification=f"Emotion changed from {previous_emotion} to {current_emotion}.",
                    jump_to_seconds=utterance.start_time_seconds,
                    confidence_score=utterance.emotion_confidence,
                )
            )
            previous_emotion = current_emotion

        organization_result = await session.exec(
            select(Organization).where(Organization.id == interaction.organization_id)
        )
        organization = organization_result.first()

        report = None
        if organization:
            try:
                report = await evaluate_interaction_triggers(
                    session=session,
                    interaction_id=interaction_id,
                    org_filter=organization.slug,
                    force_rerun=True,
                )
            except Exception:
                logger.exception("LLM trigger evaluation failed for interaction %s", interaction_id)

        if report:
            policy = await _resolve_policy_record_for_report(
                session=session,
                organization_id=interaction.organization_id,
                report=report,
            )
            compliance_score = max(0.0, min(1.0, float(report.process_adherence.efficiency_score) / 10.0))
            policy_alignment = report.nli_policy.policy_alignment_score
            if policy_alignment is None:
                policy_alignment = 1.0 if report.nli_policy.nli_category in {"Entailment", "Benign Deviation"} else 0.35
            policy_score = max(0.0, min(1.0, (compliance_score * 0.55) + (float(policy_alignment) * 0.45)))

            if report.emotion_shift.is_dissonance_detected:
                empathy_score = 0.55
            elif report.emotion_shift.insufficient_evidence:
                empathy_score = 0.82
            else:
                empathy_score = 0.95

            if report.process_adherence.is_resolved:
                resolution_score = 0.94 if not report.process_adherence.missing_sop_steps else 0.84
            else:
                resolution_score = 0.42

            overall_score = round(
                (empathy_score * 0.3) + (policy_score * 0.4) + (resolution_score * 0.3),
                4,
            )
            evidence_text = "; ".join(
                quote for quote in (
                    report.process_adherence.evidence_quotes[:1] + report.nli_policy.evidence_quotes[:1]
                )
                if quote
            ) or transcript_text[:250]

            if policy:
                is_compliant = report.nli_policy.nli_category in {"Entailment", "Benign Deviation"}
                session.add(
                    PolicyCompliance(
                        interaction_id=interaction_id,
                        policy_id=policy.id,
                        is_compliant=is_compliant,
                        compliance_score=compliance_score,
                        llm_reasoning=report.nli_policy.justification or report.process_adherence.justification,
                        evidence_text=evidence_text,
                        retrieved_policy_text=policy.policy_text,
                    )
                )

            session.add(
                InteractionScore(
                    interaction_id=interaction_id,
                    overall_score=overall_score,
                    empathy_score=empathy_score,
                    policy_score=policy_score,
                    resolution_score=resolution_score,
                    total_silence_seconds=0.0,
                    avg_response_time_seconds=1.0,
                    was_resolved=report.process_adherence.is_resolved,
                )
            )
        else:
            session.add(
                InteractionScore(
                    interaction_id=interaction_id,
                    overall_score=0.7,
                    empathy_score=0.7,
                    policy_score=0.7,
                    resolution_score=0.7,
                    total_silence_seconds=0.0,
                    avg_response_time_seconds=1.0,
                    was_resolved=False,
                )
            )

        interaction.language_detected = (analysis.get("language") or interaction.language_detected or "en")[:10]
        interaction.duration_seconds = int(max((float(seg.get("end") or 0.0) for seg in segments), default=interaction.duration_seconds or 0))
        interaction.processing_status = ProcessingStatus.completed
        session.add(interaction)

        await session.commit()

        for stage in STAGE_ORDER:
            await _set_job_status(session, interaction_id, stage, JobStatus.completed)


async def mark_interaction_failed(interaction_id: UUID, error_message: str) -> None:
    async with AsyncSession(engine, expire_on_commit=False) as session:
        interaction = await session.get(Interaction, interaction_id)
        if interaction:
            interaction.processing_status = ProcessingStatus.failed
            session.add(interaction)
            await session.commit()

        for stage in STAGE_ORDER:
            await _set_job_status(session, interaction_id, stage, JobStatus.failed, error_message)
