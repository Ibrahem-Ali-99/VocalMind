# Interactions endpoints — list and detail views for the Session Inspector.
# Performance: violation flags via LEFT JOIN subquery (no N+1 loops).

from collections import Counter
from datetime import datetime, timezone
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlmodel import select, func
from typing import Any, Literal
from uuid import UUID
import httpx
import io
import wave

from app.api.deps import SessionDep, CurrentUser
from app.core.config import settings
from app.core.emotion_fusion import fuse_emotion_signals
from pathlib import Path
from app.core.inference_contracts import is_supported_audio_filename
from app.core.interaction_processing import (
    enqueue_interaction_processing,
    create_processing_jobs,
    interaction_has_active_jobs,
    reset_interaction_for_reprocess,
    save_audio_upload,
)
from app.llm_trigger.service import evaluate_interaction_triggers
from app.models.interaction import Interaction
from app.models.interaction_score import InteractionScore
from app.models.organization import Organization
from app.models.processing import ProcessingJob
from app.models.transcript import Transcript
from app.models.utterance import Utterance
from app.models.emotion_event import EmotionEvent
from app.models.policy import CompanyPolicy, PolicyCompliance
from app.models.user import User as UserModel
from app.models.enums import ProcessingStatus, UserRole

router = APIRouter()


def _interaction_scope_filters(current_user: CurrentUser) -> list:
    filters = [Interaction.organization_id == current_user.organization_id]
    if current_user.role == UserRole.agent:
        filters.append(Interaction.agent_id == current_user.id)
    return filters


class APIModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ExplainabilitySpanResponse(APIModel):
    utteranceIndex: int | None = None
    speaker: str | None = None
    quote: str
    timestamp: str | None = None
    startSeconds: float | None = None
    endSeconds: float | None = None


class ExplainabilityPolicyReferenceResponse(APIModel):
    source: Literal["policy", "sop"]
    reference: str
    clause: str
    version: str | None = None
    category: str | None = None
    provenance: str | None = None


class TriggerAttributionResponse(APIModel):
    attributionId: str
    family: Literal["emotion", "sop", "policy"]
    triggerType: str
    title: str
    verdict: str
    confidence: float | None = None
    evidenceSpan: ExplainabilitySpanResponse | None = None
    policyReference: ExplainabilityPolicyReferenceResponse | None = None
    reasoning: str
    evidenceChain: list[str]
    supportingQuotes: list[str]


class ClaimProvenanceResponse(APIModel):
    claimId: str
    claimText: str
    claimSpan: ExplainabilitySpanResponse | None = None
    retrievedPolicy: ExplainabilityPolicyReferenceResponse | None = None
    semanticSimilarity: float | None = None
    nliVerdict: str
    confidence: float | None = None
    reasoning: str
    provenance: str
    supportingQuotes: list[str]


class EvidenceAnchoredExplainabilityResponse(APIModel):
    triggerAttributions: list[TriggerAttributionResponse] = []
    claimProvenance: list[ClaimProvenanceResponse] = []


class LLMEvidenceCitationResponse(APIModel):
    source: str
    speaker: str | None = None
    quote: str
    utteranceIndex: int | None = None


class LLMEmotionShiftResponse(APIModel):
    isDissonanceDetected: bool
    dissonanceType: str
    rootCause: str
    currentCustomerEmotion: str | None = None
    currentEmotionReasoning: str | None = None
    counterfactualCorrection: str
    evidenceQuotes: list[str]
    citations: list[LLMEvidenceCitationResponse]
    insufficientEvidence: bool | None = None
    confidenceScore: float | None = None


class LLMProcessAdherenceResponse(APIModel):
    detectedTopic: str
    isResolved: bool
    efficiencyScore: int
    justification: str
    missingSopSteps: list[str]
    evidenceQuotes: list[str]
    citations: list[LLMEvidenceCitationResponse]
    insufficientEvidence: bool | None = None
    confidenceScore: float | None = None


class LLMNliPolicyResponse(APIModel):
    nliCategory: str
    justification: str
    evidenceQuotes: list[str]
    citations: list[LLMEvidenceCitationResponse]
    policyVersion: str | None = None
    policyEffectiveAt: str | None = None
    policyCategory: str | None = None
    conflictResolutionApplied: bool | None = None
    insufficientEvidence: bool | None = None
    confidenceScore: float | None = None
    policyAlignmentScore: float | None = None


class LLMDerivedSignalsResponse(APIModel):
    customerText: str
    acousticEmotion: str
    fusedEmotion: str
    agentStatement: str


class EmotionTriggerReportResponse(APIModel):
    available: bool
    error: str | None = None
    orgFilter: str | None = None
    forcedRerun: bool | None = None
    interactionId: str | None = None
    emotionShift: LLMEmotionShiftResponse | None = None
    explainability: EvidenceAnchoredExplainabilityResponse | None = None
    derived: LLMDerivedSignalsResponse | None = None


class RagComplianceReportResponse(APIModel):
    available: bool
    error: str | None = None
    orgFilter: str | None = None
    forcedRerun: bool | None = None
    interactionId: str | None = None
    processAdherence: LLMProcessAdherenceResponse | None = None
    nliPolicy: LLMNliPolicyResponse | None = None
    explainability: EvidenceAnchoredExplainabilityResponse | None = None
    policyViolations: list["PolicyViolationResponse"] | None = None


class LLMTriggerReportResponse(APIModel):
    available: bool
    error: str | None = None
    orgFilter: str | None = None
    forcedRerun: bool | None = None
    interactionId: str | None = None
    emotionShift: LLMEmotionShiftResponse | None = None
    processAdherence: LLMProcessAdherenceResponse | None = None
    nliPolicy: LLMNliPolicyResponse | None = None
    explainability: EvidenceAnchoredExplainabilityResponse | None = None
    derived: LLMDerivedSignalsResponse | None = None


class InteractionDetailSummaryResponse(APIModel):
    id: str
    agentName: str
    agentId: str
    date: str
    time: str
    duration: str
    language: str
    overallScore: float
    empathyScore: float
    policyScore: float
    resolutionScore: float
    resolved: bool
    hasViolation: bool
    hasOverlap: bool
    responseTime: str
    status: str
    audioFilePath: str | None = None


class UtteranceResponse(APIModel):
    id: str
    interactionId: str
    speaker: str
    sequenceIndex: int | None = None
    text: str
    startTime: float
    endTime: float
    timestamp: str
    emotion: str
    confidence: float
    textEmotion: str | None = None
    textConfidence: float | None = None
    fusedEmotion: str | None = None
    fusedConfidence: float | None = None
    fusionModel: str | None = None


class EmotionEventResponse(APIModel):
    id: str
    interactionId: str
    previousEmotion: str
    newEmotion: str
    fromEmotion: str
    toEmotion: str
    jumpToSeconds: float
    timestamp: str
    confidenceScore: float
    delta: float
    speaker: str
    llmJustification: str
    justification: str


class PolicyViolationResponse(APIModel):
    id: str
    interactionId: str
    policyName: str
    policyTitle: str
    category: str
    description: str
    reasoning: str
    severity: str
    score: float


class EmotionComparisonResponse(APIModel):
    totalUtterances: int
    distributions: dict[str, Any]
    quality: dict[str, Any]
    evidence: dict[str, Any] | None = None


class InteractionDetailResponse(APIModel):
    interaction: InteractionDetailSummaryResponse
    utterances: list[UtteranceResponse]
    emotionComparison: EmotionComparisonResponse
    ragCompliance: RagComplianceReportResponse | None = None
    emotionTriggers: EmotionTriggerReportResponse | None = None
    llmTriggers: LLMTriggerReportResponse | None = None
    emotionEvents: list[EmotionEventResponse]
    policyViolations: list[PolicyViolationResponse]


RagComplianceReportResponse.model_rebuild()


@router.post("")
async def create_interaction(
    session: SessionDep,
    current_user: CurrentUser,
    file: UploadFile = File(...),
    agent_id: UUID | None = Form(default=None),
):
    """Upload a real audio call, persist it locally, and enqueue processing."""
    if not is_supported_audio_filename(file.filename):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")

    agent_query = select(UserModel).where(
        UserModel.organization_id == current_user.organization_id,
        UserModel.role == UserRole.agent,
        UserModel.is_active.is_(True),
    )
    if agent_id:
        agent_query = agent_query.where(UserModel.id == agent_id)
    agent_result = await session.exec(agent_query)
    agent = agent_result.first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found for current organization")

    org_result = await session.exec(select(Organization).where(Organization.id == current_user.organization_id))
    organization = org_result.first()
    if not organization:
        raise HTTPException(status_code=404, detail="Organization not found")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    interaction = Interaction(
        organization_id=current_user.organization_id,
        agent_id=agent.id,
        uploaded_by=current_user.id,
        audio_file_path="",
        file_size_bytes=len(content),
        duration_seconds=0,
        file_format=Path(file.filename).suffix.lstrip(".").lower() or "wav",
        interaction_date=datetime.now(timezone.utc).replace(tzinfo=None),
        processing_status=ProcessingStatus.pending,
        language_detected=None,
        has_overlap=False,
        channel_count=1,
    )
    session.add(interaction)
    await session.flush()

    saved_audio_path = await save_audio_upload(organization.slug, interaction.id, file.filename, content)
    interaction.audio_file_path = str(saved_audio_path)
    session.add(interaction)

    transcript = Transcript(interaction_id=interaction.id, full_text="", overall_confidence=None)
    session.add(transcript)
    await create_processing_jobs(session, interaction.id)
    await session.commit()
    await session.refresh(interaction)

    await enqueue_interaction_processing(interaction.id)

    jobs_result = await session.exec(
        select(ProcessingJob).where(ProcessingJob.interaction_id == interaction.id)
    )
    jobs = [
        {
            "stage": job.stage.value,
            "status": job.status.value,
            "retryCount": job.retry_count,
            "errorMessage": job.error_message,
        }
        for job in jobs_result.all()
    ]

    return {
        "interactionId": str(interaction.id),
        "status": interaction.processing_status.value,
        "audioFilePath": interaction.audio_file_path,
        "agentId": str(agent.id),
        "uploadedBy": str(current_user.id),
        "processingJobs": jobs,
    }


@router.get("/{interaction_id}/processing-status")
async def get_interaction_processing_status(
    interaction_id: UUID,
    session: SessionDep,
    current_user: CurrentUser,
):
    interaction_result = await session.exec(
        select(Interaction).where(
            Interaction.id == interaction_id,
            *_interaction_scope_filters(current_user),
        )
    )
    interaction = interaction_result.first()
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    jobs_result = await session.exec(
        select(ProcessingJob).where(ProcessingJob.interaction_id == interaction_id)
    )
    jobs = [
        {
            "stage": job.stage.value,
            "status": job.status.value,
            "retryCount": job.retry_count,
            "errorMessage": job.error_message,
            "startedAt": job.started_at.isoformat() if job.started_at else None,
            "completedAt": job.completed_at.isoformat() if job.completed_at else None,
        }
        for job in jobs_result.all()
    ]
    return {
        "interactionId": str(interaction_id),
        "status": interaction.processing_status.value,
        "jobs": jobs,
    }


@router.post("/{interaction_id}/reprocess")
async def reprocess_interaction(
    interaction_id: UUID,
    session: SessionDep,
    current_user: CurrentUser,
    force: bool = False,
):
    interaction_result = await session.exec(
        select(Interaction).where(
            Interaction.id == interaction_id,
            *_interaction_scope_filters(current_user),
        )
    )
    interaction = interaction_result.first()
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    if not force and await interaction_has_active_jobs(session, interaction_id):
        raise HTTPException(status_code=409, detail="Interaction is already processing")

    await reset_interaction_for_reprocess(session, interaction_id)
    await enqueue_interaction_processing(interaction_id)

    jobs_result = await session.exec(
        select(ProcessingJob).where(ProcessingJob.interaction_id == interaction_id)
    )
    jobs = [
        {
            "stage": job.stage.value,
            "status": job.status.value,
            "retryCount": job.retry_count,
            "errorMessage": job.error_message,
        }
        for job in jobs_result.all()
    ]
    return {
        "interactionId": str(interaction_id),
        "status": ProcessingStatus.pending.value,
        "processingJobs": jobs,
        "queued": True,
        "forced": force,
    }


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


def _map_emotion_trigger_report(report) -> dict:
    def _citation_to_dict(citation) -> dict:
        return {
            "source": citation.source,
            "speaker": citation.speaker,
            "quote": citation.quote,
            "utteranceIndex": citation.utterance_index,
        }

    def _span_to_dict(span) -> dict | None:
        if not span:
            return None
        return {
            "utteranceIndex": span.utterance_index,
            "speaker": span.speaker,
            "quote": span.quote,
            "timestamp": span.timestamp,
            "startSeconds": span.start_seconds,
            "endSeconds": span.end_seconds,
        }

    def _policy_reference_to_dict(reference) -> dict | None:
        if not reference:
            return None
        return {
            "source": reference.source,
            "reference": reference.reference,
            "clause": reference.clause,
            "version": reference.version,
            "category": reference.category,
            "provenance": reference.provenance,
        }

    def _trigger_attribution_to_dict(attribution) -> dict:
        return {
            "attributionId": attribution.attribution_id,
            "family": attribution.family,
            "triggerType": attribution.trigger_type,
            "title": attribution.title,
            "verdict": attribution.verdict,
            "confidence": attribution.confidence,
            "evidenceSpan": _span_to_dict(attribution.evidence_span),
            "policyReference": _policy_reference_to_dict(attribution.policy_reference),
            "reasoning": attribution.reasoning,
            "evidenceChain": attribution.evidence_chain,
            "supportingQuotes": attribution.supporting_quotes,
        }

    explainability = {
        "triggerAttributions": [
            _trigger_attribution_to_dict(attribution)
            for attribution in report.explainability.trigger_attributions
            if attribution.family == "emotion"
        ],
        "claimProvenance": [],
    }

    return {
        "available": True,
        "interactionId": str(report.interaction_id),
        "emotionShift": {
            "isDissonanceDetected": report.emotion_shift.is_dissonance_detected,
            "dissonanceType": report.emotion_shift.dissonance_type,
            "rootCause": report.emotion_shift.root_cause,
            "currentCustomerEmotion": report.emotion_shift.current_customer_emotion,
            "currentEmotionReasoning": report.emotion_shift.current_emotion_reasoning,
            "counterfactualCorrection": report.emotion_shift.counterfactual_correction,
            "evidenceQuotes": report.emotion_shift.evidence_quotes,
            "citations": [_citation_to_dict(c) for c in report.emotion_shift.citations],
            "insufficientEvidence": report.emotion_shift.insufficient_evidence,
            "confidenceScore": report.emotion_shift.confidence_score,
        },
        "explainability": explainability,
        "derived": {
            "customerText": report.derived_customer_text,
            "acousticEmotion": report.derived_acoustic_emotion,
            "fusedEmotion": report.derived_fused_emotion,
            "agentStatement": report.derived_agent_statement,
        },
    }


def _map_rag_compliance_report(report) -> dict:
    def _citation_to_dict(citation) -> dict:
        return {
            "source": citation.source,
            "speaker": citation.speaker,
            "quote": citation.quote,
            "utteranceIndex": citation.utterance_index,
        }

    def _span_to_dict(span) -> dict | None:
        if not span:
            return None
        return {
            "utteranceIndex": span.utterance_index,
            "speaker": span.speaker,
            "quote": span.quote,
            "timestamp": span.timestamp,
            "startSeconds": span.start_seconds,
            "endSeconds": span.end_seconds,
        }

    def _policy_reference_to_dict(reference) -> dict | None:
        if not reference:
            return None
        return {
            "source": reference.source,
            "reference": reference.reference,
            "clause": reference.clause,
            "version": reference.version,
            "category": reference.category,
            "provenance": reference.provenance,
        }

    def _trigger_attribution_to_dict(attribution) -> dict:
        return {
            "attributionId": attribution.attribution_id,
            "family": attribution.family,
            "triggerType": attribution.trigger_type,
            "title": attribution.title,
            "verdict": attribution.verdict,
            "confidence": attribution.confidence,
            "evidenceSpan": _span_to_dict(attribution.evidence_span),
            "policyReference": _policy_reference_to_dict(attribution.policy_reference),
            "reasoning": attribution.reasoning,
            "evidenceChain": attribution.evidence_chain,
            "supportingQuotes": attribution.supporting_quotes,
        }

    def _claim_provenance_to_dict(provenance) -> dict:
        return {
            "claimId": provenance.claim_id,
            "claimText": provenance.claim_text,
            "claimSpan": _span_to_dict(provenance.claim_span),
            "retrievedPolicy": _policy_reference_to_dict(provenance.retrieved_policy),
            "semanticSimilarity": provenance.semantic_similarity,
            "nliVerdict": provenance.nli_verdict,
            "confidence": provenance.confidence,
            "reasoning": provenance.reasoning,
            "provenance": provenance.provenance,
            "supportingQuotes": provenance.supporting_quotes,
        }

    explainability = {
        "triggerAttributions": [
            _trigger_attribution_to_dict(attribution)
            for attribution in report.explainability.trigger_attributions
            if attribution.family in {"sop", "policy"}
        ],
        "claimProvenance": [
            _claim_provenance_to_dict(provenance)
            for provenance in report.explainability.claim_provenance
        ],
    }

    return {
        "available": True,
        "interactionId": str(report.interaction_id),
        "processAdherence": {
            "detectedTopic": report.process_adherence.detected_topic,
            "isResolved": report.process_adherence.is_resolved,
            "efficiencyScore": report.process_adherence.efficiency_score,
            "justification": report.process_adherence.justification,
            "missingSopSteps": report.process_adherence.missing_sop_steps,
            "evidenceQuotes": report.process_adherence.evidence_quotes,
            "citations": [_citation_to_dict(c) for c in report.process_adherence.citations],
            "insufficientEvidence": report.process_adherence.insufficient_evidence,
            "confidenceScore": report.process_adherence.confidence_score,
        },
        "nliPolicy": {
            "nliCategory": report.nli_policy.nli_category,
            "justification": report.nli_policy.justification,
            "evidenceQuotes": report.nli_policy.evidence_quotes,
            "citations": [_citation_to_dict(c) for c in report.nli_policy.citations],
            "policyVersion": report.nli_policy.policy_version,
            "policyEffectiveAt": report.nli_policy.policy_effective_at,
            "policyCategory": report.nli_policy.policy_category,
            "conflictResolutionApplied": report.nli_policy.conflict_resolution_applied,
            "insufficientEvidence": report.nli_policy.insufficient_evidence,
            "confidenceScore": report.nli_policy.confidence_score,
            "policyAlignmentScore": report.nli_policy.policy_alignment_score,
        },
        "explainability": explainability,
    }


def _map_llm_trigger_report(report) -> dict:
    # Backward-compatible envelope kept for existing frontend consumers.
    emotion_payload = _map_emotion_trigger_report(report)
    rag_payload = _map_rag_compliance_report(report)
    return {
        "available": True,
        "interactionId": emotion_payload["interactionId"],
        "emotionShift": emotion_payload["emotionShift"],
        "processAdherence": rag_payload["processAdherence"],
        "nliPolicy": rag_payload["nliPolicy"],
        "explainability": {
            "triggerAttributions": [
                *emotion_payload["explainability"]["triggerAttributions"],
                *rag_payload["explainability"]["triggerAttributions"],
            ],
            "claimProvenance": rag_payload["explainability"]["claimProvenance"],
        },
        "derived": emotion_payload["derived"],
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
async def list_interactions(session: SessionDep, current_user: CurrentUser):
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
        .where(*_interaction_scope_filters(current_user))
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
            "overallScore": round(row.overall_score * 100, 0) if row.overall_score else 0,
            "empathyScore": round(row.empathy_score * 100, 0) if row.empathy_score else 0,
            "policyScore": round(row.policy_score * 100, 0) if row.policy_score else 0,
            "resolutionScore": round(row.resolution_score * 100, 0) if row.resolution_score else 0,
            "resolved": row.was_resolved or False,
            "hasViolation": row.viol_count > 0,
            "hasOverlap": row.has_overlap,
            "responseTime": f"{row.avg_response_time_seconds:.1f}s" if row.avg_response_time_seconds else "N/A",
            "status": str(row.processing_status.value) if row.processing_status else "pending",
            "audioFilePath": row.audio_file_path or None,
        })

    return interactions


@router.get("/{interaction_id}", response_model=InteractionDetailResponse)
async def get_interaction_detail(
    interaction_id: UUID,
    session: SessionDep,
    current_user: CurrentUser,
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
        .where(
            Interaction.id == interaction_id,
            *_interaction_scope_filters(current_user),
        )
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
        .join(Interaction, Utterance.interaction_id == Interaction.id)
        .where(
            Utterance.interaction_id == interaction_id,
            *_interaction_scope_filters(current_user),
        )
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
                "sequenceIndex": u.sequence_index,
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
        .join(Interaction, EmotionEvent.interaction_id == Interaction.id)
        .where(
            EmotionEvent.interaction_id == interaction_id,
            Interaction.organization_id == current_user.organization_id
        )
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
        .join(Interaction, PolicyCompliance.interaction_id == Interaction.id)
        .where(
            PolicyCompliance.interaction_id == interaction_id,
            Interaction.organization_id == current_user.organization_id,
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
    emotion_triggers = None
    rag_compliance = None
    is_processing = row.processing_status in {
        ProcessingStatus.pending,
        ProcessingStatus.processing,
    }
    if include_llm_triggers and not is_processing:
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
                force_rerun=llm_force_rerun,
                commit_cache=True,
            )
            emotion_triggers = _map_emotion_trigger_report(report)
            rag_compliance = _map_rag_compliance_report(report)
            emotion_triggers["orgFilter"] = resolved_org_filter
            emotion_triggers["forcedRerun"] = llm_force_rerun
            rag_compliance["orgFilter"] = resolved_org_filter
            rag_compliance["forcedRerun"] = llm_force_rerun
            rag_compliance["policyViolations"] = policy_violations
            llm_triggers = _map_llm_trigger_report(report)
            llm_triggers["orgFilter"] = resolved_org_filter
            llm_triggers["forcedRerun"] = llm_force_rerun
        except Exception as exc:
            emotion_triggers = {
                "available": False,
                "error": str(exc),
            }
            rag_compliance = {
                "available": False,
                "error": str(exc),
            }
            llm_triggers = {
                "available": False,
                "error": str(exc),
            }
    elif include_llm_triggers and is_processing:
        processing_message = "Interaction is still processing. LLM trigger analysis will be available once processing completes."
        emotion_triggers = {
            "available": False,
            "error": processing_message,
        }
        rag_compliance = {
            "available": False,
            "error": processing_message,
        }
        llm_triggers = {
            "available": False,
            "error": processing_message,
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
            "overallScore": round(row.overall_score * 100, 0) if row.overall_score else 0,
            "empathyScore": round(row.empathy_score * 100, 0) if row.empathy_score else 0,
            "policyScore": round(row.policy_score * 100, 0) if row.policy_score else 0,
            "resolutionScore": round(row.resolution_score * 100, 0) if row.resolution_score else 0,
            "resolved": row.was_resolved or False,
            "hasViolation": len(policy_violations) > 0,
            "hasOverlap": row.has_overlap,
            "responseTime": f"{row.avg_response_time_seconds:.1f}s" if row.avg_response_time_seconds else "N/A",
            "status": str(row.processing_status.value) if row.processing_status else "pending",
            "audioFilePath": row.audio_file_path or None,
        },
        "utterances": utterances,
        "emotionComparison": emotion_comparison,
        "ragCompliance": rag_compliance,
        "emotionTriggers": emotion_triggers,
        "llmTriggers": llm_triggers,
        "emotionEvents": emotion_events,
        "policyViolations": policy_violations,
    }


@router.get("/{interaction_id}/emotion-comparison")
async def get_interaction_emotion_comparison(interaction_id: UUID, session: SessionDep, current_user: CurrentUser):
    """Return compact acoustic vs text vs fused emotion comparison for manager panel."""
    interaction_result = await session.exec(
        select(Interaction.id).where(
            Interaction.id == interaction_id,
            *_interaction_scope_filters(current_user),
        )
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
async def get_interaction_audio(interaction_id: UUID, session: SessionDep, current_user: CurrentUser):
    """Stream the audio file for an interaction from Supabase Storage or local path."""

    # Get the audio path and duration
    result = await session.exec(
        select(Interaction.audio_file_path, Interaction.duration_seconds).where(
            Interaction.id == interaction_id,
            *_interaction_scope_filters(current_user),
        )
    )
    row = result.first()

    if not row:
        raise HTTPException(status_code=404, detail="Interaction not found")

    audio_path, duration = row

    # Prefer local audio when a filesystem path was seeded (e.g. ../AudioData/nexalink/*.mp3).
    backend_dir = Path(__file__).resolve().parents[3]
    candidates = [Path(audio_path), backend_dir / audio_path]
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.exists() and resolved.is_file():
            ext = resolved.suffix.lower()
            media_type = "audio/mpeg" if ext == ".mp3" else "audio/wav"
            content = resolved.read_bytes()
            return StreamingResponse(
                iter([content]),
                media_type=media_type,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(content)),
                },
            )

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
