from __future__ import annotations

import asyncio
import logging
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.emotion_fusion import EMOTION_NORMALIZATION, fuse_emotion_signals, infer_text_emotion_with_provider
from app.llm_trigger.chains import (
    build_emotion_shift_chain,
    build_nli_policy_chain,
    build_process_adherence_chain,
)
from app.llm_trigger.retrieval import RetrievedChunk, resolve_retrieved_sop_context, retrieve_policy_chunks
from app.llm_trigger.schemas import (
    ClaimProvenance,
    EvidenceAnchoredExplainability,
    EvidenceCitation,
    EvidenceSpan,
    EmotionShiftAnalysis,
    InteractionLLMTriggerReport,
    NLIEvaluation,
    PolicyReference,
    ProcessAdherenceReport,
    TriggerAttribution,
)
from app.models.interaction import Interaction
from app.models.llm_trigger_cache import InteractionLLMTriggerCache
from app.models.policy import CompanyPolicy, OrganizationPolicy
from app.models.transcript import Transcript
from app.models.utterance import Utterance
from app.models.user import User


logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 2
_evaluator_lock = threading.Lock()
_policy_compliance_evaluator = None


RESOLUTION_GRAPHS: dict[str, list[str]] = {
    "refund_request": [
        "Acknowledge customer issue",
        "Collect order identifier",
        "Verify refund eligibility window",
        "Confirm refund method and timeline",
        "Close with summary and next steps",
    ],
    "billing_issue": [
        "Acknowledge billing concern",
        "Verify account and charge details",
        "Explain charge source or correction",
        "Confirm customer understanding",
        "Close with follow-up path",
    ],
    "technical_support": [
        "Acknowledge the technical issue",
        "Collect device or account context",
        "Run step-by-step troubleshooting",
        "Validate issue resolution",
        "Document next escalation path",
    ],
    "account_access": [
        "Acknowledge access issue",
        "Verify user identity",
        "Guide reset or unlock steps",
        "Confirm successful login",
        "Close with prevention advice",
    ],
}


ROLLING_WINDOW_TURNS = 8
ROLLING_WINDOW_STRIDE = 4
MAX_PROCESS_WINDOWS = 3
INSUFFICIENT_EVIDENCE_LABEL = "insufficient evidence"
POLICY_PRIORITY_BUCKETS: tuple[str, ...] = ("regulatory", "legal")
DEGRADED_MODE_SUFFIX = " [DEGRADED: LLM unavailable — heuristic fallback used]"


def _log_step(interaction_id: UUID | str, step: str, **kwargs) -> None:
    extra = {"interaction_id": str(interaction_id), "pipeline_step": step, **kwargs}
    logger.info("LLM trigger pipeline step: %s", step, extra=extra)


def _services_path() -> Path:
    return Path(__file__).resolve().parents[3] / "services"


def _get_policy_compliance_evaluator():
    """
    Lazily load the transcript-level PolicyComplianceEvaluator from services/rag.

    This keeps the trigger pipeline explicit about layer boundaries:
      - retrieval/context + transcript-level compliance report are handled by
        the dedicated evaluator module,
      - claim-level checks stay in the local NLI checker.
    """
    global _policy_compliance_evaluator
    if _policy_compliance_evaluator is not None:
        return _policy_compliance_evaluator
    with _evaluator_lock:
        if _policy_compliance_evaluator is not None:
            return _policy_compliance_evaluator
        try:
            services_path = str(_services_path())
            if services_path not in sys.path:
                sys.path.append(services_path)
            from rag.evaluator import PolicyComplianceEvaluator
            _policy_compliance_evaluator = PolicyComplianceEvaluator()
        except Exception as exc:
            logger.warning("Unable to initialize PolicyComplianceEvaluator: %s", exc)
            _policy_compliance_evaluator = None
        return _policy_compliance_evaluator


@dataclass
class TranscriptWindow:
    window_id: str
    start_index: int
    end_index: int
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class ResolvedPolicyContext:
    text: str
    version: str | None = None
    effective_at: str | None = None
    category: str | None = None
    conflict_resolution_applied: bool = False


def _join_retrieved_chunk_text(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(chunk.text.strip() for chunk in chunks if chunk.text.strip())


def _policy_chunk_reference(chunk: RetrievedChunk) -> str:
    header_path = " > ".join(
        str(chunk.metadata.get(key)).strip()
        for key in ("Header 1", "Header 2", "Header 3")
        if chunk.metadata.get(key)
    )
    return header_path or str(chunk.metadata.get("source_file") or chunk.metadata.get("doc_id") or "retrieved-policy")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _sanitize_for_prompt(text: str, max_length: int = 4000) -> str:
    sanitized = (text or "").strip()
    sanitized = re.sub(r"(?i)^system\s*:", "[system]:", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"(?i)^assistant\s*:", "[assistant]:", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"(?i)^human\s*:", "[human]:", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"(?i)^user\s*:", "[user]:", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"```", "` ", sanitized)
    return _truncate_text(sanitized, max_length)


def _normalize_acoustic_label(label: str) -> str:
    base = (label or "neutral").strip().lower()
    return EMOTION_NORMALIZATION.get(base, base or "neutral")


def _emotion_polarity(label: str) -> str:
    normalized = _normalize_acoustic_label(label)
    if normalized in {"happy", "grateful"}:
        return "positive"
    if normalized in {"neutral", "unknown"}:
        return "neutral"
    if normalized in {"angry", "frustrated", "sad"}:
        return "negative"
    return "neutral"


def _detect_cross_modal_dissonance(customer_text: str, acoustic_emotion: str, text_emotion: str | None = None) -> bool:
    if text_emotion is None:
        text_emotion, _ = infer_text_emotion_with_provider(customer_text)
    text_polarity = _emotion_polarity(text_emotion)
    acoustic_polarity = _emotion_polarity(acoustic_emotion)
    if "!" in customer_text and acoustic_polarity == "negative" and text_polarity != "negative":
        return True
    return (text_polarity == "positive" and acoustic_polarity == "negative") or (
        text_polarity == "negative" and acoustic_polarity == "positive"
    )


def _detect_topic(transcript_text: str, retrieved_sop: str) -> str:
    source = f"{transcript_text}\n{retrieved_sop}".lower()
    keyword_map = {
        "refund_request": ["refund", "return", "chargeback"],
        "billing_issue": ["bill", "charge", "invoice", "payment"],
        "technical_support": ["error", "bug", "not working", "issue", "crash"],
        "account_access": ["login", "password", "reset", "locked"],
    }
    scores: dict[str, int] = {}
    for topic, keywords in keyword_map.items():
        scores[topic] = sum(1 for keyword in keywords if keyword in source)
    best_topic = max(scores, key=scores.get)
    return best_topic if scores[best_topic] > 0 else "technical_support"


def _detect_topic_from_sop_chunks(chunks: list[RetrievedChunk]) -> str | None:
    if not chunks:
        return None

    source_text = " ".join(
        str(value)
        for chunk in chunks
        for value in (
            chunk.metadata.get("source_file"),
            chunk.metadata.get("doc_id"),
            chunk.metadata.get("Header 1"),
            chunk.metadata.get("Header 2"),
            chunk.metadata.get("Header 3"),
            chunk.reference,
            chunk.provenance,
        )
        if value
    ).lower()

    hint_map = {
        "refund_request": ("01-refund", "refund request", "refund-request"),
        "billing_issue": ("02-billing", "billing issue", "billing-issue"),
        "technical_support": ("03-technical", "technical support", "technical-support"),
        "account_access": ("04-account", "account access", "account-access"),
    }
    for topic, hints in hint_map.items():
        if any(hint in source_text for hint in hints):
            return topic
    return None


def _extract_sop_steps(retrieved_sop: str) -> list[str]:
    steps: list[str] = []
    for line in retrieved_sop.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("#") or raw.startswith("<!--"):
            continue
        if "|" in raw:
            continue

        if not re.match(r"^(?:[-*]|\d+[.)])\s+", raw):
            if not raw.lower().startswith("step "):
                continue

        cleaned = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", raw).strip()
        if not cleaned:
            continue
        if len(cleaned.split()) >= 3 and len(cleaned) <= 120:
            steps.append(cleaned)
    return steps[:5]


def _step_keywords(step: str) -> set[str]:
    stop_words = {
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "for",
        "then",
        "into",
        "customer",
        "agent",
    }
    return {token for token in _tokenize(step) if len(token) > 3 and token not in stop_words}


def _trajectory_missing_steps(transcript_text: str, expected_steps: list[str]) -> list[str]:
    transcript_tokens = set(_tokenize(transcript_text))
    missing: list[str] = []
    for step in expected_steps:
        keywords = _step_keywords(step)
        if not keywords:
            continue
        overlap = len(keywords.intersection(transcript_tokens))
        threshold = max(1, len(keywords) // 3)
        if overlap < threshold:
            missing.append(step)
    return missing


def _merge_missing_steps(
    deterministic_missing: list[str],
    llm_missing: list[str],
) -> list[str]:
    if not llm_missing:
        return deterministic_missing
    if not deterministic_missing:
        return llm_missing

    normalized_deterministic = {step.lower(): step for step in deterministic_missing}
    merged = list(deterministic_missing)
    for llm_step in llm_missing:
        llm_keywords = _step_keywords(llm_step)
        matched = False
        for expected_lower, original_step in normalized_deterministic.items():
            expected_keywords = _step_keywords(expected_lower)
            if llm_keywords and expected_keywords and llm_keywords.intersection(expected_keywords):
                matched = True
                break
        if not matched and llm_step not in merged:
            merged.append(llm_step)
    return merged


def _is_resolved_heuristic(transcript_text: str) -> bool:
    text = transcript_text.lower()
    positive_endings = [
        "resolved",
        "fixed",
        "works now",
        "thank you",
        "anything else",
        "refund has been processed",
        "credit has been applied",
        "case reference",
        "ticket number",
        "follow-up",
    ]
    unresolved_markers = ["still not", "didn't work", "not fixed", "call back"]
    if any(marker in text for marker in unresolved_markers):
        return False
    return any(marker in text for marker in positive_endings)


def _efficiency_score_heuristic(transcript_text: str, missing_steps: list[str], expected_steps: list[str]) -> int:
    if not expected_steps:
        return 6
    missing_ratio = min(len(missing_steps), len(expected_steps)) / len(expected_steps)
    coverage = 1.0 - missing_ratio
    score = int(round(1 + 9 * coverage))
    turns = len([line for line in transcript_text.splitlines() if line.strip()])
    if turns > 20:
        score -= 1
    if turns > 30:
        score -= 1
    return max(1, min(10, score))


def _split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


EMOTIONAL_QUOTE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "negative": (
        "frustrat",
        "angry",
        "upset",
        "terrible",
        "unacceptable",
        "not working",
        "still",
        "again",
        "delay",
        "waiting",
        "issue",
        "problem",
        "broken",
        "failed",
        "cannot",
        "can't",
        "won't",
        "refund",
        "complaint",
    ),
    "positive": (
        "thank",
        "great",
        "resolved",
        "fixed",
        "helpful",
        "perfect",
        "appreciate",
    ),
}


def _sentence_emotion_score(sentence: str, target_emotion: str | None = None) -> float:
    text = sentence.lower()
    score = 0.0

    negative_hits = sum(1 for token in EMOTIONAL_QUOTE_KEYWORDS["negative"] if token in text)
    positive_hits = sum(1 for token in EMOTIONAL_QUOTE_KEYWORDS["positive"] if token in text)

    score += (negative_hits * 1.6)
    score += (positive_hits * 0.9)

    if "!" in sentence:
        score += 0.8
    if "?" in sentence:
        score += 0.3
    if len(sentence.split()) >= 8:
        score += 0.4

    if target_emotion:
        target_polarity = _emotion_polarity(target_emotion)
        if target_polarity == "negative":
            score += (negative_hits * 1.1)
            score -= (positive_hits * 0.4)
        elif target_polarity == "positive":
            score += (positive_hits * 0.6)

    # Down-rank identity-only lines that often pollute evidence snippets.
    if "my name is" in text or "pin is" in text:
        score -= 1.8

    return score


def _quote_candidates(text: str, max_quotes: int = 3, target_emotion: str | None = None) -> list[str]:
    if not text:
        return []

    sentences = _split_sentences(text)
    scored_sentences: list[tuple[float, int, str]] = []
    for sentence in sentences:
        cleaned = sentence.strip().strip('"')
        if len(cleaned.split()) < 4:
            continue

        score = _sentence_emotion_score(cleaned, target_emotion=target_emotion)
        scored_sentences.append((score, len(scored_sentences), cleaned))

    if scored_sentences:
        scored_sentences.sort(key=lambda item: (-item[0], item[1]))
        return [sentence for _, _, sentence in scored_sentences[:max_quotes]]

    quotes: list[str] = []
    for sentence in sentences:
        cleaned = sentence.strip().strip('"')
        if len(cleaned.split()) < 4:
            continue
        quotes.append(cleaned)
        if len(quotes) >= max_quotes:
            break

    if quotes:
        return quotes

    fallback = text.strip().replace("\n", " ")
    return [fallback[:180]] if fallback else []


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    trimmed = normalized[:max_chars].rsplit(" ", 1)[0].strip()
    return trimmed or normalized[:max_chars].strip()


def _llm_failure_reason(exc: Exception) -> str:
    message = str(exc or "").strip().lower()
    if "rate limit" in message or "rate_limit_exceeded" in message or "error code: 429" in message:
        return "the LLM provider rate limit was reached"
    if "api key" in message or "unauthorized" in message or "forbidden" in message or "authentication" in message:
        return "LLM authentication failed"
    if "timeout" in message or "timed out" in message:
        return "the LLM request timed out"
    if "connection" in message or "refused" in message or "unavailable" in message:
        return "the LLM service was unavailable"
    return "the LLM service was unavailable"


def _build_customer_emotion_reasoning(emotion: str, root_cause: str, quotes: list[str]) -> str:
    quote = quotes[0] if quotes else ""
    emotional_state = _normalize_acoustic_label(emotion)
    root = (root_cause or "").strip()
    if root and INSUFFICIENT_EVIDENCE_LABEL not in root.lower():
        return root

    baseline_reasons: dict[str, str] = {
        "angry": "Customer language and tone indicate perceived service failure and elevated agitation.",
        "frustrated": "Customer appears blocked in issue resolution and expresses repeated dissatisfaction.",
        "sad": "Customer wording suggests disappointment and low-confidence expectations.",
        "happy": "Customer language and tone indicate satisfaction with the handling and outcome.",
        "neutral": "Customer wording remains informational without strong affective escalation.",
    }
    reason = baseline_reasons.get(
        emotional_state,
        "Customer emotion is present but transcript evidence is limited for a detailed causal explanation.",
    )
    if quote:
        return f"{reason} Evidence: \"{quote}\""
    return reason


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()


def _policy_priority(category: str | None) -> int:
    normalized = (category or "").strip().lower()
    for idx, prefix in enumerate(POLICY_PRIORITY_BUCKETS):
        if normalized.startswith(prefix):
            return idx
    return len(POLICY_PRIORITY_BUCKETS)


def _iso_or_none(value: datetime | None) -> str | None:
    if not value:
        return None
    return value.isoformat()


def _ensure_minimum_policy_citation(nli_policy: NLIEvaluation, policy_text: str) -> None:
    has_policy_citation = any(c.source == "policy" and bool((c.quote or "").strip()) for c in nli_policy.citations)
    if has_policy_citation:
        return

    fallback_quotes = _quote_candidates(policy_text, max_quotes=1)
    if fallback_quotes:
        nli_policy.citations.append(
            EvidenceCitation(source="policy", speaker="system", quote=fallback_quotes[0])
        )


def _has_mapped_transcript_citation(citations: list[EvidenceCitation]) -> bool:
    return any(
        c.source == "transcript"
        and c.utterance_index is not None
        and bool((c.quote or "").strip())
        for c in citations
    )


def _backfill_transcript_citation_indices(
    citations: list[EvidenceCitation],
    utterances: list[Utterance],
) -> None:
    normalized_utterances: list[tuple[int, str, str]] = []
    for utterance in utterances:
        text = (utterance.text or "").strip()
        if not text:
            continue
        normalized = _normalize_for_match(text)
        if not normalized:
            continue
        role = utterance.speaker_role.value if utterance.speaker_role else "unknown"
        normalized_utterances.append((utterance.sequence_index, normalized, role))

    for citation in citations:
        if citation.source != "transcript" or citation.utterance_index is not None:
            continue
        quote_norm = _normalize_for_match(citation.quote)
        if not quote_norm:
            continue
        for sequence_index, utterance_norm, speaker in normalized_utterances:
            if quote_norm in utterance_norm or utterance_norm in quote_norm:
                citation.utterance_index = sequence_index
                if citation.speaker == "unknown":
                    citation.speaker = speaker  # type: ignore[assignment]
                break


async def _resolve_active_policy_context(
    session: AsyncSession,
    organization_id: UUID,
    ground_truth_policy: str,
    fallback_sop: str,
    query_text: str,
    org_filter: str | None,
) -> ResolvedPolicyContext:
    """
    Resolve the single policy context used by the claim-level NLI check.

    Retrieval happens here, but judgment does not: the returned text is later
    passed to ``run_single_claim_nli_policy_check`` for entailment-style
    classification.
    """
    if ground_truth_policy.strip():
        return ResolvedPolicyContext(
            text=ground_truth_policy.strip(),
            version="manual-override",
            category="override",
            conflict_resolution_applied=False,
        )

    retrieval_query = query_text.strip() or fallback_sop.strip()
    if retrieval_query:
        try:
            retrieved_policy_chunks = retrieve_policy_chunks(
                query_text=retrieval_query,
                org_filter=org_filter,
                top_k=2,
            )
        except Exception:
            retrieved_policy_chunks = []

        if retrieved_policy_chunks:
            primary_chunk = retrieved_policy_chunks[0]
            primary_source = str(
                primary_chunk.metadata.get("source_file")
                or primary_chunk.metadata.get("doc_id")
                or ""
            ).strip()
            contextual_chunks = [
                chunk
                for chunk in retrieved_policy_chunks
                if not primary_source
                or str(chunk.metadata.get("source_file") or chunk.metadata.get("doc_id") or "").strip() == primary_source
            ] or [primary_chunk]
            primary_reference = _policy_chunk_reference(primary_chunk)
            reference_basis = str(primary_chunk.metadata.get("source_file") or primary_chunk.metadata.get("doc_id") or primary_reference)
            unique_references = {_policy_chunk_reference(chunk) for chunk in contextual_chunks}
            return ResolvedPolicyContext(
                text=_join_retrieved_chunk_text(contextual_chunks),
                version=f"retrieved:{reference_basis}",
                category=str(primary_chunk.metadata.get("doc_type") or primary_chunk.metadata.get("category") or "retrieved"),
                conflict_resolution_applied=len(unique_references) > 1,
            )

    stmt = (
        select(
            CompanyPolicy.id,
            CompanyPolicy.policy_category,
            CompanyPolicy.policy_text,
            CompanyPolicy.updated_at,
            CompanyPolicy.created_at,
        )
        .join(OrganizationPolicy, OrganizationPolicy.policy_id == CompanyPolicy.id)
        .where(
            OrganizationPolicy.organization_id == organization_id,
            OrganizationPolicy.is_active == True,  # noqa: E712
            CompanyPolicy.is_active == True,  # noqa: E712
        )
    )
    rows = list((await session.exec(stmt)).all())

    if not rows:
        text = fallback_sop.strip() or "No ground truth policy context provided."
        return ResolvedPolicyContext(text=text, version="fallback", category="fallback")

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            _policy_priority(row.policy_category),
            -(row.updated_at or row.created_at).timestamp(),
        ),
    )
    selected = sorted_rows[0]
    categories = {str(row.policy_category or "").lower() for row in rows}
    conflict_applied = len(rows) > 1 and len(categories) > 1
    version_basis = selected.updated_at or selected.created_at
    version_token = f"{selected.id}:{version_basis.date().isoformat()}" if version_basis else str(selected.id)

    return ResolvedPolicyContext(
        text=(selected.policy_text or "").strip() or (fallback_sop.strip() or "No ground truth policy context provided."),
        version=version_token,
        effective_at=_iso_or_none(version_basis),
        category=selected.policy_category,
        conflict_resolution_applied=conflict_applied,
    )


def _format_timestamp(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    return f"{seconds_int // 60:02d}:{seconds_int % 60:02d}"


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _utterance_speaker(utterance: Utterance | None) -> str:
    if not utterance or not utterance.speaker_role:
        return "unknown"
    return utterance.speaker_role.value


def _find_utterance_by_index(
    utterances: list[Utterance],
    utterance_index: int | None,
) -> Utterance | None:
    if utterance_index is None:
        return None
    for utterance in utterances:
        if utterance.sequence_index == utterance_index:
            return utterance
    return None


def _extract_reference_label(text: str, fallback: str) -> str:
    for raw_line in text.splitlines():
        line = _clean_display_text(raw_line)
        if not line:
            continue
        bracket_match = re.match(r"^\[(.+?)\]$", line)
        if bracket_match:
            return bracket_match.group(1).strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        return line[:120]
    return fallback


def _clean_display_text(text: str | None) -> str:
    cleaned = (text or "").replace("\r", "")
    cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("â€¢", " / ").replace("Ã¢â‚¬Â¢", " / ")
    cleaned = re.sub(r"\b(image|table)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_]{2,}", "_", cleaned)
    cleaned = re.sub(r"[-|]{4,}", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _extract_display_clause(text: str, max_chars: int = 260) -> str:
    preferred_lines: list[str] = []
    fallback_lines: list[str] = []
    for raw_line in text.splitlines():
        line = _clean_display_text(raw_line)
        if not line:
            continue
        if re.match(r"^\[.+\]$", line):
            continue
        lowered = line.lower()
        if lowered.endswith(".pdf") or lowered in {
            "overview",
            "organization:",
            "version:",
            "effective date:",
            "step-by-step procedure",
            "related policies",
            "key considerations",
            "eligibility criteria",
        }:
            continue
        fallback_lines.append(line)
        if line.lower().startswith(("step ", "compliance:", "note:", "forbidden", "quote the correct timeline", "offer ", "provide ", "verify ", "apply ", "escalate ")):
            preferred_lines.append(line)

    candidate_lines = preferred_lines or fallback_lines
    if not candidate_lines:
        return _truncate_text(_clean_display_text(text), max_chars) or "No clause available."

    snippet = " ".join(candidate_lines[:2])
    return _truncate_text(snippet, max_chars) or "No clause available."


def _span_from_citation(
    citation: EvidenceCitation | None,
    utterances: list[Utterance],
) -> EvidenceSpan | None:
    if not citation or not (citation.quote or "").strip():
        return None

    utterance = _find_utterance_by_index(utterances, citation.utterance_index)
    speaker = citation.speaker
    if speaker == "unknown" and utterance:
        speaker = _utterance_speaker(utterance)  # type: ignore[assignment]

    start_seconds = utterance.start_time_seconds if utterance else None
    end_seconds = utterance.end_time_seconds if utterance else None

    return EvidenceSpan(
        utterance_index=citation.utterance_index,
        speaker=speaker,
        quote=citation.quote.strip(),
        timestamp=_format_timestamp(start_seconds) if start_seconds is not None else None,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _best_citation(
    citations: list[EvidenceCitation],
    source: str,
    speaker: str | None = None,
) -> EvidenceCitation | None:
    for citation in citations:
        if citation.source != source:
            continue
        if speaker and citation.speaker != speaker:
            continue
        if (citation.quote or "").strip():
            return citation
    return None


def _supporting_quotes(citations: list[EvidenceCitation], limit: int = 3) -> list[str]:
    quotes: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        quote = (citation.quote or "").strip()
        if not quote or quote in seen:
            continue
        seen.add(quote)
        quotes.append(quote)
        if len(quotes) >= limit:
            break
    return quotes


def _best_retrieved_chunk(chunks: list[RetrievedChunk], query_text: str) -> RetrievedChunk | None:
    if not chunks:
        return None
    return max(
        chunks,
        key=lambda chunk: (
            float(chunk.score) if chunk.score is not None else -1.0,
            _token_overlap_ratio(query_text, chunk.text),
            len(chunk.text),
        ),
    )


def _chunk_source_key(chunk: RetrievedChunk | None) -> str:
    if not chunk:
        return ""
    return str(chunk.metadata.get("source_file") or chunk.metadata.get("doc_id") or "").strip().lower()


def _filter_chunks_by_policy_context(
    chunks: list[RetrievedChunk],
    policy_context: ResolvedPolicyContext,
) -> list[RetrievedChunk]:
    version = (policy_context.version or "").strip()
    if not version.startswith("retrieved:"):
        return chunks
    expected_key = version.split("retrieved:", 1)[1].strip().lower()
    if not expected_key:
        return chunks
    filtered = [chunk for chunk in chunks if expected_key in _chunk_source_key(chunk)]
    return filtered or chunks


def _build_policy_reference(
    citation: EvidenceCitation | None,
    *,
    source_kind: str,
    source_text: str,
    fallback_reference: str,
    doc_type: str | None = None,
    policy_ref: list[str] | None = None,
    version: str | None = None,
    category: str | None = None,
    provenance: str | None = None,
) -> PolicyReference | None:
    clause = ""
    if citation and (citation.quote or "").strip():
        clause = _clean_display_text(citation.quote)
    if not clause:
        clause = _extract_display_clause(source_text)
    if not clause:
        return None

    return PolicyReference(
        source=source_kind,  # type: ignore[arg-type]
        reference=_clean_display_text(_extract_reference_label(source_text, fallback_reference)),
        clause=clause,
        doc_type=doc_type if doc_type in {"policy", "sop"} else source_kind,  # type: ignore[arg-type]
        policy_ref=policy_ref or [],
        version=version,
        category=category,
        provenance=_clean_display_text(provenance),
    )


def _build_policy_reference_from_chunk(
    chunk: RetrievedChunk | None,
    *,
    source_kind: str,
    fallback_text: str,
    fallback_reference: str,
    version: str | None = None,
    category: str | None = None,
) -> PolicyReference | None:
    if chunk and chunk.text.strip():
        clause = _extract_display_clause(chunk.text)
        chunk_doc_type = chunk.metadata.get("doc_type")
        policy_ref = chunk.metadata.get("policy_ref") or []
        if isinstance(policy_ref, str):
            policy_ref = [item.strip() for item in policy_ref.split(",") if item.strip()]
        return PolicyReference(
            source=source_kind,  # type: ignore[arg-type]
            reference=_clean_display_text(_extract_reference_label(chunk.text, chunk.reference or fallback_reference)),
            clause=clause or _truncate_text(_clean_display_text(chunk.text), 220),
            doc_type=chunk_doc_type if chunk_doc_type in {"policy", "sop"} else source_kind,  # type: ignore[arg-type]
            policy_ref=policy_ref if isinstance(policy_ref, list) else [],
            version=version,
            category=category,
            provenance=_clean_display_text(chunk.provenance or chunk.collection or chunk.source),
        )
    return _build_policy_reference(
        None,
        source_kind=source_kind,
        source_text=fallback_text,
        fallback_reference=fallback_reference,
        doc_type=source_kind,
        policy_ref=[],
        version=version,
        category=category,
        provenance=None,
    )


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / len(left_tokens.union(right_tokens))


def _emotion_signal_confidence(
    span: EvidenceSpan | None,
    utterances: list[Utterance],
) -> float | None:
    acoustic_confidence: float | None = None
    if span and span.utterance_index is not None:
        utterance = _find_utterance_by_index(utterances, span.utterance_index)
        if utterance and utterance.emotion_confidence is not None:
            acoustic_confidence = float(utterance.emotion_confidence)

    text_confidence: float | None = None
    if span and span.quote.strip():
        _label, text_confidence = infer_text_emotion_with_provider(span.quote)

    available = [score for score in [acoustic_confidence, text_confidence] if score is not None]
    if not available:
        return None
    return _clamp_unit(sum(available) / len(available))


def _select_step_citation(
    citations: list[EvidenceCitation],
    step: str,
) -> EvidenceCitation | None:
    transcript_citations = [citation for citation in citations if citation.source == "transcript"]
    if not transcript_citations:
        return None
    best = max(
        transcript_citations,
        key=lambda citation: _token_overlap_ratio(step, citation.quote),
    )
    if _token_overlap_ratio(step, best.quote) < 0.08:
        return None
    return best


def _map_nli_verdict(category: str, insufficient_evidence: bool = False) -> str:
    if insufficient_evidence:
        return "Insufficient Evidence"
    normalized = (category or "").strip().lower()
    if normalized == "entailment":
        return "Supported"
    if normalized == "contradiction":
        return "Contradiction"
    if normalized == "policy hallucination":
        return "Contradiction"
    if normalized == "benign deviation":
        return "Neutral"
    return "Neutral"


def _build_emotion_trigger_attribution(
    emotion_shift: EmotionShiftAnalysis,
    utterances: list[Utterance],
    acoustic_emotion: str,
) -> TriggerAttribution:
    transcript_citation = _best_citation(emotion_shift.citations, "transcript", speaker="customer") or _best_citation(
        emotion_shift.citations,
        "transcript",
    )
    span = _span_from_citation(transcript_citation, utterances)
    verdict = (
        "Insufficient Evidence"
        if emotion_shift.insufficient_evidence
        else "Cross-Modal Mismatch"
        if emotion_shift.is_dissonance_detected
        else "No Trigger"
    )
    evidence_chain = [
        f"Acoustic emotion signal resolved to {acoustic_emotion}.",
        f"Transcript span used for review: {(span.quote if span else 'No mapped transcript span was available.')}",
        emotion_shift.root_cause,
    ]

    return TriggerAttribution(
        attribution_id="emotion-dissonance",
        family="emotion",
        trigger_type="Acoustic-Transcript Dissonance",
        title=emotion_shift.dissonance_type or "Cross-Modal Dissonance",
        verdict=verdict,  # type: ignore[arg-type]
        confidence=emotion_shift.confidence_score or _emotion_signal_confidence(span, utterances),
        evidence_span=span,
        reasoning=emotion_shift.root_cause,
        evidence_chain=evidence_chain,
        supporting_quotes=emotion_shift.evidence_quotes or _supporting_quotes(emotion_shift.citations),
    )


def _build_emotion_transition_attributions(
    utterances: list[Utterance],
) -> list[TriggerAttribution]:
    attributions: list[TriggerAttribution] = []
    previous_by_speaker: dict[str, Utterance] = {}

    for utterance in utterances:
        if not (utterance.text or "").strip():
            continue
        speaker = _utterance_speaker(utterance)
        previous = previous_by_speaker.get(speaker)
        previous_by_speaker[speaker] = utterance
        if previous is None:
            continue

        before = _normalize_acoustic_label(previous.emotion or "neutral")
        after = _normalize_acoustic_label(utterance.emotion or "neutral")
        if before == after:
            continue

        span = EvidenceSpan(
            utterance_index=utterance.sequence_index,
            speaker=speaker,  # type: ignore[arg-type]
            quote=(utterance.text or "").strip(),
            timestamp=_format_timestamp(utterance.start_time_seconds or 0.0),
            start_seconds=utterance.start_time_seconds,
            end_seconds=utterance.end_time_seconds,
        )
        evidence_chain = [
            f"{speaker.title()} emotion changed from {before} to {after}.",
            f"Previous span: {(previous.text or '').strip() or 'No prior span available.'}",
            f"Current span: {(utterance.text or '').strip()}",
        ]
        reasoning = (
            f"{speaker.title()} emotion shifted from {before} to {after} at {span.timestamp} "
            "based on the detected emotion label for this utterance."
        )
        attributions.append(
            TriggerAttribution(
                attribution_id=f"emotion-change-{speaker}-{utterance.sequence_index}",
                family="emotion",
                trigger_type="Emotion Change",
                title=f"{speaker.title()} emotion: {before} to {after}",
                verdict="Neutral",
                confidence=_clamp_unit(float(utterance.emotion_confidence or 0.5)),
                evidence_span=span,
                reasoning=reasoning,
                evidence_chain=evidence_chain,
                supporting_quotes=[
                    quote
                    for quote in [
                        (previous.text or "").strip(),
                        (utterance.text or "").strip(),
                    ]
                    if quote
                ],
            )
        )

    return attributions


def _build_process_trigger_attributions(
    process_adherence: ProcessAdherenceReport,
    utterances: list[Utterance],
    sop_context: str,
    sop_chunks: list[RetrievedChunk],
) -> list[TriggerAttribution]:
    if not process_adherence.missing_sop_steps:
        return []

    attributions: list[TriggerAttribution] = []
    for index, step in enumerate(process_adherence.missing_sop_steps, start=1):
        transcript_citation = _select_step_citation(process_adherence.citations, step)
        span = _span_from_citation(transcript_citation, utterances)
        overlap = _token_overlap_ratio(step, transcript_citation.quote if transcript_citation else "")
        sop_citation = _best_citation(process_adherence.citations, "sop")
        best_sop_chunk = _best_retrieved_chunk(sop_chunks, step)
        policy_reference = _build_policy_reference_from_chunk(
            best_sop_chunk,
            source_kind="sop",
            fallback_text=sop_citation.quote if sop_citation else sop_context,
            fallback_reference="Retrieved SOP",
        )
        verdict = "Partial Attempt" if overlap >= 0.18 else "Contradiction"
        evidence_chain = [
            f"Expected SOP step: {step}.",
            f"Closest transcript evidence: {(span.quote if span else 'No matching utterance span was recovered.')}",
            process_adherence.justification,
        ]
        reasoning = (
            f"The review looked for explicit execution of the SOP step '{step}'. "
            f"{('A nearby transcript span was found but did not fully satisfy the step. ' if span else 'No reliable transcript span matched this step. ')}"
            f"{process_adherence.justification}"
        )

        attributions.append(
            TriggerAttribution(
                attribution_id=f"sop-step-{index}",
                family="sop",
                trigger_type="SOP Violation",
                title=step,
                verdict=verdict,  # type: ignore[arg-type]
                confidence=process_adherence.confidence_score or (best_sop_chunk.score if best_sop_chunk else None),
                evidence_span=span,
                policy_reference=PolicyReference.model_validate(policy_reference.model_dump()) if policy_reference else None,
                reasoning=reasoning,
                evidence_chain=evidence_chain,
                supporting_quotes=[quote for quote in [span.quote if span else None, step] if quote],
            )
        )

    return attributions


def _fallback_claim_citations(agent_statement: str) -> list[EvidenceCitation]:
    return [
        EvidenceCitation(source="transcript", speaker="agent", quote=quote)
        for quote in _quote_candidates(agent_statement, max_quotes=2)
    ]


def _build_claim_provenance_cards(
    nli_policy: NLIEvaluation,
    utterances: list[Utterance],
    policy_context: ResolvedPolicyContext,
    agent_statement: str,
    policy_chunks: list[RetrievedChunk],
    org_filter: str | None,
) -> list[ClaimProvenance]:
    claim_citations = [
        citation
        for citation in nli_policy.citations
        if citation.source == "transcript" and citation.speaker in {"agent", "unknown"}
    ]
    if not claim_citations:
        claim_citations = _fallback_claim_citations(agent_statement)

    cards: list[ClaimProvenance] = []
    verdict = _map_nli_verdict(nli_policy.nli_category, nli_policy.insufficient_evidence)
    for index, citation in enumerate(claim_citations[:2], start=1):
        span = _span_from_citation(citation, utterances)
        claim_text = (citation.quote or "").strip()
        policy_citation = _best_citation(nli_policy.citations, "policy")
        per_claim_chunks: list[RetrievedChunk] = []
        if claim_text:
            try:
                per_claim_chunks = retrieve_policy_chunks(
                    query_text=claim_text,
                    org_filter=org_filter,
                    top_k=2,
                )
            except Exception:
                per_claim_chunks = []
        candidate_chunks = _filter_chunks_by_policy_context(
            per_claim_chunks or policy_chunks,
            policy_context,
        )
        best_policy_chunk = _best_retrieved_chunk(candidate_chunks, claim_text)
        policy_reference = _build_policy_reference_from_chunk(
            best_policy_chunk,
            source_kind="policy",
            fallback_text=policy_citation.quote if policy_citation else policy_context.text,
            fallback_reference=f"{policy_context.category or 'Policy'} reference",
            version=policy_context.version,
            category=policy_context.category,
        )
        if not policy_reference:
            continue

        provenance = " | ".join(
            part
            for part in [
                policy_reference.reference,
                policy_reference.provenance,
            ]
            if part
        )

        cards.append(
            ClaimProvenance(
                claim_id=f"claim-{index}",
                claim_text=claim_text,
                claim_span=span,
                retrieved_policy=PolicyReference.model_validate(policy_reference.model_dump()),
                semantic_similarity=_clamp_unit(best_policy_chunk.score) if best_policy_chunk and best_policy_chunk.score is not None else None,
                nli_verdict=verdict,  # type: ignore[arg-type]
                confidence=nli_policy.confidence_score,
                reasoning=nli_policy.justification,
                provenance=provenance or "Active policy context",
                supporting_quotes=[quote for quote in [claim_text, policy_reference.clause] if quote],
            )
        )

    return cards


def _build_explainability_layer(
    emotion_shift: EmotionShiftAnalysis,
    process_adherence: ProcessAdherenceReport,
    nli_policy: NLIEvaluation,
    utterances: list[Utterance],
    acoustic_emotion: str,
    sop_context: str,
    sop_chunks: list[RetrievedChunk],
    policy_context: ResolvedPolicyContext,
    policy_chunks: list[RetrievedChunk],
    agent_statement: str,
    org_filter: str | None,
) -> EvidenceAnchoredExplainability:
    trigger_attributions: list[TriggerAttribution] = [
        _build_emotion_trigger_attribution(
            emotion_shift=emotion_shift,
            utterances=utterances,
            acoustic_emotion=acoustic_emotion,
        )
    ]
    trigger_attributions.extend(_build_emotion_transition_attributions(utterances))
    trigger_attributions.extend(
        _build_process_trigger_attributions(
            process_adherence=process_adherence,
            utterances=utterances,
            sop_context=sop_context,
            sop_chunks=sop_chunks,
        )
    )

    return EvidenceAnchoredExplainability(
        trigger_attributions=trigger_attributions,
        claim_provenance=_build_claim_provenance_cards(
            nli_policy=nli_policy,
            utterances=utterances,
            policy_context=policy_context,
            policy_chunks=policy_chunks,
            agent_statement=agent_statement,
            org_filter=org_filter,
        ),
    )


def _reconstruct_transcript(utterances: list[Utterance]) -> str:
    lines: list[str] = []
    for utterance in utterances:
        if not utterance.text:
            continue
        role = utterance.speaker_role.value if utterance.speaker_role else "unknown"
        lines.append(f"{role}: {utterance.text}")
    return "\n".join(lines)


def _build_rolling_windows(
    utterances: list[Utterance],
    window_turns: int = ROLLING_WINDOW_TURNS,
    stride: int = ROLLING_WINDOW_STRIDE,
) -> list[TranscriptWindow]:
    if not utterances:
        return []

    window_turns = max(1, window_turns)
    stride = max(1, stride)
    windows: list[TranscriptWindow] = []
    index = 0

    for start in range(0, len(utterances), stride):
        end = min(start + window_turns, len(utterances))
        chunk = utterances[start:end]
        text = _reconstruct_transcript(chunk)
        if not text.strip():
            if end == len(utterances):
                break
            continue

        first = chunk[0]
        last = chunk[-1]
        windows.append(
            TranscriptWindow(
                window_id=f"W{index}",
                start_index=start,
                end_index=end - 1,
                start_seconds=first.start_time_seconds or 0.0,
                end_seconds=last.end_time_seconds or last.start_time_seconds or 0.0,
                text=text,
            )
        )
        index += 1

        if end >= len(utterances):
            break

    return windows


def _count_role_lines(window_text: str, role: str) -> int:
    prefix = f"{role.lower()}:"
    return sum(1 for line in window_text.splitlines() if line.lower().startswith(prefix))


def _emotion_window_score(window: TranscriptWindow) -> tuple[int, int, int]:
    customer_lines = [
        line.split(":", 1)[1].strip()
        for line in window.text.splitlines()
        if line.lower().startswith("customer:") and ":" in line
    ]
    joined_customer = " ".join(customer_lines)
    text_emotion, text_confidence = infer_text_emotion_with_provider(joined_customer)
    polarity_conflict_markers = int(round(text_confidence * 10))
    if _emotion_polarity(text_emotion) == "negative":
        polarity_conflict_markers += 2
    if "!" in joined_customer:
        polarity_conflict_markers += 1
    return (
        polarity_conflict_markers,
        _count_role_lines(window.text, "customer"),
        window.end_index,
    )


def _select_emotion_window(windows: list[TranscriptWindow]) -> TranscriptWindow | None:
    if not windows:
        return None
    return max(windows, key=_emotion_window_score)


def _select_process_windows(windows: list[TranscriptWindow], max_windows: int = MAX_PROCESS_WINDOWS) -> list[TranscriptWindow]:
    if len(windows) <= max_windows:
        return windows

    selected_indices: list[int] = [0, len(windows) - 1]
    if max_windows >= 3:
        selected_indices.insert(1, len(windows) // 2)

    selected = sorted(set(selected_indices))[:max_windows]
    return [windows[idx] for idx in selected]


def _render_window_bundle(windows: list[TranscriptWindow]) -> str:
    if not windows:
        return ""

    blocks: list[str] = []
    for window in windows:
        blocks.append(
            f"[{window.window_id}] turns {window.start_index}-{window.end_index} "
            f"({_format_timestamp(window.start_seconds)}-{_format_timestamp(window.end_seconds)})\n"
            f"{window.text}"
        )
    return "\n\n".join(blocks)


def _window_citations(windows: list[TranscriptWindow]) -> list[EvidenceCitation]:
    citations: list[EvidenceCitation] = []
    for window in windows:
        snippet = window.text.splitlines()[0].strip() if window.text.splitlines() else ""
        if not snippet:
            continue
        citations.append(
            EvidenceCitation(
                source="transcript",
                speaker="unknown",
                utterance_index=window.start_index,
                quote=snippet[:220],
            )
        )
    return citations


async def analyze_emotion_shift(
    agent_context: str,
    customer_text: str,
    acoustic_emotion: str,
) -> EmotionShiftAnalysis:
    inferred_emotion = _normalize_acoustic_label(acoustic_emotion)
    _text_emotion, text_confidence = infer_text_emotion_with_provider(customer_text)
    if not _detect_cross_modal_dissonance(customer_text, acoustic_emotion, text_emotion=_text_emotion):
        quotes = _quote_candidates(customer_text, max_quotes=2, target_emotion=inferred_emotion)
        return EmotionShiftAnalysis(
            is_dissonance_detected=False,
            dissonance_type="None",
            root_cause="No strong contradiction detected between text sentiment and acoustic emotion.",
            counterfactual_correction="If the agent had continued the same supportive approach, the interaction likely would have remained stable.",
            evidence_quotes=quotes,
            current_customer_emotion=inferred_emotion,
            current_emotion_reasoning=_build_customer_emotion_reasoning(
                emotion=inferred_emotion,
                root_cause="",
                quotes=quotes,
            ),
            citations=[
                EvidenceCitation(
                    source="transcript",
                    speaker="customer",
                    quote=quote,
                )
                for quote in quotes
            ],
            confidence_score=_clamp_unit(text_confidence),
        )

    chain = build_emotion_shift_chain()
    try:
        from app.llm_trigger.chains import _invoke_chain_with_retry
        result = await _invoke_chain_with_retry(
            chain,
            {
                "agent_context": _sanitize_for_prompt(agent_context),
                "customer_text": _sanitize_for_prompt(customer_text),
                "acoustic_emotion": acoustic_emotion,
            },
        )
    except Exception as exc:
        logger.warning("Emotion shift LLM chain failed, using fallback: %s", exc)
        quotes = _quote_candidates(customer_text, max_quotes=2, target_emotion=inferred_emotion)
        return EmotionShiftAnalysis(
            is_dissonance_detected=True,
            dissonance_type="Unknown",
            root_cause=INSUFFICIENT_EVIDENCE_LABEL + DEGRADED_MODE_SUFFIX,
            counterfactual_correction="If the agent had acknowledged the concern and confirmed a clear next action, escalation risk could have decreased.",
            evidence_quotes=quotes,
            citations=[
                EvidenceCitation(source="transcript", speaker="customer", quote=quote)
                for quote in quotes
            ],
            current_customer_emotion=inferred_emotion,
            current_emotion_reasoning=_build_customer_emotion_reasoning(
                emotion=inferred_emotion,
                root_cause=INSUFFICIENT_EVIDENCE_LABEL,
                quotes=quotes,
            ),
            insufficient_evidence=True,
            confidence_score=None,
        )
    result.is_dissonance_detected = True
    if result.dissonance_type.strip().lower() == "none":
        result.dissonance_type = "Sarcasm"
    if not result.evidence_quotes:
        result.evidence_quotes = _quote_candidates(customer_text, max_quotes=3, target_emotion=inferred_emotion)
    if not result.root_cause.strip():
        result.root_cause = INSUFFICIENT_EVIDENCE_LABEL
    if not result.evidence_quotes:
        result.root_cause = INSUFFICIENT_EVIDENCE_LABEL
    if INSUFFICIENT_EVIDENCE_LABEL in result.root_cause.lower():
        result.root_cause = INSUFFICIENT_EVIDENCE_LABEL
    if not result.citations:
        result.citations = [
            EvidenceCitation(source="transcript", speaker="customer", quote=quote)
            for quote in result.evidence_quotes
        ]
    result.current_customer_emotion = inferred_emotion
    result.current_emotion_reasoning = _build_customer_emotion_reasoning(
        emotion=inferred_emotion,
        root_cause=result.root_cause,
        quotes=result.evidence_quotes,
    )
    if result.confidence_score is None:
        result.confidence_score = _clamp_unit(text_confidence)
    return result


async def evaluate_process_adherence(
    transcript_text: str,
    retrieved_sop_from_pinecone: str,
    org_filter: str | None = None,
    retrieved_sop_chunks: list[RetrievedChunk] | None = None,
) -> ProcessAdherenceReport:
    if retrieved_sop_chunks is not None:
        retrieved_sop = "\n\n---\n\n".join(chunk.text for chunk in retrieved_sop_chunks if chunk.text)
    else:
        try:
            retrieved_sop_context = resolve_retrieved_sop_context(
                transcript_text=transcript_text,
                retrieved_sop_from_pinecone=retrieved_sop_from_pinecone,
                org_filter=org_filter,
            )
            retrieved_sop = retrieved_sop_context.text
            retrieved_sop_chunks = retrieved_sop_context.chunks
        except Exception:
            retrieved_sop = ""
            retrieved_sop_chunks = []

    topic_hint = _detect_topic_from_sop_chunks(retrieved_sop_chunks or []) or _detect_topic(transcript_text, retrieved_sop)
    extracted_sop_steps = _extract_sop_steps(retrieved_sop)
    if extracted_sop_steps:
        expected_steps = extracted_sop_steps[:8]
    else:
        expected_steps = RESOLUTION_GRAPHS.get(topic_hint, []).copy()[:8]

    deterministic_missing = _trajectory_missing_steps(transcript_text, expected_steps)
    deterministic_efficiency = _efficiency_score_heuristic(
        transcript_text=transcript_text,
        missing_steps=deterministic_missing,
        expected_steps=expected_steps,
    )
    deterministic_resolved = _is_resolved_heuristic(transcript_text)

    chain = build_process_adherence_chain()
    try:
        from app.llm_trigger.chains import _invoke_chain_with_retry
        result = await _invoke_chain_with_retry(
            chain,
            {
                "topic_hint": topic_hint,
                "transcript_text": _sanitize_for_prompt(transcript_text),
                "retrieved_sop": _sanitize_for_prompt(retrieved_sop or "No SOP context found.", max_length=6000),
                "expected_resolution_graph": "\n".join(
                    f"- {step}" for step in expected_steps
                )
                or "- No explicit graph available.",
            }
        )
    except Exception as exc:
        logger.warning("Process adherence LLM chain failed, using deterministic fallback: %s", exc)
        evidence_quotes = _quote_candidates(transcript_text, max_quotes=3)
        citations = [
            EvidenceCitation(source="transcript", speaker="unknown", quote=quote)
            for quote in evidence_quotes
        ]
        sop_quote = _quote_candidates(retrieved_sop, max_quotes=1)
        if sop_quote:
            citations.append(EvidenceCitation(source="sop", speaker="system", quote=sop_quote[0]))

        return ProcessAdherenceReport(
            detected_topic=topic_hint,
            is_resolved=deterministic_resolved,
            efficiency_score=deterministic_efficiency,
            justification=(
                f"LLM analysis is temporarily unavailable because {_llm_failure_reason(exc)}. "
                "Scores are provisional estimates from transcript and SOP keyword coverage."
                + DEGRADED_MODE_SUFFIX
            ),
            missing_sop_steps=deterministic_missing,
            evidence_quotes=evidence_quotes,
            citations=citations,
            insufficient_evidence=not bool(evidence_quotes),
            confidence_score=None,
        )

    if not result.detected_topic.strip():
        result.detected_topic = topic_hint

    result.missing_sop_steps = _merge_missing_steps(
        deterministic_missing=deterministic_missing,
        llm_missing=result.missing_sop_steps,
    )
    result.efficiency_score = max(
        1,
        min(10, int(round((result.efficiency_score + deterministic_efficiency) / 2))),
    )
    result.is_resolved = result.is_resolved and deterministic_resolved
    if not result.evidence_quotes:
        result.evidence_quotes = _quote_candidates(transcript_text, max_quotes=3)
    if not result.citations:
        result.citations = [
            EvidenceCitation(source="transcript", speaker="unknown", quote=quote)
            for quote in result.evidence_quotes
        ]
        sop_quote = _quote_candidates(retrieved_sop, max_quotes=1)
        if sop_quote:
            result.citations.append(
                EvidenceCitation(source="sop", speaker="system", quote=sop_quote[0])
            )
    if result.confidence_score is None and retrieved_sop_chunks:
        scored_chunks = [chunk.score for chunk in retrieved_sop_chunks if chunk.score is not None]
        if scored_chunks:
            result.confidence_score = _clamp_unit(max(scored_chunks))
    return result


async def run_single_claim_nli_policy_check(
    agent_statement: str,
    ground_truth_policy: str,
) -> NLIEvaluation:
    """
    Single-claim NLI policy alignment check.

    This function validates one agent statement against one policy context and
    returns entailment-style categories. It is intentionally separate from
    transcript-level compliance reporting.
    """
    chain = build_nli_policy_chain()
    try:
        from app.llm_trigger.chains import _invoke_chain_with_retry
        result = await _invoke_chain_with_retry(
            chain,
            {
                "agent_statement": _sanitize_for_prompt(agent_statement),
                "ground_truth_policy": _sanitize_for_prompt(ground_truth_policy, max_length=6000),
            }
        )
    except Exception as exc:
        logger.warning("NLI policy LLM chain failed, using deterministic fallback: %s", exc)
        statement_quote = _quote_candidates(agent_statement, max_quotes=1)
        policy_quote = _quote_candidates(ground_truth_policy, max_quotes=1)
        evidence_quotes = statement_quote + [q for q in policy_quote if q not in statement_quote]
        citations: list[EvidenceCitation] = []
        if statement_quote:
            citations.append(EvidenceCitation(source="transcript", speaker="agent", quote=statement_quote[0]))
        if policy_quote:
            citations.append(EvidenceCitation(source="policy", speaker="system", quote=policy_quote[0]))

        return NLIEvaluation(
            nli_category="Benign Deviation",
            justification=(
                "Deterministic fallback was used because the LLM NLI service was unavailable. "
                "This label is provisional and should be revalidated when LLM connectivity is restored."
                + DEGRADED_MODE_SUFFIX
            ),
            evidence_quotes=evidence_quotes,
            citations=citations,
            insufficient_evidence=not bool(citations),
            confidence_score=None,
            policy_alignment_score=None,
        )
    if not result.evidence_quotes:
        result.evidence_quotes = _quote_candidates(
            f"{agent_statement}\n{ground_truth_policy}",
            max_quotes=3,
        )
    if not result.citations:
        citations: list[EvidenceCitation] = []
        statement_quote = _quote_candidates(agent_statement, max_quotes=1)
        if statement_quote:
            citations.append(
                EvidenceCitation(source="transcript", speaker="agent", quote=statement_quote[0])
            )
        policy_quote = _quote_candidates(ground_truth_policy, max_quotes=1)
        if policy_quote:
            citations.append(
                EvidenceCitation(source="policy", speaker="system", quote=policy_quote[0])
            )
        result.citations = citations
    if result.policy_alignment_score is None:
        category = (result.nli_category or "").strip().lower()
        if category == "entailment":
            result.policy_alignment_score = 0.85
        elif category == "benign deviation":
            result.policy_alignment_score = 0.45
        elif category in {"contradiction", "policy hallucination"}:
            result.policy_alignment_score = 0.1
    return result


async def run_nli_policy_check(
    agent_statement: str,
    ground_truth_policy: str,
) -> NLIEvaluation:
    """Backward-compatible alias for single-claim NLI policy alignment checks."""
    return await run_single_claim_nli_policy_check(
        agent_statement=agent_statement,
        ground_truth_policy=ground_truth_policy,
    )


def _derive_llm_inputs(
    utterances: list[Utterance],
    transcript_text: str,
    agent_name: str | None,
) -> tuple[str, str, str, str, str]:
    customer_fragments: list[str] = []
    agent_fragments: list[str] = []
    customer_emotion_weights: dict[str, float] = {}
    acoustic_emotion = "neutral"
    acoustic_confidence: float | None = None

    for index, utterance in enumerate(utterances):
        if not utterance.text:
            continue
        role = utterance.speaker_role.value if utterance.speaker_role else ""

        if role == "customer":
            customer_fragments.append(utterance.text)
            if utterance.emotion:
                normalized_emotion = _normalize_acoustic_label(utterance.emotion)
                confidence = float(utterance.emotion_confidence) if utterance.emotion_confidence is not None else 0.5
                recency_weight = 1.0 + (((index + 1) / max(1, len(utterances))) * 0.08)
                customer_emotion_weights[normalized_emotion] = customer_emotion_weights.get(normalized_emotion, 0.0) + (confidence * recency_weight)

        if role == "agent":
            agent_fragments.append(utterance.text)

    if customer_emotion_weights:
        def _emotion_rank(item: tuple[str, float]) -> tuple[float, int]:
            label, weight = item
            polarity_rank = {
                "negative": 2,
                "neutral": 1,
                "positive": 0,
            }.get(_emotion_polarity(label), 0)
            return (weight, polarity_rank)

        acoustic_emotion = max(customer_emotion_weights.items(), key=_emotion_rank)[0]
        total_weight = sum(customer_emotion_weights.values())
        acoustic_confidence = customer_emotion_weights[acoustic_emotion] / total_weight if total_weight > 0 else 0.5

    customer_text = _truncate_text("\n".join(customer_fragments), 2400)
    agent_statement = _truncate_text("\n".join(agent_fragments), 2400)

    if not customer_text:
        customer_text = _truncate_text(transcript_text, 900) if transcript_text else "No customer text available."

    if not agent_statement:
        agent_statement = _truncate_text(transcript_text, 900) if transcript_text else "No agent statement available."

    agent_label = agent_name or "Unknown Agent"
    agent_context = (
        f"Agent name: {agent_label}. "
        "Analyze full-call behavior in a customer-service quality assurance setting."
    )
    fused = fuse_emotion_signals(
        text=customer_text,
        acoustic_emotion=acoustic_emotion,
        acoustic_confidence=acoustic_confidence,
    )
    return agent_context, customer_text, acoustic_emotion, fused.emotion, agent_statement


async def _evaluate_emotion_pipeline(
    agent_context: str,
    customer_text: str,
    fused_emotion: str,
) -> EmotionShiftAnalysis:
    return await analyze_emotion_shift(
        agent_context=agent_context,
        customer_text=customer_text,
        acoustic_emotion=fused_emotion,
    )


async def _evaluate_policy_and_sop_trigger_checks(
    process_context_text: str,
    sop_context: str,
    sop_chunks: list[RetrievedChunk],
    org_filter: str | None,
    agent_statement: str,
    policy_context: str,
) -> tuple[ProcessAdherenceReport, NLIEvaluation]:
    """
    Run trigger checks that consume retrieved SOP/policy context.

    RAG has already retrieved the context before this function is called. This
    helper only performs judgment:
      - SOP process adherence for a transcript window.
      - NLI policy alignment for one agent statement.
    """
    process_task = evaluate_process_adherence(
        transcript_text=process_context_text,
        retrieved_sop_from_pinecone=sop_context,
        org_filter=org_filter,
        retrieved_sop_chunks=sop_chunks,
    )
    nli_task = run_single_claim_nli_policy_check(
        agent_statement=agent_statement,
        ground_truth_policy=policy_context,
    )
    return await asyncio.gather(process_task, nli_task)


async def _evaluate_transcript_policy_pipeline(
    transcript_text: str,
    org_filter: str | None,
) -> Any | None:
    """
    Run transcript-level compliance evaluation using the dedicated evaluator.

    This call is supplemental for trigger orchestration clarity and does not
    change the public trigger response schema.
    """
    evaluator = _get_policy_compliance_evaluator()
    if evaluator is None:
        return None
    try:
        return await asyncio.to_thread(
            evaluator.check,
            transcript_text,
            org_filter,
            False,
        )
    except Exception as exc:
        logger.warning("PolicyComplianceEvaluator failed in trigger pipeline: %s", exc)
        return None


async def _load_cached_trigger_report(
    session: AsyncSession,
    interaction_id: UUID,
    org_filter: str | None,
) -> InteractionLLMTriggerReport | None:
    cached_result = await session.exec(
        select(InteractionLLMTriggerCache).where(InteractionLLMTriggerCache.interaction_id == interaction_id)
    )
    cached = cached_result.first()
    if not cached or not cached.report_payload:
        return None
    if org_filter and cached.org_filter and cached.org_filter != org_filter:
        return None
    payload_version = cached.report_payload.get("_schema_version", 1) if isinstance(cached.report_payload, dict) else 1
    if payload_version < CACHE_SCHEMA_VERSION:
        logger.info(
            "Cached LLM trigger payload for interaction %s has schema version %d (current: %d), invalidating.",
            interaction_id,
            payload_version,
            CACHE_SCHEMA_VERSION,
        )
        return None
    try:
        return InteractionLLMTriggerReport.model_validate(cached.report_payload)
    except Exception:
        logger.warning("Ignoring invalid cached LLM trigger payload for interaction %s", interaction_id, exc_info=True)
        return None


async def _persist_trigger_report(
    session: AsyncSession,
    report: InteractionLLMTriggerReport,
    org_filter: str | None,
    *,
    commit: bool,
) -> None:
    cached_result = await session.exec(
        select(InteractionLLMTriggerCache).where(InteractionLLMTriggerCache.interaction_id == report.interaction_id)
    )
    cached = cached_result.first() or InteractionLLMTriggerCache(interaction_id=report.interaction_id)
    cached.org_filter = org_filter
    cached.report_payload = {**report.model_dump(mode="json"), "_schema_version": CACHE_SCHEMA_VERSION}
    cached.computed_at = datetime.now(timezone.utc)
    session.add(cached)
    if commit:
        await session.commit()
    else:
        await session.flush()


async def invalidate_llm_trigger_cache(
    session: AsyncSession,
    org_filter: str | None = None,
) -> int:
    from sqlalchemy import delete
    stmt = delete(InteractionLLMTriggerCache)
    if org_filter:
        stmt = stmt.where(InteractionLLMTriggerCache.org_filter == org_filter)
    result = await session.exec(stmt)
    await session.commit()
    count = result.rowcount if hasattr(result, 'rowcount') else 0
    logger.info("Invalidated %d LLM trigger cache entries (org_filter=%s)", count, org_filter)
    return count


async def evaluate_interaction_triggers(
    session: AsyncSession,
    interaction_id: UUID,
    retrieved_sop_from_pinecone: str = "",
    ground_truth_policy: str = "",
    org_filter: str | None = None,
    force_rerun: bool = False,
    commit_cache: bool = False,
) -> InteractionLLMTriggerReport:
    """
    Orchestrate the interaction-level trigger pipeline with explicit layer roles:

    1) RAG retrieval layer resolves SOP/policy grounding context.
    2) PolicyComplianceEvaluator produces transcript-level compliance report.
    3) NLI policy check validates single agent claims against policy context.
    """
    _log_step(interaction_id, "start", force_rerun=force_rerun, org_filter=org_filter)
    use_cached_report = (
        not force_rerun
        and not retrieved_sop_from_pinecone.strip()
        and not ground_truth_policy.strip()
    )
    if use_cached_report:
        cached_report = await _load_cached_trigger_report(
            session=session,
            interaction_id=interaction_id,
            org_filter=org_filter,
        )
        if cached_report is not None:
            _log_step(interaction_id, "cache_hit")
            return cached_report

    interaction_result = await session.exec(
        select(Interaction).where(Interaction.id == interaction_id)
    )
    interaction = interaction_result.first()
    if not interaction:
        raise ValueError("Interaction not found.")

    transcript_result = await session.exec(
        select(Transcript).where(Transcript.interaction_id == interaction_id)
    )
    transcript = transcript_result.first()

    utterance_result = await session.exec(
        select(Utterance)
        .where(Utterance.interaction_id == interaction_id)
        .order_by(Utterance.sequence_index)
    )
    utterances = list(utterance_result.all())

    agent_result = await session.exec(select(User).where(User.id == interaction.agent_id))
    agent = agent_result.first()

    transcript_text = (transcript.full_text if transcript and transcript.full_text else "").strip()
    if not transcript_text:
        transcript_text = _reconstruct_transcript(utterances)

    if not transcript_text:
        raise ValueError("No transcript text available for this interaction.")

    rolling_windows = _build_rolling_windows(utterances)
    selected_emotion_window = _select_emotion_window(rolling_windows)
    process_windows = _select_process_windows(rolling_windows)

    process_context_text = _render_window_bundle(process_windows) or transcript_text

    agent_context, customer_text, acoustic_emotion, fused_emotion, agent_statement = _derive_llm_inputs(
        utterances=utterances,
        transcript_text=transcript_text,
        agent_name=agent.name if agent else None,
    )

    if selected_emotion_window:
        agent_context += (
            f" Focus window: {selected_emotion_window.window_id} "
            f"(turns {selected_emotion_window.start_index}-{selected_emotion_window.end_index}, "
            f"time {_format_timestamp(selected_emotion_window.start_seconds)}-"
            f"{_format_timestamp(selected_emotion_window.end_seconds)})."
        )

    try:
        sop_resolution = resolve_retrieved_sop_context(
            transcript_text=transcript_text,
            retrieved_sop_from_pinecone=retrieved_sop_from_pinecone,
            org_filter=org_filter,
        )
        sop_context = sop_resolution.text
        sop_chunks = sop_resolution.chunks
    except Exception:
        sop_context = ""
        sop_chunks = []

    policy_context = await _resolve_active_policy_context(
        session=session,
        organization_id=interaction.organization_id,
        ground_truth_policy=ground_truth_policy,
        fallback_sop=sop_context,
        query_text=agent_statement or transcript_text,
        org_filter=org_filter,
    )

    emotion_pipeline_task = _evaluate_emotion_pipeline(
        agent_context=agent_context,
        customer_text=customer_text,
        fused_emotion=fused_emotion,
    )
    grounded_trigger_checks_task = _evaluate_policy_and_sop_trigger_checks(
        process_context_text=process_context_text,
        sop_context=sop_context,
        sop_chunks=sop_chunks,
        org_filter=org_filter,
        agent_statement=agent_statement,
        policy_context=policy_context.text,
    )
    transcript_policy_task = _evaluate_transcript_policy_pipeline(
        transcript_text=process_context_text,
        org_filter=org_filter,
    )

    emotion_shift, grounded_trigger_checks, transcript_policy_report = await asyncio.gather(
        emotion_pipeline_task,
        grounded_trigger_checks_task,
        transcript_policy_task,
    )
    _log_step(interaction_id, "llm_chains_complete")
    if transcript_policy_report is not None:
        _log_step(
            interaction_id,
            "transcript_policy_report_complete",
            compliance_score=getattr(transcript_policy_report, "compliance_score", None),
            violations=len(getattr(transcript_policy_report, "violations", []) or []),
        )
    process_adherence, nli_policy = grounded_trigger_checks

    window_citations = _window_citations(process_windows)
    if window_citations:
        if not process_adherence.citations:
            process_adherence.citations = []
        process_adherence.citations.extend(window_citations)

    if selected_emotion_window:
        emotion_window_quote = _quote_candidates(selected_emotion_window.text, max_quotes=1)
        if emotion_window_quote:
            emotion_shift.citations.append(
                EvidenceCitation(
                    source="transcript",
                    speaker="unknown",
                    utterance_index=selected_emotion_window.start_index,
                    quote=emotion_window_quote[0],
                )
            )

    _backfill_transcript_citation_indices(emotion_shift.citations, utterances)
    _backfill_transcript_citation_indices(process_adherence.citations, utterances)
    _backfill_transcript_citation_indices(nli_policy.citations, utterances)

    if not _has_mapped_transcript_citation(emotion_shift.citations):
        emotion_shift.root_cause = INSUFFICIENT_EVIDENCE_LABEL
        emotion_shift.insufficient_evidence = True

    if not _has_mapped_transcript_citation(process_adherence.citations):
        process_adherence.justification = INSUFFICIENT_EVIDENCE_LABEL
        process_adherence.insufficient_evidence = True

    _ensure_minimum_policy_citation(nli_policy, policy_context.text)
    has_policy_citation = any(c.source == "policy" and bool((c.quote or "").strip()) for c in nli_policy.citations)
    if not has_policy_citation:
        nli_policy.justification = INSUFFICIENT_EVIDENCE_LABEL
        nli_policy.insufficient_evidence = True

    nli_policy.policy_version = policy_context.version
    nli_policy.policy_effective_at = policy_context.effective_at
    nli_policy.policy_category = policy_context.category
    nli_policy.conflict_resolution_applied = policy_context.conflict_resolution_applied
    try:
        policy_chunks = retrieve_policy_chunks(
            query_text=agent_statement,
            org_filter=org_filter,
            top_k=3,
        )
    except Exception:
        policy_chunks = []
    explainability = _build_explainability_layer(
        emotion_shift=emotion_shift,
        process_adherence=process_adherence,
        nli_policy=nli_policy,
        utterances=utterances,
        acoustic_emotion=acoustic_emotion,
        sop_context=sop_context,
        sop_chunks=sop_chunks,
        policy_context=policy_context,
        policy_chunks=policy_chunks,
        agent_statement=agent_statement,
        org_filter=org_filter,
    )

    report = InteractionLLMTriggerReport(
        interaction_id=interaction_id,
        emotion_shift=emotion_shift,
        process_adherence=process_adherence,
        nli_policy=nli_policy,
        derived_customer_text=customer_text,
        derived_acoustic_emotion=acoustic_emotion,
        derived_fused_emotion=fused_emotion,
        derived_agent_statement=agent_statement,
        explainability=explainability,
    )

    if not retrieved_sop_from_pinecone.strip() and not ground_truth_policy.strip():
        await _persist_trigger_report(
            session=session,
            report=report,
            org_filter=org_filter,
            commit=commit_cache,
        )
        _log_step(interaction_id, "cache_persisted")

    _log_step(interaction_id, "complete")
    return report
