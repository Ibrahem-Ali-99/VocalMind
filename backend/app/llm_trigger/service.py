from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.emotion_fusion import EMOTION_NORMALIZATION, fuse_emotion_signals, infer_text_emotion_with_provider
from app.llm_trigger.chains import (
    build_emotion_shift_chain,
    build_nli_policy_chain,
    build_process_adherence_chain,
)
from app.llm_trigger.retrieval import resolve_retrieved_sop
from app.llm_trigger.schemas import (
    EvidenceCitation,
    EmotionShiftAnalysis,
    InteractionLLMTriggerReport,
    NLIEvaluation,
    ProcessAdherenceReport,
)
from app.models.interaction import Interaction
from app.models.transcript import Transcript
from app.models.utterance import Utterance
from app.models.user import User


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


@dataclass
class TranscriptWindow:
    window_id: str
    start_index: int
    end_index: int
    start_seconds: float
    end_seconds: float
    text: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _normalize_acoustic_label(label: str) -> str:
    base = (label or "neutral").strip().lower()
    return EMOTION_NORMALIZATION.get(base, base or "neutral")


def _emotion_polarity(label: str) -> str:
    normalized = _normalize_acoustic_label(label)
    if normalized in {"happy", "neutral"}:
        return "positive"
    if normalized in {"angry", "frustrated", "sad"}:
        return "negative"
    return "neutral"


def _detect_cross_modal_dissonance(customer_text: str, acoustic_emotion: str) -> bool:
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


def _extract_sop_steps(retrieved_sop: str) -> list[str]:
    steps: list[str] = []
    for line in retrieved_sop.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if not cleaned:
            continue
        if len(cleaned.split()) >= 4:
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


def _is_resolved_heuristic(transcript_text: str) -> bool:
    text = transcript_text.lower()
    positive_endings = ["resolved", "fixed", "works now", "thank you", "anything else"]
    unresolved_markers = ["still not", "didn't work", "not fixed", "call back"]
    if any(marker in text for marker in unresolved_markers):
        return False
    return any(marker in text for marker in positive_endings)


def _efficiency_score_heuristic(transcript_text: str, missing_steps: list[str], expected_steps: list[str]) -> int:
    if not expected_steps:
        return 6
    coverage = (len(expected_steps) - len(missing_steps)) / len(expected_steps)
    score = int(round(1 + 9 * coverage))
    turns = len([line for line in transcript_text.splitlines() if line.strip()])
    if turns > 20:
        score -= 1
    if turns > 30:
        score -= 1
    return max(1, min(10, score))


def _split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


def _quote_candidates(text: str, max_quotes: int = 3) -> list[str]:
    if not text:
        return []

    sentences = _split_sentences(text)
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


def _format_timestamp(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    return f"{seconds_int // 60:02d}:{seconds_int % 60:02d}"


def _window_text(utterances: list[Utterance]) -> str:
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
        text = _window_text(chunk)
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
    if not _detect_cross_modal_dissonance(customer_text, acoustic_emotion):
        quotes = _quote_candidates(customer_text, max_quotes=2)
        return EmotionShiftAnalysis(
            is_dissonance_detected=False,
            dissonance_type="None",
            root_cause="No strong contradiction detected between text sentiment and acoustic emotion.",
            counterfactual_correction="If the agent had continued the same supportive approach, the interaction likely would have remained stable.",
            evidence_quotes=quotes,
            citations=[
                EvidenceCitation(
                    source="transcript",
                    speaker="customer",
                    quote=quote,
                )
                for quote in quotes
            ],
        )

    chain = build_emotion_shift_chain()
    result = await chain.ainvoke(
        {
            "agent_context": agent_context,
            "customer_text": customer_text,
            "acoustic_emotion": acoustic_emotion,
        }
    )
    result.is_dissonance_detected = True
    if result.dissonance_type.strip().lower() == "none":
        result.dissonance_type = "Sarcasm"
    if not result.evidence_quotes:
        result.evidence_quotes = _quote_candidates(customer_text, max_quotes=3)
    if not result.citations:
        result.citations = [
            EvidenceCitation(source="transcript", speaker="customer", quote=quote)
            for quote in result.evidence_quotes
        ]
    return result


async def evaluate_process_adherence(
    transcript_text: str,
    retrieved_sop_from_pinecone: str,
    org_filter: str | None = None,
) -> ProcessAdherenceReport:
    try:
        retrieved_sop = resolve_retrieved_sop(
            transcript_text=transcript_text,
            retrieved_sop_from_pinecone=retrieved_sop_from_pinecone,
            org_filter=org_filter,
        )
    except Exception:
        retrieved_sop = ""

    topic_hint = _detect_topic(transcript_text, retrieved_sop)
    expected_steps = RESOLUTION_GRAPHS.get(topic_hint, []).copy()
    expected_steps.extend(step for step in _extract_sop_steps(retrieved_sop) if step not in expected_steps)
    expected_steps = expected_steps[:8]

    deterministic_missing = _trajectory_missing_steps(transcript_text, expected_steps)
    deterministic_efficiency = _efficiency_score_heuristic(
        transcript_text=transcript_text,
        missing_steps=deterministic_missing,
        expected_steps=expected_steps,
    )
    deterministic_resolved = _is_resolved_heuristic(transcript_text)

    chain = build_process_adherence_chain()
    result = await chain.ainvoke(
        {
            "topic_hint": topic_hint,
            "transcript_text": transcript_text,
            "retrieved_sop": retrieved_sop or "No SOP context found.",
            "expected_resolution_graph": "\n".join(
                f"- {step}" for step in expected_steps
            )
            or "- No explicit graph available.",
        }
    )

    if not result.detected_topic.strip():
        result.detected_topic = topic_hint

    merged_missing = list(dict.fromkeys(deterministic_missing + result.missing_sop_steps))
    result.missing_sop_steps = merged_missing
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
    return result


async def run_nli_policy_check(
    agent_statement: str,
    ground_truth_policy: str,
) -> NLIEvaluation:
    chain = build_nli_policy_chain()
    result = await chain.ainvoke(
        {
            "agent_statement": agent_statement,
            "ground_truth_policy": ground_truth_policy,
        }
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
    return result


def _reconstruct_transcript(utterances: list[Utterance]) -> str:
    lines: list[str] = []
    for utterance in utterances:
        if not utterance.text:
            continue
        speaker = utterance.speaker_role.value if utterance.speaker_role else "unknown"
        lines.append(f"{speaker}: {utterance.text}")
    return "\n".join(lines)


def _derive_llm_inputs(
    utterances: list[Utterance],
    transcript_text: str,
    agent_name: str | None,
) -> tuple[str, str, str, str, str]:
    customer_text = ""
    acoustic_emotion = "neutral"
    agent_statement = ""

    for utterance in utterances:
        if not utterance.text:
            continue
        role = utterance.speaker_role.value if utterance.speaker_role else ""

        if role == "customer" and not customer_text:
            customer_text = utterance.text
            if utterance.emotion:
                acoustic_emotion = utterance.emotion

        if role == "agent":
            agent_statement = utterance.text

    if not customer_text:
        customer_text = transcript_text[:500] if transcript_text else "No customer text available."

    if not agent_statement:
        agent_statement = transcript_text[:500] if transcript_text else "No agent statement available."

    agent_label = agent_name or "Unknown Agent"
    agent_context = (
        f"Agent name: {agent_label}. "
        "Analyze behavior in a customer-service quality assurance setting."
    )
    fused = fuse_emotion_signals(
        text=customer_text,
        acoustic_emotion=acoustic_emotion,
    )
    return agent_context, customer_text, acoustic_emotion, fused.emotion, agent_statement


async def evaluate_interaction_triggers(
    session: AsyncSession,
    interaction_id: UUID,
    retrieved_sop_from_pinecone: str = "",
    ground_truth_policy: str = "",
    org_filter: str | None = None,
) -> InteractionLLMTriggerReport:
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

    emotion_context_text = selected_emotion_window.text if selected_emotion_window else transcript_text
    process_context_text = _render_window_bundle(process_windows) or transcript_text

    agent_context, customer_text, acoustic_emotion, fused_emotion, agent_statement = _derive_llm_inputs(
        utterances=utterances,
        transcript_text=emotion_context_text,
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
        sop_context = resolve_retrieved_sop(
            transcript_text=transcript_text,
            retrieved_sop_from_pinecone=retrieved_sop_from_pinecone,
            org_filter=org_filter,
        )
    except Exception:
        sop_context = ""

    policy_context = (
        ground_truth_policy.strip()
        or sop_context.strip()
        or "No ground truth policy context provided."
    )

    emotion_task = analyze_emotion_shift(
        agent_context=agent_context,
        customer_text=customer_text,
        acoustic_emotion=fused_emotion,
    )
    process_task = evaluate_process_adherence(
        transcript_text=process_context_text,
        retrieved_sop_from_pinecone=sop_context,
        org_filter=org_filter,
    )
    nli_task = run_nli_policy_check(
        agent_statement=agent_statement,
        ground_truth_policy=policy_context,
    )

    emotion_shift, process_adherence, nli_policy = await asyncio.gather(
        emotion_task,
        process_task,
        nli_task,
    )

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

    return InteractionLLMTriggerReport(
        interaction_id=interaction_id,
        emotion_shift=emotion_shift,
        process_adherence=process_adherence,
        nli_policy=nli_policy,
        derived_customer_text=customer_text,
        derived_acoustic_emotion=acoustic_emotion,
        derived_fused_emotion=fused_emotion,
        derived_agent_statement=agent_statement,
    )
