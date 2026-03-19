from __future__ import annotations

import asyncio
import re
from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.llm_trigger.chains import (
    build_emotion_shift_chain,
    build_nli_policy_chain,
    build_process_adherence_chain,
)
from app.llm_trigger.retrieval import resolve_retrieved_sop
from app.llm_trigger.schemas import (
    EmotionShiftAnalysis,
    InteractionLLMTriggerReport,
    NLIEvaluation,
    ProcessAdherenceReport,
)
from app.models.interaction import Interaction
from app.models.transcript import Transcript
from app.models.utterance import Utterance
from app.models.user import User


POSITIVE_TEXT_WORDS = {
    "good",
    "great",
    "perfect",
    "excellent",
    "thanks",
    "thank",
    "appreciate",
    "happy",
    "glad",
    "awesome",
}

NEGATIVE_TEXT_WORDS = {
    "bad",
    "angry",
    "upset",
    "frustrated",
    "terrible",
    "awful",
    "annoying",
    "ridiculous",
    "unacceptable",
    "disappointed",
}

POSITIVE_ACOUSTIC = {"happy", "joy", "calm", "neutral", "satisfied"}
NEGATIVE_ACOUSTIC = {"anger", "angry", "frustration", "frustrated", "disgust", "sad", "fear"}

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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _text_sentiment_polarity(text: str) -> str:
    tokens = _tokenize(text)
    pos = sum(1 for token in tokens if token in POSITIVE_TEXT_WORDS)
    neg = sum(1 for token in tokens if token in NEGATIVE_TEXT_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _acoustic_polarity(acoustic_emotion: str) -> str:
    label = acoustic_emotion.strip().lower()
    if label in POSITIVE_ACOUSTIC:
        return "positive"
    if label in NEGATIVE_ACOUSTIC:
        return "negative"
    return "neutral"


def _detect_cross_modal_dissonance(customer_text: str, acoustic_emotion: str) -> bool:
    text_polarity = _text_sentiment_polarity(customer_text)
    acoustic_polarity = _acoustic_polarity(acoustic_emotion)
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


async def analyze_emotion_shift(
    agent_context: str,
    customer_text: str,
    acoustic_emotion: str,
) -> EmotionShiftAnalysis:
    if not _detect_cross_modal_dissonance(customer_text, acoustic_emotion):
        return EmotionShiftAnalysis(
            is_dissonance_detected=False,
            dissonance_type="None",
            root_cause="No strong contradiction detected between text sentiment and acoustic emotion.",
            counterfactual_correction="If the agent had continued the same supportive approach, the interaction likely would have remained stable.",
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
    return result


async def run_nli_policy_check(
    agent_statement: str,
    ground_truth_policy: str,
) -> NLIEvaluation:
    chain = build_nli_policy_chain()
    return await chain.ainvoke(
        {
            "agent_statement": agent_statement,
            "ground_truth_policy": ground_truth_policy,
        }
    )


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
) -> tuple[str, str, str, str]:
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
    return agent_context, customer_text, acoustic_emotion, agent_statement


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

    agent_context, customer_text, acoustic_emotion, agent_statement = _derive_llm_inputs(
        utterances=utterances,
        transcript_text=transcript_text,
        agent_name=agent.name if agent else None,
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
        acoustic_emotion=acoustic_emotion,
    )
    process_task = evaluate_process_adherence(
        transcript_text=transcript_text,
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

    return InteractionLLMTriggerReport(
        interaction_id=interaction_id,
        emotion_shift=emotion_shift,
        process_adherence=process_adherence,
        nli_policy=nli_policy,
        derived_customer_text=customer_text,
        derived_acoustic_emotion=acoustic_emotion,
        derived_agent_statement=agent_statement,
    )
