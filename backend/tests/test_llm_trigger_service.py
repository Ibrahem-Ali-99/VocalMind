from unittest.mock import patch

import pytest

from app.llm_trigger.schemas import EmotionShiftAnalysis, ProcessAdherenceReport
from app.llm_trigger.service import (
    _detect_cross_modal_dissonance,
    _detect_topic,
    _trajectory_missing_steps,
    analyze_emotion_shift,
    evaluate_process_adherence,
)


class _FakeProcessChain:
    async def ainvoke(self, payload):
        return ProcessAdherenceReport(
            detected_topic=payload.get("topic_hint", "refund_request"),
            is_resolved=True,
            efficiency_score=8,
            missing_sop_steps=["Confirm refund method and timeline"],
        )


@pytest.mark.asyncio
async def test_analyze_emotion_shift_skips_llm_when_no_dissonance():
    with patch("app.llm_trigger.service.build_emotion_shift_chain") as mock_builder:
        result = await analyze_emotion_shift(
            agent_context="Agent context",
            customer_text="Thank you for your help, that was great.",
            acoustic_emotion="happy",
        )

    assert isinstance(result, EmotionShiftAnalysis)
    assert result.is_dissonance_detected is False
    assert result.dissonance_type == "None"
    mock_builder.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_emotion_shift_runs_llm_when_dissonance():
    class _FakeEmotionChain:
        async def ainvoke(self, _payload):
            return EmotionShiftAnalysis(
                is_dissonance_detected=True,
                dissonance_type="None",
                root_cause="Positive lexical phrase with negative tone.",
                counterfactual_correction="If the agent had acknowledged frustration first, escalation may have dropped.",
            )

    with patch("app.llm_trigger.service.build_emotion_shift_chain", return_value=_FakeEmotionChain()):
        result = await analyze_emotion_shift(
            agent_context="Agent context",
            customer_text="That is just perfect, thanks a lot.",
            acoustic_emotion="anger",
        )

    assert result.is_dissonance_detected is True
    assert result.dissonance_type == "Sarcasm"


def test_detect_cross_modal_dissonance_heuristic():
    assert _detect_cross_modal_dissonance("Everything is perfect, thanks.", "anger") is True
    assert _detect_cross_modal_dissonance("This is unacceptable and terrible.", "happy") is True
    assert _detect_cross_modal_dissonance("Thanks for solving that quickly.", "happy") is False


def test_topic_and_trajectory_mapping_helpers():
    transcript = (
        "customer: I need a refund for my order.\n"
        "agent: Sure, let me check your order number and eligibility window.\n"
        "agent: I can submit the refund now."
    )
    sop = "1. Collect order identifier\n2. Verify refund eligibility window\n3. Confirm refund method and timeline"

    topic = _detect_topic(transcript, sop)
    assert topic == "refund_request"

    missing = _trajectory_missing_steps(
        transcript,
        [
            "Collect order identifier",
            "Verify refund eligibility window",
            "Confirm refund method and timeline",
            "Close with summary and next steps",
        ],
    )
    assert "Close with summary and next steps" in missing


@pytest.mark.asyncio
async def test_evaluate_process_adherence_merges_deterministic_and_llm_steps():
    transcript = (
        "customer: I need a refund.\n"
        "agent: I can help with that.\n"
        "agent: I checked your order and policy window."
    )

    with patch("app.llm_trigger.service.build_process_adherence_chain", return_value=_FakeProcessChain()):
        result = await evaluate_process_adherence(
            transcript_text=transcript,
            retrieved_sop_from_pinecone="",
            org_filter=None,
        )

    assert isinstance(result, ProcessAdherenceReport)
    assert result.detected_topic == "refund_request"
    assert 1 <= result.efficiency_score <= 10
    assert len(result.missing_sop_steps) >= 1
