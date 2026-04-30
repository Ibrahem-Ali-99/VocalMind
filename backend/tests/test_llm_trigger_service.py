from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.llm_trigger.retrieval import RetrievedChunk
from app.llm_trigger.schemas import EmotionShiftAnalysis, ProcessAdherenceReport
from app.llm_trigger.service import (
    _build_policy_reference_from_chunk,
    _build_emotion_transition_attributions,
    _build_rolling_windows,
    _detect_cross_modal_dissonance,
    _detect_topic,
    _detect_topic_from_sop_chunks,
    _derive_llm_inputs,
    _render_window_bundle,
    _trajectory_missing_steps,
    _window_citations,
    analyze_emotion_shift,
    evaluate_process_adherence,
)


class _FakeProcessChain:
    async def ainvoke(self, payload):
        return ProcessAdherenceReport(
            detected_topic=payload.get("topic_hint", "refund_request"),
            is_resolved=True,
            efficiency_score=8,
            justification="Agent completed key verification steps but skipped the refund confirmation closeout.",
            missing_sop_steps=["Confirm refund method and timeline"],
        )


class _FailingProcessChain:
    async def ainvoke(self, _payload):
        raise Exception("Error code: 429 - rate_limit_exceeded")


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

    with (
        patch("app.llm_trigger.service.infer_text_emotion_with_provider", return_value=("happy", 0.9)),
        patch("app.llm_trigger.service.build_emotion_shift_chain", return_value=_FakeEmotionChain()),
    ):
        result = await analyze_emotion_shift(
            agent_context="Agent context",
            customer_text="That is just perfect, thanks a lot.",
            acoustic_emotion="angry",
        )

    assert result.is_dissonance_detected is True
    assert result.dissonance_type == "Sarcasm"


def test_detect_cross_modal_dissonance_heuristic():
    with patch("app.llm_trigger.service.infer_text_emotion_with_provider", return_value=("happy", 0.95)):
        assert _detect_cross_modal_dissonance("Everything is perfect, thanks.", "angry") is True

    with patch("app.llm_trigger.service.infer_text_emotion_with_provider", return_value=("angry", 0.92)):
        assert _detect_cross_modal_dissonance("This is unacceptable and terrible.", "happy") is True

    with patch("app.llm_trigger.service.infer_text_emotion_with_provider", return_value=("happy", 0.98)):
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


def test_detect_topic_from_sop_chunk_reference_prefers_matched_document():
    chunks = [
        RetrievedChunk(
            text="[01-refund-request-processing.pdf]\nStep 1 - Open the Call & Verify Identity",
            metadata={"source_file": "01-refund-request-processing.pdf"},
            source="manual",
        )
    ]

    topic = _detect_topic_from_sop_chunks(chunks)
    assert topic == "refund_request"


def test_policy_reference_preserves_doc_type_and_policy_ref_metadata():
    chunk = RetrievedChunk(
        text="Step 4: Confirm refund timeline.",
        metadata={
            "doc_type": "sop",
            "policy_ref": ["CS-RULE-003"],
            "source_file": "refund-sop.md",
        },
        source="qdrant",
        collection="vocalmind_sop_parents",
    )

    reference = _build_policy_reference_from_chunk(
        chunk,
        source_kind="sop",
        fallback_text="",
        fallback_reference="Retrieved SOP",
    )

    assert reference is not None
    assert reference.doc_type == "sop"
    assert reference.policy_ref == ["CS-RULE-003"]


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
    assert result.detected_topic
    assert 1 <= result.efficiency_score <= 10
    assert len(result.missing_sop_steps) >= 1
    assert result.evidence_quotes
    assert result.citations


@pytest.mark.asyncio
async def test_evaluate_process_adherence_uses_sop_file_hint_for_topic():
    transcript = (
        "customer: My bill jumped after the promo expired and I want a refund.\n"
        "agent: I can check the billing error and apply the credit timeline."
    )
    chunks = [
        RetrievedChunk(
            text="[01-refund-request-processing.pdf]\nStep 1 - Open the Call & Verify Identity\nStep 7 - Apply Credit & Communicate Timeline",
            metadata={"source_file": "01-refund-request-processing.pdf"},
            source="manual",
        )
    ]

    with patch("app.llm_trigger.service.build_process_adherence_chain", return_value=_FakeProcessChain()):
        result = await evaluate_process_adherence(
            transcript_text=transcript,
            retrieved_sop_from_pinecone="",
            org_filter="nexalink",
            retrieved_sop_chunks=chunks,
        )

    assert result.detected_topic == "refund_request"


def test_build_emotion_transition_attributions_tracks_each_speaker_change():
    utterances = [
        SimpleNamespace(
            sequence_index=0,
            speaker_role=SimpleNamespace(value="customer"),
            text="I am very upset about this.",
            emotion="frustrated",
            emotion_confidence=0.82,
            start_time_seconds=0.0,
            end_time_seconds=2.0,
        ),
        SimpleNamespace(
            sequence_index=1,
            speaker_role=SimpleNamespace(value="customer"),
            text="Thank you, that helps a lot.",
            emotion="happy",
            emotion_confidence=0.88,
            start_time_seconds=18.0,
            end_time_seconds=21.0,
        ),
        SimpleNamespace(
            sequence_index=2,
            speaker_role=SimpleNamespace(value="agent"),
            text="Let me explain the next step.",
            emotion="neutral",
            emotion_confidence=0.55,
            start_time_seconds=22.0,
            end_time_seconds=24.0,
        ),
        SimpleNamespace(
            sequence_index=3,
            speaker_role=SimpleNamespace(value="agent"),
            text="Great, we have fully resolved it.",
            emotion="helpful",
            emotion_confidence=0.74,
            start_time_seconds=30.0,
            end_time_seconds=33.0,
        ),
    ]

    attributions = _build_emotion_transition_attributions(utterances)

    assert len(attributions) == 2
    assert attributions[0].family == "emotion"
    assert attributions[0].trigger_type == "Emotion Change"
    assert "customer" in attributions[0].title.lower()
    assert "agent" in attributions[1].title.lower()


@pytest.mark.asyncio
async def test_evaluate_process_adherence_fallback_mentions_rate_limit():
    transcript = (
        "customer: I need a refund.\n"
        "agent: I can help and check your eligibility now."
    )

    with patch("app.llm_trigger.service.build_process_adherence_chain", return_value=_FailingProcessChain()):
        result = await evaluate_process_adherence(
            transcript_text=transcript,
            retrieved_sop_from_pinecone="",
            org_filter=None,
        )

    assert "rate limit" in result.justification.lower()


def test_derive_llm_inputs_uses_full_call_customer_and_agent_turns():
    utterances = [
        SimpleNamespace(speaker_role=SimpleNamespace(value="customer"), text="I called yesterday.", emotion="neutral", emotion_confidence=0.3),
        SimpleNamespace(speaker_role=SimpleNamespace(value="agent"), text="I can help with that.", emotion=None, emotion_confidence=None),
        SimpleNamespace(speaker_role=SimpleNamespace(value="customer"), text="I am still frustrated because this is unresolved.", emotion="frustrated", emotion_confidence=0.82),
        SimpleNamespace(speaker_role=SimpleNamespace(value="agent"), text="I will process the refund right now.", emotion=None, emotion_confidence=None),
    ]

    agent_context, customer_text, acoustic_emotion, fused_emotion, agent_statement = _derive_llm_inputs(
        utterances=utterances,
        transcript_text="fallback transcript",
        agent_name="Sara",
    )

    assert "I called yesterday." in customer_text
    assert "still frustrated" in customer_text
    assert "I can help with that." in agent_statement
    assert "process the refund" in agent_statement
    assert acoustic_emotion == "frustrated"
    assert fused_emotion
    assert "full-call behavior" in agent_context


def test_derive_llm_inputs_uses_dominant_customer_emotion_not_last_turn_only():
    utterances = [
        SimpleNamespace(speaker_role=SimpleNamespace(value="customer"), text="This issue is still unresolved and very frustrating.", emotion="frustrated", emotion_confidence=0.94),
        SimpleNamespace(speaker_role=SimpleNamespace(value="customer"), text="I keep waiting and this is unacceptable.", emotion="angry", emotion_confidence=0.86),
        SimpleNamespace(speaker_role=SimpleNamespace(value="agent"), text="I understand your concern and will check now.", emotion=None, emotion_confidence=None),
        SimpleNamespace(speaker_role=SimpleNamespace(value="customer"), text="Thanks for checking that.", emotion="happy", emotion_confidence=0.31),
    ]

    _agent_context, _customer_text, acoustic_emotion, _fused_emotion, _agent_statement = _derive_llm_inputs(
        utterances=utterances,
        transcript_text="fallback transcript",
        agent_name="Sara",
    )

    assert acoustic_emotion in {"frustrated", "angry"}


def test_build_rolling_windows_and_bundle_for_long_transcript():
    utterances = [
        SimpleNamespace(
            speaker_role=SimpleNamespace(value="customer" if idx % 2 == 0 else "agent"),
            text=f"turn {idx} text",
            start_time_seconds=float(idx * 5),
            end_time_seconds=float((idx * 5) + 4),
        )
        for idx in range(10)
    ]

    windows = _build_rolling_windows(utterances, window_turns=4, stride=2)

    assert len(windows) == 4
    assert windows[0].start_index == 0
    assert windows[1].start_index == 2

    bundle = _render_window_bundle(windows)
    assert "[W0]" in bundle
    assert "turns 0-3" in bundle

    citations = _window_citations(windows)
    assert citations
    assert citations[0].source == "transcript"
