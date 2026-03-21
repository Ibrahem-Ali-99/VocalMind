from uuid import uuid4

from app.api.routes.interactions import _map_llm_trigger_report
from app.llm_trigger.schemas import (
    EmotionShiftAnalysis,
    EvidenceCitation,
    InteractionLLMTriggerReport,
    NLIEvaluation,
    ProcessAdherenceReport,
)


def test_map_llm_trigger_report_contains_process_and_nli_fields():
    report = InteractionLLMTriggerReport(
        interaction_id=uuid4(),
        emotion_shift=EmotionShiftAnalysis(
            is_dissonance_detected=True,
            dissonance_type="Sarcasm",
            root_cause="Customer wording conflicts with tone.",
            counterfactual_correction="If the agent had validated the customer concern first, tension may have dropped.",
            evidence_quotes=["I am fine, whatever."],
            citations=[
                EvidenceCitation(
                    source="transcript",
                    speaker="customer",
                    quote="I am fine, whatever.",
                    utterance_index=2,
                )
            ],
        ),
        process_adherence=ProcessAdherenceReport(
            detected_topic="billing_issue",
            is_resolved=False,
            efficiency_score=5,
            missing_sop_steps=["Confirm account details"],
            evidence_quotes=["Let me check your account."],
            citations=[
                EvidenceCitation(
                    source="sop",
                    speaker="system",
                    quote="Verify account and charge details",
                    utterance_index=None,
                )
            ],
        ),
        nli_policy=NLIEvaluation(
            nli_category="Contradiction",
            justification="Agent promised refund outside policy window.",
            evidence_quotes=["We can refund after 60 days.", "Refunds are allowed only within 30 days."],
            citations=[
                EvidenceCitation(
                    source="policy",
                    speaker="system",
                    quote="Refunds are allowed only within 30 days.",
                    utterance_index=None,
                )
            ],
        ),
        derived_customer_text="customer text",
        derived_acoustic_emotion="frustrated",
        derived_fused_emotion="frustrated",
        derived_agent_statement="agent statement",
    )

    payload = _map_llm_trigger_report(report)

    assert payload["available"] is True
    assert payload["processAdherence"]["detectedTopic"] == "billing_issue"
    assert payload["processAdherence"]["isResolved"] is False
    assert payload["processAdherence"]["efficiencyScore"] == 5
    assert payload["processAdherence"]["missingSopSteps"] == ["Confirm account details"]
    assert payload["nliPolicy"]["nliCategory"] == "Contradiction"
    assert payload["nliPolicy"]["justification"]
