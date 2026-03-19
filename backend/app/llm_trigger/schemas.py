from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class EmotionShiftAnalysis(BaseModel):
    is_dissonance_detected: bool = Field(
        description="Whether acoustic emotion contradicts text sentiment."
    )
    dissonance_type: str = Field(
        description='One of values like "Sarcasm", "Passive-Aggression", or "None".'
    )
    root_cause: str = Field(description="Transcript-grounded explanation of the mismatch.")
    counterfactual_correction: str = Field(
        description='Actionable rewrite starting with "If the agent had...".'
    )

    @field_validator("counterfactual_correction", mode="before")
    @classmethod
    def ensure_counterfactual_prefix(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            return "If the agent had used a clearer empathy-and-verification step, escalation risk could have decreased."
        if text.lower().startswith("if the agent had"):
            return text
        return f"If the agent had {text[0].lower() + text[1:] if len(text) > 1 else text.lower()}"


class ProcessAdherenceReport(BaseModel):
    detected_topic: str = Field(description="Detected customer-service topic.")
    is_resolved: bool = Field(description="Whether issue appears resolved by end of dialogue.")
    efficiency_score: int = Field(
        ge=1,
        le=10,
        description="Resolution efficiency score where 10 is optimal process adherence.",
    )
    missing_sop_steps: list[str] = Field(
        default_factory=list,
        description="SOP steps that are absent or weakly executed in transcript.",
    )


NLICategory = Literal[
    "Entailment",
    "Benign Deviation",
    "Contradiction",
    "Policy Hallucination",
]


class NLIEvaluation(BaseModel):
    nli_category: NLICategory
    justification: str = Field(description="Short evidence-backed rationale for the label.")


class InteractionLLMTriggerReport(BaseModel):
    interaction_id: UUID
    emotion_shift: EmotionShiftAnalysis
    process_adherence: ProcessAdherenceReport
    nli_policy: NLIEvaluation
    derived_customer_text: str
    derived_acoustic_emotion: str
    derived_agent_statement: str
