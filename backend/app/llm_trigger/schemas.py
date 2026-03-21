from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


CitationSource = Literal["transcript", "policy", "sop", "acoustic"]
CitationSpeaker = Literal["customer", "agent", "system", "unknown"]


class EvidenceCitation(BaseModel):
    source: CitationSource = Field(description="Origin of the supporting quote.")
    quote: str = Field(description="Exact quote supporting the claim.")
    speaker: CitationSpeaker = Field(default="unknown", description="Speaker tied to the quote when available.")
    utterance_index: int | None = Field(
        default=None,
        ge=0,
        description="Utterance index for transcript citations when known.",
    )


def _normalize_quote_list(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    else:
        items = value

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        quote = (item or "").strip().strip('"')
        if not quote:
            continue
        if quote in seen:
            continue
        seen.add(quote)
        normalized.append(quote)
    return normalized


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
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Short verbatim quotes from transcript used as evidence.",
    )
    citations: list[EvidenceCitation] = Field(
        default_factory=list,
        description="Structured evidence citations supporting the analysis.",
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

    @field_validator("evidence_quotes", mode="before")
    @classmethod
    def normalize_emotion_evidence_quotes(cls, value: list[str] | str | None) -> list[str]:
        return _normalize_quote_list(value)


class ProcessAdherenceReport(BaseModel):
    detected_topic: str = Field(description="Detected customer-service topic.")
    is_resolved: bool = Field(description="Whether issue appears resolved by end of dialogue.")
    efficiency_score: int = Field(
        ge=1,
        le=10,
        description="Resolution efficiency score where 10 is optimal process adherence.",
    )
    justification: str = Field(
        description="A short, quote-grounded paragraph explaining exactly why the efficiency score was given and why any steps were missed."
    )
    missing_sop_steps: list[str] = Field(
        default_factory=list,
        description="SOP steps that are absent or weakly executed in transcript.",
    )
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Verbatim transcript quotes supporting topic and adherence findings.",
    )
    citations: list[EvidenceCitation] = Field(
        default_factory=list,
        description="Structured citations mapped to transcript and SOP evidence.",
    )

    @field_validator("evidence_quotes", mode="before")
    @classmethod
    def normalize_process_evidence_quotes(cls, value: list[str] | str | None) -> list[str]:
        return _normalize_quote_list(value)


NLICategory = Literal[
    "Entailment",
    "Benign Deviation",
    "Contradiction",
    "Policy Hallucination",
]


class NLIEvaluation(BaseModel):
    nli_category: NLICategory
    justification: str = Field(description="Short evidence-backed rationale for the label.")
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Verbatim quotes from policy/agent statement supporting NLI label.",
    )
    citations: list[EvidenceCitation] = Field(
        default_factory=list,
        description="Structured policy and transcript citations used for the NLI decision.",
    )

    @field_validator("evidence_quotes", mode="before")
    @classmethod
    def normalize_nli_evidence_quotes(cls, value: list[str] | str | None) -> list[str]:
        return _normalize_quote_list(value)


class InteractionLLMTriggerReport(BaseModel):
    interaction_id: UUID
    emotion_shift: EmotionShiftAnalysis
    process_adherence: ProcessAdherenceReport
    nli_policy: NLIEvaluation
    derived_customer_text: str
    derived_acoustic_emotion: str
    derived_fused_emotion: str
    derived_agent_statement: str
