from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

from app.core.config import settings
from app.llm_trigger.prompts import (
    EMOTION_SHIFT_FEW_SHOT,
    NLI_FEW_SHOT,
    build_emotion_shift_prompt,
    build_nli_policy_prompt,
    build_process_adherence_prompt,
)
from app.llm_trigger.schemas import EmotionShiftAnalysis, NLIEvaluation, ProcessAdherenceReport


def _build_model() -> ChatGroq:
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


def build_emotion_shift_chain(model: ChatGroq | None = None):
    parser = PydanticOutputParser(pydantic_object=EmotionShiftAnalysis)
    prompt = build_emotion_shift_prompt().partial(
        format_instructions=parser.get_format_instructions(),
        few_shot=EMOTION_SHIFT_FEW_SHOT,
    )
    chain = prompt | (model or _build_model()) | parser
    return chain


def build_process_adherence_chain(model: ChatGroq | None = None):
    parser = PydanticOutputParser(pydantic_object=ProcessAdherenceReport)
    prompt = build_process_adherence_prompt().partial(
        format_instructions=parser.get_format_instructions()
    )
    chain = prompt | (model or _build_model()) | parser
    return chain


def build_nli_policy_chain(model: ChatGroq | None = None):
    parser = PydanticOutputParser(pydantic_object=NLIEvaluation)
    prompt = build_nli_policy_prompt().partial(
        format_instructions=parser.get_format_instructions(),
        few_shot=NLI_FEW_SHOT,
    )
    chain = prompt | (model or _build_model()) | parser
    return chain
