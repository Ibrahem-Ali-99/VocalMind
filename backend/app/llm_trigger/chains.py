from __future__ import annotations

import asyncio
import logging
import random
import threading

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


logger = logging.getLogger(__name__)

_lock = threading.Lock()
_shared_model: ChatGroq | None = None


def _get_shared_model() -> ChatGroq:
    global _shared_model
    if _shared_model is not None:
        return _shared_model
    with _lock:
        if _shared_model is not None:
            return _shared_model
        _shared_model = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            request_timeout=getattr(settings, "LLM_REQUEST_TIMEOUT_SECONDS", 60.0),
        )
        return _shared_model


async def _invoke_chain_with_retry(chain, inputs: dict, max_retries: int = 3) -> object:
    base_delay = 0.5
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await chain.ainvoke(inputs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_rate_limit = "rate" in msg or "429" in msg or "throttl" in msg
            is_transient = is_rate_limit or "timeout" in msg or "connection" in msg or "unavailable" in msg
            if not is_transient or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3)
            logger.warning(
                "LLM chain attempt %d/%d failed (transient), retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


def build_emotion_shift_chain(model: ChatGroq | None = None):
    parser = PydanticOutputParser(pydantic_object=EmotionShiftAnalysis)
    prompt = build_emotion_shift_prompt().partial(
        format_instructions=parser.get_format_instructions(),
        few_shot=EMOTION_SHIFT_FEW_SHOT,
    )
    chain = prompt | (model or _get_shared_model()) | parser
    return chain


def build_process_adherence_chain(model: ChatGroq | None = None):
    parser = PydanticOutputParser(pydantic_object=ProcessAdherenceReport)
    prompt = build_process_adherence_prompt().partial(
        format_instructions=parser.get_format_instructions()
    )
    chain = prompt | (model or _get_shared_model()) | parser
    return chain


def build_nli_policy_chain(model: ChatGroq | None = None):
    parser = PydanticOutputParser(pydantic_object=NLIEvaluation)
    prompt = build_nli_policy_prompt().partial(
        format_instructions=parser.get_format_instructions(),
        few_shot=NLI_FEW_SHOT,
    )
    chain = prompt | (model or _get_shared_model()) | parser
    return chain
