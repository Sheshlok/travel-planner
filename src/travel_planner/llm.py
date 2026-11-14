"""Chat models shared by the agents.

Created lazily (on first use) so importing the package doesn't need API keys.
"""

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from travel_planner import config


def _model_kwargs() -> dict:
    """Shared settings; temperature only when one is configured."""
    kwargs = {"timeout": config.LLM_TIMEOUT, "max_retries": config.LLM_MAX_RETRIES}
    if config.TEMPERATURE is not None:
        kwargs["temperature"] = config.TEMPERATURE
    if config.REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.REASONING_EFFORT
    return kwargs


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Main chat model: router, search agent, planner agent, synthesizer."""
    return ChatOpenAI(model=config.MODEL_NAME, **_model_kwargs())


@lru_cache(maxsize=1)
def get_research_model():
    """Model for the itinerary deep-research agent."""
    return init_chat_model(
        model=config.MODEL_NAME,
        model_provider="openai",
        **_model_kwargs(),
    )
