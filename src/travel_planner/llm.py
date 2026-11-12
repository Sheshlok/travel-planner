"""Chat models shared by the agents.

Created lazily (on first use) so importing the package doesn't need API keys.
"""

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from travel_planner import config


def _temperature_kwargs() -> dict:
    """Only pass temperature when one is configured."""
    if config.TEMPERATURE is None:
        return {}
    return {"temperature": config.TEMPERATURE}


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Main chat model: router, search agent, planner agent, synthesizer."""
    return ChatOpenAI(model=config.MODEL_NAME, **_temperature_kwargs())


@lru_cache(maxsize=1)
def get_research_model():
    """Model for the itinerary deep-research agent."""
    return init_chat_model(
        model=config.MODEL_NAME,
        model_provider="openai",
        **_temperature_kwargs(),
    )
