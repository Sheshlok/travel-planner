"""Itinerary agent: a deepagents deep-research agent with Tavily search.

Also exposed as a tool (`itinerary_research_agent`) so the planner agent can
call it as a sub-agent.
"""

from functools import lru_cache

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from travel_planner import config

from travel_planner.llm import get_research_model
from travel_planner.prompts import ITINERARY_RESEARCH_INSTRUCTIONS
from travel_planner.tools import get_internet_search


@lru_cache(maxsize=1)
def get_itinerary_research_agent():
    """Build (once) the itinerary deep-research agent."""
    return create_deep_agent(
        model=get_research_model(),
        system_prompt=ITINERARY_RESEARCH_INSTRUCTIONS,
        tools=[get_internet_search()],
    )


STEP_LIMIT_MESSAGE = (
    "Itinerary research stopped at its step limit before finishing. Plan the itinerary "
    "from what you know and any web search results, and say which details to verify."
)


@tool("itinerary_research_agent", description="plans travel itinerary")
def call_itinerary_research_agent(query: str):
    try:
        result = get_itinerary_research_agent().invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": config.ITINERARY_MAX_STEPS},
        )
    except GraphRecursionError:
        return STEP_LIMIT_MESSAGE
    return result["messages"][-1].content
