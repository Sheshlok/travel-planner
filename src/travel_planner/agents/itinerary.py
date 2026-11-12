"""Itinerary agent: a deepagents deep-research agent with Tavily search.

Also exposed as a tool (`itinerary_research_agent`) so the planner agent can
call it as a sub-agent.
"""

from functools import lru_cache

from deepagents import create_deep_agent
from langchain_core.tools import tool

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


@tool("itinerary_research_agent", description="plans travel itinerary")
def call_itinerary_research_agent(query: str):
    result = get_itinerary_research_agent().invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content
