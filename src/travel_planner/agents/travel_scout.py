"""Standalone Travel Scout agent (general travel questions + itinerary sub-agent).

The main graph uses its own planner node (see graph/nodes.py) with the same
prompt and tools; this standalone version is handy for testing the agent alone.
"""

from langchain.agents import create_agent

from travel_planner.agents.itinerary import call_itinerary_research_agent
from travel_planner.llm import get_llm
from travel_planner.prompts import TRAVEL_SCOUT_INSTRUCTIONS
from travel_planner.tools import get_internet_search


def get_planner_tools() -> list:
    """Tools used by the Travel Scout / planner agent."""
    return [get_internet_search(), call_itinerary_research_agent]


def build_travel_scout():
    return create_agent(
        model=get_llm(),
        system_prompt=TRAVEL_SCOUT_INSTRUCTIONS,
        tools=get_planner_tools(),
    )
