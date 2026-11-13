"""Standalone flight & hotel search agent (prebuilt ReAct agent).

The main graph uses its own search node (see graph/nodes.py) with the same
prompt and tools; this standalone version is handy for testing the agent alone.
"""

from langchain.agents import create_agent

from travel_planner.llm import get_llm
from travel_planner.prompts import FLIGHT_HOTEL_SEARCH_INSTRUCTIONS, NO_BOOKING_RULES
from travel_planner.tools import SEARCH_TOOLS


def build_search_agent():
    return create_agent(
        model=get_llm(),
        tools=SEARCH_TOOLS,
        system_prompt=FLIGHT_HOTEL_SEARCH_INSTRUCTIONS + NO_BOOKING_RULES,
    )
