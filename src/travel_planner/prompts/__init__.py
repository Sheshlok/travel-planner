"""System prompts, kept separate from the agent code."""

from travel_planner.prompts.flight_hotel import FLIGHT_HOTEL_SEARCH_INSTRUCTIONS
from travel_planner.prompts.itinerary import ITINERARY_RESEARCH_INSTRUCTIONS
from travel_planner.prompts.router import ROUTER_INSTRUCTIONS, ROUTER_MEMORY_RULES
from travel_planner.prompts.synthesizer import SYNTHESIZER_PROMPT
from travel_planner.prompts.travel_scout import TRAVEL_SCOUT_INSTRUCTIONS

__all__ = [
    "FLIGHT_HOTEL_SEARCH_INSTRUCTIONS",
    "ITINERARY_RESEARCH_INSTRUCTIONS",
    "ROUTER_INSTRUCTIONS",
    "ROUTER_MEMORY_RULES",
    "SYNTHESIZER_PROMPT",
    "TRAVEL_SCOUT_INSTRUCTIONS",
]
