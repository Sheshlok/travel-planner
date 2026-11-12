"""Tools the agents can call."""

from travel_planner.tools.flights import search_flights
from travel_planner.tools.hotels import search_hotels
from travel_planner.tools.web_search import get_internet_search

# Flight & hotel tools used by the search agent
SEARCH_TOOLS = [search_flights, search_hotels]

__all__ = ["SEARCH_TOOLS", "get_internet_search", "search_flights", "search_hotels"]
