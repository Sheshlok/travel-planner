"""Tavily web search tool, shared by the itinerary and planner agents."""

from functools import lru_cache

from langchain_tavily import TavilySearch


@lru_cache(maxsize=1)
def get_internet_search() -> TavilySearch:
    """Create the Tavily search tool (lazily, since it needs TAVILY_API_KEY)."""
    return TavilySearch(
        max_results=5,
        topic="general",
        include_images=True,
        include_image_descriptions=True,
        search_depth="advanced",
    )
