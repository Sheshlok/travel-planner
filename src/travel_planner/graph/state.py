"""Graph state definitions."""

import operator
from typing import Annotated, List, Optional, TypedDict

from langchain.messages import AnyMessage


def extend_or_reset(left: Optional[list], right: Optional[list]) -> list:
    """Reducer for per-turn lists.

    Works like operator.add (so parallel branches can write concurrently), but
    an update of None clears the list. `initial_state()` sends None at the
    start of every turn, so results and agent scratch messages from earlier
    turns don't leak into the next one when a checkpointer is used.
    """
    if right is None:
        return []
    return (left or []) + right


class TravelPlannerState(TypedDict):
    """Main graph state, built for parallel execution."""

    # Conversation history across turns: the user's questions and final answers.
    # This is the graph's memory; it is never reset within a thread.
    messages: Annotated[List[AnyMessage], operator.add]

    # Current user query
    user_query: str

    # Tasks from router, stored as plain dicts ({"source", "user_query", "focus"})
    # so they serialize cleanly in checkpoints
    tasks: List[dict]

    # Flag for synthesis
    requires_synthesis: bool

    # Per-turn results from the parallel agents (reset every turn)
    agent_results: Annotated[List[dict], extend_or_reset]

    # Final synthesized answer
    final_answer: str

    # Per-turn scratchpads for each agent's tool loop (reset every turn)
    search_messages: Annotated[List[AnyMessage], extend_or_reset]
    planner_messages: Annotated[List[AnyMessage], extend_or_reset]


def initial_state(user_query: str) -> dict:
    """Input for one user turn. None resets the per-turn lists."""
    return {
        "user_query": user_query,
        "messages": [],
        "tasks": [],
        "requires_synthesis": False,
        "agent_results": None,
        "final_answer": "",
        "search_messages": None,
        "planner_messages": None,
    }
