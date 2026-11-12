"""Graph state definitions."""

import operator
from typing import Annotated, List, TypedDict

from langchain.messages import AnyMessage

from travel_planner.schemas import AgentTask


class TravelPlannerState(TypedDict):
    """Main graph state, built for parallel execution."""

    # Conversation history
    messages: Annotated[List[AnyMessage], operator.add]

    # Current user query
    user_query: str

    # Tasks from router (replaces single classification)
    tasks: List[AgentTask]

    # Flag for synthesis
    requires_synthesis: bool

    # Parallel agent results - KEY: uses operator.add for concurrent writes
    agent_results: Annotated[List[dict], operator.add]

    # Final synthesized answer
    final_answer: str

    search_messages: Annotated[List[AnyMessage], operator.add]
    planner_messages: Annotated[List[AnyMessage], operator.add]


class AgentWorkerState(TypedDict):
    """State passed to each parallel worker."""

    task: AgentTask
    messages: List[AnyMessage]
    agent_results: Annotated[List[dict], operator.add]


def initial_state(user_query: str) -> TravelPlannerState:
    """Fresh input state for one user turn."""
    return {
        "user_query": user_query,
        "messages": [],
        "tasks": [],
        "requires_synthesis": False,
        "agent_results": [],
        "final_answer": "",
        "search_messages": [],
        "planner_messages": [],
    }
