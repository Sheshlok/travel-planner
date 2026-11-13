"""Build and compile the multi-agent travel planner graph."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from travel_planner.graph.nodes import (
    classify_query_parallel,
    dispatch_to_agents,
    planner_agent_node,
    planner_tool_node,
    search_agent_node,
    search_tool_node,
    should_continue_for_planner_agent,
    should_continue_for_search_agent,
    synthesizer_node,
)
from travel_planner.graph.state import TravelPlannerState


def build_parallel_travel_agent(checkpointer: BaseCheckpointSaver | None = None, *, persist: bool = True):
    """Build the graph with parallel execution support.

    By default an in-memory checkpointer is attached (the notebook's
    "Add Persistence" step), so calls need a `thread_id` in their config.
    Pass `persist=False` for a stateless graph, or your own checkpointer.
    """
    builder = StateGraph(TravelPlannerState)

    # Add nodes
    builder.add_node("orchestrator", classify_query_parallel)
    builder.add_node("search_agent_node", search_agent_node)
    builder.add_node("planner_agent_node", planner_agent_node)
    builder.add_node("search_tool_node", search_tool_node)
    builder.add_node("planner_tool_node", planner_tool_node)
    # defer=True: wait until every parallel branch (and its tool loop) has
    # finished, so the synthesizer runs exactly once per turn
    builder.add_node("synthesizer", synthesizer_node, defer=True)

    # Connect edges
    builder.add_edge(START, "orchestrator")

    # Router dispatches to agents via Send API (parallel execution)
    builder.add_conditional_edges(
        "orchestrator",
        dispatch_to_agents,
        ["search_agent_node", "planner_agent_node"],
    )

    # Search agent tool loop
    builder.add_conditional_edges(
        "search_agent_node",
        should_continue_for_search_agent,
        ["search_tool_node", "synthesizer"],
    )
    builder.add_edge("search_tool_node", "search_agent_node")

    # Planner agent tool loop
    builder.add_conditional_edges(
        "planner_agent_node",
        should_continue_for_planner_agent,
        ["planner_tool_node", "synthesizer"],
    )
    builder.add_edge("planner_tool_node", "planner_agent_node")

    # End after synthesis
    builder.add_edge("synthesizer", END)

    if checkpointer is None and persist:
        checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
