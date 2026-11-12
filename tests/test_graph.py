"""Offline tests: check wiring without calling any API.

Dummy keys are set so the clients can be constructed; no requests are made.
"""

import os

import pytest

for key in ("OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY"):
    os.environ.setdefault(key, "test-key")

from langchain.messages import AIMessage  # noqa: E402

from travel_planner.graph import build_parallel_travel_agent, initial_state  # noqa: E402
from travel_planner.graph.nodes import (  # noqa: E402
    dispatch_to_agents,
    should_continue_for_planner_agent,
    should_continue_for_search_agent,
    synthesizer_node,
)
from travel_planner.schemas import AgentTask  # noqa: E402


def test_graph_compiles_with_expected_nodes():
    agent = build_parallel_travel_agent()
    nodes = set(agent.get_graph().nodes)
    assert {
        "orchestrator",
        "search_agent_node",
        "search_tool_node",
        "planner_agent_node",
        "planner_tool_node",
        "synthesizer",
    } <= nodes


def test_dispatch_sends_one_task_per_agent():
    state = initial_state("Plan a week in London and find hotels")
    state["tasks"] = [
        AgentTask(source="search_agent_node", user_query="hotels in London"),
        AgentTask(source="planner_agent_node", user_query="week in London"),
    ]
    sends = dispatch_to_agents(state)
    assert [s.node for s in sends] == ["search_agent_node", "planner_agent_node"]
    assert sends[0].arg["user_query"] == "hotels in London"


@pytest.mark.parametrize(
    "fn, key, tool_node",
    [
        (should_continue_for_search_agent, "search_messages", "search_tool_node"),
        (should_continue_for_planner_agent, "planner_messages", "planner_tool_node"),
    ],
)
def test_conditional_edges(fn, key, tool_node):
    with_call = AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
    assert fn({key: [with_call]}) == tool_node
    assert fn({key: [AIMessage(content="done")]}) == "synthesizer"


def test_synthesizer_passes_single_result_through():
    state = initial_state("q")
    state["agent_results"] = [{"agent": "planner_agent", "result": "answer"}]
    assert synthesizer_node(state)["final_answer"] == "answer"
