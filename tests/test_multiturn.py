"""Multi-turn behaviour with a scripted fake model and fake tools (no API calls).

Regression tests for: the agent losing the question after a tool call,
results leaking between turns, the router lacking history, and the
synthesizer running twice when parallel branches have different lengths.
"""

import os

for key in ("OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY"):
    os.environ.setdefault(key, "test-key")

import pytest  # noqa: E402
from langchain.messages import AIMessage  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

import travel_planner.graph.nodes as nodes  # noqa: E402
from travel_planner.graph import build_parallel_travel_agent, initial_state  # noqa: E402
from travel_planner.schemas import AgentTask, ClassificationResult  # noqa: E402


class FakeModel(BaseChatModel):
    """Calls one tool on a fresh question, then answers with the question it saw.

    The search agent makes two tool calls, so the parallel branches have
    different lengths.
    """

    calls: list = []
    router_inputs: list = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def route(messages):
            self.router_inputs.append([m.content for m in messages[1:]])
            query = messages[-1].content
            if "flights" in query:
                tasks = [
                    AgentTask(source="search_agent_node", user_query="find flights"),
                    AgentTask(source="planner_agent_node", user_query="plan trip"),
                ]
                return ClassificationResult(tasks=tasks, requires_synthesis=True)
            return ClassificationResult(tasks=[AgentTask(source="planner_agent_node", user_query=query)])

        return RunnableLambda(route)

    def _generate(self, messages, stop=None, **kwargs):
        system = messages[0].content if messages[0].type == "system" else ""
        questions = [m.content for m in messages if m.type == "human"]
        last = messages[-1]
        self.calls.append(system[:20])
        n_tools = sum(m.type == "tool" for m in messages)

        if last.type == "human" and "Synthesizer" in last.content:
            msg = AIMessage(content="SYNTHESIZED")
        elif last.type == "human":
            name = "search_flights" if "travel search" in system else "tavily_search"
            msg = AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": f"c{len(self.calls)}"}])
        elif "travel search" in system and n_tools < 2:
            msg = AIMessage(content="", tool_calls=[{"name": "search_hotels", "args": {}, "id": f"h{len(self.calls)}"}])
        else:
            msg = AIMessage(content=f"answer to: {questions[-1] if questions else 'NO QUESTION'}")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool("tavily_search")
def fake_web_search() -> str:
    """Fake web search."""
    return "web result"


@tool("search_flights")
def fake_flights() -> str:
    """Fake flight search."""
    return "flight result"


@tool("search_hotels")
def fake_hotels() -> str:
    """Fake hotel search."""
    return "hotel result"


@pytest.fixture
def fake(monkeypatch):
    model = FakeModel()
    model.calls.clear()
    model.router_inputs.clear()
    monkeypatch.setattr(nodes, "get_llm", lambda: model)
    monkeypatch.setattr(nodes, "_search_model_with_tools", lambda: model)
    monkeypatch.setattr(nodes, "_planner_model_with_tools", lambda: model)
    monkeypatch.setattr(nodes, "get_planner_tools", lambda: [fake_web_search])
    monkeypatch.setattr(
        nodes, "SEARCH_TOOLS_BY_NAME", {"search_flights": fake_flights, "search_hotels": fake_hotels}
    )
    return model


def run(agent, query):
    return agent.invoke(initial_state(query), config={"configurable": {"thread_id": "t1"}})


def test_agent_keeps_question_after_tool_call(fake):
    agent = build_parallel_travel_agent()
    result = run(agent, "Best time to visit Jaipur?")
    assert result["final_answer"] == "answer to: Best time to visit Jaipur?"


def test_follow_up_turn_is_clean_and_router_sees_history(fake):
    agent = build_parallel_travel_agent()
    run(agent, "Best time to visit Jaipur?")
    result = run(agent, "Major historical sites there")

    # Only this turn's result, not the previous turn's
    assert [r["result"] for r in result["agent_results"]] == ["answer to: Major historical sites there"]
    assert result["final_answer"] == "answer to: Major historical sites there"
    # Router got the earlier question and answer before the new question
    assert fake.router_inputs[-1] == [
        "Best time to visit Jaipur?",
        "answer to: Best time to visit Jaipur?",
        "Major historical sites there",
    ]


def test_parallel_branches_synthesize_once(fake):
    agent = build_parallel_travel_agent()
    run(agent, "Best time to visit Jaipur?")
    result = run(agent, "Plan Paris and find flights")

    assert sorted(r["result"] for r in result["agent_results"]) == [
        "answer to: find flights",
        "answer to: plan trip",
    ]
    assert result["final_answer"] == "SYNTHESIZED"
    assert fake.calls.count("") == 1  # synthesizer prompt has no system message
