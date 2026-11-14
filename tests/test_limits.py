"""Research round caps: an agent that always wants more searches must stop and answer."""

import os

for key in ("OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY"):
    os.environ.setdefault(key, "test-key")

from langchain.messages import AIMessage  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

import travel_planner.graph.nodes as nodes  # noqa: E402
from test_multiturn import fake_flights, fake_hotels, fake_web_search  # noqa: E402
from travel_planner import config  # noqa: E402
from travel_planner.graph import build_parallel_travel_agent, initial_state  # noqa: E402
from travel_planner.schemas import AgentTask, ClassificationResult  # noqa: E402


class GreedyModel(BaseChatModel):
    """Asks for another search every time, unless tools are switched off."""

    tool_choice: str | None = None
    tool_calls_made: list = []

    @property
    def _llm_type(self):
        return "greedy"

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        return GreedyModel(tool_choice=tool_choice, tool_calls_made=self.tool_calls_made)

    def with_structured_output(self, schema, **kwargs):
        route = "search_agent_node" if self.tool_calls_made == ["search"] else "planner_agent_node"
        return RunnableLambda(lambda m: ClassificationResult(tasks=[AgentTask(source=route, user_query="q")]))

    def _generate(self, messages, stop=None, **kwargs):
        if self.tool_choice == "none":
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="final answer"))])
        system = messages[0].content
        name = "search_flights" if "travel search" in system else "tavily_search"
        self.tool_calls_made.append(name)
        call = {"name": name, "args": {}, "id": f"c{len(self.tool_calls_made)}"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


def _run(monkeypatch, model):
    monkeypatch.setattr(nodes, "get_llm", lambda: model)
    monkeypatch.setattr(nodes, "_search_model_with_tools", lambda: model)
    monkeypatch.setattr(nodes, "_planner_model_with_tools", lambda: model)
    monkeypatch.setattr(nodes, "get_planner_tools", lambda: [fake_web_search])
    monkeypatch.setattr(nodes, "SEARCH_TOOLS_BY_NAME", {"search_flights": fake_flights, "search_hotels": fake_hotels})
    agent = build_parallel_travel_agent(persist=False)
    return agent.invoke(initial_state("q"))


def test_planner_stops_after_its_round_limit(monkeypatch):
    monkeypatch.setattr(config, "PLANNER_MAX_TOOL_ROUNDS", 2)
    model = GreedyModel(tool_calls_made=[])
    result = _run(monkeypatch, model)
    assert model.tool_calls_made == ["tavily_search", "tavily_search"]
    assert result["final_answer"] == "final answer"


def test_search_agent_stops_after_its_round_limit(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MAX_TOOL_ROUNDS", 3)
    model = GreedyModel(tool_calls_made=["search"])  # makes the router pick the search agent
    result = _run(monkeypatch, model)
    assert model.tool_calls_made[1:] == ["search_flights"] * 3
    assert result["final_answer"] == "final answer"
