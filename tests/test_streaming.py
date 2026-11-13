"""Token streaming: the right node's tokens are streamed, and nothing else."""

import json

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from test_multiturn import FakeModel, fake  # noqa: F401  (fake is a pytest fixture)
from travel_planner.graph import build_parallel_travel_agent
from travel_planner.streaming import stream_turn


class StreamingFakeModel(FakeModel):
    """FakeModel that streams its replies word by word."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._generate(messages).generations[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            chunk = AIMessageChunk(content="", tool_call_chunks=[
                {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": 0}
            ])
            yield ChatGenerationChunk(message=chunk)
            return
        words = msg.content.split(" ")
        for i, word in enumerate(words):
            piece = word if i == 0 else " " + word
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


def _collect(agent, query):
    tokens, statuses, state = [], [], None
    for kind, value in stream_turn(agent, query, "t-stream"):
        if kind == "token":
            tokens.append(value)
        elif kind == "reset":
            tokens.clear()
        elif kind == "status":
            statuses.append(value)
        elif kind == "done":
            state = value
    return tokens, statuses, state


def _use_streaming_model(monkeypatch):
    import travel_planner.graph.nodes as nodes

    model = StreamingFakeModel()
    monkeypatch.setattr(nodes, "get_llm", lambda: model)
    monkeypatch.setattr(nodes, "_search_model_with_tools", lambda: model)
    monkeypatch.setattr(nodes, "_planner_model_with_tools", lambda: model)


def test_single_agent_answer_is_streamed(fake, monkeypatch):
    _use_streaming_model(monkeypatch)
    tokens, statuses, state = _collect(build_parallel_travel_agent(), "Best time to visit Jaipur?")

    assert len(tokens) > 1  # arrived in pieces
    assert "".join(tokens) == state["final_answer"] == "answer to: Best time to visit Jaipur?"
    assert statuses[0].endswith("Routing your request")


def test_parallel_streams_only_the_synthesizer(fake, monkeypatch):
    _use_streaming_model(monkeypatch)
    tokens, _, state = _collect(build_parallel_travel_agent(), "Plan Paris and find flights")

    # The agents' own answers are not streamed, only the combined answer
    assert "".join(tokens) == state["final_answer"] == "SYNTHESIZED"
