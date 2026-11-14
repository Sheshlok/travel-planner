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


def test_retry_from_checkpoint_replaces_the_last_turn(fake):
    from travel_planner.streaming import current_checkpoint_id

    agent = build_parallel_travel_agent()
    _collect(agent, "Best time to visit Jaipur?")
    before_turn_2 = current_checkpoint_id(agent, "t-stream")
    _collect(agent, "Major historical sites there")

    # Retry turn 2 from the checkpoint taken before it
    fake.router_inputs.clear()
    for _ in stream_turn(agent, "Major historical sites there", "t-stream", before_turn_2):
        pass

    # The router saw turn 1 only, not the first attempt at turn 2
    assert fake.router_inputs[-1] == [
        "Best time to visit Jaipur?",
        "answer to: Best time to visit Jaipur?",
        "Major historical sites there",
    ]
    final = agent.get_state({"configurable": {"thread_id": "t-stream"}}).values
    assert [m.content for m in final["messages"]].count("Major historical sites there") == 1


def test_speech_text_strips_markdown():
    from travel_planner.ui_actions import to_speech_text

    text = "## Flights\n- **IndiGo** 6E 2164 — DEL → JAI\n[Source](https://example.com) `x`"
    assert to_speech_text(text) == "Flights\nIndiGo 6E 2164 — DEL to JAI\nSource x"
