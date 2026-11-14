"""Run one turn of the graph and yield progress + answer tokens as they happen.

Used by the Streamlit UI. Events:
    ("status", label)   a graph node finished (for a progress box)
    ("token", text)     a piece of the final answer
    ("reset", None)     discard tokens shown so far (the agent decided to call a tool)
    ("done", state)     final graph state (state["final_answer"] is the full answer)
"""

from collections.abc import Iterator

from langchain_core.messages import AIMessageChunk

from travel_planner.graph import initial_state

NODE_LABELS = {
    "orchestrator": "🧭 Routing your request",
    "search_agent_node": "✈️ Search agent is working",
    "search_tool_node": "🔎 Searching flights and hotels",
    "planner_agent_node": "🗺️ Planner agent is working",
    "planner_tool_node": "🌐 Researching (web search / itinerary agent)",
    "synthesizer": "📝 Writing the final answer",
}


def current_checkpoint_id(agent, thread_id: str) -> str | None:
    """Id of the thread's latest checkpoint (None for a new thread)."""
    snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})
    return snapshot.config.get("configurable", {}).get("checkpoint_id")


def stream_turn(
    agent, query: str, thread_id: str, from_checkpoint: str | None = None
) -> Iterator[tuple[str, object]]:
    """Run one turn. With from_checkpoint, the turn starts from that earlier
    checkpoint instead of the latest one (used to retry the last answer)."""
    cfg = {"configurable": {"thread_id": thread_id}}
    run_cfg = cfg
    if from_checkpoint:
        run_cfg = {"configurable": {"thread_id": thread_id, "checkpoint_id": from_checkpoint}}

    # Which node produces the final answer's tokens:
    # - one agent dispatched  -> that agent's last LLM call (synthesizer just passes it through)
    # - several agents        -> the synthesizer's LLM call
    answer_node = None

    for mode, data in agent.stream(
        initial_state(query), config=run_cfg, stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            for node, update in data.items():
                update = update or {}
                if node == "orchestrator":
                    tasks = update.get("tasks", [])
                    answer_node = tasks[0]["source"] if len(tasks) == 1 else "synthesizer"
                if node in NODE_LABELS:
                    yield ("status", NODE_LABELS[node])
                # The answering agent asked for a tool: anything it streamed was not the answer
                if node == answer_node and node != "synthesizer":
                    msgs = update.get("search_messages") or update.get("planner_messages") or []
                    if msgs and getattr(msgs[-1], "tool_calls", None):
                        yield ("reset", None)

        else:  # "messages": LLM tokens from any node (including nested agents)
            chunk, metadata = data
            if (
                answer_node
                and isinstance(chunk, AIMessageChunk)
                and metadata.get("langgraph_node") == answer_node
            ):
                text = chunk.text
                if text:
                    yield ("token", text)

    yield ("done", agent.get_state(cfg).values)
