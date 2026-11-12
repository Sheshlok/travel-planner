"""Command-line entry point.

Examples:
    uv run travel-planner "What is the best time to visit Jaipur?"
    uv run travel-planner                 # interactive chat (same thread)
    uv run travel-planner -v "Plan 5 days in Paris and find flights JFK->CDG on 2026-12-21"
    uv run travel-planner --draw-graph graph.png
"""

import argparse
import uuid

from travel_planner import config
from travel_planner.graph import build_parallel_travel_agent, initial_state


def print_details(result: dict) -> None:
    """Print the dispatched tasks and each agent's raw result (notebook Step 7)."""
    print("=" * 60)
    print("TASKS DISPATCHED:")
    for task in result.get("tasks", []):
        print(f"  • {task.source}: {task.user_query}")

    print("\n" + "=" * 60)
    print("AGENT RESULTS:")
    for r in result.get("agent_results", []):
        print(f"\n[{r['agent'].upper()}]")
        print("-" * 40)
        print(r["result"])
    print("\n" + "=" * 60)
    print("FINAL RESPONSE:")
    print("=" * 60)


def ask(agent, query: str, thread_id: str, verbose: bool = False) -> str:
    """Run one user turn through the graph and return the final answer."""
    result = agent.invoke(
        initial_state(query),
        config={"configurable": {"thread_id": thread_id}},
    )
    if verbose:
        print_details(result)
    return result["final_answer"]


def draw_graph(path: str) -> None:
    """Save the graph diagram as a PNG (uses the mermaid.ink web service)."""
    agent = build_parallel_travel_agent(persist=False)
    with open(path, "wb") as f:
        f.write(agent.get_graph().draw_mermaid_png())
    print(f"Saved graph diagram to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent travel planner")
    parser.add_argument("query", nargs="*", help="Travel question. Omit for interactive mode.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show routed tasks and per-agent results")
    parser.add_argument("--thread", default=None, help="Conversation thread id (default: random)")
    parser.add_argument("--draw-graph", metavar="PNG", help="Save the graph diagram to a PNG and exit")
    args = parser.parse_args()

    config.check_keys()

    if args.draw_graph:
        draw_graph(args.draw_graph)
        return

    agent = build_parallel_travel_agent()
    thread_id = args.thread or f"user-{uuid.uuid4().hex[:8]}"

    if args.query:
        print(ask(agent, " ".join(args.query), thread_id, args.verbose))
        return

    print("Travel planner — type your question, or 'exit' to quit.")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit", "q"}:
            break
        if query:
            print("\nPlanner:\n" + ask(agent, query, thread_id, args.verbose))


if __name__ == "__main__":
    main()
