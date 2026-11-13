# Multi-Agent Travel Planner

A LangGraph multi-agent system, converted from the Colab notebook in `notebooks/`.

```
START → orchestrator ──Send──► search_agent_node ⇄ search_tool_node ─┐
                     └─Send──► planner_agent_node ⇄ planner_tool_node ┴─► synthesizer → END
```

- **Orchestrator**: an LLM router with structured output. It sends each request to one or both agents, in parallel.
- **Search agent**: finds flights and hotels using SerpAPI (Google Flights and Google Hotels).
- **Planner agent (Travel Scout)**: answers general questions with Tavily web search. For day-by-day itineraries it calls the itinerary deep-research agent (deepagents) as a tool.
- **Synthesizer**: combines the agents' outputs into one answer.

## Setup

```bash
uv sync                      # creates .venv and installs everything from uv.lock
cp .env.example .env         # then add your OpenAI, Tavily and SerpAPI keys
```

In VS Code, open this folder. The interpreter is set to `.venv/bin/python` in `.vscode/settings.json`.

## Run

```bash
uv run travel-planner                                       # interactive chat (memory within the session)
uv run travel-planner "What is the best time to visit Jaipur?"
uv run travel-planner -v "Plan 5 days in Paris and find flights JFK→CDG on 2026-12-21"   # -v shows routing + per-agent output
uv run travel-planner --draw-graph graph.png                # save the graph diagram
uv run pytest                                               # offline tests (no API calls)
```

The **Run and Debug** panel has two launch configurations, so you can set breakpoints inside nodes.

## Layout

```
src/travel_planner/
├── main.py            CLI entry point
├── config.py          .env loading, model name/temperature
├── llm.py             shared chat models (created lazily)
├── schemas.py         AgentTask, ClassificationResult
├── prompts/           all system prompts (verbatim from the notebook)
├── tools/             flights.py, hotels.py (SerpAPI), web_search.py (Tavily)
├── agents/            itinerary deep agent (+ its tool wrapper), standalone search agent and travel scout
└── graph/             state.py, nodes.py, builder.py
notebooks/             original Colab notebook
tests/                 offline wiring tests
```

## Using it from a notebook

```python
from travel_planner.graph import build_parallel_travel_agent, initial_state

agent = build_parallel_travel_agent()
result = agent.invoke(initial_state("Plan 2 days in Agra"), config={"configurable": {"thread_id": "t1"}})
print(result["final_answer"])
```

Run `uv add --dev ipykernel` once if it isn't installed, then pick the `.venv` kernel in VS Code.

## Changes from the Colab version

- `!pip install` lines are now dependencies in `pyproject.toml`.
- `google.colab.userdata` and `getpass` are replaced by a `.env` file.
- Models, tools and agents are created lazily, so importing the package needs no keys.
- The notebook defined `build_parallel_travel_agent()` twice. There is now one version, and the in-memory checkpointer is on by default.
- Test cells and `display(Image(...))` were removed. Use `--draw-graph` for the diagram.

## Fixes to the notebook's graph logic

- **The agents keep the question.** Each agent's first message (the query) is saved in its scratchpad, so the agent still sees the question after a tool call.
- **Each turn starts clean.** `agent_results`, `search_messages` and `planner_messages` are reset every turn (the `extend_or_reset` reducer), so earlier answers aren't synthesized again.
- **Memory works through the router.** The router sees the last few questions and answers, and rewrites follow-ups ("sites there") into self-contained queries for the agents.
- **The synthesizer runs once.** It is added with `defer=True`, so it waits for every parallel branch to finish.
- **Tasks are stored as plain dicts,** which removes the checkpoint deserialization warning.

- **Round-trip flights include return options.** `search_flights` makes a second SerpAPI call with the best outbound flight's `departure_token` to fetch its return flights.
- **Follow-ups keep earlier constraints.** The router carries forward dates, travelers, star rating and budget unless the user changes them.
- **No booking promises.** The search agent and synthesizer are told they can search and recommend but not book.

`tests/test_multiturn.py` and `tests/test_flights.py` check this with a fake model and a mocked SerpAPI.

