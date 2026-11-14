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

## Web UI (Streamlit)

```bash
uv run streamlit run src/travel_planner/app.py
```

Opens at http://localhost:8501. It's a chat with the same memory as the CLI (one conversation per browser tab), live progress for each graph step, the final answer streamed token by token, and an optional "routing and agent details" panel. Under each answer there's an action bar: copy (the original markdown), read aloud (browser speech), 👍/👎 feedback (logged to the terminal / Cloud logs) and retry for the latest answer, which re-runs it from the conversation state before that answer. Set `APP_PASSWORD` to require a password.

**Answers run in the background.** Each question runs as a job on the server (`src/travel_planner/jobs.py`), not inside the browser connection, so closing the tab or locking the phone doesn't stop it. Each conversation has its own link (`?c=…`): reopen it to see live progress or the finished answer. Turns, feedback and conversation memory are stored in SQLite (`.data/travel_planner.sqlite`, git-ignored). Like a chat app, you can keep typing while an answer runs: a new message waits its turn and then runs, and **Stop** ends the current answer (keeping what was written so far). Each turn builds on the memory of the last *completed* answer, so stopped or failed turns never leak into it. A turn cut off by an app restart is shown as interrupted, with a Retry button. At most `TRAVEL_PLANNER_MAX_JOBS` (default 3) questions run at once across all users; others wait.

Anyone with a conversation link can open that conversation, so treat links like the password.

Token streaming is in `src/travel_planner/streaming.py`. With one agent it streams that agent's final answer; with several it streams the synthesizer.

### Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (`.env` is git-ignored, so your keys stay local).
2. On share.streamlit.io, create an app from the repo, branch `main`, main file `src/travel_planner/app.py`. Under Advanced settings, choose Python 3.12 and paste your secrets in TOML format:
   ```toml
   OPENAI_API_KEY = "..."
   TAVILY_API_KEY = "..."
   SERPAPI_API_KEY = "..."
   TRAVEL_PLANNER_MODEL = "gpt-5"
   APP_PASSWORD = "..."
   ```
3. Community Cloud installs dependencies from `uv.lock`.

On Community Cloud the SQLite file lives on the app's temporary disk, so conversations survive closing the tab but not a reboot or sleep. For permanent history, point the app at an external database. After a push that changes files other than `app.py`, reboot the app (Manage app → Reboot) so it loads the new modules.

## Layout

```
src/travel_planner/
├── main.py            CLI entry point
├── app.py             Streamlit web UI
├── streaming.py       per-turn progress + answer-token stream (used by the UI)
├── ui_actions.py      answer action bar: copy, read aloud, feedback, retry
├── jobs.py            background turns + SQLite store (used by the UI)
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

## Research limits (speed)

Most of an answer's time goes into research rounds, so each agent gets a fixed budget before it must answer:

| Setting | Default | What it limits |
|---|---|---|
| `TRAVEL_PLANNER_PLANNER_MAX_ROUNDS` | 2 | planner tool-calling rounds (web search / itinerary agent) |
| `TRAVEL_PLANNER_SEARCH_MAX_ROUNDS` | 3 | flight/hotel search rounds |
| `TRAVEL_PLANNER_ITINERARY_MAX_STEPS` | 30 | steps in one call to the itinerary deep-research agent |

When an agent reaches its limit it is called with `tool_choice="none"`, so it must answer with what it has.

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

