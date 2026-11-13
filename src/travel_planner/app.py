"""Streamlit chat UI for the travel planner.

Run locally:  uv run streamlit run src/travel_planner/app.py
"""

import hmac
import logging
import os
import time
import uuid

import streamlit as st

from travel_planner import config  # loads .env
from travel_planner.graph import build_parallel_travel_agent
from travel_planner.streaming import stream_turn

st.set_page_config(page_title="Travel Planner", page_icon="✈️", layout="centered")

# Progress log in the terminal running Streamlit (helps diagnose slow or stuck turns)
log = logging.getLogger("travel_planner.ui")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [ui] %(message)s", "%H:%M:%S"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)

EXAMPLES = [
    "What is the best time to visit Jaipur?",
    "Plan a 3-day trip to Jaipur for history and food",
    "Find flights from DEL to JAI on 2026-12-19, returning 2026-12-22, for 1 adult",
]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@st.cache_resource
def get_agent():
    """Build the graph once per server process.

    Its in-memory checkpointer is shared; each browser session uses its own
    thread_id, so conversations stay separate.
    """
    return build_parallel_travel_agent()


def check_password() -> None:
    """Optional gate: if APP_PASSWORD is set, ask for it before showing the app."""
    password = os.getenv("APP_PASSWORD")
    if not password or st.session_state.get("authenticated"):
        return
    st.title("✈️ Travel Planner")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered, password):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


def new_conversation() -> None:
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.history = []  # [{"role", "content", "details"}]


def md(text: str) -> str:
    """Escape $ so prices like $80 ... $96 aren't rendered as LaTeX math."""
    return text.replace("$", r"\$")


def render_details(details: dict) -> None:
    """Show how the router split the request and each agent's raw answer."""
    with st.expander("Routing and agent details"):
        for task in details.get("tasks", []):
            st.markdown(f"**{task['source']}** → {md(task['user_query'])}")
        for result in details.get("agent_results", []):
            st.divider()
            st.markdown(f"**{result['agent']}**")
            st.markdown(md(result["result"]))


def run_turn(query: str) -> tuple[str, dict]:
    """Run one turn: live progress in a status box, answer streamed token by token."""
    status = st.status("Planning…", expanded=False)
    placeholder = st.empty()
    streamed = ""
    state: dict = {}
    t0 = time.time()
    log.info("turn started (thread %s): %s", st.session_state.thread_id, query[:80])

    for kind, value in stream_turn(get_agent(), query, st.session_state.thread_id):
        if kind != "token":
            log.info("%6.1fs %s %s", time.time() - t0, kind, value if kind == "status" else "")
        elif not streamed:
            log.info("%6.1fs first answer token", time.time() - t0)
        if kind == "status":
            status.update(label=value)
            status.write(value)
        elif kind == "token":
            streamed += value
            placeholder.markdown(md(streamed) + " ▌")
        elif kind == "reset":
            streamed = ""
            placeholder.empty()
        elif kind == "done":
            state = value

    status.update(label="Done", state="complete")
    # The stored final answer is authoritative (also covers models that don't stream)
    answer = state.get("final_answer") or streamed or "Sorry, I couldn't produce an answer."
    placeholder.markdown(md(answer))
    details = {"tasks": state.get("tasks", []), "agent_results": state.get("agent_results", [])}
    return answer, details


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

check_password()

if "thread_id" not in st.session_state:
    new_conversation()

with st.sidebar:
    st.title("✈️ Travel Planner")
    st.caption(f"Model: {config.MODEL_NAME}")
    show_details = st.toggle("Show routing and agent details")
    if st.button("New conversation", width="stretch"):
        new_conversation()
        st.rerun()
    st.markdown("**Try asking:**")
    for example in EXAMPLES:
        st.caption(f"• {example}")

missing = config.missing_keys()
if missing:
    st.error(f"Missing API keys: {', '.join(missing)}. Add them to .env (or app secrets) and restart.")
    st.stop()

st.title("Where do you want to go?")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(md(msg["content"]))
        if show_details and msg.get("details"):
            render_details(msg["details"])

if prompt := st.chat_input("Ask about a trip, flights, hotels or an itinerary…"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(md(prompt))

    with st.chat_message("assistant"):
        try:
            answer, details = run_turn(prompt)
        except Exception as e:  # show API errors in the page instead of a traceback
            log.exception("turn failed")
            st.error(f"Something went wrong: {e}")
            st.stop()
        if show_details:
            render_details(details)

    st.session_state.history.append({"role": "assistant", "content": answer, "details": details})
