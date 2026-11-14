"""Streamlit chat UI for the travel planner.

Run locally:  uv run streamlit run src/travel_planner/app.py

Questions run as background jobs (see jobs.py): the page only shows their
progress, so closing the tab or locking the phone doesn't lose the answer.
Each conversation has its own link (?c=<id>) to come back to.
"""

import hmac
import logging
import os

import streamlit as st

from travel_planner import config  # loads .env
from travel_planner import jobs
from travel_planner.ui_actions import action_bar

st.set_page_config(page_title="Travel Planner", page_icon="✈️", layout="centered")

# Progress log in the terminal running Streamlit / Cloud logs (UI + background jobs)
_logger = logging.getLogger("travel_planner")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
log = logging.getLogger("travel_planner.ui")

EXAMPLES = [
    "What is the best time to visit Jaipur?",
    "Plan a 3-day trip to Jaipur for history and food",
    "Find flights from DEL to JAI on 2026-12-19, returning 2026-12-22, for 1 adult",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def current_conversation() -> str:
    """Conversation from the ?c= link, or a new one (written into the link)."""
    conv_id = st.query_params.get("c")
    if not conv_id or not jobs.get_conversation(conv_id):
        conv_id = jobs.new_conversation()
        st.query_params["c"] = conv_id
    return conv_id


def start_new_conversation() -> None:
    st.query_params["c"] = jobs.new_conversation()


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


def record_feedback(msg: dict) -> None:
    """Called when 👍/👎 changes; saved with the turn and logged."""
    value = st.session_state.get(f"fb-{msg['id']}")
    jobs.set_feedback(msg["id"], value)
    label = {1: "thumbs_up", 0: "thumbs_down"}.get(value, "cleared")
    log.info("feedback %s (turn %s): %s", label, msg["id"], msg.get("question", "")[:80])


def retry(msg: dict) -> None:
    jobs.retry_turn(msg["id"])
    log.info("retry requested (turn %s): %s", msg["id"], msg.get("question", "")[:80])


@st.fragment(run_every=1.0)
def live_turn(turn_id: str) -> None:
    """Shows a queued/running turn's progress; refreshes itself every second."""
    turn = jobs.get_turn(turn_id)
    if turn is None or turn["status"] not in jobs.ACTIVE:
        st.rerun()  # finished, stopped or replaced: redraw the page with the result
    # Not used as a `with` block: that would mark it complete (✓) when the block ends
    st.status(turn["progress"] or "Planning…", state="running", expanded=False)
    if turn["partial"]:
        st.markdown(md(turn["partial"]) + " ▌")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.button("Stop", icon=":material/stop_circle:", key=f"stop-{turn_id}", type="tertiary",
                  on_click=jobs.cancel_turn, args=(turn_id,))
        if turn["status"] == "running":
            st.caption("You can close this page or lock your phone; reopen this link to see the answer.")


def render_turn(turn: dict, *, is_last: bool, show_details: bool) -> None:
    with st.chat_message("user"):
        st.markdown(md(turn["question"]))
    with st.chat_message("assistant"):
        status = turn["status"]
        if status in jobs.ACTIVE:
            live_turn(turn["id"])
            return
        msg = {"id": turn["id"], "question": turn["question"]}
        if status == "done" or (status == "cancelled" and turn["answer"]):
            st.markdown(md(turn["answer"]))
            if status == "cancelled":
                st.caption("Stopped. This answer isn't kept in the conversation memory.")
            elif show_details and turn["details"]:
                render_details(turn["details"])
            msg.update(content=turn["answer"])
            # Show saved feedback when the conversation is reopened
            st.session_state.setdefault(f"fb-{turn['id']}", turn["feedback"])
        else:
            if status == "cancelled":
                text = "Stopped before an answer was written."
            elif status == "interrupted":
                text = "This answer was interrupted because the app restarted."
            else:
                text = f"Something went wrong: {turn['error']}"
            (st.caption if status == "cancelled" else st.error)(text)
            msg.update(content=text, error=True)
        action_bar(msg, is_last=is_last, on_feedback=record_feedback,
                   on_retry=lambda m=msg: retry(m))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

check_password()

missing = config.missing_keys()
if missing:
    st.error(f"Missing API keys: {', '.join(missing)}. Add them to .env (or app secrets) and restart.")
    st.stop()

conv_id = current_conversation()

with st.sidebar:
    st.title("✈️ Travel Planner")
    st.caption(f"Model: {config.MODEL_NAME}")
    show_details = st.toggle("Show routing and agent details")
    st.button("New conversation", width="stretch", on_click=start_new_conversation)
    st.caption("Bookmark or share this page's link to come back to this conversation.")
    st.markdown("**Try asking:**")
    for example in EXAMPLES:
        st.caption(f"• {example}")

st.title("Where do you want to go?")

# Like a chat app: you can always type. A message sent while an answer is
# running waits its turn and then runs (shown as "Waiting for the previous answer").
busy = jobs.is_busy(conv_id)
prompt = st.chat_input(
    "Type your next message…" if busy else "Ask about a trip, flights, hotels or an itinerary…"
)
if prompt:
    jobs.start_turn(conv_id, prompt)
    st.rerun()  # redraw with the new turn's live progress

turns = jobs.list_turns(conv_id)
for i, turn in enumerate(turns):
    render_turn(turn, is_last=(i == len(turns) - 1), show_details=show_details)
