"""Background turns for the web UI.

Each question runs in a worker thread on the server, not in the browser
connection, so closing the tab or locking the phone doesn't stop it. Progress,
the partial answer and the final answer are written to SQLite; the page just
reads them. Conversation memory (the LangGraph checkpointer) lives in the same
SQLite file.

Turns in a conversation run one at a time, in order: a message sent while an
answer is running waits ("queued") and then runs. Each turn builds on the memory
of the last *completed* turn, so a stopped or failed turn never leaks into it.

Turn statuses: queued -> running -> done | error | cancelled. A queued/running
turn that belongs to an earlier server process (the app restarted) is reported
as "interrupted".
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from travel_planner import config
from travel_planner.graph import build_parallel_travel_agent
from travel_planner.streaming import current_checkpoint_id, stream_turn

log = logging.getLogger("travel_planner.jobs")

# Identifies this server process, to spot turns cut off by a restart
BOOT_ID = uuid.uuid4().hex
ACTIVE = ("queued", "running")
_slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_JOBS)
_write_lock = threading.Lock()
_cancel_requested: set[str] = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    thread_id  TEXT NOT NULL,      -- LangGraph memory thread
    created    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id                 TEXT PRIMARY KEY,
    conv_id            TEXT NOT NULL REFERENCES conversations(id),
    seq                INTEGER NOT NULL,
    question           TEXT NOT NULL,
    status             TEXT NOT NULL,
    progress           TEXT,
    partial            TEXT,
    answer             TEXT,
    details            TEXT,
    error              TEXT,
    parent_checkpoint  TEXT,
    thread_id          TEXT,          -- memory thread this turn ran on
    end_checkpoint     TEXT,          -- memory state after this turn's answer
    feedback           INTEGER,
    boot_id            TEXT NOT NULL,
    created            REAL NOT NULL,
    updated            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_by_conv ON turns(conv_id, seq);
"""


def _connect() -> sqlite3.Connection:
    path = Path(config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@lru_cache(maxsize=1)
def _db() -> sqlite3.Connection:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Databases created by the previous version lack these columns
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(turns)")}
    for column in ("thread_id", "end_checkpoint"):
        if column not in existing:
            conn.execute(f"ALTER TABLE turns ADD COLUMN {column} TEXT")
    conn.commit()
    return conn


def _write(sql: str, params: tuple = ()) -> None:
    with _write_lock:
        conn = _db()
        conn.execute(sql, params)
        conn.commit()


def _read(sql: str, params: tuple = ()) -> list[dict]:
    with _write_lock:
        return [dict(r) for r in _db().execute(sql, params).fetchall()]


_agent = None
_agent_lock = threading.Lock()


def get_agent():
    """One compiled graph per server process, with memory stored in SQLite."""
    global _agent
    with _agent_lock:  # workers may ask at the same time; build it only once
        if _agent is None:
            saver = SqliteSaver(_connect())
            saver.setup()
            _agent = build_parallel_travel_agent(checkpointer=saver)
        return _agent


# ---------------------------------------------------------------------------
# Conversations and turns
# ---------------------------------------------------------------------------

def new_conversation() -> str:
    conv_id = uuid.uuid4().hex[:12]
    _write("INSERT INTO conversations (id, thread_id, created) VALUES (?, ?, ?)",
           (conv_id, f"{conv_id}-0", time.time()))
    return conv_id


def get_conversation(conv_id: str) -> dict | None:
    rows = _read("SELECT * FROM conversations WHERE id = ?", (conv_id,))
    return rows[0] if rows else None


def _with_status(turn: dict) -> dict:
    if turn["status"] in ACTIVE and turn["boot_id"] != BOOT_ID:
        turn["status"] = "interrupted"
    turn["details"] = json.loads(turn["details"]) if turn.get("details") else None
    return turn


def list_turns(conv_id: str) -> list[dict]:
    return [_with_status(t) for t in _read("SELECT * FROM turns WHERE conv_id = ? ORDER BY seq", (conv_id,))]


def get_turn(turn_id: str) -> dict | None:
    rows = _read("SELECT * FROM turns WHERE id = ?", (turn_id,))
    return _with_status(rows[0]) if rows else None


def is_busy(conv_id: str) -> bool:
    turns = list_turns(conv_id)
    return bool(turns) and turns[-1]["status"] in ACTIVE


def start_turn(conv_id: str, question: str) -> str:
    """Add a question to a conversation. It runs after any earlier turns finish."""
    return _start(conv_id, question)


def retry_turn(turn_id: str) -> str | None:
    """Replace the latest turn with a fresh run (from the memory before it)."""
    turn = get_turn(turn_id)
    if not turn or turn["status"] in ACTIVE:
        return None
    _write("DELETE FROM turns WHERE id = ?", (turn_id,))
    return _start(turn["conv_id"], turn["question"])


def cancel_turn(turn_id: str) -> None:
    """Stop a running turn (keeping what was written so far) or drop a queued one."""
    _cancel_requested.add(turn_id)
    turn = get_turn(turn_id)
    if turn and turn["status"] == "queued":
        _update(turn_id, status="cancelled", progress="Stopped")


def set_feedback(turn_id: str, value: int | None) -> None:
    _write("UPDATE turns SET feedback = ?, updated = ? WHERE id = ?", (value, time.time(), turn_id))


def _start(conv_id: str, question: str) -> str:
    turn_id = uuid.uuid4().hex[:12]
    seq = _read("SELECT COALESCE(MAX(seq), -1) + 1 AS n FROM turns WHERE conv_id = ?", (conv_id,))[0]["n"]
    now = time.time()
    _write(
        "INSERT INTO turns (id, conv_id, seq, question, status, progress, partial,"
        " boot_id, created, updated) VALUES (?, ?, ?, ?, 'queued', 'Waiting to start…', '', ?, ?, ?)",
        (turn_id, conv_id, seq, question, BOOT_ID, now, now),
    )
    threading.Thread(target=_run, args=(turn_id, conv_id, seq, question),
                     name=f"turn-{turn_id}", daemon=True).start()
    return turn_id


def _update(turn_id: str, **fields) -> None:
    cols = ", ".join(f"{k} = ?" for k in fields)
    _write(f"UPDATE turns SET {cols}, updated = ? WHERE id = ?", (*fields.values(), time.time(), turn_id))


def _wait_for_earlier_turns(turn_id: str, conv_id: str, seq: int) -> bool:
    """Block until earlier turns in this conversation are finished. False if cancelled meanwhile."""
    while True:
        if turn_id in _cancel_requested:
            return False
        earlier = [t for t in list_turns(conv_id) if t["seq"] < seq and t["status"] in ACTIVE]
        if not earlier:
            return True
        _update(turn_id, progress="Waiting for the previous answer to finish…")
        time.sleep(0.5)


def _memory_start(conv_id: str, seq: int) -> tuple[str, str | None]:
    """(memory thread, checkpoint) to build on: the state after the last completed turn.

    With no completed turn yet, use a brand-new thread, so leftovers from stopped,
    failed or retried attempts are never remembered.
    """
    done = [t for t in list_turns(conv_id) if t["seq"] < seq and t["status"] == "done" and t["end_checkpoint"]]
    if done:
        return done[-1]["thread_id"], done[-1]["end_checkpoint"]
    return f"{conv_id}-{uuid.uuid4().hex[:6]}", None


def _run(turn_id: str, conv_id: str, seq: int, question: str) -> None:
    """Worker: wait for its turn, run the graph, keep the turn's row up to date."""
    if not _wait_for_earlier_turns(turn_id, conv_id, seq):
        _cancel_requested.discard(turn_id)
        return
    with _slots:  # limits how many turns run at once across all conversations
        if turn_id in _cancel_requested:
            _cancel_requested.discard(turn_id)
            _update(turn_id, status="cancelled", progress="Stopped")
            return
        thread_id, parent = _memory_start(conv_id, seq)
        t0 = time.time()
        log.info("turn %s started: %s", turn_id, question[:80])
        _update(turn_id, status="running", progress="Planning…", thread_id=thread_id, parent_checkpoint=parent)
        streamed, last_write = "", 0.0
        try:
            events = stream_turn(get_agent(), question, thread_id, parent)
            for kind, value in events:
                if turn_id in _cancel_requested:
                    events.close()  # stop the graph; keep what was written so far
                    log.info("turn %s stopped by user after %.1fs", turn_id, time.time() - t0)
                    _update(turn_id, status="cancelled", progress="Stopped", answer=streamed, partial=streamed)
                    return
                if kind == "status":
                    log.info("turn %s %6.1fs %s", turn_id, time.time() - t0, value)
                    _update(turn_id, progress=value)
                elif kind == "token":
                    streamed += value
                    if time.time() - last_write > 0.25:  # don't write on every token
                        _update(turn_id, partial=streamed)
                        last_write = time.time()
                elif kind == "reset":
                    streamed = ""
                    _update(turn_id, partial="")
                elif kind == "done":
                    answer = value.get("final_answer") or streamed or "Sorry, I couldn't produce an answer."
                    details = {"tasks": value.get("tasks", []), "agent_results": value.get("agent_results", [])}
                    _update(turn_id, status="done", answer=answer, partial=answer, details=json.dumps(details),
                            progress="Done", end_checkpoint=current_checkpoint_id(get_agent(), thread_id))
            log.info("turn %s done in %.1fs", turn_id, time.time() - t0)
        except Exception as e:
            log.exception("turn %s failed", turn_id)
            _update(turn_id, status="error", error=str(e), progress="Failed")
        finally:
            _cancel_requested.discard(turn_id)
