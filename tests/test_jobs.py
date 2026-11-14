"""Background jobs: turns run in worker threads and are saved in SQLite."""

import time

import pytest

from test_multiturn import fake  # noqa: F401  (pytest fixture: fake model and tools)


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    from travel_planner import config
    import travel_planner.jobs as jobs_mod

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "app.sqlite"))
    jobs_mod._db.cache_clear()
    jobs_mod._agent = None
    yield jobs_mod
    jobs_mod._db.cache_clear()
    jobs_mod._agent = None


def wait(jobs, turn_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        turn = jobs.get_turn(turn_id)
        if turn and turn["status"] not in jobs.ACTIVE:
            return turn
        time.sleep(0.05)
    raise AssertionError("turn did not finish")


def test_turn_runs_in_background_and_is_saved(fake, jobs):
    conv = jobs.new_conversation()
    turn_id = jobs.start_turn(conv, "Best time to visit Jaipur?")
    turn = wait(jobs, turn_id)
    assert turn["status"] == "done"
    assert turn["answer"] == "answer to: Best time to visit Jaipur?"
    assert turn["details"]["tasks"][0]["source"] == "planner_agent_node"
    assert not jobs.is_busy(conv)


def test_follow_up_uses_saved_memory_and_retry_replaces_turn(fake, jobs):
    conv = jobs.new_conversation()
    wait(jobs, jobs.start_turn(conv, "Best time to visit Jaipur?"))
    second = wait(jobs, jobs.start_turn(conv, "Major sites there"))
    assert fake.router_inputs[-1][:2] == ["Best time to visit Jaipur?", "answer to: Best time to visit Jaipur?"]

    fake.router_inputs.clear()
    retried = wait(jobs, jobs.retry_turn(second["id"]))
    assert retried["status"] == "done"
    assert [t["question"] for t in jobs.list_turns(conv)] == ["Best time to visit Jaipur?", "Major sites there"]
    # The retry saw turn 1 but not the first attempt at turn 2
    assert fake.router_inputs[-1] == [
        "Best time to visit Jaipur?", "answer to: Best time to visit Jaipur?", "Major sites there",
    ]


def test_retry_of_first_turn_starts_fresh_memory(fake, jobs):
    conv = jobs.new_conversation()
    first = wait(jobs, jobs.start_turn(conv, "Hello"))
    fake.router_inputs.clear()
    wait(jobs, jobs.retry_turn(first["id"]))
    assert fake.router_inputs[-1] == ["Hello"]  # no memory of the first attempt


def test_errors_and_restarts_are_reported(fake, jobs, monkeypatch):
    import travel_planner.graph.nodes as nodes

    class Boom:
        def with_structured_output(self, *a, **k):
            raise RuntimeError("Error code: 403")

    monkeypatch.setattr(nodes, "get_llm", lambda: Boom())
    conv = jobs.new_conversation()
    failed = wait(jobs, jobs.start_turn(conv, "hi"))
    assert failed["status"] == "error" and "403" in failed["error"]

    # A running turn from an earlier server process shows as interrupted
    jobs._write("UPDATE turns SET status = 'running', boot_id = 'old-process' WHERE id = ?", (failed["id"],))
    assert jobs.get_turn(failed["id"])["status"] == "interrupted"
    assert not jobs.is_busy(conv)


def test_feedback_is_saved(fake, jobs):
    conv = jobs.new_conversation()
    turn = wait(jobs, jobs.start_turn(conv, "Hi"))
    jobs.set_feedback(turn["id"], 1)
    assert jobs.get_turn(turn["id"])["feedback"] == 1


def _slow_streaming(monkeypatch):
    """Planner answers word by word, slowly enough to stop mid-answer."""
    import travel_planner.graph.nodes as nodes
    from test_streaming import StreamingFakeModel

    class Slow(StreamingFakeModel):
        def _stream(self, *a, **kw):
            for chunk in super()._stream(*a, **kw):
                time.sleep(0.3)
                yield chunk

    model = Slow()
    model.router_inputs.clear()
    for name in ("get_llm", "_search_model_with_tools", "_planner_model_with_tools"):
        monkeypatch.setattr(nodes, name, lambda: model)
    return model


def test_message_sent_while_busy_waits_then_runs_with_memory(fake, jobs, monkeypatch):
    model = _slow_streaming(monkeypatch)
    conv = jobs.new_conversation()
    first = jobs.start_turn(conv, "Best time to visit Jaipur please")
    second = jobs.start_turn(conv, "Major sites there")  # sent while the first is running
    time.sleep(0.3)
    assert jobs.get_turn(second)["status"] == "queued"
    wait(jobs, first)
    assert wait(jobs, second)["status"] == "done"
    # The second turn ran after the first, with the first answer in memory
    assert model.router_inputs[-1][:2] == ["Best time to visit Jaipur please",
                                           "answer to: Best time to visit Jaipur please"]


def test_stop_keeps_partial_answer_and_leaves_memory_clean(fake, jobs, monkeypatch):
    model = _slow_streaming(monkeypatch)
    conv = jobs.new_conversation()
    wait(jobs, jobs.start_turn(conv, "Hi"))
    running = jobs.start_turn(conv, "one two three four five six seven eight nine ten")
    deadline = time.time() + 10
    while not jobs.get_turn(running)["partial"] and time.time() < deadline:
        time.sleep(0.05)
    jobs.cancel_turn(running)
    stopped = wait(jobs, running)
    assert stopped["status"] == "cancelled"
    full = "answer to: one two three four five six seven eight nine ten"
    assert stopped["answer"] and full.startswith(stopped["answer"]) and stopped["answer"] != full

    # The next turn builds on "Hi" only; the stopped question isn't in memory
    model.router_inputs.clear()
    wait(jobs, jobs.start_turn(conv, "Next"))
    assert model.router_inputs[-1] == ["Hi", "answer to: Hi", "Next"]


def test_queued_message_can_be_dropped(fake, jobs, monkeypatch):
    _slow_streaming(monkeypatch)
    conv = jobs.new_conversation()
    first = jobs.start_turn(conv, "one two three four five")
    queued = jobs.start_turn(conv, "never mind")
    jobs.cancel_turn(queued)
    assert jobs.get_turn(queued)["status"] == "cancelled"
    assert wait(jobs, first)["status"] == "done"
    time.sleep(0.6)
    assert jobs.get_turn(queued)["status"] == "cancelled"  # never ran
