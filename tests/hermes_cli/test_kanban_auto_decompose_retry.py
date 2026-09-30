"""Durable retry policy for the gateway's paid auto-decompose path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_dispatcher as kwd
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as kd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kb.init_db()
    return home


def _dispatcher(monkeypatch: pytest.MonkeyPatch, *, failure_limit: int = 2):
    monkeypatch.setattr(kwd, "_board_slugs", lambda _kb: ["default"])
    settings = kwd._DispatcherSettings(
        60.0, None, None, failure_limit, 0, True, None, None,
    )
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def _failed(
    task_id: str,
    reason: str = "LLM returned malformed JSON",
    *,
    input_token=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_id=task_id,
        ok=False,
        fanout=False,
        child_ids=None,
        reason=reason,
        input_token=input_token,
    )


def _succeed_single(task_id: str, *, author: str | None) -> SimpleNamespace:
    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn,
            task_id,
            body="rewritten by decomposer",
            author=author,
        )
    return SimpleNamespace(
        task_id=task_id,
        ok=ok,
        fanout=False,
        child_ids=None,
        reason="" if ok else "not in triage",
    )


def _complete_decomposition_body() -> str:
    sections = (
        "Goal",
        "Known context",
        "Ownership",
        "Approach",
        "Dependencies",
        "Acceptance criteria",
        "Acceptance evidence",
        "Stop conditions",
        "Unverified / unknowns",
    )
    return "\n\n".join(f"**{heading}**\n{heading} details." for heading in sections)


@pytest.mark.parametrize("fanout", [False, True])
def test_success_for_input_edited_during_model_call_is_rejected_without_retry_debt(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    fanout: bool,
) -> None:
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="original title",
            body="original body",
            triage=True,
        )

    parsed = (
        {
            "fanout": True,
            "rationale": "split the work",
            "tasks": [{
                "title": "stale generated child",
                "body": _complete_decomposition_body(),
                "assignee": None,
                "parents": [],
            }],
        }
        if fanout
        else {
            "fanout": False,
            "rationale": "one unit of work",
            "title": "stale generated title",
            "body": _complete_decomposition_body(),
            "assignee": None,
        }
    )

    def edit_then_return_success(_verb: str, edited_task_id: str, **_kwargs):
        assert edited_task_id == task_id
        with kbc.connect_closing() as conn, kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET title = ?, body = ? WHERE id = ?",
                ("human title", "human clarified specification", task_id),
            )
            kb._append_event(conn, task_id, "edited", {"fields": ["title", "body"]})
        return json.dumps(parsed), ""

    monkeypatch.setattr(
        kd,
        "_load_routing",
        lambda **_kwargs: kd._Routing("orchestrator", "worker", False, [], set()),
    )
    monkeypatch.setattr(kd, "_call_aux", edit_then_return_success)
    real_decompose_task = kd.decompose_task
    outcomes = []

    def capture_outcome(*args, **kwargs):
        outcome = real_decompose_task(*args, **kwargs)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(kd, "decompose_task", capture_outcome)
    assert _dispatcher(monkeypatch).auto_decompose_tick(1, now=900) == 0

    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    assert outcomes[0].auto_retry_skipped is True
    assert "changed" in outcomes[0].reason
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        task_ids = [row.id for row in kb.list_tasks(conn, include_archived=True)]
        event_kinds = [event.kind for event in kb.list_events(conn, task_id)]
    assert task is not None
    assert task.status == "triage"
    assert task.title == "human title"
    assert task.body == "human clarified specification"
    assert task_ids == [task_id]
    assert "specified" not in event_kinds
    assert "decomposed" not in event_kinds
    assert "auto_decompose_failed" not in event_kinds
    assert kd.list_triage_ids(auto_retry_due_only=True, now=900) == [task_id]


def test_cooling_failure_does_not_repeat_or_starve_later_task(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect_closing() as conn:
        failing = kb.create_task(conn, title="bad model output", triage=True, priority=10)
        later = kb.create_task(conn, title="later valid task", triage=True, priority=0)

    attempted: list[str] = []

    def fake_decompose(task_id: str, author: str | None = None) -> SimpleNamespace:
        attempted.append(task_id)
        if task_id == failing:
            return _failed(task_id)
        return _succeed_single(task_id, author=author)

    monkeypatch.setattr(kd, "decompose_task", fake_decompose)
    assert _dispatcher(monkeypatch).auto_decompose_tick(1, now=1_000) == 0
    # A fresh dispatcher still reads the durable cooldown and spends its slot
    # on the later task instead of forgetting state after a gateway restart.
    assert _dispatcher(monkeypatch).auto_decompose_tick(1, now=1_001) == 1
    assert attempted == [failing, later]

    with kbc.connect_closing() as conn:
        failed_task = kb.get_task(conn, failing)
        later_task = kb.get_task(conn, later)
        failure_events = [
            event for event in kb.list_events(conn, failing)
            if event.kind == "auto_decompose_failed"
        ]
    assert failed_task is not None
    assert failed_task.status == "triage"
    assert failed_task.consecutive_failures == 0
    assert failed_task.last_failure_error is None
    assert failure_events[-1].payload["next_attempt_at"] == 1_300
    assert failure_events[-1].payload["parked"] is False
    assert later_task is not None and later_task.status == "ready"


def test_failure_recorded_after_selection_is_rechecked_before_model_call(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect_closing() as conn:
        selected_first = kb.create_task(
            conn,
            title="selected before another dispatcher fails",
            triage=True,
            priority=10,
        )
        later = kb.create_task(conn, title="later eligible task", triage=True, priority=0)

    real_list_triage_ids = kd.list_triage_ids
    failure_injected = False

    def list_then_record_concurrent_failure(**kwargs) -> list[str]:
        nonlocal failure_injected
        task_ids = real_list_triage_ids(**kwargs)
        if not failure_injected:
            failure_injected = True
            assert not kd.record_auto_decompose_failure(
                selected_first,
                "another dispatcher observed invalid output",
                failure_limit=2,
                now=1_000,
            )
        return task_ids

    model_calls: list[str] = []

    def fail_aux_call(_verb: str, task_id: str, **_kwargs):
        model_calls.append(task_id)
        return None, "synthetic auxiliary failure"

    monkeypatch.setattr(kd, "list_triage_ids", list_then_record_concurrent_failure)
    monkeypatch.setattr(
        kd,
        "_load_routing",
        lambda **_kwargs: kd._Routing("default", "default", False, [], set()),
    )
    monkeypatch.setattr(kd, "_call_aux", fail_aux_call)

    # The first id was eligible at selection time but cooled down before task
    # load. It must not spend a call or consume the one-attempt tick cap; the
    # next eligible id gets the slot instead.
    assert _dispatcher(monkeypatch).auto_decompose_tick(1, now=1_000) == 0
    assert model_calls == [later]

    with kbc.connect_closing() as conn:
        selected_failures = [
            event for event in kb.list_events(conn, selected_first)
            if event.kind == "auto_decompose_failed"
        ]
        later_failures = [
            event for event in kb.list_events(conn, later)
            if event.kind == "auto_decompose_failed"
        ]
    assert len(selected_failures) == 1
    assert selected_failures[0].payload["failures"] == 1
    assert len(later_failures) == 1


def test_auto_filter_reaches_eligible_task_after_thousand_parked_cards(
    kanban_home: Path,
) -> None:
    parked_ids = [f"t_parked_{idx:04d}" for idx in range(1_001)]
    with kbc.connect_closing() as conn, kb.write_txn(conn):
        conn.executemany(
            "INSERT INTO tasks (id, title, status, priority, created_at) "
            "VALUES (?, ?, 'triage', 10, ?)",
            [(task_id, task_id, idx) for idx, task_id in enumerate(parked_ids)],
        )
        for task_id in parked_ids:
            kb._append_event(
                conn,
                task_id,
                "auto_decompose_failed",
                {
                    "error": "invalid structured output",
                    "failures": 2,
                    "effective_limit": 2,
                    "next_attempt_at": None,
                    "parked": True,
                },
            )
        conn.execute(
            "INSERT INTO tasks (id, title, status, priority, created_at) "
            "VALUES ('t_eligible', 'eligible', 'triage', 0, 2000)",
        )

    assert kd.list_triage_ids(
        exclude_loop_detected=True,
        auto_retry_due_only=True,
        now=10_000,
    ) == ["t_eligible"]


def test_edit_resets_cooldown_and_success_clears_retry_state(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="rough idea", triage=True)

    attempts = 0

    def fail_then_succeed(task_id: str, author: str | None = None) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _failed(task_id, "decomposer field rationale must be a non-empty string")
        return _succeed_single(task_id, author=author)

    monkeypatch.setattr(kd, "decompose_task", fail_then_succeed)
    dispatcher = _dispatcher(monkeypatch)

    assert dispatcher.auto_decompose_tick(1, now=2_000) == 0
    with kbc.connect_closing() as conn, kb.write_txn(conn):
        conn.execute("UPDATE tasks SET body = ? WHERE id = ?", ("human clarified input", task_id))
        kb._append_event(conn, task_id, "edited", {"fields": ["body"]})

    # The edit changes the decomposer input, so it explicitly starts a fresh
    # automatic attempt instead of waiting for the stale-output cooldown.
    assert dispatcher.auto_decompose_tick(1, now=2_001) == 1
    assert attempts == 2
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready"
    assert task.consecutive_failures == 0
    assert task.last_failure_error is None


def test_retry_limit_parks_task_and_manual_edit_restores_clean_budget(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="always invalid", triage=True)

    attempted: list[str] = []

    def always_fail(task_id: str, author: str | None = None) -> SimpleNamespace:
        attempted.append(task_id)
        return _failed(task_id)

    monkeypatch.setattr(kd, "decompose_task", always_fail)
    dispatcher = _dispatcher(monkeypatch, failure_limit=2)

    assert dispatcher.auto_decompose_tick(1, now=3_000) == 0
    # Manual one-shot use bypasses the automatic cooldown and does not consume
    # or reset the automatic retry budget when it also fails.
    assert not kd.decompose_task(task_id, author="human").ok
    with kbc.connect_closing() as conn:
        assert len([
            event for event in kb.list_events(conn, task_id)
            if event.kind == "auto_decompose_failed"
        ]) == 1
    assert dispatcher.auto_decompose_tick(1, now=3_299) == 0
    assert dispatcher.auto_decompose_tick(1, now=3_300) == 0
    assert attempted == [task_id, task_id, task_id]

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        failures = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "auto_decompose_failed"
        ]
        assert task is not None
        assert task.status == "triage"
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        assert failures[-1].payload["failures"] == 2
        assert failures[-1].payload["parked"] is True
        assert f"hermes kanban decompose {task_id}" in failures[-1].payload["action"]

    # Even after the old cooldown would have elapsed, a parked task is absent
    # only from the automatic view; manual one-shot decomposition can still see
    # and act on it with its established semantics.
    assert dispatcher.auto_decompose_tick(1, now=9_999) == 0
    assert attempted == [task_id, task_id, task_id]
    assert kd.list_triage_ids() == [task_id]
    assert kd.list_triage_ids(auto_retry_due_only=True, now=9_999) == []

    # Changing the decomposer input is an explicit human reset.  The next
    # failure begins a fresh bounded stint at one rather than immediately
    # parking again.
    with kbc.connect_closing() as conn, kb.write_txn(conn):
        conn.execute("UPDATE tasks SET body = ? WHERE id = ?", ("human correction", task_id))
        kb._append_event(conn, task_id, "edited", {"fields": ["body"]})
    assert dispatcher.auto_decompose_tick(1, now=3_301) == 0
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        failures = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "auto_decompose_failed"
        ]
    assert task is not None
    assert task.status == "triage"
    assert failures[-1].payload["failures"] == 1
    assert failures[-1].payload["parked"] is False


def test_task_edited_during_call_does_not_record_stale_failure(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="operator may promote", triage=True)

    def edit_then_fail(task_id: str, author: str | None = None) -> SimpleNamespace:
        with kbc.connect_closing() as conn:
            input_token = kd._auto_decompose_input_token(conn, task_id)
        with kbc.connect_closing() as conn, kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET body = ? WHERE id = ?",
                ("operator supplied new decomposition input", task_id),
            )
            kb._append_event(conn, task_id, "edited", {"fields": ["body"]})
        return _failed(task_id, input_token=input_token)

    monkeypatch.setattr(kd, "decompose_task", edit_then_fail)
    assert _dispatcher(monkeypatch).auto_decompose_tick(1, now=3_500) == 0

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        failures = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "auto_decompose_failed"
        ]
    assert task is not None and task.status == "triage"
    assert failures == []


def test_edit_before_model_read_records_failure_for_new_input(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="operator may clarify", triage=True)

    def edit_then_read_and_fail(task_id: str, author: str | None = None) -> SimpleNamespace:
        with kbc.connect_closing() as conn, kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET body = ? WHERE id = ?",
                ("new input evaluated by this call", task_id),
            )
            kb._append_event(conn, task_id, "edited", {"fields": ["body"]})
        with kbc.connect_closing() as conn:
            input_token = kd._auto_decompose_input_token(conn, task_id)
        return _failed(task_id, input_token=input_token)

    monkeypatch.setattr(kd, "decompose_task", edit_then_read_and_fail)
    assert _dispatcher(monkeypatch).auto_decompose_tick(1, now=3_600) == 0

    with kbc.connect_closing() as conn:
        failures = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "auto_decompose_failed"
        ]
    assert len(failures) == 1
    assert failures[0].payload["failures"] == 1
    assert failures[0].payload["next_attempt_at"] == 3_900


def test_successful_manual_fanout_closes_automatic_failure_stint(
    kanban_home: Path,
) -> None:
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="fan out after one failure", triage=True)
    assert not kd.record_auto_decompose_failure(
        task_id,
        "LLM returned malformed JSON",
        failure_limit=2,
        now=4_000,
    )

    with kbc.connect_closing() as conn:
        child_ids = kd.decompose_triage_task(
            conn,
            task_id,
            root_assignee="orchestrator",
            children=[{
                "title": "First child",
                "body": "complete the first child",
                "assignee": "worker",
                "parents": [],
            }],
            author="human",
            auto_promote=False,
        )
        task = kb.get_task(conn, task_id)
        retry_stint_closed = kd.auto_decompose_retry_due(conn, task_id, now=4_001)

    assert child_ids
    assert task is not None
    assert task.status == "todo"
    assert task.consecutive_failures == 0
    assert task.last_failure_error is None
    assert retry_stint_closed
