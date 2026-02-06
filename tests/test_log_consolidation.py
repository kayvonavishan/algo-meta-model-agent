from __future__ import annotations

import json
from pathlib import Path


def test_make_log_record_fields() -> None:
    from agentic_experimentation.tree_runner import _make_log_record

    rec = _make_log_record(
        level="info",
        component="tree_runner",
        run_id="run_123",
        eval_id="0001",
        node_id="0000",
        event="scheduler_started",
        message="scheduler event",
        payload={"foo": "bar"},
    )
    assert rec["level"] == "info"
    assert rec["component"] == "tree_runner"
    assert rec["run_id"] == "run_123"
    assert rec["eval_id"] == "0001"
    assert rec["node_id"] == "0000"
    assert rec["event"] == "scheduler_started"
    assert rec["message"] == "scheduler event"
    assert rec["payload"] == {"foo": "bar"}


def test_run_event_log_paths_consolidated(tmp_path: Path) -> None:
    from agentic_experimentation.tree_runner import _run_event_log_paths

    paths = _run_event_log_paths(tmp_path)
    assert len(paths) == 1
    assert str(paths[0]).endswith(str(Path("logs") / "run_events.jsonl"))


def test_conversation_turn_log_paths_consolidated(tmp_path: Path) -> None:
    from agentic_experimentation.tree_runner import _conversation_turn_log_paths

    paths = _conversation_turn_log_paths(tmp_path, "node_0000")
    assert str(paths["primary"]).endswith(str(Path("logs") / "conversations" / "node_0000.jsonl"))
    assert paths["secondary"] is None


def test_append_jsonl_writes_valid_json(tmp_path: Path) -> None:
    from agentic_experimentation.tree_runner import _append_jsonl

    log_path = tmp_path / "events.jsonl"
    payload = {"event": "test", "value": 123}
    _append_jsonl(log_path, payload)
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == payload
