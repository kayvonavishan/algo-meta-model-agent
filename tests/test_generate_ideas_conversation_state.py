from __future__ import annotations

import json
from pathlib import Path

from agentic_experimentation.idea_generation.generate_ideas import (
    _compact_conversation_state,
    _find_turn_index_by_operation_id,
    _load_conversation_state,
    _render_conversation_replay_block,
)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def test_compaction_keeps_recent_and_writes_summary() -> None:
    state = {
        "turns": [
            {"turn_id": "turn_0001", "output_title": "one", "output_hash": "a1"},
            {"turn_id": "turn_0002", "output_title": "two", "output_hash": "a2"},
            {"turn_id": "turn_0003", "output_title": "three", "output_hash": "a3"},
            {"turn_id": "turn_0004", "output_title": "four", "output_hash": "a4"},
            {"turn_id": "turn_0005", "output_title": "five", "output_hash": "a5"},
        ],
        "latest_turn_id": "turn_0005",
        "compact_summary_text": "",
    }
    _compact_conversation_state(state=state, keep_recent_turns=2, summary_max_chars=160)

    turns = state["turns"]
    assert len(turns) == 2
    assert turns[0]["turn_id"] == "turn_0004"
    assert turns[1]["turn_id"] == "turn_0005"
    assert state["latest_turn_id"] == "turn_0005"
    assert state["compact_summary_text"]
    assert len(state["compact_summary_text"]) <= 160


def test_replay_block_respects_char_budget() -> None:
    raw_payload_len = len(("alpha" * 50) + ("beta" * 50) + ("gamma" * 50))
    state = {
        "turns": [
            {"turn_id": "turn_0001", "timestamp": "t1", "output_idea_path": "ideas/one.md", "output_text": "alpha" * 50},
            {"turn_id": "turn_0002", "timestamp": "t2", "output_idea_path": "ideas/two.md", "output_text": "beta" * 50},
            {"turn_id": "turn_0003", "timestamp": "t3", "output_idea_path": "ideas/three.md", "output_text": "gamma" * 50},
        ],
        "compact_summary_text": "",
        "compact_summary_updated_at": None,
    }
    block = _render_conversation_replay_block(conversation_state=state, max_turns=2, max_chars=300)
    assert "IDEA CONVERSATION MEMORY (REPLAY)" in block
    assert "turn_0003" in block
    assert len(block) < raw_payload_len + 400


def test_replay_block_zero_budget_disables_memory() -> None:
    state = {
        "turns": [
            {"turn_id": "turn_0001", "timestamp": "t1", "output_idea_path": "ideas/one.md", "output_text": "alpha"},
        ],
        "compact_summary_text": "",
        "compact_summary_updated_at": None,
    }
    block = _render_conversation_replay_block(conversation_state=state, max_turns=8, max_chars=0)
    assert block == ""


def test_replay_block_negative_budget_is_unbounded() -> None:
    state = {
        "turns": [
            {"turn_id": "turn_0001", "timestamp": "t1", "output_idea_path": "ideas/one.md", "output_text": "alpha"},
            {"turn_id": "turn_0002", "timestamp": "t2", "output_idea_path": "ideas/two.md", "output_text": "beta"},
        ],
        "compact_summary_text": "",
        "compact_summary_updated_at": None,
    }
    block = _render_conversation_replay_block(conversation_state=state, max_turns=8, max_chars=-1)
    assert "turn_0001" in block
    assert "turn_0002" in block
    assert "alpha" in block
    assert "beta" in block


def test_load_state_backfills_operation_map_and_lookup(tmp_path: Path) -> None:
    state_path = tmp_path / "conversation.json"
    raw_state = {
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "requested_mode": "replay",
        "next_turn_index": 2,
        "turns": [
            {
                "turn_id": "turn_0001",
                "prompt_hash": "abc123",
                "output_hash": "out123",
                "output_idea_path_relative_to_repo": "agentic_experimentation/worktrees/tree_runs/t/node_ideas/0000/000.md",
            }
        ],
        "operation_id_turn_map": {},
    }
    _write_json(state_path, raw_state)

    loaded = _load_conversation_state(state_path, requested_mode="replay")
    assert loaded["turns"][0].get("operation_id")
    op_map = loaded["operation_id_turn_map"]
    assert isinstance(op_map, dict)
    assert len(op_map) == 1
    op_id = next(iter(op_map.keys()))
    assert op_map[op_id] == "turn_0001"
    assert _find_turn_index_by_operation_id(loaded, op_id) == 0
