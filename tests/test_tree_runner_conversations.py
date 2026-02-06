from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_experimentation.tree_runner import (
    _ensure_manifest_conversation_schema,
    _ensure_node_conversation,
    _sync_node_conversation_latest_turn,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        idea_conversation_mode="auto",
        idea_history_window_turns=12,
        idea_history_max_chars=20000,
    )


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_schema_migration_adds_conversation_fields(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    manifest = {
        "manifest_version": 2,
        "run_config": {},
        "nodes": {
            "0000": {"node_id": "0000", "parent_node_id": None, "depth": 0},
            "0001": {"node_id": "0001", "parent_node_id": "0000", "depth": 1},
        },
        "evaluations": {"0001": {"eval_id": "0001"}},
    }

    _ensure_manifest_conversation_schema(manifest=manifest, run_root=run_root, args=_args())

    assert manifest["manifest_version"] == 3
    conv_cfg = manifest["conversation_config"]
    assert conv_cfg["mode"] == "auto"
    assert conv_cfg["history_window_turns"] == 12
    assert conv_cfg["history_max_chars"] == 20000
    assert str(conv_cfg["debug_log_jsonl_path"]).replace("\\", "/").endswith(
        "logs/conversations/conversation_debug.jsonl"
    )

    nodes = manifest["nodes"]
    assert nodes["0000"]["conversation_id"] == "node_0000"
    assert nodes["0001"]["conversation_id"] == "node_0001"
    assert "latest_conversation_turn_id" in nodes["0000"]
    assert "expansion_seed_turn_id" in nodes["0000"]

    evals = manifest["evaluations"]
    assert evals["0001"]["idea_generation_conversation_id"] is None
    assert evals["0001"]["idea_generation_turn_id"] is None

    conversations = manifest["conversations"]
    assert "node_0000" in conversations
    assert "node_0001" in conversations
    root_state = Path(conversations["node_0000"]["state_json_path"])
    child_state = Path(conversations["node_0001"]["state_json_path"])
    assert root_state.exists()
    assert child_state.exists()


def test_conversation_fork_and_sync_latest_turn(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    manifest = {
        "manifest_version": 3,
        "run_config": {
            "idea_conversation_mode": "auto",
            "idea_history_window_turns": 12,
            "idea_history_max_chars": 20000,
        },
        "conversation_config": {
            "mode": "auto",
            "history_window_turns": 12,
            "history_max_chars": 20000,
            "debug_log_jsonl_path": str(run_root / "logs" / "conversations" / "conversation_debug.jsonl"),
        },
        "conversations": {},
        "nodes": {
            "0000": {"node_id": "0000", "parent_node_id": None, "depth": 0},
            "0001": {"node_id": "0001", "parent_node_id": "0000", "depth": 1},
        },
        "evaluations": {},
    }

    root_conv = _ensure_node_conversation(
        manifest=manifest,
        run_root=run_root,
        node_id="0000",
        parent_node_id=None,
        fork_from_turn_id=None,
    )
    root_state_path = Path(manifest["conversations"][root_conv]["state_json_path"])
    root_state = _read_json(root_state_path)
    root_state["turns"] = [
        {"turn_id": "turn_0001", "output_hash": "a", "output_idea_path": "ideas/a.md"},
        {"turn_id": "turn_0002", "output_hash": "b", "output_idea_path": "ideas/b.md"},
    ]
    root_state["latest_turn_id"] = "turn_0002"
    root_state["next_turn_index"] = 3
    _write_json(root_state_path, root_state)

    child_conv = _ensure_node_conversation(
        manifest=manifest,
        run_root=run_root,
        node_id="0001",
        parent_node_id="0000",
        fork_from_turn_id="turn_0001",
    )
    child_state_path = Path(manifest["conversations"][child_conv]["state_json_path"])
    child_state = _read_json(child_state_path)

    assert child_state["parent_conversation_id"] == root_conv
    assert child_state["fork_from_turn_id"] == "turn_0001"
    assert [t["turn_id"] for t in child_state["turns"]] == ["turn_0001"]
    assert child_state["latest_turn_id"] == "turn_0001"
    assert int(child_state["next_turn_index"]) == 2

    child_state["turns"].append({"turn_id": "turn_0002", "output_hash": "c", "output_idea_path": "ideas/c.md"})
    child_state["latest_turn_id"] = "turn_0002"
    child_state["next_turn_index"] = 3
    _write_json(child_state_path, child_state)

    latest = _sync_node_conversation_latest_turn(manifest=manifest, run_root=run_root, node_id="0001")
    assert latest == "turn_0002"
    assert manifest["nodes"]["0001"]["latest_conversation_turn_id"] == "turn_0002"


def test_branch_lineage_is_isolated_across_siblings(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    manifest = {
        "manifest_version": 3,
        "run_config": {
            "idea_conversation_mode": "auto",
            "idea_history_window_turns": 12,
            "idea_history_max_chars": 20000,
        },
        "conversation_config": {
            "mode": "auto",
            "history_window_turns": 12,
            "history_max_chars": 20000,
            "debug_log_jsonl_path": str(run_root / "logs" / "conversations" / "conversation_debug.jsonl"),
        },
        "conversations": {},
        "nodes": {
            "0000": {"node_id": "0000", "parent_node_id": None, "depth": 0, "expansion_seed_turn_id": "turn_0001"},
            "0001": {"node_id": "0001", "parent_node_id": "0000", "depth": 1, "expansion_seed_turn_id": "turn_0001"},
            "0002": {"node_id": "0002", "parent_node_id": "0000", "depth": 1, "expansion_seed_turn_id": "turn_0001"},
            "0003": {"node_id": "0003", "parent_node_id": "0000", "depth": 1, "expansion_seed_turn_id": "turn_0001"},
            "0101": {"node_id": "0101", "parent_node_id": "0001", "depth": 2},
            "0102": {"node_id": "0102", "parent_node_id": "0002", "depth": 2},
            "0103": {"node_id": "0103", "parent_node_id": "0003", "depth": 2},
        },
        "evaluations": {},
    }

    root_conv = _ensure_node_conversation(
        manifest=manifest,
        run_root=run_root,
        node_id="0000",
        parent_node_id=None,
        fork_from_turn_id=None,
    )
    root_state_path = Path(manifest["conversations"][root_conv]["state_json_path"])
    root_state = _read_json(root_state_path)
    root_state["turns"] = [{"turn_id": "turn_0001", "output_idea_path": "ideas/root.md", "output_hash": "root"}]
    root_state["latest_turn_id"] = "turn_0001"
    root_state["next_turn_index"] = 2
    _write_json(root_state_path, root_state)

    child_nodes = ["0001", "0002", "0003"]
    for idx, nid in enumerate(child_nodes, start=1):
        child_conv = _ensure_node_conversation(
            manifest=manifest,
            run_root=run_root,
            node_id=nid,
            parent_node_id="0000",
            fork_from_turn_id="turn_0001",
        )
        child_state_path = Path(manifest["conversations"][child_conv]["state_json_path"])
        child_state = _read_json(child_state_path)
        child_state["turns"].append(
            {
                "turn_id": "turn_0002",
                "output_idea_path": f"ideas/{nid}.md",
                "output_hash": f"child-{idx}",
            }
        )
        child_state["latest_turn_id"] = "turn_0002"
        child_state["next_turn_index"] = 3
        _write_json(child_state_path, child_state)
        _sync_node_conversation_latest_turn(manifest=manifest, run_root=run_root, node_id=nid)

    for parent_nid, child_nid in [("0001", "0101"), ("0002", "0102"), ("0003", "0103")]:
        parent_conv = manifest["nodes"][parent_nid]["conversation_id"]
        fork_turn = manifest["nodes"][parent_nid]["latest_conversation_turn_id"]
        _ensure_node_conversation(
            manifest=manifest,
            run_root=run_root,
            node_id=child_nid,
            parent_node_id=parent_nid,
            fork_from_turn_id=fork_turn,
        )
        gc_conv = manifest["nodes"][child_nid]["conversation_id"]
        gc_state = _read_json(Path(manifest["conversations"][gc_conv]["state_json_path"]))
        joined_paths = "\n".join(str(t.get("output_idea_path")) for t in gc_state["turns"])
        assert f"ideas/{parent_nid}.md" in joined_paths
        sibling_ids = {"0001", "0002", "0003"} - {parent_nid}
        for sid in sibling_ids:
            assert f"ideas/{sid}.md" not in joined_paths


def test_resume_preserves_existing_conversation_lineage(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    manifest = {
        "manifest_version": 2,
        "run_config": {},
        "nodes": {
            "0000": {"node_id": "0000", "parent_node_id": None, "depth": 0},
        },
        "evaluations": {},
    }

    _ensure_manifest_conversation_schema(manifest=manifest, run_root=run_root, args=_args())
    conv_id_before = manifest["nodes"]["0000"]["conversation_id"]
    state_path = Path(manifest["conversations"][conv_id_before]["state_json_path"])
    state = _read_json(state_path)
    state["turns"] = [{"turn_id": "turn_0001", "output_idea_path": "ideas/root.md", "output_hash": "x"}]
    state["latest_turn_id"] = "turn_0001"
    state["next_turn_index"] = 2
    _write_json(state_path, state)

    _ensure_manifest_conversation_schema(manifest=manifest, run_root=run_root, args=_args())
    conv_id_after = manifest["nodes"]["0000"]["conversation_id"]
    assert conv_id_after == conv_id_before

    _sync_node_conversation_latest_turn(manifest=manifest, run_root=run_root, node_id="0000")
    assert manifest["nodes"]["0000"]["latest_conversation_turn_id"] == "turn_0001"
