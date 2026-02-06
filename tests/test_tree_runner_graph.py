from __future__ import annotations

from pathlib import Path


def test_tree_graph_markdown_includes_nodes_edges_and_frontier(tmp_path: Path) -> None:
    from agentic_experimentation.tree_runner import _tree_graph_markdown

    manifest = {
        "tree_run_id": "t1",
        "state": {"frontier_node_ids": ["0001"]},
        "nodes": {
            "0000": {"node_id": "0000", "depth": 0, "parent_node_id": None, "artifacts": {}},
            "0001": {
                "node_id": "0001",
                "depth": 1,
                "parent_node_id": "0000",
                "conversation_id": "node_0001",
                "artifacts": {"promoted_from_eval_id": "0003"},
            },
        },
        "conversations": {
            "node_0001": {
                "conversation_id": "node_0001",
                "parent_conversation_id": "node_0000",
                "fork_from_turn_id": "turn_0004",
            }
        },
        "evaluations": {
            "0003": {
                "eval_id": "0003",
                "root_relative": {
                    "recommendation_summary": {"score": 0.05, "grade": "promising"},
                    "primary_delta": 0.1,
                },
            }
        },
    }

    md = _tree_graph_markdown(run_root=tmp_path, manifest=manifest)
    assert "```mermaid" in md
    assert "N0000" in md
    assert "N0001" in md
    assert "N0000 -->|eval 0003" in md
    assert "class N0001 frontier;" in md
    assert "conv node_0000->node_0001 @turn_0004" in md
