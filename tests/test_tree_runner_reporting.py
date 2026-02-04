from __future__ import annotations

from pathlib import Path

from agentic_experimentation.tree_runner import _tree_summary_markdown, _validate_run


def test_tree_summary_includes_nodes_and_best_path(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    artifacts = run_root / "artifacts"
    artifacts.mkdir(parents=True)

    root_csv = artifacts / "root_baseline.csv"
    root_csv.write_text("config_id,status\n0,ok\n", encoding="utf-8")
    parent_csv = artifacts / "baseline_node_0000.csv"
    parent_csv.write_text("config_id,status\n0,ok\n", encoding="utf-8")
    cand_csv = artifacts / "candidate_eval_0001.csv"
    cand_csv.write_text("config_id,status\n0,ok\n", encoding="utf-8")

    manifest = {
        "tree_run_id": "t0",
        "run_config": {
            "ideas_per_node": 2,
            "max_depth": 2,
            "beam_width": 1,
            "sweep_config_limit": 25,
            "ideas_context_strategy": "node_plus_ancestors",
        },
        "state": {"stop_reason": None},
        "nodes": {
            "0000": {
                "node_id": "0000",
                "parent_node_id": None,
                "depth": 0,
                "commit": "c0",
                "ref_name": "",
                "worktree_path": "",
                "baseline_results_csv_path": str(root_csv),
                "node_ideas_dir": str(run_root / "node_ideas" / "0000"),
                "idea_chain": [],
                "artifacts": {},
            },
            "0001": {
                "node_id": "0001",
                "parent_node_id": "0000",
                "depth": 1,
                "commit": "c1",
                "ref_name": "",
                "worktree_path": "",
                "baseline_results_csv_path": str(cand_csv),
                "node_ideas_dir": str(run_root / "node_ideas" / "0001"),
                "idea_chain": ["idea_a.md"],
                "artifacts": {"promoted_from_eval_id": "0001"},
            },
        },
        "evaluations": {
            "0001": {
                "eval_id": "0001",
                "parent_node_id": "0000",
                "depth": 0,
                "idea_path": "idea_a.md",
                "experiment_dir": str(run_root / "eval" / "0001" / "experiment"),
                "candidate_results_csv_path": str(cand_csv),
                "parent_baseline_provenance": {"copied_to_path": str(parent_csv), "sha256": "x"},
                "strict_completeness": {"ok_count": 25, "expected_count": 25},
                "decision": {"passed_gate": True, "promotion_reason": None, "rank_score": 1.0},
                "root_relative": {"recommendation_summary": {"grade": "good", "should_explore": True}},
                "parent_relative": {"candidate_rows_used": 25},
            }
        },
    }

    md = _tree_summary_markdown(run_root=run_root, manifest=manifest)
    assert "TREE_SUMMARY" in md
    assert "| 1 | 0001 | 0000 |" in md
    assert "Best Path" in md
    assert "best_node_id" in md


def test_validate_run_flags_paths_outside_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    artifacts = run_root / "artifacts"
    artifacts.mkdir(parents=True)

    outside = tmp_path / "outside.csv"
    outside.write_text("config_id,status\n0,ok\n", encoding="utf-8")

    manifest = {
        "tree_run_id": "t0",
        "root": {"root_baseline_csv_path": str(outside)},
        "nodes": {"0000": {"baseline_results_csv_path": str(outside), "ref_name": "", "commit": ""}},
        "evaluations": {"0001": {"candidate_results_csv_path": str(outside)}},
    }

    issues = _validate_run(repo_root=tmp_path, run_root=run_root, manifest=manifest)
    kinds = {i["kind"] for i in issues}
    assert "path_scope" in kinds


def test_validate_run_flags_missing_eval_paths(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "artifacts").mkdir(parents=True)

    manifest = {
        "tree_run_id": "t0",
        "nodes": {"0000": {"baseline_results_csv_path": str(run_root / "artifacts" / "root.csv"), "ref_name": "", "commit": "", "depth": 0}},
        "evaluations": {
            "0001": {
                "eval_id": "0001",
                "experiment_dir": str(run_root / "eval" / "0001" / "experiment"),
                "parent_relative": {"summary_json_path": str(run_root / "eval" / "0001" / "experiment" / "0001" / "summary.json")},
            }
        },
    }

    issues = _validate_run(repo_root=tmp_path, run_root=run_root, manifest=manifest)
    messages = {i["message"] for i in issues}
    assert "Experiment directory missing" in messages
    assert "summary_json_path missing" in messages


def test_validate_run_flags_frontier_depth_holes(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    artifacts = run_root / "artifacts"
    artifacts.mkdir(parents=True)
    csv0 = artifacts / "n0.csv"
    csv0.write_text("config_id,status\n0,ok\n", encoding="utf-8")
    csv1 = artifacts / "n1.csv"
    csv1.write_text("config_id,status\n0,ok\n", encoding="utf-8")

    manifest = {
        "tree_run_id": "t0",
        "state": {
            "current_depth": 2,
            "frontier_node_ids": ["0001"],
            "expanded_node_ids_by_depth": {"0": ["0000"]},
        },
        "nodes": {
            "0000": {"baseline_results_csv_path": str(csv0), "ref_name": "", "commit": "", "depth": 0},
            # depth=1 node exists but not expanded at depth 1, yet current_depth is 2 -> hole
            "0001": {"baseline_results_csv_path": str(csv1), "ref_name": "", "commit": "", "depth": 1},
        },
    }

    issues = _validate_run(repo_root=tmp_path, run_root=run_root, manifest=manifest)
    msgs = [i["message"] for i in issues if i.get("kind") == "state_inconsistency"]
    assert "frontier_node_depth_mismatch" in msgs
    assert "unexpanded_nodes_before_current_depth" in msgs
