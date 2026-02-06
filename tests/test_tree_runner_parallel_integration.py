from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from agentic_experimentation import tree_runner


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_resume_run(
    *,
    runs_root: Path,
    tree_run_id: str,
    config_path: Path,
    baseline_csv: Path,
    ideas_per_node: int,
    beam_width: int,
    max_depth: int,
    max_parallel_evals: int,
    max_parallel_per_node: int | None = None,
    eval_retries: int = 0,
    sweep_config_limit: int = 5,
    preexisting_evaluations: dict[str, dict[str, Any]] | None = None,
    task_plan_depth0: list[dict[str, Any]] | None = None,
) -> Path:
    run_root = runs_root / tree_run_id
    artifacts_root = run_root / "artifacts"
    node_ideas_dir = run_root / "node_ideas" / "0000"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    node_ideas_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(1, ideas_per_node + 1):
        idea_path = node_ideas_dir / f"idea_{idx:03d}.md"
        idea_path.write_text(
            "\n".join(
                [
                    f"IDEA: integration test idea {idx}",
                    "RATIONALE: test",
                    "REQUIRED_CHANGES: test",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    root_baseline_copy = artifacts_root / "root_baseline.csv"
    root_baseline_copy.write_text(baseline_csv.read_text(encoding="utf-8"), encoding="utf-8")

    evals = dict(preexisting_evaluations or {})
    max_eval_id = 0
    for eval_id in evals.keys():
        try:
            max_eval_id = max(max_eval_id, int(eval_id))
        except Exception:
            pass

    manifest = {
        "manifest_version": 3,
        "tree_run_id": tree_run_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "run_config": {
            "ideas_per_node": int(ideas_per_node),
            "max_depth": int(max_depth),
            "beam_width": int(beam_width),
            "max_parallel_evals": int(max_parallel_evals),
            "max_parallel_per_node": (int(max_parallel_per_node) if max_parallel_per_node is not None else None),
            "parallel_backend": "threadpool",
            "eval_retries": int(eval_retries),
            "eval_timeout_seconds": None,
            "strict_fail_depth": False,
            "sweep_config_limit": int(sweep_config_limit),
            "max_total_idea_evals": None,
            "stop_on_empty_frontier": True,
            "resume": True,
            "lock_stale_seconds": 600,
            "artifact_policy": "copy_to_run_root",
            "node_ideas_root_dir": str(run_root / "node_ideas"),
            "ideas_context_strategy": "node_plus_ancestors",
            "idea_conversation_mode": "auto",
            "idea_history_window_turns": 12,
            "idea_history_max_chars": 20000,
            "agent_config_path": str(config_path),
            "agent_config_snapshot": {"baseline_csv": str(baseline_csv)},
            "runs_root": str(runs_root),
            "run_root": str(run_root),
            "wt_root": str(run_root / "wt"),
            "cand_root": str(run_root / "cand"),
            "eval_root": str(run_root / "eval"),
            "conversations_root": str(run_root / "conversations"),
        },
        "conversation_config": {
            "mode": "auto",
            "history_window_turns": 12,
            "history_max_chars": 20000,
            "debug_log_jsonl_path": str(run_root / "conversations" / "conversation_debug.jsonl"),
        },
        "root": {
            "root_commit": "dry_root_commit",
            "root_ref_name": "tree/test/n0000",
            "root_baseline_csv_path": str(root_baseline_copy),
            "root_baseline_provenance": {
                "source_path": str(baseline_csv),
                "copied_to_path": str(root_baseline_copy),
                "sha256": "x",
            },
        },
        "state": {
            "current_depth": 0,
            "frontier_node_ids": ["0000"],
            "expanded_node_ids_by_depth": {},
            "completed_depths": [],
            "next_node_id": 1,
            "next_eval_id": int(max_eval_id + 1 if max_eval_id > 0 else 1),
            "deferred_cleanup": [],
            "task_plan_by_depth": {"0": list(task_plan_depth0 or [])},
        },
        "events": [],
        "conversations": {},
        "nodes": {
            "0000": {
                "node_id": "0000",
                "parent_node_id": None,
                "depth": 0,
                "commit": "dry_root_commit",
                "ref_name": "tree/test/n0000",
                "worktree_path": str(run_root / "wt" / "0000"),
                "baseline_results_csv_path": str(root_baseline_copy),
                "node_ideas_dir": str(node_ideas_dir),
                "idea_chain": [],
                "conversation_id": None,
                "expansion_seed_turn_id": None,
                "latest_conversation_turn_id": None,
                "artifacts": {},
                "created_at": "2026-01-01T00:00:00+00:00",
                "status": "ready",
            }
        },
        "evaluations": evals,
    }

    _write_json(run_root / "manifest.json", manifest)
    return run_root


def _run_tree_resume_dry(*, runs_root: Path, tree_run_id: str, config_path: Path) -> dict[str, Any]:
    agentic_dir = (Path(__file__).resolve().parents[1] / "agentic_experimentation").resolve()
    added = False
    if str(agentic_dir) not in sys.path:
        sys.path.insert(0, str(agentic_dir))
        added = True
    rc = tree_runner.main(
        [
            "--tree-run-id",
            tree_run_id,
            "--runs-root",
            str(runs_root),
            "--resume",
            "--dry-run",
            "--config",
            str(config_path),
        ]
    )
    if added:
        try:
            sys.path.remove(str(agentic_dir))
        except ValueError:
            pass
    assert rc == 0
    return _read_json(runs_root / tree_run_id / "manifest.json")


def _promoted_eval_ids(manifest: dict[str, Any]) -> list[str]:
    nodes = manifest.get("nodes") or {}
    promoted: list[str] = []
    if isinstance(nodes, dict):
        for node_id, node_rec in nodes.items():
            if str(node_id) == "0000" or not isinstance(node_rec, dict):
                continue
            artifacts = node_rec.get("artifacts") or {}
            if isinstance(artifacts, dict):
                eval_id = artifacts.get("promoted_from_eval_id")
                if eval_id:
                    promoted.append(str(eval_id))
    return sorted(promoted)


def _stub_completed_worker(**kwargs: Any) -> dict[str, Any]:
    eval_id = str(kwargs["eval_id"])
    run_root = Path(str(kwargs["run_root"]))
    node_baseline_csv_path = Path(str(kwargs["node_baseline_csv_path"]))
    sweep_limit = kwargs.get("sweep_config_limit")
    scenario = tree_runner._dry_run_scenario(eval_id=eval_id)

    artifacts_root = run_root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    candidate_csv = artifacts_root / f"candidate_eval_{eval_id}.csv"
    limit = int(sweep_limit) if sweep_limit is not None else 1
    lines = ["config_id,status,mean_topN_avg_return_per_trade_pct_oos"]
    strict_complete = bool(scenario.get("strict_complete", True))
    for idx in range(limit):
        status = "ok" if strict_complete else ("error" if idx == 0 else "ok")
        lines.append(f"{idx},{status},0.0")
    candidate_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    parent_primary = float(scenario.get("primary_delta") or 0.0)
    rank_score = scenario.get("rank_score")
    should_explore = bool(scenario.get("should_explore", False))
    grade = str(scenario.get("grade") or "")

    updates: dict[str, Any] = {
        "candidate_commit": f"dry_commit_{eval_id}",
        "candidate_results_csv_path": str(candidate_csv),
        "parent_baseline_provenance": {
            "source_path": str(node_baseline_csv_path),
            "copied_to_path": str(node_baseline_csv_path),
            "sha256": "x",
        },
        "candidate_results_provenance": {
            "source_path": str(candidate_csv),
            "copied_to_path": str(candidate_csv),
            "sha256": "x",
        },
        "parent_relative": {
            "recommendation_summary": {
                "should_explore": should_explore,
                "grade": grade,
                "score": rank_score,
                "reasons": ["stub"],
            },
            "primary_delta": parent_primary,
            "baseline_rows_used": limit,
            "candidate_rows_used": limit,
            "score": rank_score,
            "summary_json_path": "",
        },
        "root_relative": {
            "recommendation_summary": {
                "should_explore": should_explore,
                "grade": grade,
                "score": rank_score,
                "reasons": ["stub"],
            },
            "primary_delta": parent_primary,
            "baseline_rows_used": limit,
            "candidate_rows_used": limit,
            "score": rank_score,
        },
    }
    if sweep_limit is not None:
        updates["strict_completeness"] = {
            "ok_count": (limit if strict_complete else max(0, limit - 1)),
            "expected_count": limit,
            "is_complete": strict_complete,
        }

    return {
        "status": "completed",
        "error": None,
        "timed_out": False,
        "updates": updates,
        "cleanup_requests": [],
    }


def test_parallel_dry_run_promotions_match_serial(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tree_runner, "_execute_eval_task_worker", _stub_completed_worker)

    runs_root = tmp_path / "runs"
    baseline_csv = tmp_path / "baseline.csv"
    baseline_csv.write_text("config_id,status\n0,ok\n1,ok\n2,ok\n", encoding="utf-8")
    config_path = tmp_path / "agent_config.json"
    _write_json(config_path, {"baseline_csv": str(baseline_csv), "scoring": {"score_column": "core_topN_sharpe"}})

    _bootstrap_resume_run(
        runs_root=runs_root,
        tree_run_id="serial_case",
        config_path=config_path,
        baseline_csv=baseline_csv,
        ideas_per_node=6,
        beam_width=2,
        max_depth=1,
        max_parallel_evals=1,
        max_parallel_per_node=None,
        eval_retries=0,
        sweep_config_limit=5,
    )
    serial_manifest = _run_tree_resume_dry(runs_root=runs_root, tree_run_id="serial_case", config_path=config_path)

    _bootstrap_resume_run(
        runs_root=runs_root,
        tree_run_id="parallel_case",
        config_path=config_path,
        baseline_csv=baseline_csv,
        ideas_per_node=6,
        beam_width=2,
        max_depth=1,
        max_parallel_evals=3,
        max_parallel_per_node=1,
        eval_retries=0,
        sweep_config_limit=5,
    )
    parallel_manifest = _run_tree_resume_dry(runs_root=runs_root, tree_run_id="parallel_case", config_path=config_path)

    assert _promoted_eval_ids(serial_manifest) == _promoted_eval_ids(parallel_manifest)
    assert str((serial_manifest.get("state") or {}).get("stop_reason")) == "max_depth_reached"
    assert str((parallel_manifest.get("state") or {}).get("stop_reason")) == "max_depth_reached"


def test_retry_requeues_then_completes(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    baseline_csv = tmp_path / "baseline.csv"
    baseline_csv.write_text("config_id,status\n0,ok\n1,ok\n", encoding="utf-8")
    config_path = tmp_path / "agent_config.json"
    _write_json(config_path, {"baseline_csv": str(baseline_csv), "scoring": {"score_column": "core_topN_sharpe"}})

    _bootstrap_resume_run(
        runs_root=runs_root,
        tree_run_id="retry_case",
        config_path=config_path,
        baseline_csv=baseline_csv,
        ideas_per_node=1,
        beam_width=1,
        max_depth=1,
        max_parallel_evals=1,
        eval_retries=1,
        sweep_config_limit=2,
    )

    original_worker = _stub_completed_worker
    seen_attempts: dict[str, int] = {}

    def flaky_worker(**kwargs: Any) -> dict[str, Any]:
        eval_id = str(kwargs.get("eval_id") or "")
        seen_attempts[eval_id] = int(seen_attempts.get(eval_id, 0)) + 1
        if seen_attempts[eval_id] == 1:
            return {
                "status": "failed",
                "error": "forced_failure_once",
                "timed_out": False,
                "updates": {},
                "cleanup_requests": [],
            }
        return original_worker(**kwargs)

    monkeypatch.setattr(tree_runner, "_execute_eval_task_worker", flaky_worker)
    manifest = _run_tree_resume_dry(runs_root=runs_root, tree_run_id="retry_case", config_path=config_path)

    eval_rec = (manifest.get("evaluations") or {}).get("0001") or {}
    assert eval_rec.get("status") == "completed"
    assert int(eval_rec.get("attempt") or 0) == 2

    events = manifest.get("events") or []
    requeue_events = [
        ev
        for ev in events
        if isinstance(ev, dict)
        and str(ev.get("type")) == "scheduler_requeued"
        and str((ev.get("details") or {}).get("eval_id")) == "0001"
        and str((ev.get("details") or {}).get("reason")) == "retry"
    ]
    assert len(requeue_events) >= 1


def test_resume_running_eval_is_reused_without_duplication(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tree_runner, "_execute_eval_task_worker", _stub_completed_worker)

    runs_root = tmp_path / "runs"
    baseline_csv = tmp_path / "baseline.csv"
    baseline_csv.write_text("config_id,status\n0,ok\n1,ok\n", encoding="utf-8")
    config_path = tmp_path / "agent_config.json"
    _write_json(config_path, {"baseline_csv": str(baseline_csv), "scoring": {"score_column": "core_topN_sharpe"}})

    idea_001_path = runs_root / "resume_recover_case" / "node_ideas" / "0000" / "idea_001.md"
    preexisting = {
        "0001": {
            "eval_id": "0001",
            "parent_node_id": "0000",
            "depth": 0,
            "idea_path": str(idea_001_path),
            "eval_output_root": "",
            "experiment_dir": "",
            "candidate_ref_name": "tree/test/e0001",
            "status": "running",
            "task_state": "running",
            "attempt": 1,
            "error": None,
            "decision": {"gate_basis": "parent_relative", "rank_basis": "root_relative"},
        }
    }
    _bootstrap_resume_run(
        runs_root=runs_root,
        tree_run_id="resume_recover_case",
        config_path=config_path,
        baseline_csv=baseline_csv,
        ideas_per_node=1,
        beam_width=1,
        max_depth=1,
        max_parallel_evals=1,
        eval_retries=0,
        sweep_config_limit=2,
        preexisting_evaluations=preexisting,
        task_plan_depth0=[
            {
                "eval_id": "0001",
                "parent_node_id": "0000",
                "idea_path": str(idea_001_path),
                "depth": 0,
                "task_state": "running",
            }
        ],
    )
    manifest = _run_tree_resume_dry(runs_root=runs_root, tree_run_id="resume_recover_case", config_path=config_path)

    evals = manifest.get("evaluations") or {}
    assert set(evals.keys()) == {"0001"}
    eval_rec = (manifest.get("evaluations") or {}).get("0001") or {}
    assert eval_rec.get("status") == "completed"
    assert eval_rec.get("task_state") == "completed"
    assert int(eval_rec.get("attempt") or 0) >= 2
