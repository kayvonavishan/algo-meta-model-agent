from __future__ import annotations

import argparse
import concurrent.futures as _futures
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tree / beam-search runner (phase 1: manifest + resume + lock).",
    )
    parser.add_argument(
        "--tree-run-id",
        default=None,
        help="Explicit tree run id (default: current timestamp).",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Root directory for tree runs (default: agentic_experimentation/worktrees/tree_runs).",
    )
    parser.add_argument(
        "--ideas-per-node",
        type=int,
        default=7,
        help="Number of new ideas to generate+evaluate per node expansion.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum search depth (number of expansion rounds).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Beam width (max number of nodes to continue per depth).",
    )
    parser.add_argument(
        "--max-parallel-evals",
        type=int,
        default=1,
        help="Global cap for concurrently running eval tasks at a depth.",
    )
    parser.add_argument(
        "--max-parallel-per-node",
        type=int,
        default=None,
        help="Optional fairness cap for concurrent eval tasks per parent node.",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["threadpool"],
        default="threadpool",
        help="Parallel execution backend for eval tasks.",
    )
    parser.add_argument(
        "--eval-retries",
        type=int,
        default=0,
        help="Number of retries for failed eval tasks at the same depth.",
    )
    parser.add_argument(
        "--eval-timeout-seconds",
        type=int,
        default=None,
        help="Optional timeout per eval task (seconds).",
    )
    parser.add_argument(
        "--strict-fail-depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, stop scheduling remaining tasks at this depth after a non-retriable failure.",
    )
    parser.add_argument(
        "--sweep-config-limit",
        type=int,
        default=None,
        help="If set, only evaluate config_id < N for each sweep (deterministic subset).",
    )
    parser.add_argument(
        "--max-total-idea-evals",
        type=int,
        default=None,
        help="Hard cap on total idea evaluations across the entire run.",
    )
    parser.add_argument(
        "--stop-on-empty-frontier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop early if no candidates pass the promotion gate.",
    )
    parser.add_argument(
        "--keep-rejected-worktrees",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep worktrees for rejected candidates (debugging).",
    )
    parser.add_argument(
        "--keep-failed-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep artifacts for failed evaluations (debugging).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing manifest if found.",
    )
    parser.add_argument(
        "--rerun-evals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow re-running completed evaluations (not used in phase 1).",
    )
    parser.add_argument(
        "--force-regenerate-ideas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate node ideas even if ideas already exist on disk.",
    )
    parser.add_argument(
        "--lock-stale-seconds",
        type=int,
        default=600,
        help="Treat an existing run.lock.json as stale after this many seconds without a heartbeat.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore an existing non-stale lock and proceed (records a lock takeover event).",
    )
    parser.add_argument(
        "--config",
        default="agentic_experimentation/agent_config.json",
        help="Path to agent config JSON for multi_agent_runner execution.",
    )
    parser.add_argument(
        "--idea-conversation-mode",
        choices=["off", "auto", "native", "replay"],
        default="auto",
        help="Idea-generation conversation continuation mode.",
    )
    parser.add_argument(
        "--idea-history-window-turns",
        type=int,
        default=12,
        help="Replay: max recent turns to include in idea-generation conversation memory.",
    )
    parser.add_argument(
        "--idea-history-max-chars",
        type=int,
        default=20000,
        help="Replay memory char budget (-1 = unbounded, 0 = disable replay memory).",
    )
    parser.add_argument(
        "--dedupe-scope",
        choices=["node_plus_ancestors", "global", "none"],
        default="node_plus_ancestors",
        help="Idea repeat-avoidance scope when selecting ideas to evaluate.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Simulate evals/promotions deterministically without calling LLMs or running sweeps.",
    )
    parser.add_argument(
        "--report-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate TREE_SUMMARY.md from an existing manifest and exit.",
    )
    parser.add_argument(
        "--validate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Validate manifest+artifacts for an existing run and exit.",
    )
    return parser.parse_args(argv)


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8", errors="replace")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        # Windows PowerShell `Set-Content -Encoding UTF8` can emit a UTF-8 BOM.
        return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def _resolve_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "META_MODEL_GUIDE.md").exists() and (p / "adaptive_vol_momentum.py").exists():
            return p
    return start.resolve()


def _run_git(repo_root: Path, args: list[str]) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {cp.stderr.strip() or cp.stdout.strip()}")
    return (cp.stdout or "").strip()


def _delete_branch(*, repo_root: Path, ref_name: str) -> None:
    # Best-effort; treat "branch not found" as a no-op.
    cp = subprocess.run(
        ["git", "branch", "-D", str(ref_name)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode == 0:
        return
    msg = (cp.stderr or cp.stdout or "").strip().lower()
    if "not found" in msg:
        return
    raise RuntimeError(f"git branch -D {ref_name} failed: {cp.stderr.strip() or cp.stdout.strip()}")


def _assert_clean_working_tree(repo_root: Path) -> None:
    status = _run_git(repo_root, ["status", "--porcelain"])
    if status.strip():
        raise RuntimeError(
            "Root working tree must be clean to start a tree run. "
            "Commit or stash your changes first."
        )


def _ensure_standard_run_dirs(run_root: Path) -> dict[str, Path]:
    # Keep paths short on Windows (avoid long folder names).
    wt_root = run_root / "wt"
    cand_root = run_root / "cand"
    node_ideas_root = run_root / "node_ideas"
    artifacts_root = run_root / "artifacts"
    eval_root = run_root / "eval"
    conversations_root = run_root / "conversations"
    for p in (wt_root, cand_root, node_ideas_root, artifacts_root, eval_root, conversations_root):
        _safe_mkdir(p)
    return {
        "wt_root": wt_root,
        "cand_root": cand_root,
        "node_ideas_root": node_ideas_root,
        "artifacts_root": artifacts_root,
        "eval_root": eval_root,
        "conversations_root": conversations_root,
    }


def _node_ref_name(tree_run_id: str, node_id: str) -> str:
    # Must be globally unique per worktree.
    return f"tree/{tree_run_id}/n{node_id}"


def _eval_ref_name(tree_run_id: str, eval_id: str) -> str:
    # Must be globally unique per worktree.
    return f"tree/{tree_run_id}/e{eval_id}"


def _format_node_id(n: int) -> str:
    return f"{n:04d}"


def _format_eval_id(n: int) -> str:
    return f"{n:04d}"


def _create_branch_at_commit(repo_root: Path, ref_name: str, commit: str) -> None:
    _run_git(repo_root, ["branch", "-f", ref_name, commit])


def _create_worktree_at_ref(*, repo_root: Path, worktree_path: Path, ref_name: str) -> None:
    if worktree_path.exists():
        raise RuntimeError(f"Worktree path already exists: {worktree_path}")
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _run_git(repo_root, ["worktree", "add", str(worktree_path), ref_name])


def create_node_worktree(*, repo_root: Path, run_root: Path, tree_run_id: str, node_id: str, commit: str) -> dict[str, Any]:
    paths = _ensure_standard_run_dirs(run_root)
    ref_name = _node_ref_name(tree_run_id, node_id)
    worktree_path = paths["wt_root"] / node_id
    _create_branch_at_commit(repo_root, ref_name, commit)
    _create_worktree_at_ref(repo_root=repo_root, worktree_path=worktree_path, ref_name=ref_name)
    return {"node_id": node_id, "commit": commit, "ref_name": ref_name, "worktree_path": str(worktree_path)}


def create_candidate_worktree(*, repo_root: Path, run_root: Path, tree_run_id: str, eval_id: str, parent_commit: str) -> dict[str, Any]:
    paths = _ensure_standard_run_dirs(run_root)
    ref_name = _eval_ref_name(tree_run_id, eval_id)
    worktree_path = paths["cand_root"] / eval_id
    _create_branch_at_commit(repo_root, ref_name, parent_commit)
    _create_worktree_at_ref(repo_root=repo_root, worktree_path=worktree_path, ref_name=ref_name)
    return {"eval_id": eval_id, "commit": parent_commit, "ref_name": ref_name, "worktree_path": str(worktree_path)}


def cleanup_worktree(*, repo_root: Path, worktree_path: Path, retries: int = 6) -> None:
    # Windows-friendly: attempt retries/backoff to handle transient file locks.
    delay = 0.25
    last_err: Exception | None = None
    for _ in range(max(1, int(retries))):
        try:
            _run_git(repo_root, ["worktree", "remove", "--force", str(worktree_path)])
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(delay)
            delay = min(delay * 2.0, 5.0)
    raise RuntimeError(f"Failed to remove worktree after retries: {worktree_path}") from last_err


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _git_status_porcelain(worktree_path: Path) -> str:
    cp = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(worktree_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"git status failed: {cp.stderr.strip() or cp.stdout.strip()}")
    return (cp.stdout or "").rstrip("\n")


def _git_commit_all(*, worktree_path: Path, message: str) -> str:
    # Stage everything and commit, using local config so it works in fresh environments.
    subprocess.run(["git", "add", "-A"], cwd=str(worktree_path), check=True)
    cp = subprocess.run(
        [
            "git",
            "-c",
            "user.name=tree_runner",
            "-c",
            "user.email=tree_runner@local",
            "commit",
            "-m",
            message,
        ],
        cwd=str(worktree_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        out = (cp.stderr or cp.stdout or "").strip()
        # If there is nothing to commit, treat as error for Phase 3 (candidate commit required for identity).
        raise RuntimeError(f"git commit failed: {out}")
    return _run_git(Path(worktree_path), ["rev-parse", "HEAD"])


def _run_python(*, repo_root: Path, args: list[str], env: dict[str, str]) -> int:
    cp = subprocess.run([sys.executable, *args], cwd=str(repo_root), env=env, check=False)
    return int(cp.returncode)


def _append_tree_log(run_root: Path, message: str) -> None:
    """
    Best-effort append-only log for `tree_runner.py` itself.

    This is intentionally simple (no global logging configuration) so tests/imports
    have no side-effects.
    """
    try:
        ts = _utc_now_iso()
        p = run_root / "tree_runner.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8", errors="replace") as f:
            f.write(f"[{ts}] {message.rstrip()}\n")
    except Exception:
        return


def _run_python_logged(
    *,
    repo_root: Path,
    args: list[str],
    env: dict[str, str],
    log_path: Path,
    echo: bool = True,
    timeout_seconds: Optional[int] = None,
) -> int:
    """
    Run a Python subprocess and tee stdout/stderr into a log file.

    `tree_runner.py` orchestrates other scripts (idea generation + multi-agent loop). Those
    scripts print useful diagnostics to stdout/stderr; without teeing, those logs are lost
    unless the user manually redirects output.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, *args]
    started = time.time()
    write_lock = threading.Lock()

    def _write(line: str) -> None:
        with write_lock:
            with log_path.open("a", encoding="utf-8", errors="replace") as f:
                f.write(line)

    _write(
        "\n".join(
            [
                "==============================",
                f"ts: {_utc_now_iso()}",
                f"cwd: {str(repo_root)}",
                f"cmd: {' '.join(str(c) for c in cmd)}",
                "==============================",
                "",
            ]
        )
        + "\n"
    )

    cp = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    done_q: queue.Queue[tuple[str, Optional[BaseException]]] = queue.Queue()

    def _pump(stream, label: str, console_stream) -> None:  # noqa: ANN001
        exc = None
        try:
            for line in iter(stream.readline, ""):
                if line == "":
                    break
                _write(f"[{label}] {line}")
                if echo:
                    try:
                        console_stream.write(line)
                        console_stream.flush()
                    except Exception:
                        pass
        except BaseException as e:  # noqa: BLE001
            exc = e
        finally:
            try:
                stream.close()
            except Exception:
                pass
            done_q.put((label, exc))

    threads = []
    if cp.stdout is not None:
        threads.append(threading.Thread(target=_pump, args=(cp.stdout, "stdout", sys.stdout), daemon=True))
    if cp.stderr is not None:
        threads.append(threading.Thread(target=_pump, args=(cp.stderr, "stderr", sys.stderr), daemon=True))
    for t in threads:
        t.start()

    timed_out = False
    try:
        if timeout_seconds is not None and int(timeout_seconds) > 0:
            rc = int(cp.wait(timeout=float(timeout_seconds)))
        else:
            rc = int(cp.wait())
    except subprocess.TimeoutExpired:
        timed_out = True
        with contextlib.suppress(Exception):
            cp.kill()
        with contextlib.suppress(Exception):
            cp.wait(timeout=5)
        rc = 124

    # Wait for pump threads to finish (short timeout avoids rare hangs).
    for _ in threads:
        try:
            done_q.get(timeout=5)
        except Exception:
            continue
    for t in threads:
        try:
            t.join(timeout=0.1)
        except Exception:
            pass

    dur = time.time() - started
    _write(f"\n[exit] code={rc} duration_s={dur:.3f} timed_out={str(timed_out).lower()}\n")
    return rc


def _dry_run_write_idea_file(*, node_ideas_dir: Path, node_id: str, idx: int) -> Path:
    node_ideas_dir.mkdir(parents=True, exist_ok=True)
    p = node_ideas_dir / f"idea_{idx:03d}.md"
    if p.exists():
        return p
    # Keep deterministic, short content for hashing.
    text = f"""IDEA: dry_run idea {idx} for node {node_id}
RATIONALE: simulate tree mechanics
REQUIRED_CHANGES: none (dry-run)
"""
    p.write_text(text.strip() + "\n", encoding="utf-8")
    return p


def _dry_run_ensure_ideas(*, node_ideas_dir: Path, node_id: str, desired_k: int) -> list[Path]:
    ideas = _list_markdown_files(node_ideas_dir)
    if len(ideas) >= desired_k:
        return ideas[:desired_k]
    for i in range(1, desired_k + 1):
        _dry_run_write_idea_file(node_ideas_dir=node_ideas_dir, node_id=node_id, idx=i)
    return _list_markdown_files(node_ideas_dir)[:desired_k]


def _dry_run_scenario(*, eval_id: str) -> dict[str, Any]:
    """
    Deterministic mix of outcomes:
    - some fail completeness
    - some regress primary
    - some are mixed-but-non-regressing
    - some have missing score to exercise fallback ranking
    """
    try:
        n = int(eval_id)
    except Exception:
        n = sum(ord(c) for c in str(eval_id))

    mod = n % 7
    if mod == 0:
        return {"should_explore": True, "grade": "good", "primary_delta": 0.02, "rank_score": 1.0, "strict_complete": True, "rows_used": None}
    if mod == 1:
        return {"should_explore": False, "grade": "mixed", "primary_delta": 0.0, "rank_score": 0.8, "strict_complete": True, "rows_used": None}
    if mod == 2:
        return {"should_explore": False, "grade": "weak", "primary_delta": 0.01, "rank_score": 0.4, "strict_complete": True, "rows_used": None}
    if mod == 3:
        return {"should_explore": True, "grade": "good", "primary_delta": -0.0001, "rank_score": 0.9, "strict_complete": True, "rows_used": None}
    if mod == 4:
        return {"should_explore": True, "grade": "good", "primary_delta": 0.01, "rank_score": None, "strict_complete": True, "rows_used": None}
    if mod == 5:
        return {"should_explore": True, "grade": "good", "primary_delta": 0.01, "rank_score": 0.7, "strict_complete": False, "rows_used": None}
    return {"should_explore": True, "grade": "good", "primary_delta": 0.005, "rank_score": 0.6, "strict_complete": True, "rows_used": None}


def _dry_run_write_candidate_csv(*, path: Path, config_id_limit: Optional[int], ok: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(config_id_limit) if config_id_limit is not None else 1
    rows = ["config_id,status,mean_topN_avg_return_per_trade_pct_oos"]
    for i in range(n):
        status = "ok" if ok else ("error" if i == 0 else "ok")
        rows.append(f"{i},{status},0.0")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _execute_eval_task_worker(
    *,
    repo_root: Path,
    run_root: Path,
    tree_run_id: str,
    eval_id: str,
    node_id: str,
    idea_path: Path,
    node_commit: str,
    node_baseline_csv_path: Path,
    eval_output_root: Path,
    experiment_dir: Path,
    sweep_output_dir: Path,
    sweep_results_csv: Path,
    config_path: Path,
    sweep_config_limit: Optional[int],
    eval_timeout_seconds: Optional[int],
    score_column: Optional[str],
    root_baseline_csv: Path,
    dry_run: bool,
    keep_rejected_worktrees: bool,
    keep_failed_artifacts: bool,
    compute_score_fn: Any,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    cleanup_requests: list[dict[str, str]] = []
    status = "failed"
    error: Optional[str] = None
    timed_out = False

    if dry_run:
        try:
            artifacts_root = _ensure_standard_run_dirs(run_root)["artifacts_root"]
            parent_baseline_prov = _ensure_artifact_copy(
                run_root=run_root,
                source_path=node_baseline_csv_path,
                dest_name=f"baseline_node_{node_id}.csv",
            )
            updates["parent_baseline_provenance"] = parent_baseline_prov

            scenario = _dry_run_scenario(eval_id=eval_id)
            candidate_art_path = artifacts_root / f"candidate_eval_{eval_id}.csv"
            _dry_run_write_candidate_csv(
                path=candidate_art_path,
                config_id_limit=sweep_config_limit,
                ok=bool(scenario.get("strict_complete", True)),
            )
            cand_prov = _copy_with_provenance(source_path=candidate_art_path, dest_path=candidate_art_path)
            updates["candidate_results_provenance"] = cand_prov
            updates["candidate_results_csv_path"] = str(candidate_art_path)
            updates["candidate_commit"] = node_commit

            if sweep_config_limit is not None:
                strict = _strict_completeness_counts(csv_path=candidate_art_path, config_id_limit=int(sweep_config_limit))
                updates["strict_completeness"] = strict

            should_explore = bool(scenario.get("should_explore"))
            grade = scenario.get("grade")
            primary_delta = float(scenario.get("primary_delta") or 0.0)
            rank_score = scenario.get("rank_score")

            updates["parent_relative"] = {
                "recommendation_summary": {
                    "should_explore": should_explore,
                    "grade": grade,
                    "score": None,
                    "reasons": ["dry_run"],
                },
                "primary_delta": primary_delta,
                "baseline_rows_used": (int(sweep_config_limit) if sweep_config_limit is not None else 1),
                "candidate_rows_used": (int(sweep_config_limit) if sweep_config_limit is not None else 1),
                "score": None,
                "summary_json_path": "",
            }
            updates["root_relative"] = {
                "recommendation_summary": {
                    "should_explore": should_explore,
                    "grade": grade,
                    "score": rank_score,
                    "reasons": ["dry_run"],
                },
                "primary_delta": primary_delta,
                "baseline_rows_used": (int(sweep_config_limit) if sweep_config_limit is not None else 1),
                "candidate_rows_used": (int(sweep_config_limit) if sweep_config_limit is not None else 1),
                "score": None,
            }
            status = "completed"
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        return {
            "status": status,
            "error": error,
            "timed_out": False,
            "updates": updates,
            "cleanup_requests": cleanup_requests,
        }

    cand_dir = _ensure_standard_run_dirs(run_root)["cand_root"] / eval_id
    try:
        if cand_dir.exists():
            try:
                cleanup_worktree(repo_root=repo_root, worktree_path=cand_dir)
            except Exception as exc:  # noqa: BLE001
                cleanup_requests.append({"path": str(cand_dir), "kind": "worktree", "reason": str(exc)})
                raise

        cand_info = create_candidate_worktree(
            repo_root=repo_root,
            run_root=run_root,
            tree_run_id=tree_run_id,
            eval_id=eval_id,
            parent_commit=node_commit,
        )
        updates["candidate_worktree_path"] = cand_info["worktree_path"]

        env = dict(os.environ)
        env["AGENTIC_OUTPUT_DIR"] = str(sweep_output_dir)
        env["AGENTIC_RESULTS_CSV"] = str(sweep_results_csv)
        worker_tmp_dir = Path(eval_output_root) / "tmp"
        worker_tmp_dir.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(worker_tmp_dir)
        env["TMP"] = str(worker_tmp_dir)
        env["TEMP"] = str(worker_tmp_dir)
        mar_args = [
            "agentic_experimentation/multi_agent_runner.py",
            "--in-place",
            "--worktree-path",
            str(cand_info["worktree_path"]),
            "--idea-path",
            str(idea_path),
            "--config",
            str(config_path),
            "--experiments-root",
            str(experiment_dir),
            "--run-id",
            eval_id,
            "--baseline-csv",
            str(node_baseline_csv_path),
            "--no-archive-ideas",
        ]
        if sweep_config_limit is not None:
            mar_args.extend(["--sweep-config-limit", str(int(sweep_config_limit))])

        mar_log_path = eval_output_root / "multi_agent_runner.subprocess.log"
        _append_tree_log(run_root, f"multi_agent_runner eval_id={eval_id} log={mar_log_path}")
        rc = _run_python_logged(
            repo_root=repo_root,
            args=mar_args,
            env=env,
            log_path=mar_log_path,
            echo=True,
            timeout_seconds=eval_timeout_seconds,
        )
        if rc == 124 and eval_timeout_seconds is not None:
            timed_out = True
            raise TimeoutError(f"multi_agent_runner timed out after {int(eval_timeout_seconds)} seconds")
        if rc != 0:
            raise RuntimeError(f"multi_agent_runner failed (exit {rc})")

        exp_dir = Path(experiment_dir) / eval_id
        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Missing summary.json at {summary_path}")
        summary = _read_json(summary_path)
        if not isinstance(summary, dict):
            raise RuntimeError(f"Invalid summary.json at {summary_path}")
        approved = bool(summary.get("approved"))
        sweep_exit = summary.get("sweep_exit_code")
        candidate_csv = exp_dir / "meta_config_sweep_results.csv"
        if not approved or sweep_exit != 0 or not candidate_csv.exists():
            raise RuntimeError("Idea did not reach approved+sweep-success state; no candidate results.")

        cand_wt_path = Path(str(cand_info["worktree_path"]))
        msg = f"tree_run {tree_run_id} eval {eval_id} idea {idea_path.name}"
        candidate_commit = _git_commit_all(worktree_path=cand_wt_path, message=msg)
        updates["candidate_commit"] = candidate_commit

        parent_baseline_prov = _ensure_artifact_copy(
            run_root=run_root,
            source_path=node_baseline_csv_path,
            dest_name=f"baseline_node_{node_id}.csv",
        )
        cand_prov = _copy_with_provenance(
            source_path=candidate_csv,
            dest_path=_ensure_standard_run_dirs(run_root)["artifacts_root"] / f"candidate_eval_{eval_id}.csv",
        )
        updates["parent_baseline_provenance"] = parent_baseline_prov
        updates["candidate_results_provenance"] = cand_prov
        copied_candidate_csv = Path(str(cand_prov["copied_to_path"]))
        updates["candidate_results_csv_path"] = str(copied_candidate_csv)

        parent_score = compute_score_fn(
            Path(str(parent_baseline_prov["copied_to_path"])),
            copied_candidate_csv,
            score_column,
            config_id_limit=sweep_config_limit,
        )
        root_score = compute_score_fn(
            root_baseline_csv,
            copied_candidate_csv,
            score_column,
            config_id_limit=sweep_config_limit,
        )
        updates["parent_relative"] = {
            "recommendation_summary": _recommendation_summary(parent_score),
            "primary_delta": _primary_delta(parent_score),
            "baseline_rows_used": parent_score.get("baseline_rows_used"),
            "candidate_rows_used": parent_score.get("candidate_rows_used"),
            "score": parent_score.get("score"),
            "summary_json_path": str(summary_path),
        }
        updates["root_relative"] = {
            "recommendation_summary": _recommendation_summary(root_score),
            "primary_delta": _primary_delta(root_score),
            "baseline_rows_used": root_score.get("baseline_rows_used"),
            "candidate_rows_used": root_score.get("candidate_rows_used"),
            "score": root_score.get("score"),
        }

        if sweep_config_limit is not None:
            strict = _strict_completeness_counts(csv_path=copied_candidate_csv, config_id_limit=int(sweep_config_limit))
            updates["strict_completeness"] = strict
            if not strict.get("is_complete"):
                updates["decision_updates"] = {"passed_gate": False, "promotion_reason": "incomplete_or_failed_rows"}
            else:
                updates["decision_updates"] = {"passed_gate": None}

        status = "completed"
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        _ = keep_rejected_worktrees
        _ = keep_failed_artifacts

    return {
        "status": status,
        "error": error,
        "timed_out": bool(timed_out),
        "updates": updates,
        "cleanup_requests": cleanup_requests,
    }

def _list_markdown_files(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob("*.md") if p.is_file()], key=lambda p: p.name.lower())


def _normalize_idea_text(text: str) -> str:
    # Deterministic normalization for dedupe hashing.
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    # Drop empty lines and normalize common bullet prefixes.
    normalized = []
    for ln in lines:
        if not ln:
            continue
        if ln.startswith("* "):
            ln = "- " + ln[2:]
        normalized.append(ln)
    return "\n".join(normalized).strip().lower()


def _idea_text_hash(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    norm = _normalize_idea_text(raw)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _collect_dedupe_hashes(
    manifest: dict[str, Any],
    *,
    node_id: str,
    scope: str,
) -> set[str]:
    nodes = manifest.get("nodes") or {}
    if scope == "none":
        return set()

    exclude_dir = None
    node_rec = (nodes or {}).get(node_id) if isinstance(nodes, dict) else None
    if isinstance(node_rec, dict) and node_rec.get("node_ideas_dir"):
        exclude_dir = str(node_rec.get("node_ideas_dir"))

    idea_dirs: list[str] = []
    if scope == "node_plus_ancestors":
        idea_dirs = _collect_context_idea_dirs(manifest, node_id)
    elif scope == "global":
        for rec in (nodes or {}).values():
            if not isinstance(rec, dict):
                continue
            d = rec.get("node_ideas_dir")
            if d:
                idea_dirs.append(str(d))
        # Stable order to keep deterministic behavior.
        idea_dirs = sorted(list(dict.fromkeys(idea_dirs)))
    else:
        idea_dirs = _collect_context_idea_dirs(manifest, node_id)

    if exclude_dir:
        idea_dirs = [d for d in idea_dirs if str(d) != exclude_dir]

    hashes: set[str] = set()
    for d in idea_dirs:
        for p in _list_markdown_files(Path(d)):
            try:
                hashes.add(_idea_text_hash(p))
            except Exception:
                continue
    return hashes


def _collect_ancestor_node_ids(manifest: dict[str, Any], node_id: str) -> list[str]:
    nodes = manifest.get("nodes") or {}
    cur = node_id
    chain = []
    while cur:
        chain.append(cur)
        rec = nodes.get(cur) or {}
        cur = rec.get("parent_node_id")
    return list(reversed(chain))


def _collect_context_idea_dirs(manifest: dict[str, Any], node_id: str) -> list[str]:
    nodes = manifest.get("nodes") or {}
    chain = _collect_ancestor_node_ids(manifest, node_id)
    dirs: list[str] = []
    for nid in chain:
        rec = nodes.get(nid) or {}
        d = rec.get("node_ideas_dir")
        if d and str(d) not in dirs:
            dirs.append(str(d))
    return dirs


def _strict_completeness_counts(*, csv_path: Path, config_id_limit: int) -> dict[str, Any]:
    import csv  # local import to keep module import time low

    expected = int(config_id_limit)
    ok_count = 0
    error_count = 0
    seen: set[int] = set()
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Empty CSV or missing header: {csv_path}")
        if "config_id" not in reader.fieldnames or "status" not in reader.fieldnames:
            raise RuntimeError(f"CSV missing required columns config_id/status: {csv_path}")
        for row in reader:
            try:
                cid = int(str(row.get("config_id", "")).strip())
            except Exception:
                continue
            if cid < expected:
                seen.add(cid)
                status = str(row.get("status", "")).strip().lower()
                if status == "ok":
                    ok_count += 1
                else:
                    error_count += 1
    return {
        "expected_count": expected,
        "observed_unique_config_ids": len(seen),
        "ok_count": ok_count,
        "error_count": error_count,
        "is_complete": (len(seen) == expected and error_count == 0 and ok_count == expected),
    }


def _recommendation_summary(score_obj: dict[str, Any]) -> dict[str, Any]:
    rec = (score_obj or {}).get("recommendation") or {}
    return {
        "should_explore": bool(rec.get("should_explore")),
        "grade": rec.get("grade"),
        "score": rec.get("score"),
        "reasons": rec.get("reasons"),
    }


def _primary_delta(score_obj: dict[str, Any], *, primary_col: str = "core_topN_sharpe") -> Optional[float]:
    deltas = (score_obj or {}).get("column_deltas") or {}
    rec = deltas.get(primary_col)
    if not isinstance(rec, dict):
        return None
    try:
        v = float(rec.get("delta"))
    except Exception:
        return None
    return v


_IDEA_GEN_BASELINE_METRICS: tuple[str, ...] = (
    "core_topN_sharpe",
    "mean_topN_avg_return_per_trade_pct_oos",
    "mean_topN_avg_return_per_trade_pct",
    "core_topN_sortino",
    "core_topN_calmar",
    "core_topN_max_drawdown",
)


def _extract_baseline_metrics_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    score = summary.get("score") or {}
    if not isinstance(score, dict):
        return {}
    col_deltas = score.get("column_deltas") or {}
    if not isinstance(col_deltas, dict):
        return {}

    out: dict[str, float] = {}
    for name in _IDEA_GEN_BASELINE_METRICS:
        rec = col_deltas.get(name)
        if not isinstance(rec, dict):
            continue
        v = _coerce_float(rec.get("candidate_mean"))
        if v is None:
            continue
        out[name] = v
    return out


def _baseline_metrics_from_csv_self(
    *,
    compute_score: Any,
    csv_path: Path,
    sweep_config_limit: Optional[int],
    score_column: str = "core_topN_sharpe",
) -> dict[str, float]:
    score_obj = compute_score(csv_path, csv_path, score_column, config_id_limit=sweep_config_limit)
    if not isinstance(score_obj, dict):
        return {}
    col_deltas = score_obj.get("column_deltas") or {}
    if not isinstance(col_deltas, dict):
        return {}
    out: dict[str, float] = {}
    for name in _IDEA_GEN_BASELINE_METRICS:
        rec = col_deltas.get(name)
        if not isinstance(rec, dict):
            continue
        v = _coerce_float(rec.get("candidate_mean"))
        if v is None:
            continue
        out[name] = v
    return out


def _find_node_promotion_summary_path(*, run_root: Path, manifest: dict[str, Any], node_rec: dict[str, Any]) -> Optional[Path]:
    direct = node_rec.get("baseline_summary_json_path")
    if direct:
        p = Path(str(direct)).expanduser()
        return p if p.exists() else p

    art = node_rec.get("artifacts") or {}
    if not isinstance(art, dict):
        art = {}
    ev_id = art.get("promoted_from_eval_id")
    if not ev_id:
        return None

    evals = manifest.get("evaluations") or {}
    ev = evals.get(str(ev_id)) if isinstance(evals, dict) else None
    if isinstance(ev, dict):
        parent_rel = ev.get("parent_relative") or {}
        if isinstance(parent_rel, dict) and parent_rel.get("summary_json_path"):
            return Path(str(parent_rel["summary_json_path"])).expanduser()
        if ev.get("eval_output_root"):
            # Default layout: <eval_output_root>/experiment/<eval_id>/summary.json
            base = Path(str(ev.get("eval_output_root"))).expanduser()
            return base / "experiment" / str(ev_id) / "summary.json"

    # Fallback to standard location.
    return run_root / "eval" / str(ev_id) / "experiment" / str(ev_id) / "summary.json"


def _build_baseline_context_for_node(
    *,
    run_root: Path,
    repo_root: Path,
    manifest: dict[str, Any],
    node_id: str,
    node_rec: dict[str, Any],
    compute_score: Any,
    sweep_config_limit: Optional[int],
) -> dict[str, Any]:
    baseline_csv = str(node_rec.get("baseline_results_csv_path") or "").strip()
    baseline_csv_path = Path(baseline_csv).expanduser() if baseline_csv else None

    summary_path = _find_node_promotion_summary_path(run_root=run_root, manifest=manifest, node_rec=node_rec)
    summary = None
    if summary_path is not None and summary_path.exists():
        try:
            raw = _read_json(summary_path)
            if isinstance(raw, dict):
                summary = raw
        except Exception:
            summary = None

    def _summary_to_artifact_paths(summary_obj: dict[str, Any]) -> tuple[str, str, str]:
        sweep_csv = str(summary_obj.get("agentic_run_results_csv") or summary_obj.get("results_csv") or "").strip()
        agentic_root = str(summary_obj.get("agentic_output_root") or "").strip()
        plots_dir = ""
        if sweep_csv:
            try:
                run_dir = Path(sweep_csv).expanduser().resolve().parent
                plots_dir = str(run_dir / "avg_trade_return_plots")
            except Exception:
                plots_dir = ""
        elif agentic_root:
            plots_dir = str(Path(agentic_root).expanduser() / "run_0" / "avg_trade_return_plots")
        return sweep_csv, agentic_root, plots_dir

    baseline_metrics: dict[str, float] = {}
    sweep_results_csv = ""
    agentic_output_root = ""
    avg_trade_return_plots_dir = ""
    source = "unknown"

    if summary is not None:
        baseline_metrics = _extract_baseline_metrics_from_summary(summary)
        sweep_results_csv, agentic_output_root, avg_trade_return_plots_dir = _summary_to_artifact_paths(summary)
        source = "promoted_eval_summary_json"

    if not baseline_metrics and baseline_csv_path is not None and baseline_csv_path.exists():
        baseline_metrics = _baseline_metrics_from_csv_self(
            compute_score=compute_score,
            csv_path=baseline_csv_path,
            sweep_config_limit=sweep_config_limit,
        )
        if not sweep_results_csv and baseline_csv_path is not None:
            sweep_results_csv = str(baseline_csv_path)
        source = "csv_self_score"

    # Prefer original configured root baseline source path for depth=0 prompts,
    # so users see the canonical baseline location rather than internal copies.
    baseline_source_csv = ""
    try:
        if int(node_rec.get("depth") or 0) == 0:
            root_obj = manifest.get("root") or {}
            if isinstance(root_obj, dict):
                prov = root_obj.get("root_baseline_provenance") or {}
                if isinstance(prov, dict):
                    baseline_source_csv = str(prov.get("source_path") or "").strip()
    except Exception:
        baseline_source_csv = ""

    nodes = manifest.get("nodes") or {}
    chain = _collect_ancestor_node_ids(manifest, str(node_id))

    # Build a compact, chronological branch timeline with major metrics and per-step deltas.
    timeline: list[dict[str, Any]] = []
    prev_metrics: Optional[dict[str, float]] = None
    for nid in chain:
        nrec = (nodes.get(nid) or {}) if isinstance(nodes, dict) else {}
        if not isinstance(nrec, dict):
            continue

        n_depth = nrec.get("depth")
        n_parent = nrec.get("parent_node_id")
        n_baseline_csv = str(nrec.get("baseline_results_csv_path") or "").strip()
        try:
            if int(n_depth or 0) == 0:
                root_obj = manifest.get("root") or {}
                if isinstance(root_obj, dict):
                    prov = root_obj.get("root_baseline_provenance") or {}
                    if isinstance(prov, dict):
                        src = str(prov.get("source_path") or "").strip()
                        if src:
                            n_baseline_csv = src
        except Exception:
            pass
        n_baseline_csv_path = Path(n_baseline_csv).expanduser() if n_baseline_csv else None
        n_idea_chain = list(nrec.get("idea_chain") or []) if isinstance(nrec.get("idea_chain"), list) else []
        applied_idea_path = n_idea_chain[-1] if n_idea_chain else None

        n_summary_path = _find_node_promotion_summary_path(run_root=run_root, manifest=manifest, node_rec=nrec)
        n_summary = None
        if n_summary_path is not None and n_summary_path.exists():
            try:
                raw = _read_json(n_summary_path)
                if isinstance(raw, dict):
                    n_summary = raw
            except Exception:
                n_summary = None

        n_metrics: dict[str, float] = {}
        n_deltas: dict[str, float] = {}
        n_sweep_csv = ""
        n_agentic_root = ""
        n_plots_dir = ""
        n_source = "unknown"

        if n_summary is not None:
            n_metrics = _extract_baseline_metrics_from_summary(n_summary)
            n_sweep_csv, n_agentic_root, n_plots_dir = _summary_to_artifact_paths(n_summary)
            # Prefer deltas from the promotion summary (parent -> this node).
            col_deltas = ((n_summary.get("score") or {}) if isinstance(n_summary.get("score"), dict) else {}).get("column_deltas") or {}
            if isinstance(col_deltas, dict):
                for k in _IDEA_GEN_BASELINE_METRICS:
                    rec = col_deltas.get(k)
                    if isinstance(rec, dict):
                        dv = _coerce_float(rec.get("delta"))
                        if dv is not None:
                            n_deltas[k] = dv
            n_source = "promoted_eval_summary_json"

        if not n_metrics and n_baseline_csv_path is not None and n_baseline_csv_path.exists():
            n_metrics = _baseline_metrics_from_csv_self(
                compute_score=compute_score,
                csv_path=n_baseline_csv_path,
                sweep_config_limit=sweep_config_limit,
            )
            n_source = "csv_self_score"

        # Fallback delta: current - previous (if summary delta missing).
        if prev_metrics is not None:
            for k in _IDEA_GEN_BASELINE_METRICS:
                if k in n_deltas:
                    continue
                if k in n_metrics and k in prev_metrics:
                    try:
                        n_deltas[k] = float(n_metrics[k]) - float(prev_metrics[k])
                    except Exception:
                        continue

        timeline.append(
            {
                "node_id": nid,
                "depth": n_depth,
                "parent_node_id": n_parent,
                "applied_idea_path": applied_idea_path,
                "idea_chain_len": len(n_idea_chain),
                "baseline_results_csv_path": n_baseline_csv,
                "promotion_summary_json_path": (str(n_summary_path) if n_summary_path is not None else ""),
                "metrics": n_metrics,
                "metric_deltas_vs_parent": n_deltas,
                "agentic_output_root": n_agentic_root,
                "sweep_results_csv": n_sweep_csv,
                "avg_trade_return_plots_dir": n_plots_dir,
                "source": n_source,
            }
        )
        prev_metrics = n_metrics if n_metrics else prev_metrics

    # Collect rejected (non-selected) evals along this branch.
    rejected: list[dict[str, Any]] = []
    evals = manifest.get("evaluations") or {}
    if isinstance(nodes, dict) and isinstance(evals, dict) and len(chain) >= 2:
        chain_set = set(chain)
        # For each parent node in the chain, exclude the one eval that produced the next node.
        selected_eval_ids: set[str] = set()
        for idx in range(len(chain) - 1):
            child_id = chain[idx + 1]
            child_rec = nodes.get(child_id) or {}
            if not isinstance(child_rec, dict):
                continue
            art = child_rec.get("artifacts") or {}
            if isinstance(art, dict) and art.get("promoted_from_eval_id"):
                selected_eval_ids.add(str(art.get("promoted_from_eval_id")))

        for ev in evals.values():
            if not isinstance(ev, dict):
                continue
            parent_id = str(ev.get("parent_node_id") or "")
            if parent_id not in chain_set:
                continue
            ev_id = str(ev.get("eval_id") or "")
            if not ev_id or ev_id in selected_eval_ids:
                continue
            if str(ev.get("status") or "").lower() != "completed":
                continue

            summary_path = ""
            pr = ev.get("parent_relative") or {}
            if isinstance(pr, dict) and pr.get("summary_json_path"):
                summary_path = str(pr.get("summary_json_path") or "")

            summary = None
            if summary_path:
                sp = Path(summary_path).expanduser()
                if sp.exists():
                    try:
                        raw = _read_json(sp)
                        if isinstance(raw, dict):
                            summary = raw
                    except Exception:
                        summary = None

            ev_metrics: dict[str, float] = {}
            ev_deltas: dict[str, float] = {}
            ev_sweep_csv = ""
            ev_agentic_root = ""
            ev_plots_dir = ""
            ev_source = "unknown"

            if summary is not None:
                ev_metrics = _extract_baseline_metrics_from_summary(summary)
                ev_sweep_csv, ev_agentic_root, ev_plots_dir = _summary_to_artifact_paths(summary)
                col_deltas = ((summary.get("score") or {}) if isinstance(summary.get("score"), dict) else {}).get("column_deltas") or {}
                if isinstance(col_deltas, dict):
                    for k in _IDEA_GEN_BASELINE_METRICS:
                        rec = col_deltas.get(k)
                        if isinstance(rec, dict):
                            dv = _coerce_float(rec.get("delta"))
                            if dv is not None:
                                ev_deltas[k] = dv
                ev_source = "summary_json"

            # Candidate results CSV copy (kept under run_root/artifacts) is the canonical local reference.
            candidate_csv = str(ev.get("candidate_results_csv_path") or "").strip()

            rejected.append(
                {
                    "eval_id": ev_id,
                    "parent_node_id": parent_id,
                    "depth": ev.get("depth"),
                    "idea_path": str(ev.get("idea_path") or "").strip(),
                    "promotion_summary_json_path": summary_path,
                    "baseline_results_csv_path": candidate_csv,
                    "metrics": ev_metrics,
                    "metric_deltas_vs_parent": ev_deltas,
                    "agentic_output_root": ev_agentic_root,
                    "sweep_results_csv": ev_sweep_csv,
                    "avg_trade_return_plots_dir": ev_plots_dir,
                    "source": ev_source,
                }
            )

    return {
        "tree_run_id": str(manifest.get("tree_run_id") or ""),
        "node_id": str(node_id),
        "parent_node_id": node_rec.get("parent_node_id"),
        "depth": node_rec.get("depth"),
        "node_worktree_path": str(node_rec.get("worktree_path") or ""),
        "parent_node_worktree_path": (
            str(((nodes.get(str(node_rec.get("parent_node_id") or "")) or {}).get("worktree_path") or ""))
            if isinstance(nodes, dict)
            else ""
        ),
        "idea_chain": list(node_rec.get("idea_chain") or []) if isinstance(node_rec.get("idea_chain"), list) else [],
        "changes_applied_count": len(list(node_rec.get("idea_chain") or [])) if isinstance(node_rec.get("idea_chain"), list) else 0,
        "sweep_config_limit": sweep_config_limit,
        "baseline_source_csv": baseline_source_csv,
        "baseline_context_source": source,
        "baseline_summary_json_path": (str(summary_path) if summary_path is not None else ""),
        "baseline_metrics": baseline_metrics,
        "agentic_output_root": agentic_output_root,
        "sweep_results_csv": sweep_results_csv,
        "avg_trade_return_plots_dir": avg_trade_return_plots_dir,
        "branch_timeline": timeline,
        "rejected_branch_evals": rejected,
        "artifact_docs_root": str((repo_root / "agentic_experimentation" / "artifact_docs").resolve()),
    }


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _is_under_dir(path: Path, parent: Path) -> bool:
    try:
        path = path.resolve()
        parent = parent.resolve()
    except Exception:
        return False
    try:
        path.relative_to(parent)
        return True
    except Exception:
        return False


def _compute_gate_and_reason(
    eval_rec: dict[str, Any],
    *,
    artifacts_root: Path,
    sweep_config_limit: Optional[int],
) -> tuple[bool, Optional[str], Optional[bool]]:
    """
    Promotion gate (parent-relative):
    - must be completed with candidate commit + candidate CSV artifact
    - primary delta must not be negative (strict)
    - should_explore must be true OR grade == "mixed" (with non-regressing primary)
    - if sweep_config_limit is set, strict completeness must be satisfied
    """
    if (eval_rec.get("status") or "").lower() != "completed":
        return False, "eval_not_completed", None

    cand_commit = str(eval_rec.get("candidate_commit") or "").strip()
    if not cand_commit:
        return False, "missing_candidate_commit", None

    cand_csv = str(eval_rec.get("candidate_results_csv_path") or "").strip()
    if not cand_csv:
        return False, "missing_candidate_results_csv", None
    cand_csv_path = Path(cand_csv).expanduser()
    if not cand_csv_path.exists():
        return False, "candidate_results_csv_missing_on_disk", None
    if not _is_under_dir(cand_csv_path, artifacts_root):
        return False, "candidate_results_csv_not_in_artifacts", None

    baseline_prov = eval_rec.get("parent_baseline_provenance") or {}
    if not isinstance(baseline_prov, dict) or not baseline_prov.get("copied_to_path"):
        return False, "missing_parent_baseline_artifact", None
    parent_baseline_path = Path(str(baseline_prov.get("copied_to_path"))).expanduser()
    if not parent_baseline_path.exists():
        return False, "parent_baseline_artifact_missing_on_disk", None
    if not _is_under_dir(parent_baseline_path, artifacts_root):
        return False, "parent_baseline_not_in_artifacts", None

    if sweep_config_limit is not None:
        strict = eval_rec.get("strict_completeness") or {}
        if not isinstance(strict, dict) or not strict.get("is_complete"):
            return False, "incomplete_or_failed_rows", None

        parent_rel = eval_rec.get("parent_relative") or {}
        if not isinstance(parent_rel, dict):
            parent_rel = {}
        used = parent_rel.get("candidate_rows_used")
        try:
            if int(used) != int(sweep_config_limit):
                return False, "candidate_rows_used_mismatch", None
        except Exception:
            return False, "missing_candidate_rows_used", None

    parent_rel = eval_rec.get("parent_relative") or {}
    if not isinstance(parent_rel, dict):
        parent_rel = {}
    primary_delta = _coerce_float(parent_rel.get("primary_delta"))
    if primary_delta is None:
        return False, "missing_primary_delta", None
    primary_regressed = bool(primary_delta < 0.0)
    if primary_regressed:
        return False, "primary_regressed", True

    rec = parent_rel.get("recommendation_summary") or {}
    if not isinstance(rec, dict):
        rec = {}
    should_explore = bool(rec.get("should_explore"))
    grade = str(rec.get("grade") or "").strip().lower() if rec.get("grade") is not None else ""

    if should_explore:
        return True, None, False
    if grade == "mixed":
        return True, None, False
    return False, "should_explore_false", False


def _rank_score(eval_rec: dict[str, Any]) -> float:
    root_rel = eval_rec.get("root_relative") or {}
    if not isinstance(root_rel, dict):
        root_rel = {}

    rec = root_rel.get("recommendation_summary") or {}
    if not isinstance(rec, dict):
        rec = {}

    score = _coerce_float(rec.get("score"))
    if score is not None:
        return float(score)

    # Fallback: root-relative primary delta (still root-relative ranking).
    pd = _coerce_float(root_rel.get("primary_delta"))
    if pd is not None:
        return float(pd)

    return float("-inf")


def _root_primary_delta(eval_rec: dict[str, Any]) -> float:
    root_rel = eval_rec.get("root_relative") or {}
    if not isinstance(root_rel, dict):
        return float("-inf")
    pd = _coerce_float(root_rel.get("primary_delta"))
    return float(pd) if pd is not None else float("-inf")


def _select_global_beam(eval_recs: list[dict[str, Any]], *, beam_width: int) -> list[dict[str, Any]]:
    width = max(0, int(beam_width))
    if width == 0:
        return []

    # Deterministic: highest scores first; tie-break eval_id ascending.
    def _key(r: dict[str, Any]) -> tuple[float, float, str]:
        return (-_rank_score(r), -_root_primary_delta(r), str(r.get("eval_id") or ""))

    ranked = sorted(eval_recs, key=_key)
    return ranked[:width]


def _conversation_id_for_node(node_id: str) -> str:
    return f"node_{str(node_id)}"


def _conversation_paths(*, run_root: Path, conversation_id: str) -> dict[str, Path]:
    conv_root = _ensure_standard_run_dirs(run_root)["conversations_root"]
    cid = str(conversation_id)
    return {
        "state_json_path": conv_root / f"{cid}.json",
        "turn_log_jsonl_path": conv_root / f"{cid}.jsonl",
        "summary_md_path": conv_root / f"{cid}_summary.md",
    }


def _conversation_debug_log_path(*, manifest: dict[str, Any], run_root: Path) -> Optional[Path]:
    cfg = manifest.get("conversation_config") or {}
    if not isinstance(cfg, dict):
        return None
    raw = str(cfg.get("debug_log_jsonl_path") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = run_root / p
    return p


def _append_conversation_debug_log(*, manifest: dict[str, Any], run_root: Path, event: dict[str, Any]) -> None:
    path = _conversation_debug_log_path(manifest=manifest, run_root=run_root)
    if path is None:
        return
    rec = {"timestamp": _utc_now_iso(), **(event or {})}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return


def _read_json_if_exists(
    path: Path,
    *,
    run_root: Optional[Path] = None,
    manifest: Optional[dict[str, Any]] = None,
    context: Optional[str] = None,
) -> Any:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception as exc:  # noqa: BLE001
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        corrupt_path = path.with_suffix(path.suffix + f".corrupt_{ts}")
        moved = False
        try:
            os.replace(path, corrupt_path)
            moved = True
        except Exception:
            moved = False

        context_s = str(context or path)
        if moved:
            warn = f"[conversation] warning: invalid JSON recovered for {context_s}: {path} -> {corrupt_path} ({type(exc).__name__})"
        else:
            warn = f"[conversation] warning: invalid JSON for {context_s}: {path} ({type(exc).__name__})"
        print(warn, file=sys.stderr)
        if run_root is not None:
            _append_tree_log(
                run_root,
                (
                    f"conversation_json_recovery context={context_s}"
                    + f" path={path}"
                    + (f" moved_to={corrupt_path}" if moved else "")
                    + f" error={type(exc).__name__}:{exc}"
                ),
            )
        if isinstance(manifest, dict):
            events = manifest.setdefault("events", [])
            if isinstance(events, list):
                events.append(
                    {
                        "timestamp": _utc_now_iso(),
                        "type": "conversation_json_recovery",
                        "context": context_s,
                        "path": str(path),
                        "recovered_to": (str(corrupt_path) if moved else None),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if run_root is not None:
                _append_conversation_debug_log(
                    manifest=manifest,
                    run_root=run_root,
                    event={
                        "type": "conversation_json_recovery",
                        "context": context_s,
                        "path": str(path),
                        "recovered_to": (str(corrupt_path) if moved else None),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
        return None


def _conversation_state_template(
    *,
    manifest: dict[str, Any],
    conversation_id: str,
    node_id: str,
    parent_conversation_id: Optional[str],
    fork_from_turn_id: Optional[str],
) -> dict[str, Any]:
    conv_cfg = manifest.get("conversation_config") or {}
    if not isinstance(conv_cfg, dict):
        conv_cfg = {}
    history_window_turns = conv_cfg.get("history_window_turns", 12)
    history_max_chars = conv_cfg.get("history_max_chars", 20000)
    mode = str(conv_cfg.get("mode") or "auto")
    return {
        "schema_version": 1,
        "conversation_id": str(conversation_id),
        "node_id": str(node_id),
        "parent_conversation_id": (str(parent_conversation_id) if parent_conversation_id else None),
        "fork_from_turn_id": (str(fork_from_turn_id) if fork_from_turn_id else None),
        "requested_mode": mode,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "latest_turn_id": None,
        "next_turn_index": 1,
        "history_window_turns": int(history_window_turns),
        "history_max_chars": int(history_max_chars),
        "provider": {
            "name": "claude_agent_sdk",
            "supports_native_continuation": False,
            "session_id": None,
            "last_response_id": None,
        },
        "turns": [],
        "idea_file_turn_map": {},
    }


def _conversation_turn_index(turn_id: Optional[str]) -> Optional[int]:
    if turn_id is None:
        return None
    s = str(turn_id).strip()
    m = re.match(r"^turn_(\d+)$", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _fork_state_from_parent(
    *,
    parent_state: dict[str, Any],
    child_state: dict[str, Any],
    fork_from_turn_id: Optional[str],
) -> dict[str, Any]:
    parent_turns = parent_state.get("turns") or []
    if not isinstance(parent_turns, list):
        parent_turns = []
    copied_turns = [dict(t) for t in parent_turns if isinstance(t, dict)]
    if fork_from_turn_id:
        keep_to = None
        for i, turn in enumerate(copied_turns):
            if str(turn.get("turn_id") or "") == str(fork_from_turn_id):
                keep_to = i
                break
        if keep_to is not None:
            copied_turns = copied_turns[: keep_to + 1]
    child_state["turns"] = copied_turns

    latest_turn_id = None
    next_turn_index = 1
    kept_turn_ids: set[str] = set()
    for turn in copied_turns:
        tid = str(turn.get("turn_id") or "")
        if not tid:
            continue
        latest_turn_id = tid
        kept_turn_ids.add(tid)
        idx = _conversation_turn_index(tid)
        if idx is not None and idx >= next_turn_index:
            next_turn_index = idx + 1
    child_state["latest_turn_id"] = latest_turn_id
    child_state["next_turn_index"] = next_turn_index

    parent_map = parent_state.get("idea_file_turn_map") or {}
    child_map: dict[str, Any] = {}
    if isinstance(parent_map, dict):
        for key, value in parent_map.items():
            if isinstance(value, str) and value in kept_turn_ids:
                child_map[str(key)] = value
    child_state["idea_file_turn_map"] = child_map

    parent_provider = parent_state.get("provider") or {}
    if isinstance(parent_provider, dict):
        provider = child_state.get("provider") or {}
        if not isinstance(provider, dict):
            provider = {}
        provider["session_id"] = parent_provider.get("session_id")
        provider["last_response_id"] = parent_provider.get("last_response_id")
        provider["supports_native_continuation"] = bool(parent_provider.get("supports_native_continuation", False))
        provider["name"] = str(parent_provider.get("name") or "claude_agent_sdk")
        child_state["provider"] = provider

    child_state["updated_at"] = _utc_now_iso()
    return child_state


def _ensure_node_conversation(
    *,
    manifest: dict[str, Any],
    run_root: Path,
    node_id: str,
    parent_node_id: Optional[str],
    fork_from_turn_id: Optional[str],
) -> str:
    nodes = manifest.setdefault("nodes", {})
    if not isinstance(nodes, dict):
        raise RuntimeError("Manifest nodes map is invalid.")
    node = nodes.get(str(node_id))
    if not isinstance(node, dict):
        raise RuntimeError(f"Missing node record for conversation attach: {node_id}")

    conversations = manifest.setdefault("conversations", {})
    if not isinstance(conversations, dict):
        conversations = {}
        manifest["conversations"] = conversations

    conversation_id = str(node.get("conversation_id") or _conversation_id_for_node(str(node_id)))
    parent_conversation_id = None
    if parent_node_id:
        parent_node = nodes.get(str(parent_node_id))
        if isinstance(parent_node, dict):
            parent_conversation_id = str(parent_node.get("conversation_id") or _conversation_id_for_node(str(parent_node_id)))

    paths = _conversation_paths(run_root=run_root, conversation_id=conversation_id)
    conv_rec = conversations.get(conversation_id)
    state_path = paths["state_json_path"]
    turn_log_path = paths["turn_log_jsonl_path"]
    summary_path = paths["summary_md_path"]

    created_now = False
    if not isinstance(conv_rec, dict):
        created_now = True
        conv_rec = {
            "conversation_id": conversation_id,
            "node_id": str(node_id),
            "parent_conversation_id": parent_conversation_id,
            "fork_from_turn_id": (str(fork_from_turn_id) if fork_from_turn_id else None),
            "state_json_path": str(state_path),
            "turn_log_jsonl_path": str(turn_log_path),
            "summary_md_path": str(summary_path),
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "latest_turn_id": None,
        }
        conversations[conversation_id] = conv_rec
    else:
        conv_rec.setdefault("conversation_id", conversation_id)
        conv_rec.setdefault("node_id", str(node_id))
        conv_rec.setdefault("parent_conversation_id", parent_conversation_id)
        conv_rec.setdefault("fork_from_turn_id", (str(fork_from_turn_id) if fork_from_turn_id else None))
        conv_rec["state_json_path"] = str(state_path)
        conv_rec["turn_log_jsonl_path"] = str(turn_log_path)
        conv_rec["summary_md_path"] = str(summary_path)
        conv_rec.setdefault("created_at", _utc_now_iso())
        conv_rec["updated_at"] = _utc_now_iso()

    state = _read_json_if_exists(
        state_path,
        run_root=run_root,
        manifest=manifest,
        context=f"conversation_state:{conversation_id}",
    )
    if not isinstance(state, dict):
        state = _conversation_state_template(
            manifest=manifest,
            conversation_id=conversation_id,
            node_id=str(node_id),
            parent_conversation_id=parent_conversation_id,
            fork_from_turn_id=fork_from_turn_id,
        )
        if parent_conversation_id:
            parent_conv = conversations.get(parent_conversation_id)
            if isinstance(parent_conv, dict):
                parent_state_path = Path(str(parent_conv.get("state_json_path") or "")).expanduser()
                parent_state = _read_json_if_exists(
                    parent_state_path,
                    run_root=run_root,
                    manifest=manifest,
                    context=f"parent_conversation_state:{parent_conversation_id}",
                )
                if isinstance(parent_state, dict):
                    state = _fork_state_from_parent(
                        parent_state=parent_state,
                        child_state=state,
                        fork_from_turn_id=fork_from_turn_id,
                    )
    else:
        # Ensure required fields on older state shapes.
        state.setdefault("schema_version", 1)
        state.setdefault("conversation_id", conversation_id)
        state.setdefault("node_id", str(node_id))
        state.setdefault("parent_conversation_id", parent_conversation_id)
        state.setdefault("fork_from_turn_id", (str(fork_from_turn_id) if fork_from_turn_id else None))
        state.setdefault("requested_mode", str((manifest.get("conversation_config") or {}).get("mode") or "auto"))
        state.setdefault("created_at", _utc_now_iso())
        state["updated_at"] = _utc_now_iso()
        state.setdefault("latest_turn_id", None)
        state.setdefault("next_turn_index", 1)
        state.setdefault("history_window_turns", int((manifest.get("conversation_config") or {}).get("history_window_turns") or 12))
        _history_max_chars_raw = (manifest.get("conversation_config") or {}).get("history_max_chars", 20000)
        state.setdefault("history_max_chars", int(20000 if _history_max_chars_raw is None else _history_max_chars_raw))
        state.setdefault("provider", {})
        if not isinstance(state.get("provider"), dict):
            state["provider"] = {}
        state["provider"].setdefault("name", "claude_agent_sdk")
        state["provider"].setdefault("supports_native_continuation", False)
        state["provider"].setdefault("session_id", None)
        state["provider"].setdefault("last_response_id", None)
        state.setdefault("turns", [])
        if not isinstance(state.get("turns"), list):
            state["turns"] = []
        state.setdefault("idea_file_turn_map", {})
        if not isinstance(state.get("idea_file_turn_map"), dict):
            state["idea_file_turn_map"] = {}

        # Recompute next_turn_index from observed turn ids to maintain monotonicity.
        next_idx = 1
        latest_turn_id = None
        for turn in state.get("turns") or []:
            if not isinstance(turn, dict):
                continue
            tid = str(turn.get("turn_id") or "")
            if not tid:
                continue
            latest_turn_id = tid
            idx = _conversation_turn_index(tid)
            if idx is not None and idx >= next_idx:
                next_idx = idx + 1
        state["latest_turn_id"] = latest_turn_id
        state["next_turn_index"] = int(max(next_idx, int(state.get("next_turn_index") or 1)))

    _atomic_write_json(state_path, state)
    if not turn_log_path.exists():
        turn_log_path.parent.mkdir(parents=True, exist_ok=True)
        turn_log_path.write_text("", encoding="utf-8")
    if not summary_path.exists():
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            "\n".join(
                [
                    f"# Conversation {conversation_id}",
                    "",
                    f"node_id: {node_id}",
                    f"parent_conversation_id: {parent_conversation_id or '(none)'}",
                    f"fork_from_turn_id: {fork_from_turn_id or '(none)'}",
                    "",
                    "Conversation summary placeholder.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    latest_turn = state.get("latest_turn_id")
    conv_rec["latest_turn_id"] = latest_turn
    conv_rec["updated_at"] = _utc_now_iso()
    node["conversation_id"] = conversation_id
    node["latest_conversation_turn_id"] = latest_turn
    if node.get("expansion_seed_turn_id") is None:
        node["expansion_seed_turn_id"] = (str(fork_from_turn_id) if fork_from_turn_id else latest_turn)
    if created_now and node.get("expansion_seed_turn_id") is None:
        node["expansion_seed_turn_id"] = latest_turn
    _append_conversation_debug_log(
        manifest=manifest,
        run_root=run_root,
        event={
            "type": "ensure_node_conversation",
            "node_id": str(node_id),
            "conversation_id": conversation_id,
            "parent_conversation_id": parent_conversation_id,
            "fork_from_turn_id": (str(fork_from_turn_id) if fork_from_turn_id else None),
            "latest_turn_id": latest_turn,
            "created": bool(created_now),
            "state_json_path": str(state_path),
            "turn_log_jsonl_path": str(turn_log_path),
        },
    )
    return conversation_id


def _sync_node_conversation_latest_turn(*, manifest: dict[str, Any], run_root: Path, node_id: str) -> Optional[str]:
    nodes = manifest.get("nodes") or {}
    if not isinstance(nodes, dict):
        return None
    node = nodes.get(str(node_id))
    if not isinstance(node, dict):
        return None
    conversation_id = str(node.get("conversation_id") or "")
    if not conversation_id:
        return None
    conversations = manifest.get("conversations") or {}
    if not isinstance(conversations, dict):
        return None
    conv_rec = conversations.get(conversation_id)
    if not isinstance(conv_rec, dict):
        return None
    state_path = Path(str(conv_rec.get("state_json_path") or "")).expanduser()
    state = _read_json_if_exists(
        state_path,
        run_root=run_root,
        manifest=manifest,
        context=f"sync_node_latest_turn:{conversation_id}",
    )
    if not isinstance(state, dict):
        return None
    latest_turn = state.get("latest_turn_id")
    conv_rec["latest_turn_id"] = latest_turn
    conv_rec["updated_at"] = _utc_now_iso()
    node["latest_conversation_turn_id"] = latest_turn
    return str(latest_turn) if latest_turn is not None else None


def _lookup_conversation_turn_for_idea(
    *,
    manifest: dict[str, Any],
    run_root: Path,
    conversation_id: Optional[str],
    idea_path: Path,
) -> Optional[str]:
    if not conversation_id:
        return None
    conversations = manifest.get("conversations") or {}
    if not isinstance(conversations, dict):
        return None
    conv_rec = conversations.get(str(conversation_id))
    if not isinstance(conv_rec, dict):
        return None
    state_path = Path(str(conv_rec.get("state_json_path") or "")).expanduser()
    state = _read_json_if_exists(
        state_path,
        run_root=run_root,
        manifest=manifest,
        context=f"lookup_turn_for_idea:{conversation_id}",
    )
    if not isinstance(state, dict):
        return None

    idea_map = state.get("idea_file_turn_map") or {}
    if not isinstance(idea_map, dict):
        idea_map = {}

    candidates: set[str] = set()
    candidates.add(str(idea_path))
    try:
        candidates.add(str(idea_path.resolve()))
    except Exception:
        pass
    try:
        candidates.add(str(idea_path.resolve().relative_to(Path.cwd().resolve())))
    except Exception:
        pass

    def _norm(s: str) -> str:
        return str(s).replace("\\", "/").strip().lower()

    norm_candidates = {_norm(c) for c in candidates if c}
    for key, value in idea_map.items():
        if not isinstance(value, str):
            continue
        if _norm(str(key)) in norm_candidates:
            return value

    turns = state.get("turns") or []
    if isinstance(turns, list):
        for turn in reversed(turns):
            if not isinstance(turn, dict):
                continue
            turn_id = turn.get("turn_id")
            if not isinstance(turn_id, str):
                continue
            for k in ("output_idea_path", "output_idea_path_resolved", "output_idea_path_relative_to_repo"):
                v = turn.get(k)
                if not isinstance(v, str):
                    continue
                if _norm(v) in norm_candidates:
                    return turn_id
    return None


def _ensure_manifest_conversation_schema(
    *,
    manifest: dict[str, Any],
    run_root: Path,
    args: argparse.Namespace,
) -> None:
    current_version = int(manifest.get("manifest_version") or 0)
    if current_version < 3:
        manifest["manifest_version"] = 3

    run_config = manifest.setdefault("run_config", {})
    if not isinstance(run_config, dict):
        run_config = {}
        manifest["run_config"] = run_config
    run_config.setdefault("idea_conversation_mode", str(getattr(args, "idea_conversation_mode", "auto")))
    run_config.setdefault("idea_history_window_turns", int(getattr(args, "idea_history_window_turns", 12)))
    run_config.setdefault("idea_history_max_chars", int(getattr(args, "idea_history_max_chars", 20000)))
    run_config.setdefault("max_parallel_evals", int(getattr(args, "max_parallel_evals", 1)))
    _max_parallel_per_node = getattr(args, "max_parallel_per_node", None)
    run_config.setdefault("max_parallel_per_node", (int(_max_parallel_per_node) if _max_parallel_per_node is not None else None))
    run_config.setdefault("parallel_backend", str(getattr(args, "parallel_backend", "threadpool")))
    run_config.setdefault("eval_retries", int(getattr(args, "eval_retries", 0)))
    _eval_timeout = getattr(args, "eval_timeout_seconds", None)
    run_config.setdefault("eval_timeout_seconds", (int(_eval_timeout) if _eval_timeout is not None else None))
    run_config.setdefault("strict_fail_depth", bool(getattr(args, "strict_fail_depth", False)))

    conversation_config = manifest.setdefault("conversation_config", {})
    if not isinstance(conversation_config, dict):
        conversation_config = {}
        manifest["conversation_config"] = conversation_config
    conversation_config.setdefault("mode", str(run_config.get("idea_conversation_mode") or "auto"))
    conversation_config.setdefault("history_window_turns", int(run_config.get("idea_history_window_turns") or 12))
    _idea_history_max_chars_raw = run_config.get("idea_history_max_chars", 20000)
    conversation_config.setdefault("history_max_chars", int(20000 if _idea_history_max_chars_raw is None else _idea_history_max_chars_raw))
    conversation_config.setdefault(
        "debug_log_jsonl_path",
        str(_ensure_standard_run_dirs(run_root)["conversations_root"] / "conversation_debug.jsonl"),
    )

    conversations = manifest.setdefault("conversations", {})
    if not isinstance(conversations, dict):
        conversations = {}
        manifest["conversations"] = conversations

    nodes = manifest.setdefault("nodes", {})
    if not isinstance(nodes, dict):
        nodes = {}
        manifest["nodes"] = nodes
    for node_id, node in list(nodes.items()):
        if not isinstance(node, dict):
            continue
        node.setdefault("conversation_id", None)
        node.setdefault("expansion_seed_turn_id", None)
        node.setdefault("latest_conversation_turn_id", None)

    evals = manifest.setdefault("evaluations", {})
    if not isinstance(evals, dict):
        evals = {}
        manifest["evaluations"] = evals
    for _, rec in list(evals.items()):
        if not isinstance(rec, dict):
            continue
        rec.setdefault("idea_generation_conversation_id", None)
        rec.setdefault("idea_generation_turn_id", None)

    # Ensure one conversation per existing node so resume can continue cleanly.
    ordered_nodes: list[tuple[int, str, dict[str, Any]]] = []
    for node_id, node in list(nodes.items()):
        if not isinstance(node, dict):
            continue
        try:
            depth_i = int(node.get("depth") or 0)
        except Exception:
            depth_i = 0
        ordered_nodes.append((depth_i, str(node_id), node))
    ordered_nodes.sort(key=lambda item: (item[0], item[1]))

    for _, node_id, node in ordered_nodes:
        parent_node_id = node.get("parent_node_id")
        parent_node_id_s = str(parent_node_id) if parent_node_id is not None else None
        fork_turn_id = node.get("expansion_seed_turn_id")
        fork_turn_id_s = str(fork_turn_id) if fork_turn_id is not None else None
        _ensure_node_conversation(
            manifest=manifest,
            run_root=run_root,
            node_id=str(node_id),
            parent_node_id=parent_node_id_s,
            fork_from_turn_id=fork_turn_id_s,
        )

def _manifest_write(run_root: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now_iso()
    _atomic_write_json(run_root / "manifest.json", manifest)


def _record_scheduler_event(
    *,
    manifest: dict[str, Any],
    run_root: Path,
    event_type: str,
    depth: int,
    eval_id: str,
    parent_node_id: Optional[str] = None,
    attempt: Optional[int] = None,
    reason: Optional[str] = None,
) -> None:
    events = manifest.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        manifest["events"] = events
    details: dict[str, Any] = {
        "depth": int(depth),
        "eval_id": str(eval_id),
    }
    if parent_node_id is not None:
        details["parent_node_id"] = str(parent_node_id)
    if attempt is not None:
        details["attempt"] = int(attempt)
    if reason:
        details["reason"] = str(reason)
    events.append({"ts": _utc_now_iso(), "type": f"scheduler_{event_type}", "details": details})
    _append_tree_log(
        run_root,
        "scheduler_event"
        + f" type={event_type}"
        + f" depth={depth}"
        + f" eval_id={eval_id}"
        + (f" parent_node_id={parent_node_id}" if parent_node_id else "")
        + (f" attempt={attempt}" if attempt is not None else "")
        + (f" reason={reason}" if reason else ""),
    )


def _compute_depth_progress(*, manifest: dict[str, Any], depth: int) -> dict[str, Any]:
    depth_key = str(int(depth))
    state = manifest.get("state") or {}
    if not isinstance(state, dict):
        state = {}
    task_plan = state.get("task_plan_by_depth") or {}
    if not isinstance(task_plan, dict):
        task_plan = {}
    task_records = task_plan.get(depth_key) or []
    total_tasks = len(task_records) if isinstance(task_records, list) else 0

    queued = 0
    running = 0
    completed = 0
    failed = 0
    attempts_total = 0

    evals = manifest.get("evaluations") or {}
    if isinstance(evals, dict):
        for rec in evals.values():
            if not isinstance(rec, dict):
                continue
            try:
                depth_val = rec.get("depth")
                rec_depth = int(-1 if depth_val is None else depth_val)
            except Exception:
                continue
            if rec_depth != int(depth):
                continue
            status = str(rec.get("task_state") or rec.get("status") or "").lower().strip()
            if status == "queued":
                queued += 1
            elif status == "running":
                running += 1
            elif status == "completed":
                completed += 1
            elif status == "failed":
                failed += 1
            attempts_total += int(rec.get("attempt") or 0)

    requeued = 0
    events = manifest.get("events") or []
    if isinstance(events, list):
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if str(ev.get("type") or "") != "scheduler_requeued":
                continue
            details = ev.get("details") or {}
            if not isinstance(details, dict):
                continue
            try:
                ev_depth = int(details.get("depth"))
            except Exception:
                continue
            if ev_depth == int(depth):
                requeued += 1

    done = completed + failed
    pending = max(0, total_tasks - done)
    return {
        "depth": int(depth),
        "total_tasks": int(total_tasks),
        "queued": int(queued),
        "running": int(running),
        "completed": int(completed),
        "failed": int(failed),
        "done": int(done),
        "pending": int(pending),
        "requeued": int(requeued),
        "attempts_total": int(attempts_total),
        "updated_at": _utc_now_iso(),
    }


def _refresh_depth_progress(*, manifest: dict[str, Any], depth: int) -> None:
    state = manifest.setdefault("state", {})
    if not isinstance(state, dict):
        state = {}
        manifest["state"] = state
    by_depth = state.setdefault("depth_progress_by_depth", {})
    if not isinstance(by_depth, dict):
        by_depth = {}
        state["depth_progress_by_depth"] = by_depth
    by_depth[str(int(depth))] = _compute_depth_progress(manifest=manifest, depth=int(depth))


def _tree_summary_markdown(*, run_root: Path, manifest: dict[str, Any]) -> str:
    state = manifest.get("state") or {}
    run_config = manifest.get("run_config") or {}
    conversation_config = manifest.get("conversation_config") or {}
    if not isinstance(conversation_config, dict):
        conversation_config = {}
    conversations = manifest.get("conversations") or {}
    if not isinstance(conversations, dict):
        conversations = {}
    nodes = manifest.get("nodes") or {}
    evals = manifest.get("evaluations") or {}

    def _eval_for_node(n: dict[str, Any]) -> Optional[dict[str, Any]]:
        art = n.get("artifacts") or {}
        if not isinstance(art, dict):
            return None
        ev_id = art.get("promoted_from_eval_id")
        if not ev_id:
            return None
        ev = evals.get(str(ev_id))
        return ev if isinstance(ev, dict) else None

    # Best node selection (root-relative).
    best_node_id: Optional[str] = None
    best_key: tuple[float, float, str] | None = None
    for nid, nrec in (nodes or {}).items():
        if str(nid) == "0000":
            continue
        if not isinstance(nrec, dict):
            continue
        ev = _eval_for_node(nrec)
        if not ev:
            continue
        decision = ev.get("decision") or {}
        if not isinstance(decision, dict):
            decision = {}
        rank_score = _coerce_float(decision.get("rank_score"))
        if rank_score is None:
            rank_score = _coerce_float(_rank_score(ev))
        primary = _root_primary_delta(ev)
        key = (float(rank_score) if rank_score is not None else float("-inf"), float(primary), str(nid))
        if best_key is None or key > best_key:
            best_key = key
            best_node_id = str(nid)

    # Path to best.
    best_path: list[str] = []
    if best_node_id and isinstance(nodes.get(best_node_id), dict):
        cur = best_node_id
        while cur:
            best_path.append(cur)
            parent = (nodes.get(cur) or {}).get("parent_node_id")
            cur = str(parent) if parent else ""
        best_path = list(reversed(best_path))
    else:
        best_path = ["0000"] if "0000" in nodes else []

    lines: list[str] = []
    lines.append(f"# TREE_SUMMARY ({manifest.get('tree_run_id')})")
    lines.append("")
    lines.append("## Run Config")
    lines.append(f"- ideas_per_node: {run_config.get('ideas_per_node')}")
    lines.append(f"- max_depth: {run_config.get('max_depth')}")
    lines.append(f"- beam_width: {run_config.get('beam_width')}")
    lines.append(f"- max_parallel_evals: {run_config.get('max_parallel_evals')}")
    lines.append(f"- max_parallel_per_node: {run_config.get('max_parallel_per_node')}")
    lines.append(f"- parallel_backend: {run_config.get('parallel_backend')}")
    lines.append(f"- eval_retries: {run_config.get('eval_retries')}")
    lines.append(f"- eval_timeout_seconds: {run_config.get('eval_timeout_seconds')}")
    lines.append(f"- strict_fail_depth: {run_config.get('strict_fail_depth')}")
    lines.append(f"- sweep_config_limit: {run_config.get('sweep_config_limit')}")
    lines.append(f"- ideas_context_strategy: {run_config.get('ideas_context_strategy')}")
    lines.append(f"- idea_conversation_mode: {conversation_config.get('mode')}")
    lines.append(f"- idea_history_window_turns: {conversation_config.get('history_window_turns')}")
    lines.append(f"- idea_history_max_chars: {conversation_config.get('history_max_chars')}")
    lines.append(f"- conversation_debug_log_jsonl_path: {conversation_config.get('debug_log_jsonl_path')}")
    lines.append(f"- stop_reason: {state.get('stop_reason')}")
    lines.append("")

    depth_progress = state.get("depth_progress_by_depth") or {}
    if isinstance(depth_progress, dict) and depth_progress:
        lines.append("## Depth Progress")
        lines.append("")
        lines.append("| depth | total_tasks | queued | running | completed | failed | done | pending | requeued | attempts_total | updated_at |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for d in sorted(depth_progress.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            rec = depth_progress.get(d) or {}
            if not isinstance(rec, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rec.get("depth", d)),
                        str(rec.get("total_tasks", "")),
                        str(rec.get("queued", "")),
                        str(rec.get("running", "")),
                        str(rec.get("completed", "")),
                        str(rec.get("failed", "")),
                        str(rec.get("done", "")),
                        str(rec.get("pending", "")),
                        str(rec.get("requeued", "")),
                        str(rec.get("attempts_total", "")),
                        str(rec.get("updated_at", "")),
                    ]
                )
                + " |"
            )
        lines.append("")

    # Per-depth table.
    lines.append("## Nodes")
    lines.append("")
    lines.append("| depth | node_id | parent | conversation_id | parent_conversation_id | latest_turn_id | root_rank_score | root_grade | root_should_explore | parent_gate | ok/expected | candidate_rows_used | baseline_csv | candidate_csv | experiment_dir |")
    lines.append("|---:|---:|---:|---|---|---|---:|---|---|---|---|---:|---|---|---|")
    # Stable node order: by depth then node_id.
    def _node_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        nid, rec = item
        try:
            depth = int((rec or {}).get("depth") or 0)
        except Exception:
            depth = 0
        return (depth, str(nid))

    for nid, nrec in sorted([(str(k), v) for k, v in (nodes or {}).items() if isinstance(v, dict)], key=_node_sort_key):
        depth = int(nrec.get("depth") or 0)
        parent = nrec.get("parent_node_id") or ""
        baseline_csv = nrec.get("baseline_results_csv_path") or ""
        cand_csv = nrec.get("baseline_results_csv_path") or ""
        exp_dir = ""

        root_rank = ""
        root_grade = ""
        root_se = ""
        gate = ""
        ok_expected = ""
        cand_rows = ""
        conversation_id = str(nrec.get("conversation_id") or "")
        conversation_latest_turn = str(nrec.get("latest_conversation_turn_id") or "")
        parent_conversation_id = ""
        if conversation_id:
            conv_rec = conversations.get(conversation_id)
            if isinstance(conv_rec, dict):
                parent_conversation_id = str(conv_rec.get("parent_conversation_id") or "")
                if not conversation_latest_turn:
                    conversation_latest_turn = str(conv_rec.get("latest_turn_id") or "")

        if nid == "0000":
            gate = "ROOT"
        else:
            ev = _eval_for_node(nrec) or {}
            decision = ev.get("decision") or {}
            if not isinstance(decision, dict):
                decision = {}
            root_rel = ev.get("root_relative") or {}
            if not isinstance(root_rel, dict):
                root_rel = {}
            root_rec = root_rel.get("recommendation_summary") or {}
            if not isinstance(root_rec, dict):
                root_rec = {}
            root_rank = decision.get("rank_score")
            root_grade = root_rec.get("grade") or ""
            root_se = root_rec.get("should_explore")
            gate = "pass" if bool(decision.get("passed_gate")) else f"fail:{decision.get('promotion_reason') or ''}"
            strict = ev.get("strict_completeness") or {}
            if isinstance(strict, dict) and strict.get("expected_count") is not None:
                ok_expected = f"{strict.get('ok_count')}/{strict.get('expected_count')}"
            parent_rel = ev.get("parent_relative") or {}
            if isinstance(parent_rel, dict):
                cand_rows = parent_rel.get("candidate_rows_used") or ""
            exp_dir = ev.get("experiment_dir") or ""
            # For promoted nodes, candidate CSV is the node baseline itself.
            cand_csv = nrec.get("baseline_results_csv_path") or ""
            baseline_csv = str((ev.get("parent_baseline_provenance") or {}).get("copied_to_path") or "")

        lines.append(
            f"| {depth} | {nid} | {parent} | {conversation_id} | {parent_conversation_id} | {conversation_latest_turn} | {root_rank} | {root_grade} | {root_se} | {gate} | {ok_expected} | {cand_rows} | {baseline_csv} | {cand_csv} | {exp_dir} |"
        )

    lines.append("")
    lines.append("## Best Path")
    lines.append("")
    if best_node_id:
        lines.append(f"- best_node_id: `{best_node_id}`")
    else:
        lines.append("- best_node_id: (none promoted)")
    lines.append("")
    for nid in best_path:
        nrec = nodes.get(nid) or {}
        if not isinstance(nrec, dict):
            continue
        chain = nrec.get("idea_chain") or []
        lines.append(f"- `{nid}` depth={nrec.get('depth')} parent={nrec.get('parent_node_id') or ''} ideas={len(chain)}")
        for idea in chain[-3:] if isinstance(chain, list) else []:
            lines.append(f"  - {idea}")

    lines.append("")
    lines.append(f"_Generated at: {_utc_now_iso()}_")
    lines.append("")

    lines.append("## Evaluations")
    lines.append("")
    lines.append("| eval_id | depth | parent_node_id | status | task_state | attempt | started_at | finished_at | duration_s | error_type | error |")
    lines.append("|---:|---:|---:|---|---|---:|---|---|---:|---|---|")
    eval_items = [(str(k), v) for k, v in (evals or {}).items() if isinstance(v, dict)]
    eval_items.sort(key=lambda item: str(item[0]))
    for eval_id, ev in eval_items:
        depth = ev.get("depth")
        parent_id = ev.get("parent_node_id")
        status = ev.get("status")
        task_state = ev.get("task_state")
        attempt = ev.get("attempt")
        started_at = str(ev.get("started_at") or "")
        finished_at = str(ev.get("finished_at") or "")
        duration_s = ""
        if started_at and finished_at:
            try:
                start_dt = _dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                finish_dt = _dt.datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                duration_s = f"{max(0.0, (finish_dt - start_dt).total_seconds()):.3f}"
            except Exception:
                duration_s = ""
        error_payload = ev.get("error_payload") or {}
        error_type = ""
        if isinstance(error_payload, dict):
            error_type = str(error_payload.get("error_type") or "")
        error = str(ev.get("error") or "")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(eval_id),
                    str(depth),
                    str(parent_id),
                    str(status),
                    str(task_state),
                    str(attempt),
                    started_at,
                    finished_at,
                    duration_s,
                    error_type,
                    error.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _escape_mermaid_label(text: str) -> str:
    # Mermaid labels are quoted; keep it simple and predictable.
    s = str(text or "")
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    return s


def _fmt_float(v: Any, *, digits: int = 4) -> str:
    try:
        f = float(v)
    except Exception:
        return "na"
    if not math.isfinite(f):
        return "na"
    return f"{f:.{digits}g}"


def _tree_graph_markdown(*, run_root: Path, manifest: dict[str, Any]) -> str:
    """
    Mermaid graph of the full tree state.

    Useful for quickly visualizing nodes/edges and which nodes are currently in the frontier.
    """
    state = manifest.get("state") or {}
    nodes = manifest.get("nodes") or {}
    evals = manifest.get("evaluations") or {}
    conversations = manifest.get("conversations") or {}

    if not isinstance(nodes, dict):
        nodes = {}
    if not isinstance(evals, dict):
        evals = {}
    if not isinstance(conversations, dict):
        conversations = {}

    frontier = set(str(x) for x in (state.get("frontier_node_ids") or []))

    def _eval_for_node(n: dict[str, Any]) -> Optional[dict[str, Any]]:
        art = n.get("artifacts") or {}
        if not isinstance(art, dict):
            return None
        ev_id = art.get("promoted_from_eval_id")
        if not ev_id:
            return None
        ev = evals.get(str(ev_id))
        return ev if isinstance(ev, dict) else None

    # Best node selection (root-relative).
    best_node_id: Optional[str] = None
    best_key: tuple[float, float, str] | None = None
    for nid, nrec in nodes.items():
        if str(nid) == "0000":
            continue
        if not isinstance(nrec, dict):
            continue
        ev = _eval_for_node(nrec)
        if not ev:
            continue
        key = (_rank_score(ev), _root_primary_delta(ev), str(nid))
        if best_key is None or key > best_key:
            best_key = key
            best_node_id = str(nid)

    # Path to best.
    best_path: list[str] = []
    if best_node_id and isinstance(nodes.get(best_node_id), dict):
        cur = best_node_id
        while cur:
            best_path.append(cur)
            parent = (nodes.get(cur) or {}).get("parent_node_id")
            cur = str(parent) if parent else ""
        best_path = list(reversed(best_path))

    lines: list[str] = []
    lines.append(f"# TREE_GRAPH ({manifest.get('tree_run_id')})")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")

    # Nodes (stable order).
    for nid in sorted(nodes.keys(), key=lambda x: str(x)):
        nrec = nodes.get(nid)
        if not isinstance(nrec, dict):
            continue
        node_ident = f"N{str(nid)}"
        depth = nrec.get("depth")
        label_lines = [f"{nid} (d={depth})"]
        if str(nid) == "0000":
            label_lines.append("root")
        else:
            ev = _eval_for_node(nrec)
            if ev:
                rr = ev.get("root_relative") or {}
                if isinstance(rr, dict):
                    rrs = rr.get("recommendation_summary") or {}
                    if not isinstance(rrs, dict):
                        rrs = {}
                    label_lines.append(f"rank={_fmt_float(rrs.get('score'))} grade={rrs.get('grade')}")
                    label_lines.append(f"root_primary={_fmt_float(rr.get('primary_delta'))}")
        label = _escape_mermaid_label("\\n".join(label_lines))
        lines.append(f'  {node_ident}["{label}"]')

    # Edges.
    for nid in sorted(nodes.keys(), key=lambda x: str(x)):
        nrec = nodes.get(nid)
        if not isinstance(nrec, dict):
            continue
        parent = str(nrec.get("parent_node_id") or "")
        if not parent:
            continue
        ev = _eval_for_node(nrec)
        ev_id = str(ev.get("eval_id") or "") if ev else ""
        rank = _fmt_float(_rank_score(ev)) if ev else "na"
        edge_lines = [f"eval {ev_id}".strip(), f"rank {rank}".strip()]
        conv_id = str(nrec.get("conversation_id") or "")
        conv_rec = conversations.get(conv_id) if conv_id else None
        parent_conv_id = ""
        fork_turn_id = ""
        if isinstance(conv_rec, dict):
            parent_conv_id = str(conv_rec.get("parent_conversation_id") or "")
            fork_turn_id = str(conv_rec.get("fork_from_turn_id") or "")
        if conv_id:
            conv_line = f"conv {parent_conv_id or '?'}->{conv_id}"
            if fork_turn_id:
                conv_line += f" @{fork_turn_id}"
            edge_lines.append(conv_line)
        edge_label = _escape_mermaid_label("\\n".join([ln for ln in edge_lines if ln]))
        lines.append(f"  N{parent} -->|{edge_label}| N{nid}")

    # Styling.
    lines.append("")
    lines.append("classDef frontier fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px;")
    lines.append("classDef best fill:#e8f5e9,stroke:#43a047,stroke-width:2px;")
    if frontier:
        lines.append("class " + ",".join([f"N{nid}" for nid in sorted(frontier)]) + " frontier;")
    if best_path:
        lines.append("class " + ",".join([f"N{nid}" for nid in best_path]) + " best;")

    lines.append("```")
    lines.append("")
    lines.append("Legend:")
    lines.append("- Green: best path so far (by root-relative rank)")
    lines.append("- Blue: current frontier")
    lines.append("")
    return "\n".join(lines)


def _write_tree_graph(*, run_root: Path, manifest: dict[str, Any]) -> Path:
    md = _tree_graph_markdown(run_root=run_root, manifest=manifest)
    out_path = run_root / "TREE_GRAPH.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path


def _write_tree_summary(*, run_root: Path, manifest: dict[str, Any]) -> Path:
    md = _tree_summary_markdown(run_root=run_root, manifest=manifest)
    out_path = run_root / "TREE_SUMMARY.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path


def _validate_run(*, repo_root: Path, run_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    artifacts_root = _ensure_standard_run_dirs(run_root)["artifacts_root"]
    state = manifest.get("state") or {}
    if not isinstance(state, dict):
        state = {}

    def _issue(kind: str, msg: str, **details: Any) -> None:
        issues.append({"kind": kind, "message": msg, "details": details})

    root = manifest.get("root") or {}
    if isinstance(root, dict):
        root_csv = root.get("root_baseline_csv_path")
        if root_csv:
            p = Path(str(root_csv)).expanduser()
            if not p.exists():
                _issue("missing_file", "Root baseline CSV missing", path=str(p))
        prov = root.get("root_baseline_provenance")
        if isinstance(prov, dict) and prov.get("copied_to_path") and prov.get("sha256"):
            p = Path(str(prov["copied_to_path"])).expanduser()
            if not p.exists():
                _issue("missing_file", "Root baseline provenance target missing", path=str(p))
            else:
                if _sha256_file(p) != str(prov["sha256"]):
                    _issue("sha_mismatch", "Root baseline sha256 mismatch", path=str(p))

    nodes = manifest.get("nodes") or {}
    conversations = manifest.get("conversations") or {}
    if not isinstance(conversations, dict):
        conversations = {}
    if isinstance(nodes, dict):
        for nid, nrec in nodes.items():
            if not isinstance(nrec, dict):
                continue
            csv_path = nrec.get("baseline_results_csv_path")
            if csv_path:
                p = Path(str(csv_path)).expanduser()
                if not p.exists():
                    _issue("missing_file", "Node baseline CSV missing", node_id=str(nid), path=str(p))
                elif not _is_under_dir(p, artifacts_root):
                    _issue("path_scope", "Node baseline CSV not under artifacts", node_id=str(nid), path=str(p))

            ref_name = nrec.get("ref_name")
            commit = nrec.get("commit")
            if ref_name and commit:
                try:
                    ref_commit = _run_git(repo_root, ["rev-parse", str(ref_name)])
                    if str(ref_commit).strip() != str(commit).strip():
                        _issue("ref_mismatch", "Node ref does not point at commit", node_id=str(nid), ref_name=str(ref_name), commit=str(commit), ref_commit=str(ref_commit))
                except Exception as exc:  # noqa: BLE001
                    _issue("ref_error", "Failed to resolve node ref", node_id=str(nid), ref_name=str(ref_name), error=str(exc))

            conv_id = str(nrec.get("conversation_id") or "").strip()
            if conv_id:
                conv_rec = conversations.get(conv_id)
                if not isinstance(conv_rec, dict):
                    _issue("state_inconsistency", "node_conversation_missing_record", node_id=str(nid), conversation_id=conv_id)
                else:
                    sp = conv_rec.get("state_json_path")
                    if sp:
                        cp = Path(str(sp)).expanduser()
                        if not cp.exists():
                            _issue("missing_file", "Conversation state JSON missing", node_id=str(nid), conversation_id=conv_id, path=str(cp))

    evals = manifest.get("evaluations") or {}
    if isinstance(evals, dict):
        for eid, ev in evals.items():
            if not isinstance(ev, dict):
                continue

            exp_dir = ev.get("experiment_dir")
            if exp_dir:
                p = Path(str(exp_dir)).expanduser()
                if not p.exists():
                    _issue("missing_file", "Experiment directory missing", eval_id=str(eid), path=str(p))

            parent_rel = ev.get("parent_relative") or {}
            if isinstance(parent_rel, dict):
                sp = parent_rel.get("summary_json_path")
                if sp:
                    p = Path(str(sp)).expanduser()
                    if not p.exists():
                        _issue("missing_file", "summary_json_path missing", eval_id=str(eid), path=str(p))

            cand_csv = ev.get("candidate_results_csv_path")
            if cand_csv:
                p = Path(str(cand_csv)).expanduser()
                if not p.exists():
                    _issue("missing_file", "Candidate results CSV missing", eval_id=str(eid), path=str(p))
                elif not _is_under_dir(p, artifacts_root):
                    _issue("path_scope", "Candidate results CSV not under artifacts", eval_id=str(eid), path=str(p))
            prov = ev.get("candidate_results_provenance")
            if isinstance(prov, dict) and prov.get("copied_to_path") and prov.get("sha256"):
                p = Path(str(prov["copied_to_path"])).expanduser()
                if p.exists() and _sha256_file(p) != str(prov["sha256"]):
                    _issue("sha_mismatch", "Candidate results sha256 mismatch", eval_id=str(eid), path=str(p))
            base_prov = ev.get("parent_baseline_provenance")
            if isinstance(base_prov, dict) and base_prov.get("copied_to_path") and base_prov.get("sha256"):
                p = Path(str(base_prov["copied_to_path"])).expanduser()
                if p.exists() and _sha256_file(p) != str(base_prov["sha256"]):
                    _issue("sha_mismatch", "Parent baseline sha256 mismatch", eval_id=str(eid), path=str(p))

    # Frontier/expanded consistency checks.
    try:
        current_depth = int(state.get("current_depth") or 0)
    except Exception:
        current_depth = 0

    frontier = list(state.get("frontier_node_ids") or [])
    expanded_by_depth = state.get("expanded_node_ids_by_depth") or {}
    if not isinstance(expanded_by_depth, dict):
        expanded_by_depth = {}

    # Frontier nodes must exist and match current depth.
    for nid in frontier:
        nrec = (nodes or {}).get(nid) if isinstance(nodes, dict) else None
        if not isinstance(nrec, dict):
            _issue("state_inconsistency", "frontier_node_missing", node_id=str(nid))
            continue
        try:
            nd = int(nrec.get("depth") or 0)
        except Exception:
            nd = None
        if nd is None or nd != current_depth:
            _issue(
                "state_inconsistency",
                "frontier_node_depth_mismatch",
                node_id=str(nid),
                node_depth=nd,
                current_depth=current_depth,
            )

    # Expanded nodes must exist and match their depth bucket, and depth buckets must be well-formed.
    for depth_key, ids in expanded_by_depth.items():
        try:
            d = int(depth_key)
        except Exception:
            _issue("state_inconsistency", "expanded_by_depth_key_not_int", key=str(depth_key))
            continue
        if not isinstance(ids, list):
            _issue("state_inconsistency", "expanded_by_depth_value_not_list", depth=d, value_type=str(type(ids)))
            continue
        seen: set[str] = set()
        for nid in ids:
            sid = str(nid)
            if sid in seen:
                _issue("state_inconsistency", "expanded_node_duplicate", depth=d, node_id=sid)
                continue
            seen.add(sid)
            nrec = (nodes or {}).get(sid) if isinstance(nodes, dict) else None
            if not isinstance(nrec, dict):
                _issue("state_inconsistency", "expanded_node_missing", depth=d, node_id=sid)
                continue
            try:
                nd = int(nrec.get("depth") or 0)
            except Exception:
                nd = None
            if nd is None or nd != d:
                _issue("state_inconsistency", "expanded_node_depth_mismatch", depth=d, node_id=sid, node_depth=nd)

    # "No holes": for any depth strictly before current_depth, all nodes at that depth should be marked expanded.
    # (If the run hasn't advanced to a depth, it should not have left unexpanded nodes behind.)
    for d in range(0, max(0, current_depth)):
        nodes_at_depth: set[str] = set()
        if isinstance(nodes, dict):
            for nid, nrec in nodes.items():
                if not isinstance(nrec, dict):
                    continue
                try:
                    nd = int(nrec.get("depth") or 0)
                except Exception:
                    continue
                if nd == d:
                    nodes_at_depth.add(str(nid))
        expanded_ids = set(str(x) for x in (expanded_by_depth.get(str(d)) or []) if isinstance(expanded_by_depth.get(str(d)), list))
        missing = sorted(nodes_at_depth - expanded_ids)
        if missing:
            _issue("state_inconsistency", "unexpanded_nodes_before_current_depth", depth=d, node_ids=missing)

    return issues


def _write_validation_report(*, run_root: Path, issues: list[dict[str, Any]]) -> Path:
    out = run_root / "VALIDATION_REPORT.md"
    lines = ["# VALIDATION_REPORT", "", f"- issues: {len(issues)}", ""]
    for i, rec in enumerate(issues, start=1):
        lines.append(f"## Issue {i}")
        lines.append(f"- kind: {rec.get('kind')}")
        lines.append(f"- message: {rec.get('message')}")
        details = rec.get("details") or {}
        if isinstance(details, dict) and details:
            for k in sorted(details.keys()):
                lines.append(f"- {k}: {details[k]}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _queue_deferred_cleanup(manifest: dict[str, Any], *, path: str, kind: str, reason: str) -> None:
    state = manifest.setdefault("state", {})
    dq = state.setdefault("deferred_cleanup", [])
    now = time.time()
    dq.append(
        {
            "path": str(path),
            "kind": str(kind),
            "reason": str(reason),
            "attempts": 1,
            "next_retry_at": _dt.datetime.fromtimestamp(now + 60, tz=_dt.timezone.utc).isoformat(),
        }
    )


def _ensure_artifact_copy(*, run_root: Path, source_path: Path, dest_name: str) -> dict[str, Any]:
    artifacts_root = _ensure_standard_run_dirs(run_root)["artifacts_root"]
    dest = artifacts_root / dest_name
    # If the file is already in artifacts, keep it as-is.
    try:
        if source_path.resolve().parent == artifacts_root.resolve() and source_path.name == dest.name:
            return {"source_path": str(source_path), "copied_to_path": str(source_path), "sha256": _sha256_file(source_path)}
    except Exception:
        pass
    return _copy_with_provenance(source_path=source_path, dest_path=dest)

@dataclasses.dataclass
class _RunLock:
    path: Path
    stale_seconds: int
    force: bool
    tree_run_id: str
    acquired: bool = False
    _stop: threading.Event = dataclasses.field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def _lock_payload(self, *, created_at: str, last_heartbeat_at: str, takeover: bool) -> dict[str, Any]:
        return {
            "lock_version": 1,
            "tree_run_id": self.tree_run_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "created_at": created_at,
            "last_heartbeat_at": last_heartbeat_at,
            "takeover": bool(takeover),
        }

    def _load_existing(self) -> Optional[dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            obj = _read_json(self.path)
        except Exception:
            return {"_corrupt": True}
        if not isinstance(obj, dict):
            return {"_corrupt": True}
        return obj

    def _is_stale(self, lock_obj: dict[str, Any], *, now: float) -> bool:
        try:
            ts = lock_obj.get("last_heartbeat_at") or lock_obj.get("created_at")
            if not ts:
                return True
            dt = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            age = now - dt.timestamp()
            return age >= float(self.stale_seconds)
        except Exception:
            return True

    def acquire(self) -> dict[str, Any]:
        _safe_mkdir(self.path.parent)
        now = time.time()
        existing = self._load_existing()
        takeover = False
        if existing is not None:
            if existing.get("_corrupt"):
                takeover = True
            elif not self._is_stale(existing, now=now) and not self.force:
                raise RuntimeError(
                    f"Run lock is active (not stale): {self.path}. "
                    "Use --force to take over or wait for it to become stale."
                )
            else:
                takeover = True

        created_at = _utc_now_iso()
        payload = self._lock_payload(created_at=created_at, last_heartbeat_at=created_at, takeover=takeover)
        _atomic_write_json(self.path, payload)
        self.acquired = True
        self._start_heartbeat_thread()
        return {"takeover": takeover, "lock": payload}

    def _start_heartbeat_thread(self) -> None:
        if self._thread is not None:
            return

        def _loop() -> None:
            # Required heartbeat policy:
            # - update at least every 30 seconds while running.
            while not self._stop.is_set():
                self.heartbeat()
                self._stop.wait(30.0)

        self._thread = threading.Thread(target=_loop, name="tree_run_lock_heartbeat", daemon=True)
        self._thread.start()

    def heartbeat(self) -> None:
        if not self.acquired:
            return
        try:
            existing = self._load_existing() or {}
            if not isinstance(existing, dict):
                existing = {}
            existing["last_heartbeat_at"] = _utc_now_iso()
            _atomic_write_json(self.path, existing)
        except Exception:
            # Best-effort only; heartbeat failures should not crash the run.
            return

    def release(self) -> None:
        if not self.acquired:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with contextlib.suppress(Exception):
            self.path.unlink()


def _default_tree_run_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _copy_with_provenance(*, source_path: Path, dest_path: Path) -> dict[str, Any]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    return {
        "source_path": str(source_path),
        "copied_to_path": str(dest_path),
        "sha256": _sha256_file(dest_path),
    }


def _init_or_resume_manifest(
    *,
    run_root: Path,
    repo_root: Path,
    agentic_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest_path = run_root / "manifest.json"

    if manifest_path.exists():
        if not bool(args.resume):
            raise RuntimeError(f"manifest.json already exists at {manifest_path} (use --resume).")
        _ensure_standard_run_dirs(run_root)
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Invalid manifest format at {manifest_path}")
        if str(manifest.get("tree_run_id") or "") != str(args.tree_run_id):
            raise RuntimeError(
                f"Manifest tree_run_id mismatch: manifest has {manifest.get('tree_run_id')!r} "
                f"but args.tree_run_id is {args.tree_run_id!r}"
            )
        _ensure_manifest_conversation_schema(manifest=manifest, run_root=run_root, args=args)
        _manifest_write(run_root, manifest)
        return manifest

    # New run: enforce clean root tree (decision).
    _assert_clean_working_tree(repo_root)

    paths = _ensure_standard_run_dirs(run_root)

    # Load agent config (store for audit).
    config_path = (repo_root / str(args.config)).resolve() if not Path(str(args.config)).is_absolute() else Path(str(args.config)).resolve()
    if not config_path.exists():
        raise RuntimeError(f"Agent config not found: {config_path}")
    config_obj = _read_json(config_path)
    if not isinstance(config_obj, dict):
        raise RuntimeError(f"Invalid agent config JSON: {config_path}")

    baseline_csv = config_obj.get("baseline_csv")
    if not baseline_csv:
        raise RuntimeError("agent_config.json must define baseline_csv")
    baseline_csv_path = (repo_root / str(baseline_csv)).resolve() if not Path(str(baseline_csv)).is_absolute() else Path(str(baseline_csv)).resolve()
    if not baseline_csv_path.exists():
        raise RuntimeError(f"Root baseline CSV not found: {baseline_csv_path}")

    artifact_policy = "copy_to_run_root"
    root_baseline_copy = paths["artifacts_root"] / "root_baseline.csv"
    baseline_prov = _copy_with_provenance(source_path=baseline_csv_path, dest_path=root_baseline_copy)

    root_commit = _run_git(repo_root, ["rev-parse", "HEAD"])
    node_id = _format_node_id(0)
    node_wt = create_node_worktree(repo_root=repo_root, run_root=run_root, tree_run_id=str(args.tree_run_id), node_id=node_id, commit=root_commit)
    root_ref_name = str(node_wt["ref_name"])
    node_ideas_dir = paths["node_ideas_root"] / node_id
    _safe_mkdir(node_ideas_dir)
    node_record = {
        "node_id": node_id,
        "parent_node_id": None,
        "depth": 0,
        "commit": root_commit,
        "ref_name": root_ref_name,
        "worktree_path": str(node_wt["worktree_path"]),
        "baseline_results_csv_path": str(root_baseline_copy),
        "node_ideas_dir": str(node_ideas_dir),
        "idea_chain": [],
        "conversation_id": None,
        "expansion_seed_turn_id": None,
        "latest_conversation_turn_id": None,
        "artifacts": {"root_baseline_provenance": baseline_prov},
        "created_at": _utc_now_iso(),
        "status": "ready",
    }

    tree_run_id = str(args.tree_run_id)
    manifest = {
        "manifest_version": 3,
        "tree_run_id": tree_run_id,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "run_config": {
            "ideas_per_node": int(args.ideas_per_node),
            "max_depth": int(args.max_depth),
            "beam_width": int(args.beam_width),
            "max_parallel_evals": int(args.max_parallel_evals),
            "max_parallel_per_node": (int(args.max_parallel_per_node) if args.max_parallel_per_node is not None else None),
            "parallel_backend": str(args.parallel_backend),
            "eval_retries": int(args.eval_retries),
            "eval_timeout_seconds": (int(args.eval_timeout_seconds) if args.eval_timeout_seconds is not None else None),
            "strict_fail_depth": bool(args.strict_fail_depth),
            "sweep_config_limit": (int(args.sweep_config_limit) if args.sweep_config_limit is not None else None),
            "max_total_idea_evals": (int(args.max_total_idea_evals) if args.max_total_idea_evals is not None else None),
            "stop_on_empty_frontier": bool(args.stop_on_empty_frontier),
            "resume": bool(args.resume),
            "lock_stale_seconds": int(args.lock_stale_seconds),
            "artifact_policy": artifact_policy,
            "node_ideas_root_dir": str(paths["node_ideas_root"]),
            "ideas_context_strategy": "node_plus_ancestors",
            "idea_conversation_mode": str(args.idea_conversation_mode),
            "idea_history_window_turns": int(args.idea_history_window_turns),
            "idea_history_max_chars": int(args.idea_history_max_chars),
            "agent_config_path": str(config_path),
            "agent_config_snapshot": config_obj,
            "runs_root": str(run_root.parent),
            "run_root": str(run_root),
            "wt_root": str(paths["wt_root"]),
            "cand_root": str(paths["cand_root"]),
            "eval_root": str(paths["eval_root"]),
            "conversations_root": str(paths["conversations_root"]),
        },
        "conversation_config": {
            "mode": str(args.idea_conversation_mode),
            "history_window_turns": int(args.idea_history_window_turns),
            "history_max_chars": int(args.idea_history_max_chars),
            "debug_log_jsonl_path": str(paths["conversations_root"] / "conversation_debug.jsonl"),
        },
        "root": {
            "root_commit": root_commit,
            "root_ref_name": root_ref_name,
            "root_baseline_csv_path": str(root_baseline_copy),
            "root_baseline_provenance": baseline_prov,
        },
        "state": {
            "current_depth": 0,
            "frontier_node_ids": [node_id],
            "expanded_node_ids_by_depth": {},
            "completed_depths": [],
            "next_node_id": 1,
            "next_eval_id": 1,
            "deferred_cleanup": [],
        },
        "events": [],
        "conversations": {},
        "nodes": {node_id: node_record},
        "evaluations": {},
    }

    _ensure_manifest_conversation_schema(manifest=manifest, run_root=run_root, args=args)
    _ensure_node_conversation(
        manifest=manifest,
        run_root=run_root,
        node_id=node_id,
        parent_node_id=None,
        fork_from_turn_id=None,
    )
    _manifest_write(run_root, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    if int(args.idea_history_max_chars) < -1:
        raise ValueError("--idea-history-max-chars must be -1, 0, or a positive integer.")
    if int(args.max_parallel_evals) < 1:
        raise ValueError("--max-parallel-evals must be >= 1.")
    if args.max_parallel_per_node is not None and int(args.max_parallel_per_node) < 1:
        raise ValueError("--max-parallel-per-node must be >= 1 when set.")
    if int(args.eval_retries) < 0:
        raise ValueError("--eval-retries must be >= 0.")
    if args.eval_timeout_seconds is not None and int(args.eval_timeout_seconds) <= 0:
        raise ValueError("--eval-timeout-seconds must be > 0 when set.")
    # Local import: keeps module import side-effects minimal.
    from scoring_hooks import compute_score  # type: ignore

    repo_root = _resolve_repo_root(Path(__file__).resolve())
    agentic_root = repo_root / "agentic_experimentation"
    tree_run_id = str(args.tree_run_id or _default_tree_run_id())
    args.tree_run_id = tree_run_id

    runs_root = Path(args.runs_root).expanduser().resolve() if args.runs_root else (agentic_root / "worktrees" / "tree_runs")
    run_root = runs_root / tree_run_id
    _safe_mkdir(run_root)

    lock = _RunLock(
        path=run_root / "run.lock.json",
        stale_seconds=int(args.lock_stale_seconds),
        force=bool(args.force),
        tree_run_id=tree_run_id,
    )

    lock_info = None
    manifest = None
    try:
        lock_info = lock.acquire()
        manifest = _init_or_resume_manifest(run_root=run_root, repo_root=repo_root, agentic_root=agentic_root, args=args)
        _append_tree_log(
            run_root,
            "start"
            + f" tree_run_id={tree_run_id}"
            + f" pid={os.getpid()}"
            + f" resume={bool(args.resume)}"
            + f" force={bool(args.force)}"
            + f" validate_only={bool(args.validate_only)}"
            + f" report_only={bool(args.report_only)}",
        )

        # Record lock takeover events in the manifest for auditability.
        if lock_info and lock_info.get("takeover"):
            manifest.setdefault("events", []).append(
                {
                    "ts": _utc_now_iso(),
                    "type": "lock_takeover",
                    "details": {"lock_path": str(lock.path), "lock": lock_info.get("lock")},
                }
            )
            _manifest_write(run_root, manifest)

        if bool(args.validate_only) or bool(args.report_only):
            issues = _validate_run(repo_root=repo_root, run_root=run_root, manifest=manifest)
            report_path = _write_validation_report(run_root=run_root, issues=issues)
            summary_path = None
            graph_path = None
            if bool(args.report_only):
                summary_path = _write_tree_summary(run_root=run_root, manifest=manifest)
                graph_path = _write_tree_graph(run_root=run_root, manifest=manifest)
            print(
                json.dumps(
                    {
                        "tree_run_id": tree_run_id,
                        "run_root": str(run_root),
                        "manifest_path": str(run_root / "manifest.json"),
                        "tree_summary_path": (str(summary_path) if summary_path else None),
                        "tree_graph_path": (str(graph_path) if graph_path else None),
                        "validation_report_path": str(report_path),
                        "issues": len(issues),
                    },
                    indent=2,
                )
            )
            return 0

        # Phase 3: expand frontier nodes at current depth, evaluate K ideas per node.
        run_config = manifest.get("run_config") or {}
        sweep_config_limit = run_config.get("sweep_config_limit")
        max_total_idea_evals = run_config.get("max_total_idea_evals")
        max_depth = int(run_config.get("max_depth") or 0)
        beam_width = int(run_config.get("beam_width") or 1)
        max_parallel_evals = int(run_config.get("max_parallel_evals") or 1)
        _max_parallel_per_node_raw = run_config.get("max_parallel_per_node")
        max_parallel_per_node = (int(_max_parallel_per_node_raw) if _max_parallel_per_node_raw is not None else None)
        parallel_backend = str(run_config.get("parallel_backend") or "threadpool")
        eval_retries = int(run_config.get("eval_retries") or 0)
        _eval_timeout_raw = run_config.get("eval_timeout_seconds")
        eval_timeout_seconds = (int(_eval_timeout_raw) if _eval_timeout_raw is not None else None)
        strict_fail_depth = bool(run_config.get("strict_fail_depth", False))
        stop_on_empty_frontier = bool(run_config.get("stop_on_empty_frontier", True))
        conversation_config = manifest.get("conversation_config") or {}
        if not isinstance(conversation_config, dict):
            conversation_config = {}
        idea_conversation_mode = str(conversation_config.get("mode") or "auto")
        idea_history_window_turns = int(conversation_config.get("history_window_turns") or 12)
        _idea_history_max_chars_raw = conversation_config.get("history_max_chars", 20000)
        idea_history_max_chars = int(20000 if _idea_history_max_chars_raw is None else _idea_history_max_chars_raw)
        config_path = Path(run_config.get("agent_config_path") or args.config).resolve()
        agent_cfg = _read_json(config_path)
        if not isinstance(agent_cfg, dict):
            raise RuntimeError(f"Invalid agent config JSON: {config_path}")
        score_column = (agent_cfg.get("scoring") or {}).get("score_column")

        state = manifest.get("state") or {}
        current_depth = int(state.get("current_depth") or 0)
        frontier = sorted(str(x) for x in (state.get("frontier_node_ids") or []))
        _append_tree_log(run_root, f"depth_start depth={current_depth} frontier={frontier}")

        if current_depth >= max_depth:
            state["stop_reason"] = "max_depth_reached"
            _manifest_write(run_root, manifest)
            try:
                issues = _validate_run(repo_root=repo_root, run_root=run_root, manifest=manifest)
                _write_validation_report(run_root=run_root, issues=issues)
            except Exception:
                pass
            try:
                _write_tree_summary(run_root=run_root, manifest=manifest)
            except Exception:
                pass
            try:
                _write_tree_graph(run_root=run_root, manifest=manifest)
            except Exception:
                pass
            print(
                json.dumps(
                    {
                        "tree_run_id": tree_run_id,
                        "run_root": str(run_root),
                        "manifest_path": str(run_root / "manifest.json"),
                        "current_depth": current_depth,
                        "frontier_size": len(frontier),
                        "evaluations_total": len((manifest.get("evaluations") or {})),
                        "stop_reason": state.get("stop_reason"),
                    },
                    indent=2,
                )
            )
            return 0

        expanded_by_depth = state.setdefault("expanded_node_ids_by_depth", {})
        expanded_set = set(expanded_by_depth.get(str(current_depth)) or [])
        if expanded_set:
            _append_tree_log(run_root, f"depth_resume depth={current_depth} already_expanded={sorted(expanded_set)}")

        # Auto-repair: if a previous run ended with empty_frontier at this depth but left
        # gate/rank decisions unset (common after a crash or a logic bug), reconstruct the
        # frontier from expanded nodes and re-run promotion logic.
        #
        # This is intentionally narrow: it only triggers when `passed_gate` is still None.
        if str(state.get("stop_reason") or "") == "empty_frontier" and not frontier and current_depth < max_depth and expanded_set:
            candidate_frontier = sorted(str(x) for x in expanded_set)
            cand_frontier_set = set(candidate_frontier)
            needs_repair = False
            for rec in (manifest.get("evaluations") or {}).values():
                if not isinstance(rec, dict):
                    continue
                depth_val = rec.get("depth")
                if depth_val is None:
                    continue
                try:
                    rec_depth = int(depth_val)
                except Exception:
                    continue
                if rec_depth != current_depth:
                    continue
                if str(rec.get("parent_node_id") or "") not in cand_frontier_set:
                    continue
                decision = rec.get("decision") or {}
                if not isinstance(decision, dict) or decision.get("passed_gate") is None:
                    needs_repair = True
                    break

            if needs_repair:
                _append_tree_log(
                    run_root,
                    f"repair_empty_frontier depth={current_depth} restored_frontier={candidate_frontier}",
                )
                manifest.setdefault("events", []).append(
                    {
                        "ts": _utc_now_iso(),
                        "type": "repair_empty_frontier",
                        "details": {
                            "depth": current_depth,
                            "restored_frontier": candidate_frontier,
                        },
                    }
                )
                state.pop("stop_reason", None)
                frontier = candidate_frontier
                state["frontier_node_ids"] = list(candidate_frontier)
                _manifest_write(run_root, manifest)

        # Count eval records for max_total_idea_evals enforcement.
        evals = manifest.setdefault("evaluations", {})
        completed_evals = [e for e in evals.values() if isinstance(e, dict) and e.get("status") == "completed"]

        stop_reason = state.get("stop_reason")
        depth_task_records: list[dict[str, Any]] = []
        task_plan_by_depth = state.setdefault("task_plan_by_depth", {})
        if not isinstance(task_plan_by_depth, dict):
            task_plan_by_depth = {}
            state["task_plan_by_depth"] = task_plan_by_depth
        previous_depth_plan = list(task_plan_by_depth.get(str(current_depth)) or [])
        depth_dispatch_queue: list[dict[str, Any]] = []
        for node_id in frontier:
            if stop_reason:
                break
            if node_id in expanded_set:
                continue

            node = (manifest.get("nodes") or {}).get(node_id)
            if not isinstance(node, dict):
                continue

            node_commit = str(node.get("commit") or "")
            node_baseline_csv_path = Path(str(node.get("baseline_results_csv_path") or "")).expanduser()
            if not node_commit or not node_baseline_csv_path.exists():
                raise RuntimeError(f"Node {node_id} missing commit/baseline csv")

            node_ideas_dir = Path(str(node.get("node_ideas_dir") or "")).expanduser()
            if not node_ideas_dir:
                node_ideas_dir = _ensure_standard_run_dirs(run_root)["node_ideas_root"] / str(node_id)
                _safe_mkdir(node_ideas_dir)
                node["node_ideas_dir"] = str(node_ideas_dir)

            parent_node_id = node.get("parent_node_id")
            parent_node_id_s = str(parent_node_id) if parent_node_id is not None else None
            fork_from_turn_id = node.get("expansion_seed_turn_id")
            fork_from_turn_id_s = str(fork_from_turn_id) if fork_from_turn_id is not None else None
            node_conversation_id = _ensure_node_conversation(
                manifest=manifest,
                run_root=run_root,
                node_id=str(node_id),
                parent_node_id=parent_node_id_s,
                fork_from_turn_id=fork_from_turn_id_s,
            )
            conversations = manifest.get("conversations") or {}
            if not isinstance(conversations, dict):
                conversations = {}
                manifest["conversations"] = conversations
            node_conv = conversations.get(node_conversation_id) or {}
            if not isinstance(node_conv, dict):
                node_conv = {}
            node_conv_state_raw = str(node_conv.get("state_json_path") or "").strip()
            node_conv_turn_log_raw = str(node_conv.get("turn_log_jsonl_path") or "").strip()
            node_conv_state_path = Path(node_conv_state_raw).expanduser() if node_conv_state_raw else None
            node_conv_turn_log_path = Path(node_conv_turn_log_raw).expanduser() if node_conv_turn_log_raw else None

            context_dirs = _collect_context_idea_dirs(manifest, node_id)
            node["context_ideas_dirs"] = context_dirs

            # Build baseline artifact context for the idea generator (small JSON + prompt block).
            baseline_ctx_path = node_ideas_dir / "baseline_context.json"
            baseline_ctx = _build_baseline_context_for_node(
                run_root=run_root,
                repo_root=repo_root,
                manifest=manifest,
                node_id=str(node_id),
                node_rec=node,
                compute_score=compute_score,
                sweep_config_limit=(int(sweep_config_limit) if sweep_config_limit is not None else None),
            )
            _atomic_write_json(baseline_ctx_path, baseline_ctx)
            node["baseline_context_json_path"] = str(baseline_ctx_path)

            # Generate missing ideas (if needed).
            desired_k = int(run_config.get("ideas_per_node") or int(args.ideas_per_node))
            existing_ideas = _list_markdown_files(node_ideas_dir)
            if len(existing_ideas) < desired_k or bool(args.force_regenerate_ideas):
                missing = desired_k - len(existing_ideas)
                if missing > 0:
                    if bool(args.dry_run):
                        _dry_run_ensure_ideas(node_ideas_dir=node_ideas_dir, node_id=str(node_id), desired_k=desired_k)
                    else:
                        gen_log_path = node_ideas_dir / "generate_ideas.subprocess.log"
                        _append_tree_log(run_root, f"generate_ideas node_id={node_id} missing={missing} log={gen_log_path}")
                        gen_args = [
                            "agentic_experimentation/idea_generation/generate_ideas.py",
                            "--ideas-dir",
                            str(node_ideas_dir),
                            "--count",
                            str(missing),
                        ]
                        gen_args.extend(["--baseline-context-json", str(baseline_ctx_path)])
                        if idea_conversation_mode != "off":
                            gen_args.extend(["--conversation-mode", str(idea_conversation_mode)])
                            if node_conv_state_path is not None:
                                gen_args.extend(["--conversation-state-in", str(node_conv_state_path)])
                                gen_args.extend(["--conversation-state-out", str(node_conv_state_path)])
                            if node_conv_turn_log_path is not None:
                                gen_args.extend(["--emit-turn-log", str(node_conv_turn_log_path)])
                            gen_args.extend(["--conversation-history-window-turns", str(int(idea_history_window_turns))])
                            gen_args.extend(["--conversation-history-max-chars", str(int(idea_history_max_chars))])
                            if fork_from_turn_id_s:
                                gen_args.extend(["--fork-from-turn-id", str(fork_from_turn_id_s)])
                        for d in context_dirs:
                            gen_args.extend(["--context-ideas-dir", str(d)])
                        env = dict(os.environ)
                        rc = _run_python_logged(repo_root=repo_root, args=gen_args, env=env, log_path=gen_log_path, echo=True)
                        if rc != 0:
                            raise RuntimeError(f"generate_ideas.py failed for node {node_id} (exit {rc})")
                existing_ideas = _list_markdown_files(node_ideas_dir)

            latest_turn_id = _sync_node_conversation_latest_turn(manifest=manifest, run_root=run_root, node_id=str(node_id))
            if latest_turn_id:
                node["expansion_seed_turn_id"] = latest_turn_id

            selected_ideas = sorted(existing_ideas[:desired_k], key=lambda p: str(p))
            node["generated_idea_files"] = [str(p) for p in selected_ideas]
            # Record context file hashes for reproducibility (the set included by generate_ideas is dynamic,
            # but at least persist what dirs exist and their file hashes at eval time).
            context_files = []
            for d in context_dirs:
                for p in _list_markdown_files(Path(d)):
                    context_files.append({"path": str(p), "sha256": _sha256_file(p)})
            node["context_idea_files"] = context_files

            # Dedupe seed: avoid repeating an identical idea already present in ancestors (and optionally globally).
            dedupe_scope = str(getattr(args, "dedupe_scope", "node_plus_ancestors"))
            seed_hashes = _collect_dedupe_hashes(manifest, node_id=node_id, scope=dedupe_scope)
            node.setdefault("dedupe", {})
            if isinstance(node.get("dedupe"), dict):
                node["dedupe"]["policy"] = {
                    "scope": dedupe_scope,
                    "normalization": "lower+trim+drop_blank+normalize_bullets",
                }
                node["dedupe"].setdefault("skipped", [])
            seen_hashes = set(seed_hashes)
            _manifest_write(run_root, manifest)
            lock.heartbeat()

            # Build queued eval tasks in deterministic filename order.
            for idea_path in selected_ideas:
                # Deterministic repeat-avoidance (skip duplicate idea text within this node and vs ancestors).
                try:
                    ih = _idea_text_hash(idea_path)
                except Exception:
                    ih = None
                if ih and ih in seen_hashes:
                    if isinstance(node.get("dedupe"), dict) and isinstance(node["dedupe"].get("skipped"), list):
                        node["dedupe"]["skipped"].append({"path": str(idea_path), "hash": ih, "reason": "duplicate_idea_text"})
                        _manifest_write(run_root, manifest)
                    continue
                if ih:
                    seen_hashes.add(ih)

                eval_record_count = len([e for e in evals.values() if isinstance(e, dict)])
                if max_total_idea_evals is not None and eval_record_count >= int(max_total_idea_evals):
                    state["stop_reason"] = "max_total_idea_evals_reached"
                    stop_reason = state["stop_reason"]
                    _manifest_write(run_root, manifest)
                    break

                idea_generation_turn_id = _lookup_conversation_turn_for_idea(
                    manifest=manifest,
                    run_root=run_root,
                    conversation_id=str(node.get("conversation_id") or ""),
                    idea_path=idea_path,
                )

                # Skip if already evaluated for this node+idea (resume).
                matching = []
                for ev in evals.values():
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("parent_node_id") == node_id and str(ev.get("idea_path") or "") == str(idea_path):
                        matching.append(ev)
                any_completed_for_key = any(str(ev.get("status") or "") == "completed" for ev in matching)
                # Prefer the latest eval_id for the same (node, idea).
                existing_eval = None
                if matching:
                    matching.sort(key=lambda e: str(e.get("eval_id") or ""))
                    existing_eval = matching[-1]

                if existing_eval is not None and not bool(args.rerun_evals):
                    # If any prior eval for this idempotency key completed, never duplicate it.
                    if any_completed_for_key:
                        _manifest_write(run_root, manifest)
                        continue
                    # Prior attempts failed/interrupted/running/queued -> requeue in-place.
                    existing_eval["idea_generation_conversation_id"] = str(node.get("conversation_id") or "")
                    if idea_generation_turn_id:
                        existing_eval["idea_generation_turn_id"] = idea_generation_turn_id
                    eval_id = str(existing_eval.get("eval_id"))
                    eval_rec = existing_eval
                    eval_output_root = Path(str(eval_rec.get("eval_output_root") or "")).expanduser()
                    if not eval_output_root:
                        eval_output_root = _ensure_standard_run_dirs(run_root)["eval_root"] / eval_id
                        eval_rec["eval_output_root"] = str(eval_output_root)
                    experiment_dir = Path(str(eval_rec.get("experiment_dir") or "")).expanduser()
                    if not experiment_dir:
                        experiment_dir = eval_output_root / "experiment"
                        eval_rec["experiment_dir"] = str(experiment_dir)
                    _safe_mkdir(experiment_dir)
                else:
                    next_eval_num = int(state.get("next_eval_id") or 1)
                    eval_id = _format_eval_id(next_eval_num)
                    state["next_eval_id"] = next_eval_num + 1

                    eval_output_root = _ensure_standard_run_dirs(run_root)["eval_root"] / eval_id
                    experiment_dir = eval_output_root / "experiment"
                    _safe_mkdir(experiment_dir)

                    eval_rec = {
                        "eval_id": eval_id,
                        "parent_node_id": node_id,
                        "depth": current_depth,
                        "idea_path": str(idea_path),
                        "idea_generation_conversation_id": str(node.get("conversation_id") or ""),
                        "idea_generation_turn_id": (str(idea_generation_turn_id) if idea_generation_turn_id else None),
                        "eval_output_root": str(eval_output_root),
                        "experiment_dir": str(experiment_dir),
                        "candidate_worktree_path": None,
                        "candidate_ref_name": _eval_ref_name(tree_run_id, eval_id),
                        "candidate_commit": None,
                        "candidate_results_csv_path": None,
                        "status": "queued",
                        "task_state": "queued",
                        "queued_at": None,
                        "started_at": None,
                        "finished_at": None,
                        "worker_pid": None,
                        "attempt": 0,
                        "error": None,
                        "parent_relative": {},
                        "root_relative": {},
                        "decision": {
                            "gate_basis": "parent_relative",
                            "rank_basis": "root_relative",
                            "passed_gate": None,
                            "rank_score": None,
                            "promotion_reason": None,
                            "primary_regressed": None,
                        },
                    }
                    evals[eval_id] = eval_rec

                eval_rec.setdefault(
                    "decision",
                    {
                        "gate_basis": "parent_relative",
                        "rank_basis": "root_relative",
                        "passed_gate": None,
                        "rank_score": None,
                        "promotion_reason": None,
                        "primary_regressed": None,
                    },
                )
                eval_rec["status"] = "queued"
                eval_rec["task_state"] = "queued"
                eval_rec["error"] = None
                eval_rec.setdefault("attempt", 0)
                queued_ts = _utc_now_iso()
                eval_rec["queued_at"] = queued_ts
                # Clear previous run markers when re-queuing.
                eval_rec["started_at"] = None
                eval_rec["finished_at"] = None
                eval_rec["worker_pid"] = None
                eval_rec["idea_generation_conversation_id"] = str(node.get("conversation_id") or "")
                if idea_generation_turn_id:
                    eval_rec["idea_generation_turn_id"] = str(idea_generation_turn_id)

                task_rec = {
                    "eval_id": str(eval_id),
                    "parent_node_id": str(node_id),
                    "idea_path": str(idea_path),
                    "depth": int(current_depth),
                    "task_state": "queued",
                    "queued_at": queued_ts,
                    "started_at": None,
                    "finished_at": None,
                    "worker_pid": None,
                    "attempt": int(eval_rec.get("attempt") or 0),
                    "node_commit": node_commit,
                    "node_baseline_csv_path": str(node_baseline_csv_path),
                }
                depth_task_records.append(task_rec)
                depth_dispatch_queue.append(task_rec)
                _record_scheduler_event(
                    manifest=manifest,
                    run_root=run_root,
                    event_type="queued",
                    depth=int(current_depth),
                    eval_id=str(eval_id),
                    parent_node_id=str(node_id),
                    attempt=int(eval_rec.get("attempt") or 0),
                )

            expanded_set.add(node_id)
            expanded_by_depth[str(current_depth)] = sorted(expanded_set)
            _manifest_write(run_root, manifest)

        # Resume recovery: include prior queued/running tasks for this depth that were not
        # reconstructed above (e.g., interrupted run before task reconstruction completed).
        existing_task_eval_ids = {str(rec.get("eval_id") or "") for rec in depth_task_records}
        for old_task in previous_depth_plan:
            if not isinstance(old_task, dict):
                continue
            old_eval_id = str(old_task.get("eval_id") or "")
            if not old_eval_id or old_eval_id in existing_task_eval_ids:
                continue
            old_eval = (evals or {}).get(old_eval_id)
            if not isinstance(old_eval, dict):
                continue
            try:
                old_depth = int(old_eval.get("depth") or -1)
            except Exception:
                continue
            if old_depth != int(current_depth):
                continue
            old_parent = str(old_eval.get("parent_node_id") or "")
            if old_parent not in frontier:
                continue
            if str(old_eval.get("status") or "").lower() == "completed":
                continue

            parent_node = (manifest.get("nodes") or {}).get(old_parent) or {}
            if not isinstance(parent_node, dict):
                continue
            node_commit = str(parent_node.get("commit") or "")
            node_baseline_csv_path = str(parent_node.get("baseline_results_csv_path") or "")
            if not node_commit or not node_baseline_csv_path:
                continue

            queued_ts = _utc_now_iso()
            old_eval["status"] = "queued"
            old_eval["task_state"] = "queued"
            old_eval["queued_at"] = queued_ts
            old_eval["started_at"] = None
            old_eval["finished_at"] = None
            old_eval["worker_pid"] = None
            old_eval["error"] = None

            recovered = {
                "eval_id": old_eval_id,
                "parent_node_id": old_parent,
                "idea_path": str(old_eval.get("idea_path") or ""),
                "depth": int(current_depth),
                "task_state": "queued",
                "queued_at": queued_ts,
                "started_at": None,
                "finished_at": None,
                "worker_pid": None,
                "attempt": int(old_eval.get("attempt") or 0),
                "node_commit": node_commit,
                "node_baseline_csv_path": node_baseline_csv_path,
            }
            depth_task_records.append(recovered)
            depth_dispatch_queue.append(recovered)
            existing_task_eval_ids.add(old_eval_id)
            _append_tree_log(run_root, f"depth_resume_requeue depth={current_depth} eval_id={old_eval_id}")
            _record_scheduler_event(
                manifest=manifest,
                run_root=run_root,
                event_type="requeued",
                depth=int(current_depth),
                eval_id=str(old_eval_id),
                parent_node_id=str(old_parent),
                attempt=int(old_eval.get("attempt") or 0),
                reason="resume_recover",
            )

        task_plan_by_depth[str(current_depth)] = depth_task_records
        if depth_task_records:
            _append_tree_log(run_root, f"depth_task_plan depth={current_depth} queued_tasks={len(depth_task_records)}")
        _refresh_depth_progress(manifest=manifest, depth=int(current_depth))
        _manifest_write(run_root, manifest)
        lock.heartbeat()

        # Execute queued depth tasks (parallel coordinator + worker model).
        if parallel_backend != "threadpool":
            raise RuntimeError(f"Unsupported parallel backend: {parallel_backend}")

        root_baseline_csv = Path(str((manifest.get("root") or {}).get("root_baseline_csv_path") or "")).expanduser()
        completed_eval_ids = {str(rec.get("eval_id") or "") for rec in completed_evals if isinstance(rec, dict)}
        pending_tasks = list(depth_dispatch_queue)
        running_by_parent: dict[str, int] = {}
        futures: dict[Any, tuple[dict[str, Any], dict[str, Any]]] = {}
        strict_halt_depth = False

        # Preflight uniqueness checks for per-eval isolation guarantees.
        seen_output_roots: set[str] = set()
        seen_ref_names: set[str] = set()
        for planned in pending_tasks:
            eval_id = str(planned.get("eval_id") or "")
            eval_rec = (evals or {}).get(eval_id)
            if not isinstance(eval_rec, dict):
                continue
            output_root = str(eval_rec.get("eval_output_root") or "")
            ref_name = str(eval_rec.get("candidate_ref_name") or "")
            if output_root:
                if output_root in seen_output_roots:
                    raise RuntimeError(f"Duplicate eval_output_root detected at depth {current_depth}: {output_root}")
                seen_output_roots.add(output_root)
            if ref_name:
                if ref_name in seen_ref_names:
                    raise RuntimeError(f"Duplicate candidate_ref_name detected at depth {current_depth}: {ref_name}")
                seen_ref_names.add(ref_name)

        def _start_task(*, planned: dict[str, Any], pool: _futures.Executor) -> None:
            eval_id = str(planned.get("eval_id") or "")
            node_id = str(planned.get("parent_node_id") or "")
            idea_path = Path(str(planned.get("idea_path") or "")).expanduser()
            node_commit = str(planned.get("node_commit") or "")
            node_baseline_csv_path = Path(str(planned.get("node_baseline_csv_path") or "")).expanduser()
            if not eval_id or not node_id or not node_commit or not node_baseline_csv_path.exists():
                return

            eval_rec = (evals or {}).get(eval_id)
            if not isinstance(eval_rec, dict):
                return

            eval_output_root = Path(str(eval_rec.get("eval_output_root") or "")).expanduser()
            if not eval_output_root:
                eval_output_root = _ensure_standard_run_dirs(run_root)["eval_root"] / eval_id
                eval_rec["eval_output_root"] = str(eval_output_root)
            experiment_dir = Path(str(eval_rec.get("experiment_dir") or "")).expanduser()
            if not experiment_dir:
                experiment_dir = eval_output_root / "experiment"
                eval_rec["experiment_dir"] = str(experiment_dir)
            sweep_output_dir = eval_output_root
            sweep_results_csv = eval_output_root / "meta_config_sweep_results.csv"
            _safe_mkdir(experiment_dir)

            eval_rec["status"] = "running"
            eval_rec["task_state"] = "running"
            eval_rec["attempt"] = int(eval_rec.get("attempt") or 0) + 1
            start_ts = _utc_now_iso()
            eval_rec["started_at"] = start_ts
            eval_rec["worker_pid"] = int(os.getpid())
            eval_rec["error"] = None
            planned["task_state"] = "running"
            planned["attempt"] = int(eval_rec.get("attempt") or 0)
            planned["started_at"] = start_ts
            planned["worker_pid"] = int(os.getpid())
            _manifest_write(run_root, manifest)
            lock.heartbeat()
            _append_tree_log(run_root, f"eval_start eval_id={eval_id} parent_node_id={node_id}")
            _record_scheduler_event(
                manifest=manifest,
                run_root=run_root,
                event_type="started",
                depth=int(current_depth),
                eval_id=str(eval_id),
                parent_node_id=str(node_id),
                attempt=int(eval_rec.get("attempt") or 0),
            )
            _refresh_depth_progress(manifest=manifest, depth=int(current_depth))
            _manifest_write(run_root, manifest)
            lock.heartbeat()

            fut = pool.submit(
                _execute_eval_task_worker,
                repo_root=repo_root,
                run_root=run_root,
                tree_run_id=tree_run_id,
                eval_id=eval_id,
                node_id=node_id,
                idea_path=idea_path,
                node_commit=node_commit,
                node_baseline_csv_path=node_baseline_csv_path,
                eval_output_root=eval_output_root,
                experiment_dir=experiment_dir,
                sweep_output_dir=sweep_output_dir,
                sweep_results_csv=sweep_results_csv,
                config_path=config_path,
                sweep_config_limit=(int(sweep_config_limit) if sweep_config_limit is not None else None),
                eval_timeout_seconds=(int(eval_timeout_seconds) if eval_timeout_seconds is not None else None),
                score_column=score_column,
                root_baseline_csv=root_baseline_csv,
                dry_run=bool(args.dry_run),
                keep_rejected_worktrees=bool(args.keep_rejected_worktrees),
                keep_failed_artifacts=bool(args.keep_failed_artifacts),
                compute_score_fn=compute_score,
            )
            futures[fut] = (planned, eval_rec)
            running_by_parent[node_id] = running_by_parent.get(node_id, 0) + 1

        def _next_dispatch_index() -> Optional[int]:
            if not pending_tasks:
                return None
            if max_parallel_per_node is None:
                return 0
            for idx, task in enumerate(pending_tasks):
                parent = str(task.get("parent_node_id") or "")
                if running_by_parent.get(parent, 0) < int(max_parallel_per_node):
                    return idx
            return None

        def _apply_worker_result(*, planned: dict[str, Any], eval_rec: dict[str, Any], result: dict[str, Any]) -> None:
            nonlocal strict_halt_depth
            updates = result.get("updates") or {}
            if not isinstance(updates, dict):
                updates = {}

            for key, value in updates.items():
                if key == "decision_updates":
                    continue
                eval_rec[key] = value

            decision_updates = updates.get("decision_updates")
            if isinstance(decision_updates, dict):
                decision = eval_rec.setdefault("decision", {})
                if isinstance(decision, dict):
                    decision.update(decision_updates)

            status_val = str(result.get("status") or "failed")
            error_val = result.get("error")
            timed_out = bool(result.get("timed_out", False))
            finish_ts = _utc_now_iso()
            eval_rec["status"] = status_val
            eval_rec["task_state"] = status_val
            eval_rec["finished_at"] = finish_ts
            eval_rec["error"] = (str(error_val) if error_val else None)
            planned["task_state"] = status_val
            planned["finished_at"] = finish_ts
            eval_rec["worker_pid"] = None
            planned["worker_pid"] = None

            attempt_val = int(eval_rec.get("attempt") or 0)
            terminal_failure = False
            if status_val != "completed":
                can_retry = attempt_val <= int(eval_retries)
                eval_rec["error_payload"] = {
                    "error_type": ("TimeoutError" if timed_out else "EvalError"),
                    "message": str(error_val or ""),
                    "attempt": attempt_val,
                    "max_retries": int(eval_retries),
                    "will_retry": bool(can_retry),
                    "timed_out": bool(timed_out),
                }
                if can_retry:
                    queued_ts = _utc_now_iso()
                    eval_rec["status"] = "queued"
                    eval_rec["task_state"] = "queued"
                    eval_rec["queued_at"] = queued_ts
                    eval_rec["started_at"] = None
                    eval_rec["finished_at"] = None
                    eval_rec["worker_pid"] = None
                    eval_rec["error"] = None
                    planned["task_state"] = "queued"
                    planned["queued_at"] = queued_ts
                    planned["started_at"] = None
                    planned["finished_at"] = None
                    planned["worker_pid"] = None
                    planned["attempt"] = int(eval_rec.get("attempt") or 0)
                    pending_tasks.append(planned)
                    eval_id_local = str(eval_rec.get("eval_id") or "")
                    _append_tree_log(
                        run_root,
                        f"eval_requeued eval_id={eval_id_local} attempt={attempt_val} max_retries={int(eval_retries)}",
                    )
                    _record_scheduler_event(
                        manifest=manifest,
                        run_root=run_root,
                        event_type="requeued",
                        depth=int(current_depth),
                        eval_id=str(eval_id_local),
                        parent_node_id=str(eval_rec.get("parent_node_id") or ""),
                        attempt=int(attempt_val),
                        reason="retry",
                    )
                else:
                    terminal_failure = True

            for req in (result.get("cleanup_requests") or []):
                if not isinstance(req, dict):
                    continue
                _queue_deferred_cleanup(
                    manifest,
                    path=str(req.get("path") or ""),
                    kind=str(req.get("kind") or "unknown"),
                    reason=str(req.get("reason") or "cleanup_failed"),
                )

            eval_id_local = str(eval_rec.get("eval_id") or "")
            if status_val == "completed" and eval_id_local not in completed_eval_ids:
                completed_evals.append(eval_rec)
                completed_eval_ids.add(eval_id_local)

            if terminal_failure and bool(strict_fail_depth) and not strict_halt_depth:
                strict_halt_depth = True
                state["stop_reason"] = "strict_depth_failure"
                state["stop_reason_detail"] = f"eval_id={eval_id_local}"
                now_ts = _utc_now_iso()
                while pending_tasks:
                    skipped = pending_tasks.pop(0)
                    skipped_eval_id = str(skipped.get("eval_id") or "")
                    skipped_rec = (evals or {}).get(skipped_eval_id)
                    skipped["task_state"] = "failed"
                    skipped["finished_at"] = now_ts
                    if isinstance(skipped_rec, dict):
                        skipped_rec["status"] = "failed"
                        skipped_rec["task_state"] = "failed"
                        skipped_rec["finished_at"] = now_ts
                        skipped_rec["error"] = "strict_fail_depth: skipped after terminal failure at same depth"
                        skipped_rec["error_payload"] = {
                            "error_type": "StrictDepthFailure",
                            "message": "Skipped because strict_fail_depth halted remaining tasks at depth.",
                            "attempt": int(skipped_rec.get("attempt") or 0),
                            "max_retries": int(eval_retries),
                            "will_retry": False,
                            "timed_out": False,
                        }
                        _record_scheduler_event(
                            manifest=manifest,
                            run_root=run_root,
                            event_type="failed",
                            depth=int(current_depth),
                            eval_id=str(skipped_eval_id),
                            parent_node_id=str(skipped_rec.get("parent_node_id") or ""),
                            attempt=int(skipped_rec.get("attempt") or 0),
                            reason="strict_fail_depth_skipped",
                        )

            if status_val == "completed":
                _record_scheduler_event(
                    manifest=manifest,
                    run_root=run_root,
                    event_type="completed",
                    depth=int(current_depth),
                    eval_id=str(eval_id_local),
                    parent_node_id=str(eval_rec.get("parent_node_id") or ""),
                    attempt=int(eval_rec.get("attempt") or 0),
                )
            elif terminal_failure:
                _record_scheduler_event(
                    manifest=manifest,
                    run_root=run_root,
                    event_type="failed",
                    depth=int(current_depth),
                    eval_id=str(eval_id_local),
                    parent_node_id=str(eval_rec.get("parent_node_id") or ""),
                    attempt=int(eval_rec.get("attempt") or 0),
                    reason=("timeout" if timed_out else "terminal_failure"),
                )

            _refresh_depth_progress(manifest=manifest, depth=int(current_depth))
            _manifest_write(run_root, manifest)
            lock.heartbeat()
            _append_tree_log(run_root, f"eval_done eval_id={eval_id_local} status={status_val}")

        with _futures.ThreadPoolExecutor(max_workers=int(max_parallel_evals)) as pool:
            while pending_tasks or futures:
                while len(futures) < int(max_parallel_evals):
                    next_idx = _next_dispatch_index()
                    if next_idx is None:
                        break
                    planned = pending_tasks.pop(next_idx)
                    _start_task(planned=planned, pool=pool)

                if not futures:
                    break

                done, _ = _futures.wait(list(futures.keys()), return_when=_futures.FIRST_COMPLETED)
                for fut in done:
                    planned, eval_rec = futures.pop(fut)
                    parent_id = str(planned.get("parent_node_id") or "")
                    if parent_id:
                        running_by_parent[parent_id] = max(0, running_by_parent.get(parent_id, 1) - 1)
                        if running_by_parent[parent_id] == 0:
                            running_by_parent.pop(parent_id, None)

                    try:
                        result = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "updates": {},
                            "cleanup_requests": [],
                        }

                    _apply_worker_result(planned=planned, eval_rec=eval_rec, result=result)

        # Phase 4: decide promotions and advance the frontier (global beam).
        stop_reason = state.get("stop_reason")
        if not stop_reason:
            artifacts_root = _ensure_standard_run_dirs(run_root)["artifacts_root"]
            evals = manifest.setdefault("evaluations", {})
            nodes = manifest.setdefault("nodes", {})

            depth_eval_recs: list[dict[str, Any]] = []
            frontier_set = set(str(x) for x in frontier)
            for rec in evals.values():
                if not isinstance(rec, dict):
                    continue
                depth_val = rec.get("depth")
                if depth_val is None:
                    continue
                try:
                    rec_depth = int(depth_val)
                except Exception:
                    continue
                if rec_depth != current_depth:
                    continue
                if str(rec.get("parent_node_id") or "") not in frontier_set:
                    continue
                depth_eval_recs.append(rec)
            depth_eval_recs = sorted(depth_eval_recs, key=lambda rec: str(rec.get("eval_id") or ""))

            _append_tree_log(
                run_root,
                f"phase4_candidates depth={current_depth} frontier={list(frontier)} depth_eval_recs={len(depth_eval_recs)}",
            )

            # Compute gate + rank for all evals at this depth.
            for rec in depth_eval_recs:
                decision = rec.setdefault("decision", {})
                passed, reason, primary_regressed = _compute_gate_and_reason(
                    rec,
                    artifacts_root=artifacts_root,
                    sweep_config_limit=(int(sweep_config_limit) if sweep_config_limit is not None else None),
                )
                decision["passed_gate"] = bool(passed)
                decision["promotion_reason"] = reason
                decision["primary_regressed"] = primary_regressed
                decision["rank_score"] = _rank_score(rec)

            passed_gate = [r for r in depth_eval_recs if bool((r.get("decision") or {}).get("passed_gate"))]
            selected = _select_global_beam(passed_gate, beam_width=beam_width)
            selected_eval_ids = {str(r.get("eval_id") or "") for r in selected}

            promoted_node_ids: list[str] = []
            promotion_events: list[dict[str, Any]] = []
            for rec in selected:
                eval_id = str(rec.get("eval_id") or "")
                parent_node_id = str(rec.get("parent_node_id") or "")
                parent_node = nodes.get(parent_node_id) or {}
                if not isinstance(parent_node, dict):
                    continue

                decision = rec.setdefault("decision", {})
                already_node_id = decision.get("promoted_to_node_id")
                if already_node_id and isinstance(nodes.get(str(already_node_id)), dict):
                    promoted_node_ids.append(str(already_node_id))
                    continue

                next_node_num = int(state.get("next_node_id") or 1)
                new_node_id = _format_node_id(next_node_num)
                state["next_node_id"] = next_node_num + 1

                cand_commit = str(rec.get("candidate_commit") or "").strip()
                if not cand_commit:
                    continue

                try:
                    if bool(args.dry_run):
                        node_wt = {"ref_name": "", "worktree_path": ""}
                    else:
                        node_wt = create_node_worktree(
                            repo_root=repo_root,
                            run_root=run_root,
                            tree_run_id=tree_run_id,
                            node_id=new_node_id,
                            commit=cand_commit,
                        )

                    paths = _ensure_standard_run_dirs(run_root)
                    node_ideas_dir = paths["node_ideas_root"] / new_node_id
                    _safe_mkdir(node_ideas_dir)

                    idea_chain = list(parent_node.get("idea_chain") or [])
                    idea_chain.append(str(rec.get("idea_path") or ""))

                    promotion_summary_path = ""
                    parent_rel = rec.get("parent_relative") or {}
                    if isinstance(parent_rel, dict) and parent_rel.get("summary_json_path"):
                        promotion_summary_path = str(parent_rel["summary_json_path"])
                    parent_seed_turn_id = str(
                        parent_node.get("expansion_seed_turn_id")
                        or parent_node.get("latest_conversation_turn_id")
                        or ""
                    ).strip()

                    node_record = {
                        "node_id": new_node_id,
                        "parent_node_id": parent_node_id,
                        "depth": int(current_depth) + 1,
                        "commit": cand_commit,
                        "ref_name": str(node_wt["ref_name"]),
                        "worktree_path": str(node_wt["worktree_path"]),
                        "baseline_results_csv_path": str(rec.get("candidate_results_csv_path") or ""),
                        "baseline_summary_json_path": promotion_summary_path,
                        "node_ideas_dir": str(node_ideas_dir),
                        "idea_chain": idea_chain,
                        "conversation_id": None,
                        "expansion_seed_turn_id": (parent_seed_turn_id or None),
                        "latest_conversation_turn_id": None,
                        "created_at": _utc_now_iso(),
                        "status": "ready",
                        "artifacts": {
                            "promoted_from_eval_id": eval_id,
                            "promoted_candidate_results_provenance": rec.get("candidate_results_provenance"),
                        },
                    }
                    nodes[new_node_id] = node_record
                    _ensure_node_conversation(
                        manifest=manifest,
                        run_root=run_root,
                        node_id=new_node_id,
                        parent_node_id=parent_node_id,
                        fork_from_turn_id=(parent_seed_turn_id or None),
                    )
                    _sync_node_conversation_latest_turn(manifest=manifest, run_root=run_root, node_id=new_node_id)
                    promoted_node_ids.append(new_node_id)

                    decision["promoted_to_node_id"] = new_node_id
                    decision["promoted_at"] = _utc_now_iso()

                    promotion_events.append(
                        {
                            "ts": _utc_now_iso(),
                            "type": "promotion",
                            "details": {
                                "eval_id": eval_id,
                                "parent_node_id": parent_node_id,
                                "new_node_id": new_node_id,
                                "candidate_commit": cand_commit,
                            },
                        }
                    )

                    # If the candidate worktree was kept (debug), remove it after promotion.
                    cand_wt = rec.get("candidate_worktree_path")
                    if cand_wt and Path(str(cand_wt)).exists():
                        try:
                            cleanup_worktree(repo_root=repo_root, worktree_path=Path(str(cand_wt)))
                        except Exception as exc:  # noqa: BLE001
                            _queue_deferred_cleanup(manifest, path=str(cand_wt), kind="worktree", reason=str(exc))

                    # Optionally delete candidate ref after node ref exists (avoid dangling commits).
                    if not bool(args.dry_run) and not bool(args.keep_rejected_worktrees):
                        ref_name = str(rec.get("candidate_ref_name") or _eval_ref_name(tree_run_id, eval_id))
                        try:
                            _delete_branch(repo_root=repo_root, ref_name=ref_name)
                        except Exception as exc:  # noqa: BLE001
                            _queue_deferred_cleanup(manifest, path=ref_name, kind="ref", reason=str(exc))
                except Exception as exc:  # noqa: BLE001
                    decision["passed_gate"] = False
                    decision["promotion_reason"] = f"promotion_failed: {type(exc).__name__}"
                    decision["promotion_error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    # Persist after each promotion attempt so resume is safe.
                    _manifest_write(run_root, manifest)
                    lock.heartbeat()

            # Prune rejected candidate refs/worktrees (best-effort).
            for rec in depth_eval_recs:
                if bool(args.dry_run):
                    continue
                keep_candidate = bool(args.keep_rejected_worktrees) or (rec.get("status") != "completed" and bool(args.keep_failed_artifacts))
                if keep_candidate:
                    continue

                decision = rec.get("decision") or {}
                if not isinstance(decision, dict):
                    decision = {}
                eval_id = str(rec.get("eval_id") or "")
                ref_name = str(rec.get("candidate_ref_name") or _eval_ref_name(tree_run_id, eval_id))
                try:
                    _delete_branch(repo_root=repo_root, ref_name=ref_name)
                except Exception as exc:  # noqa: BLE001
                    _queue_deferred_cleanup(manifest, path=ref_name, kind="ref", reason=str(exc))

                wt_path = rec.get("candidate_worktree_path")
                if wt_path and Path(str(wt_path)).exists():
                    try:
                        cleanup_worktree(repo_root=repo_root, worktree_path=Path(str(wt_path)))
                    except Exception as exc:  # noqa: BLE001
                        _queue_deferred_cleanup(manifest, path=str(wt_path), kind="worktree", reason=str(exc))

            # Record promotions and advance the state.
            if promotion_events:
                manifest.setdefault("events", []).extend(promotion_events)

            completed_depths = state.setdefault("completed_depths", [])
            if current_depth not in completed_depths:
                completed_depths.append(current_depth)

            if promoted_node_ids:
                state["frontier_node_ids"] = promoted_node_ids
                state["current_depth"] = int(current_depth) + 1
                if int(state["current_depth"]) >= int(max_depth):
                    state["stop_reason"] = "max_depth_reached"
            else:
                state["frontier_node_ids"] = []
                if stop_on_empty_frontier:
                    state["stop_reason"] = "empty_frontier"

            _manifest_write(run_root, manifest)
            lock.heartbeat()

        # Phase 5: write summary + validation reports (best-effort).
        try:
            issues = _validate_run(repo_root=repo_root, run_root=run_root, manifest=manifest)
            _write_validation_report(run_root=run_root, issues=issues)
        except Exception:
            pass
        try:
            _write_tree_summary(run_root=run_root, manifest=manifest)
        except Exception:
            pass
        try:
            _write_tree_graph(run_root=run_root, manifest=manifest)
        except Exception:
            pass

        print(
            json.dumps(
                {
                    "tree_run_id": tree_run_id,
                    "run_root": str(run_root),
                    "manifest_path": str(run_root / "manifest.json"),
                    "current_depth": int((manifest.get("state") or {}).get("current_depth") or current_depth),
                    "frontier_size": len(list((manifest.get("state") or {}).get("frontier_node_ids") or [])),
                    "evaluations_total": len((manifest.get("evaluations") or {})),
                    "stop_reason": state.get("stop_reason"),
                },
                indent=2,
            )
        )
        return 0
    finally:
        # Ensure we heartbeat after any state transition.
        if lock is not None:
            lock.heartbeat()
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
