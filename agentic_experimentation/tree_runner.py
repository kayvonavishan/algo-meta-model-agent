from __future__ import annotations

import argparse
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
    for p in (wt_root, cand_root, node_ideas_root, artifacts_root, eval_root):
        _safe_mkdir(p)
    return {
        "wt_root": wt_root,
        "cand_root": cand_root,
        "node_ideas_root": node_ideas_root,
        "artifacts_root": artifacts_root,
        "eval_root": eval_root,
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

    rc = int(cp.wait())

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
    _write(f"\n[exit] code={rc} duration_s={dur:.3f}\n")
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


def _manifest_write(run_root: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now_iso()
    _atomic_write_json(run_root / "manifest.json", manifest)


def _tree_summary_markdown(*, run_root: Path, manifest: dict[str, Any]) -> str:
    state = manifest.get("state") or {}
    run_config = manifest.get("run_config") or {}
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
    lines.append(f"- sweep_config_limit: {run_config.get('sweep_config_limit')}")
    lines.append(f"- ideas_context_strategy: {run_config.get('ideas_context_strategy')}")
    lines.append(f"- stop_reason: {state.get('stop_reason')}")
    lines.append("")

    # Per-depth table.
    lines.append("## Nodes")
    lines.append("")
    lines.append("| depth | node_id | parent | root_rank_score | root_grade | root_should_explore | parent_gate | ok/expected | candidate_rows_used | baseline_csv | candidate_csv | experiment_dir |")
    lines.append("|---:|---:|---:|---:|---|---|---|---|---:|---|---|---|")
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
            f"| {depth} | {nid} | {parent} | {root_rank} | {root_grade} | {root_se} | {gate} | {ok_expected} | {cand_rows} | {baseline_csv} | {cand_csv} | {exp_dir} |"
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
    return "\n".join(lines)


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
        "artifacts": {"root_baseline_provenance": baseline_prov},
        "created_at": _utc_now_iso(),
        "status": "ready",
    }

    tree_run_id = str(args.tree_run_id)
    manifest = {
        "manifest_version": 2,
        "tree_run_id": tree_run_id,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "run_config": {
            "ideas_per_node": int(args.ideas_per_node),
            "max_depth": int(args.max_depth),
            "beam_width": int(args.beam_width),
            "sweep_config_limit": (int(args.sweep_config_limit) if args.sweep_config_limit is not None else None),
            "max_total_idea_evals": (int(args.max_total_idea_evals) if args.max_total_idea_evals is not None else None),
            "stop_on_empty_frontier": bool(args.stop_on_empty_frontier),
            "resume": bool(args.resume),
            "lock_stale_seconds": int(args.lock_stale_seconds),
            "artifact_policy": artifact_policy,
            "node_ideas_root_dir": str(paths["node_ideas_root"]),
            "ideas_context_strategy": "node_plus_ancestors",
            "agent_config_path": str(config_path),
            "agent_config_snapshot": config_obj,
            "runs_root": str(run_root.parent),
            "run_root": str(run_root),
            "wt_root": str(paths["wt_root"]),
            "cand_root": str(paths["cand_root"]),
            "eval_root": str(paths["eval_root"]),
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
        "nodes": {node_id: node_record},
        "evaluations": {},
    }

    _atomic_write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
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
            if bool(args.report_only):
                summary_path = _write_tree_summary(run_root=run_root, manifest=manifest)
            print(
                json.dumps(
                    {
                        "tree_run_id": tree_run_id,
                        "run_root": str(run_root),
                        "manifest_path": str(run_root / "manifest.json"),
                        "tree_summary_path": (str(summary_path) if summary_path else None),
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
        stop_on_empty_frontier = bool(run_config.get("stop_on_empty_frontier", True))
        config_path = Path(run_config.get("agent_config_path") or args.config).resolve()
        agent_cfg = _read_json(config_path)
        if not isinstance(agent_cfg, dict):
            raise RuntimeError(f"Invalid agent config JSON: {config_path}")
        score_column = (agent_cfg.get("scoring") or {}).get("score_column")

        state = manifest.get("state") or {}
        current_depth = int(state.get("current_depth") or 0)
        frontier = list(state.get("frontier_node_ids") or [])
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

        # Count eval records for max_total_idea_evals enforcement.
        evals = manifest.setdefault("evaluations", {})
        completed_evals = [e for e in evals.values() if isinstance(e, dict) and e.get("status") == "completed"]

        stop_reason = state.get("stop_reason")
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

            context_dirs = _collect_context_idea_dirs(manifest, node_id)
            node["context_ideas_dirs"] = context_dirs

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
                        for d in context_dirs:
                            gen_args.extend(["--context-ideas-dir", str(d)])
                        env = dict(os.environ)
                        rc = _run_python_logged(repo_root=repo_root, args=gen_args, env=env, log_path=gen_log_path, echo=True)
                        if rc != 0:
                            raise RuntimeError(f"generate_ideas.py failed for node {node_id} (exit {rc})")
                existing_ideas = _list_markdown_files(node_ideas_dir)

            selected_ideas = existing_ideas[:desired_k]
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

            # Evaluate each idea in deterministic filename order.
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

                # Skip if already evaluated for this node+idea (resume).
                matching = []
                for ev in evals.values():
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("parent_node_id") == node_id and str(ev.get("idea_path") or "") == str(idea_path):
                        matching.append(ev)
                # Prefer the latest eval_id for the same (node, idea).
                existing_eval = None
                if matching:
                    matching.sort(key=lambda e: str(e.get("eval_id") or ""))
                    existing_eval = matching[-1]

                if existing_eval is not None and not bool(args.rerun_evals):
                    # Only skip completed evals. If a prior attempt failed or was interrupted,
                    # rerun using the same eval_id/output paths to preserve determinism.
                    if existing_eval.get("status") == "completed":
                        continue
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
                    sweep_output_dir = eval_output_root
                    sweep_results_csv = eval_output_root / "meta_config_sweep_results.csv"
                    _safe_mkdir(experiment_dir)
                    eval_rec["status"] = "running"
                    eval_rec["error"] = None
                    _manifest_write(run_root, manifest)
                    lock.heartbeat()
                else:
                    next_eval_num = int(state.get("next_eval_id") or 1)
                    eval_id = _format_eval_id(next_eval_num)
                    state["next_eval_id"] = next_eval_num + 1

                    eval_output_root = _ensure_standard_run_dirs(run_root)["eval_root"] / eval_id
                    experiment_dir = eval_output_root / "experiment"
                    sweep_output_dir = eval_output_root
                    sweep_results_csv = eval_output_root / "meta_config_sweep_results.csv"
                    _safe_mkdir(experiment_dir)

                    eval_rec = {
                        "eval_id": eval_id,
                        "parent_node_id": node_id,
                        "depth": current_depth,
                        "idea_path": str(idea_path),
                        "eval_output_root": str(eval_output_root),
                        "experiment_dir": str(experiment_dir),
                        "candidate_worktree_path": None,
                        "candidate_ref_name": _eval_ref_name(tree_run_id, eval_id),
                        "candidate_commit": None,
                        "candidate_results_csv_path": None,
                        "status": "running",
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
                    _manifest_write(run_root, manifest)
                    lock.heartbeat()

                # Dry-run: simulate evaluation without creating worktrees or running sweeps.
                if bool(args.dry_run):
                    try:
                        artifacts_root = _ensure_standard_run_dirs(run_root)["artifacts_root"]
                        parent_baseline_prov = _ensure_artifact_copy(
                            run_root=run_root,
                            source_path=node_baseline_csv_path,
                            dest_name=f"baseline_node_{node_id}.csv",
                        )
                        eval_rec["parent_baseline_provenance"] = parent_baseline_prov

                        scenario = _dry_run_scenario(eval_id=eval_id)
                        candidate_art_path = artifacts_root / f"candidate_eval_{eval_id}.csv"
                        _dry_run_write_candidate_csv(
                            path=candidate_art_path,
                            config_id_limit=(int(sweep_config_limit) if sweep_config_limit is not None else None),
                            ok=bool(scenario.get("strict_complete", True)),
                        )
                        cand_prov = _copy_with_provenance(source_path=candidate_art_path, dest_path=candidate_art_path)
                        eval_rec["candidate_results_provenance"] = cand_prov
                        eval_rec["candidate_results_csv_path"] = str(candidate_art_path)
                        eval_rec["candidate_commit"] = node_commit

                        # Strict completeness counters (simulate failures deterministically).
                        if sweep_config_limit is not None:
                            strict = _strict_completeness_counts(csv_path=candidate_art_path, config_id_limit=int(sweep_config_limit))
                            eval_rec["strict_completeness"] = strict

                        should_explore = bool(scenario.get("should_explore"))
                        grade = scenario.get("grade")
                        primary_delta = float(scenario.get("primary_delta") or 0.0)
                        rank_score = scenario.get("rank_score")

                        eval_rec["parent_relative"] = {
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
                        eval_rec["root_relative"] = {
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

                        eval_rec["status"] = "completed"
                        eval_rec["error"] = None
                    except Exception as exc:  # noqa: BLE001
                        eval_rec["status"] = "failed"
                        eval_rec["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        _manifest_write(run_root, manifest)
                        lock.heartbeat()
                    continue

                # Candidate worktree at node commit.
                cand_dir = _ensure_standard_run_dirs(run_root)["cand_root"] / eval_id
                try:
                    if cand_dir.exists():
                        # Leftover from crash; try to remove first.
                        try:
                            cleanup_worktree(repo_root=repo_root, worktree_path=cand_dir)
                        except Exception as exc:  # noqa: BLE001
                            _queue_deferred_cleanup(manifest, path=str(cand_dir), kind="worktree", reason=str(exc))
                            raise

                    cand_info = create_candidate_worktree(
                        repo_root=repo_root,
                        run_root=run_root,
                        tree_run_id=tree_run_id,
                        eval_id=eval_id,
                        parent_commit=node_commit,
                    )
                    eval_rec["candidate_worktree_path"] = cand_info["worktree_path"]

                    # Run multi-agent loop in-place.
                    env = dict(os.environ)
                    env["AGENTIC_OUTPUT_DIR"] = str(sweep_output_dir)
                    env["AGENTIC_RESULTS_CSV"] = str(sweep_results_csv)
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
                    rc = _run_python_logged(repo_root=repo_root, args=mar_args, env=env, log_path=mar_log_path, echo=True)
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

                    # Commit candidate changes (required identity).
                    cand_wt_path = Path(str(cand_info["worktree_path"]))
                    msg = f"tree_run {tree_run_id} eval {eval_id} idea {idea_path.name}"
                    candidate_commit = _git_commit_all(worktree_path=cand_wt_path, message=msg)
                    eval_rec["candidate_commit"] = candidate_commit

                    # Copy artifacts with provenance.
                    parent_baseline_prov = _ensure_artifact_copy(
                        run_root=run_root,
                        source_path=node_baseline_csv_path,
                        dest_name=f"baseline_node_{node_id}.csv",
                    )
                    cand_prov = _copy_with_provenance(
                        source_path=candidate_csv,
                        dest_path=_ensure_standard_run_dirs(run_root)["artifacts_root"] / f"candidate_eval_{eval_id}.csv",
                    )
                    eval_rec["parent_baseline_provenance"] = parent_baseline_prov
                    eval_rec["candidate_results_provenance"] = cand_prov
                    copied_candidate_csv = Path(str(cand_prov["copied_to_path"]))
                    eval_rec["candidate_results_csv_path"] = str(copied_candidate_csv)

                    # Scoring: parent-relative and root-relative.
                    parent_score = compute_score(
                        Path(str(parent_baseline_prov["copied_to_path"])),
                        copied_candidate_csv,
                        score_column,
                        config_id_limit=sweep_config_limit,
                    )
                    root_baseline_csv = Path(str((manifest.get("root") or {}).get("root_baseline_csv_path") or "")).expanduser()
                    root_score = compute_score(
                        root_baseline_csv,
                        copied_candidate_csv,
                        score_column,
                        config_id_limit=sweep_config_limit,
                    )
                    eval_rec["parent_relative"] = {
                        "recommendation_summary": _recommendation_summary(parent_score),
                        "primary_delta": _primary_delta(parent_score),
                        "baseline_rows_used": parent_score.get("baseline_rows_used"),
                        "candidate_rows_used": parent_score.get("candidate_rows_used"),
                        "score": parent_score.get("score"),
                        "summary_json_path": str(summary_path),
                    }
                    eval_rec["root_relative"] = {
                        "recommendation_summary": _recommendation_summary(root_score),
                        "primary_delta": _primary_delta(root_score),
                        "baseline_rows_used": root_score.get("baseline_rows_used"),
                        "candidate_rows_used": root_score.get("candidate_rows_used"),
                        "score": root_score.get("score"),
                    }

                    # Strict completeness counters.
                    if sweep_config_limit is not None:
                        strict = _strict_completeness_counts(csv_path=copied_candidate_csv, config_id_limit=int(sweep_config_limit))
                        eval_rec["strict_completeness"] = strict
                        if not strict.get("is_complete"):
                            eval_rec["decision"]["passed_gate"] = False
                            eval_rec["decision"]["promotion_reason"] = "incomplete_or_failed_rows"
                        else:
                            eval_rec["decision"]["passed_gate"] = None

                    eval_rec["status"] = "completed"
                    completed_evals.append(eval_rec)
                except Exception as exc:  # noqa: BLE001
                    eval_rec["status"] = "failed"
                    eval_rec["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    _manifest_write(run_root, manifest)
                    lock.heartbeat()

                    # Cleanup candidate worktree unless requested to keep.
                    keep = bool(args.keep_rejected_worktrees) or (eval_rec.get("status") != "completed" and bool(args.keep_failed_artifacts))
                    if not keep and eval_rec.get("candidate_worktree_path"):
                        try:
                            cleanup_worktree(repo_root=repo_root, worktree_path=Path(str(eval_rec["candidate_worktree_path"])))
                        except Exception as exc:  # noqa: BLE001
                            _queue_deferred_cleanup(manifest, path=str(eval_rec["candidate_worktree_path"]), kind="worktree", reason=str(exc))
                            _manifest_write(run_root, manifest)

            expanded_set.add(node_id)
            expanded_by_depth[str(current_depth)] = sorted(expanded_set)
            _manifest_write(run_root, manifest)

        # Phase 4: decide promotions and advance the frontier (global beam).
        stop_reason = state.get("stop_reason")
        if not stop_reason:
            artifacts_root = _ensure_standard_run_dirs(run_root)["artifacts_root"]
            evals = manifest.setdefault("evaluations", {})
            nodes = manifest.setdefault("nodes", {})

            depth_eval_recs: list[dict[str, Any]] = []
            for rec in evals.values():
                if not isinstance(rec, dict):
                    continue
                try:
                    if int(rec.get("depth") or -1) != current_depth:
                        continue
                except Exception:
                    continue
                if str(rec.get("parent_node_id") or "") not in set(frontier):
                    continue
                depth_eval_recs.append(rec)

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

                    node_record = {
                        "node_id": new_node_id,
                        "parent_node_id": parent_node_id,
                        "depth": int(current_depth) + 1,
                        "commit": cand_commit,
                        "ref_name": str(node_wt["ref_name"]),
                        "worktree_path": str(node_wt["worktree_path"]),
                        "baseline_results_csv_path": str(rec.get("candidate_results_csv_path") or ""),
                        "node_ideas_dir": str(node_ideas_dir),
                        "idea_chain": idea_chain,
                        "created_at": _utc_now_iso(),
                        "status": "ready",
                        "artifacts": {
                            "promoted_from_eval_id": eval_id,
                            "promoted_candidate_results_provenance": rec.get("candidate_results_provenance"),
                        },
                    }
                    nodes[new_node_id] = node_record
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
