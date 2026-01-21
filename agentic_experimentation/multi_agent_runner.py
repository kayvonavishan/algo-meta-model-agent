import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from agent_runner import (
    _build_repo_context,
    _load_ideas,
    _load_json,
    _read_text,
    _render_prompt,
    _resolve_path,
    _resolve_repo_root,
    _run_sweep,
    _write_json,
    _write_text,
)
from git_worktree import apply_patch_text, create_worktree, remove_worktree, sync_working_tree
from llm_clients import DummyClient, build_llm_client
from scoring_hooks import compute_score


SYSTEM_PROMPT = "You are a coding agent improving a meta model. Keep changes small and runnable."


def main():
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = _load_json(config_path)

    repo_root = _resolve_repo_root(config, config_path)
    experiments_root = _resolve_path(repo_root, config.get("experiments_root"))
    worktree_root = _resolve_path(repo_root, config.get("worktree_root"))
    baseline_csv = _resolve_path(repo_root, config.get("baseline_csv"))
    results_csv = Path(config.get("results_csv")).expanduser()
    ideas_path = _resolve_path(repo_root, config.get("ideas_file"))
    if not ideas_path:
        raise RuntimeError("ideas_file must be set in the config.")
    if not baseline_csv.exists():
        raise RuntimeError(f"Baseline CSV not found at {baseline_csv}. Run with --refresh-baseline.")

    ideas = _load_ideas(ideas_path)
    if not ideas:
        raise RuntimeError(f"ideas_file provided but no ideas found at {ideas_path}")

    agents_cfg = config.get("agents", {})
    coordinator_llm = _build_llm_for_role(agents_cfg, "coordinator", args.dry_run_llm)
    coder_llm = _build_llm_for_role(agents_cfg, "coder", args.dry_run_llm)
    reviewer_llm = _build_llm_for_role(agents_cfg, "reviewer", args.dry_run_llm)

    prompts_cfg = config.get("prompts", {})
    coord_prompt_path = _resolve_path(repo_root, prompts_cfg.get("coordinator"))
    coder_prompt_path = _resolve_path(repo_root, prompts_cfg.get("coder"))
    reviewer_prompt_path = _resolve_path(repo_root, prompts_cfg.get("reviewer"))
    if not (coord_prompt_path and coder_prompt_path and reviewer_prompt_path):
        raise RuntimeError("Coordinator, coder, and reviewer prompts must be set in config['prompts'].")

    iterations_requested = args.iterations
    if iterations_requested is None:
        iterations_requested = config.get("iterations", len(ideas))
    iterations = int(iterations_requested)
    if iterations > len(ideas):
        raise RuntimeError(
            f"Requested {iterations} iterations but only {len(ideas)} ideas in {ideas_path}"
        )
    experiments_root.mkdir(parents=True, exist_ok=True)

    for i in range(iterations):
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{i:02d}"
        exp_dir = experiments_root / run_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        worktree_path = create_worktree(repo_root, worktree_root, run_id)
        try:
            if config.get("base_on_working_tree", False):
                sync_working_tree(repo_root, worktree_path)

            repo_context = _build_repo_context(repo_root)
            idea_text = ideas[i]
            _write_text(exp_dir / "idea.md", idea_text)

            # Coordinator: produce plan
            coord_prompt = _render_prompt(
                _read_text(coord_prompt_path),
                idea_text=idea_text,
                repo_context=repo_context,
            )
            _write_text(exp_dir / "coordinator_prompt.txt", coord_prompt)
            plan_text = coordinator_llm.generate(coord_prompt, system_prompt=SYSTEM_PROMPT)
            _write_text(exp_dir / "plan.md", plan_text)

            # Coder: produce patch
            coder_prompt = _render_prompt(
                _read_text(coder_prompt_path),
                idea_text=idea_text,
                plan_text=plan_text,
                repo_context=repo_context,
            )
            _write_text(exp_dir / "coder_prompt.txt", coder_prompt)
            patch_text = coder_llm.generate(coder_prompt, system_prompt=SYSTEM_PROMPT)
            _write_text(exp_dir / "patch.diff", patch_text)

            patch_applied, patch_error = _apply_patch_safe(worktree_path, patch_text)

            # Reviewer: evaluate idea, plan, and diff
            review_prompt = _render_prompt(
                _read_text(reviewer_prompt_path),
                idea_text=idea_text,
                plan_text=plan_text,
                patch_text=patch_text,
            )
            _write_text(exp_dir / "reviewer_prompt.txt", review_prompt)
            review_text = reviewer_llm.generate(review_prompt, system_prompt=SYSTEM_PROMPT)
            _write_text(exp_dir / "review.md", review_text)
            verdict = _parse_reviewer_verdict(review_text)

            tests_exit = None
            sweep_exit = None
            candidate_csv = exp_dir / "meta_config_sweep_results.csv"
            score_result = None

            proceed = patch_applied and verdict == "APPROVE"

            # Tests
            test_cmd = config.get("test_command")
            if proceed and test_cmd:
                tests_log = exp_dir / "tests.log"
                test_cwd_value = config.get("test_cwd", ".")
                test_cwd = Path(test_cwd_value)
                if not test_cwd.is_absolute():
                    test_cwd = Path(worktree_path) / test_cwd
                tests_exit = _run_command(test_cmd, test_cwd, tests_log)
                proceed = proceed and tests_exit == 0

            # Sweep
            if proceed:
                sweep_log = exp_dir / "sweep.log"
                sweep_exit = _run_sweep(config, config_path, worktree_path, sweep_log)

                if results_csv.exists():
                    shutil.copy2(results_csv, candidate_csv)

                if candidate_csv.exists():
                    score_cfg = config.get("scoring", {})
                    score_result = compute_score(
                        baseline_csv,
                        candidate_csv,
                        score_cfg.get("score_column"),
                        score_cfg.get("higher_is_better", True),
                    )

            summary = {
                "run_id": run_id,
                "idea": idea_text.strip(),
                "plan": plan_text.strip(),
                "patch_applied": patch_applied,
                "patch_error": patch_error,
                "review_verdict": verdict,
                "tests_exit_code": tests_exit,
                "sweep_exit_code": sweep_exit,
                "results_csv": str(candidate_csv) if candidate_csv.exists() else None,
                "score": score_result,
            }
            _write_json(exp_dir / "summary.json", summary)
            print(f"Finished multi-agent iteration {i + 1}/{iterations}: {run_id}")
        finally:
            if not config.get("keep_worktrees", False) and not args.keep_worktrees:
                remove_worktree(repo_root, worktree_path)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="agentic_experimentation/agent_config.json",
        help="Path to agent config JSON.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--dry-run-llm", action="store_true")
    parser.add_argument("--keep-worktrees", action="store_true")
    return parser.parse_args()


def _build_llm_for_role(cfg_map, role, dry_run):
    cfg = cfg_map.get(role, {})
    return DummyClient(cfg) if dry_run else build_llm_client(cfg)


def _apply_patch_safe(worktree_path, patch_text):
    if not patch_text.strip():
        return False, "empty patch"
    try:
        apply_patch_text(worktree_path, patch_text)
        return True, None
    except Exception as exc:  # pylint: disable=broad-except
        return False, str(exc)


def _parse_reviewer_verdict(text):
    first_line = (text or "").strip().splitlines()[0] if text else ""
    upper = first_line.upper()
    if "APPROVE" in upper:
        return "APPROVE"
    if "REJECT" in upper:
        return "REJECT"
    return "UNKNOWN"


def _run_command(cmd, cwd, log_path):
    with open(log_path, "w", encoding="utf-8") as f:
        if isinstance(cmd, list):
            proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=False)
        else:
            proc = subprocess.run(cmd, cwd=cwd, shell=True, stdout=f, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


if __name__ == "__main__":
    main()
