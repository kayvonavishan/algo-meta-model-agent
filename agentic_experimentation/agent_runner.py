import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from git_worktree import apply_patch_text, create_worktree, remove_worktree, sync_working_tree
from llm_clients import build_llm_client, DummyClient
from scoring_hooks import compute_score


SYSTEM_PROMPT = (
    "You are a coding agent improving a meta model. Keep changes small and runnable."
)


def main():
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = _load_json(config_path)

    # Load .env files (config directory first, then repo root)
    _load_env_files([config_path.parent / ".env"])

    repo_root = _resolve_repo_root(config, config_path)
    experiments_root = _resolve_path(repo_root, config.get("experiments_root"))
    worktree_root = _resolve_path(repo_root, config.get("worktree_root"))
    baseline_csv = _resolve_path(repo_root, config.get("baseline_csv"))
    results_csv = Path(config.get("results_csv")).expanduser()
    ideas_path = _resolve_path(repo_root, config.get("ideas_file"))
    if not ideas_path:
        raise RuntimeError("ideas_file must be set in the config.")

    if args.refresh_baseline:
        _refresh_baseline(results_csv, baseline_csv)
        print(f"Baseline copied to {baseline_csv}")
        return

    # Repo-level .env (if present)
    _load_env_files([repo_root / ".env"])

    if not baseline_csv.exists():
        raise RuntimeError(f"Baseline CSV not found at {baseline_csv}. Run with --refresh-baseline.")

    ideas = _load_ideas(ideas_path)
    if not ideas:
        raise RuntimeError(f"ideas_file provided but no ideas found at {ideas_path}")

    llm_config = config.get("llm", {})
    llm = DummyClient(llm_config) if args.dry_run_llm else build_llm_client(llm_config)

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
            idea_note = f"Idea loaded from {ideas_path} ({i + 1}/{len(ideas)})."
            _write_text(exp_dir / "idea_source.txt", idea_note)
            _write_text(exp_dir / "idea.md", idea_text)

            patch_prompt = _render_prompt(
                _read_text(_resolve_path(repo_root, config["prompts"]["patch"])),
                idea_text=idea_text,
                repo_context=repo_context,
            )
            _write_text(exp_dir / "patch_prompt.txt", patch_prompt)
            patch_text = llm.generate(patch_prompt, system_prompt=SYSTEM_PROMPT)
            _write_text(exp_dir / "patch.diff", patch_text)

            patch_applied = apply_patch_text(worktree_path, patch_text)

            sweep_log = exp_dir / "sweep.log"
            sweep_code = _run_sweep(config, config_path, worktree_path, sweep_log)

            candidate_csv = exp_dir / "meta_config_sweep_results.csv"
            if results_csv.exists():
                shutil.copy2(results_csv, candidate_csv)

            score_result = None
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
                "patch_applied": patch_applied,
                "sweep_exit_code": sweep_code,
                "results_csv": str(candidate_csv) if candidate_csv.exists() else None,
                "score": score_result,
            }
            _write_json(exp_dir / "summary.json", summary)
            print(f"Finished iteration {i + 1}/{iterations}: {run_id}")
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
    parser.add_argument("--refresh-baseline", action="store_true")
    parser.add_argument("--dry-run-llm", action="store_true")
    parser.add_argument("--keep-worktrees", action="store_true")
    return parser.parse_args()


def _refresh_baseline(results_csv, baseline_csv):
    if not results_csv.exists():
        raise RuntimeError(f"Results CSV not found at {results_csv}")
    baseline_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results_csv, baseline_csv)


def _run_sweep(config, config_path, worktree_path, log_path, env_extra=None):
    sweep_command = config.get("sweep_command")
    if not sweep_command:
        raise RuntimeError("sweep_command missing in config.")
    sweep_cwd_value = config.get("sweep_cwd", ".")
    sweep_cwd_path = Path(sweep_cwd_value)
    if not sweep_cwd_path.is_absolute():
        sweep_cwd_path = Path(worktree_path) / sweep_cwd_path
    sweep_cwd = sweep_cwd_path
    with open(log_path, "w", encoding="utf-8") as f:
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        if isinstance(sweep_command, list):
            proc = subprocess.run(
                sweep_command,
                cwd=sweep_cwd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        else:
            proc = subprocess.run(
                sweep_command,
                cwd=sweep_cwd,
                shell=True,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
    return proc.returncode


def _build_repo_context(repo_root):
    files = _list_repo_files(repo_root)
    lines = [
        f"Repo root: {repo_root}",
        "Key entrypoint: adaptive_vol_momentum.py",
        "Files:",
        "\n".join(files),
    ]
    return "\n".join(lines)


def _list_repo_files(repo_root):
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except subprocess.SubprocessError:
        return _list_files_fallback(repo_root)


def _list_files_fallback(repo_root):
    try:
        result = subprocess.run(
            ["rg", "--files"],
            cwd=repo_root,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except subprocess.SubprocessError:
        return [str(p.relative_to(repo_root)) for p in Path(repo_root).rglob("*") if p.is_file()]


def _resolve_repo_root(config, config_path):
    repo_root = config.get("repo_root", ".")
    return _resolve_path(config_path.parent, repo_root)


def _resolve_path(base_dir, path_value):
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")


def _render_prompt(template, **kwargs):
    return template.format(**kwargs)


def _load_ideas(path):
    if path.is_dir():
        ideas = []
        for p in sorted(path.glob("*.md")):
            content = _read_text(p).strip()
            if content:
                ideas.append(content)
        return ideas
    content = _read_text(path)
    blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
    return blocks


def _load_env_files(paths):
    for p in paths:
        if not p:
            continue
        try:
            data = _read_text(p)
        except FileNotFoundError:
            continue
        for line in data.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val


if __name__ == "__main__":
    main()
