import os
import shutil
import subprocess
from pathlib import Path


def create_worktree(repo_root, worktree_root, run_id, base_ref="HEAD"):
    worktree_root = Path(worktree_root)
    worktree_root.mkdir(parents=True, exist_ok=True)
    worktree_path = worktree_root / run_id
    _run(["git", "worktree", "add", "--detach", str(worktree_path), base_ref], cwd=repo_root)
    return worktree_path


def remove_worktree(repo_root, worktree_path):
    if not Path(worktree_path).exists():
        return
    _run(["git", "worktree", "remove", "--force", str(worktree_path)], cwd=repo_root)


def apply_patch_text(worktree_path, patch_text):
    if not patch_text.strip():
        return False
    _run(
        ["git", "apply", "--whitespace=nowarn"],
        cwd=worktree_path,
        input_text=patch_text,
    )
    return True


def sync_working_tree(repo_root, worktree_path):
    _apply_diff(repo_root, worktree_path, cached=False)
    _apply_diff(repo_root, worktree_path, cached=True)
    _copy_untracked(repo_root, worktree_path)


def _apply_diff(repo_root, worktree_path, cached):
    args = ["git", "diff"]
    if cached:
        args.append("--cached")
    result = _run(args, cwd=repo_root, capture_output=True)
    patch_text = result.stdout
    if patch_text.strip():
        _run(
            ["git", "apply", "--whitespace=nowarn"],
            cwd=worktree_path,
            input_text=patch_text,
        )


def _copy_untracked(repo_root, worktree_path):
    result = _run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True)
    for line in result.stdout.splitlines():
        if not line.startswith("?? "):
            continue
        rel_path = line[3:].strip()
        src = Path(repo_root) / rel_path
        dst = Path(worktree_path) / rel_path
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _run(cmd, cwd, input_text=None, capture_output=False):
    kwargs = {
        "cwd": cwd,
        "text": True,
        "check": True,
    }
    if input_text is not None:
        kwargs["input"] = input_text
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    return subprocess.run(cmd, **kwargs)
