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
    stripped = patch_text.lstrip()
    if stripped.startswith("*** Begin Patch"):
        _apply_codex_cli_patch(Path(worktree_path), stripped)
        return True
    try:
        _run(
            ["git", "apply", "--whitespace=nowarn"],
            cwd=worktree_path,
            input_text=patch_text,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(detail or str(exc)) from exc
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
        try:
            _run(
                ["git", "apply", "--whitespace=nowarn", "--3way"],
                cwd=worktree_path,
                input_text=patch_text,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(detail or str(exc)) from exc


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


def _apply_codex_cli_patch(worktree_path, patch_text):
    lines = (patch_text or "").splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise RuntimeError("Invalid patch: expected '*** Begin Patch' header.")

    idx = 1
    while idx < len(lines):
        header = lines[idx].strip()
        if header == "*** End Patch":
            return

        if header.startswith("*** Update File: "):
            rel_path = header[len("*** Update File: ") :].strip()
            idx += 1
            move_to = None
            if idx < len(lines) and lines[idx].strip().startswith("*** Move to: "):
                move_to = lines[idx].strip()[len("*** Move to: ") :].strip()
                idx += 1
            content, idx = _collect_until_next_header(lines, idx)
            _apply_update_file(worktree_path, rel_path, content, move_to)
            continue

        if header.startswith("*** Add File: "):
            rel_path = header[len("*** Add File: ") :].strip()
            idx += 1
            content, idx = _collect_until_next_header(lines, idx)
            _apply_add_file(worktree_path, rel_path, content)
            continue

        if header.startswith("*** Delete File: "):
            rel_path = header[len("*** Delete File: ") :].strip()
            idx += 1
            _apply_delete_file(worktree_path, rel_path)
            continue

        raise RuntimeError(f"Invalid patch header: {lines[idx]}")

    raise RuntimeError("Invalid patch: missing '*** End Patch'.")


def _collect_until_next_header(lines, idx):
    content = []
    while idx < len(lines):
        if lines[idx].startswith("*** "):
            break
        content.append(lines[idx])
        idx += 1
    return content, idx


def _read_file_preserve_newline(path):
    raw = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    return raw, raw.splitlines(), newline


def _write_file_with_newline(path, lines, newline):
    text = newline.join(lines)
    if lines:
        text += newline
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_codex_hunks(content_lines):
    hunks = []
    current = []
    for line in content_lines:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
                current = []
            continue
        if line.strip() == "*** End of File":
            continue
        if not line:
            continue
        prefix = line[0]
        if prefix in (" ", "+", "-"):
            current.append((prefix, line[1:]))
        else:
            current.append((" ", line))
    if current:
        hunks.append(current)
    return hunks


def _apply_update_file(worktree_path, rel_path, content_lines, move_to):
    src_path = Path(worktree_path) / rel_path
    if not src_path.exists():
        raise RuntimeError(f"Update target not found: {rel_path}")

    raw, file_lines, newline = _read_file_preserve_newline(src_path)
    hunks = _parse_codex_hunks(content_lines)
    if not hunks:
        raise RuntimeError("No valid hunks found in patch.")

    search_from = 0
    for hunk in hunks:
        before = [text for op, text in hunk if op in (" ", "-")]
        after = [text for op, text in hunk if op in (" ", "+")]

        if not before:
            file_lines[search_from:search_from] = after
            search_from += len(after)
            continue

        match_idx = _find_subsequence(file_lines, before, start=search_from)
        if match_idx is None:
            match_idx = _find_subsequence(file_lines, before, start=0)
        if match_idx is None:
            excerpt = "\n".join(before[:10])
            raise RuntimeError(f"Failed to apply hunk to {rel_path}. Hunk starts with:\n{excerpt}")

        file_lines[match_idx : match_idx + len(before)] = after
        search_from = match_idx + len(after)

    _write_file_with_newline(src_path, file_lines, newline)

    if move_to:
        dst_path = Path(worktree_path) / move_to
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.replace(dst_path)


def _apply_add_file(worktree_path, rel_path, content_lines):
    path = Path(worktree_path) / rel_path
    if path.exists():
        raise RuntimeError(f"Add target already exists: {rel_path}")

    out_lines = []
    for line in content_lines:
        if not line:
            continue
        if line[0] != "+":
            raise RuntimeError(f"Invalid add-file line (expected '+'): {line}")
        out_lines.append(line[1:])

    _write_file_with_newline(path, out_lines, "\n")


def _apply_delete_file(worktree_path, rel_path):
    path = Path(worktree_path) / rel_path
    if path.exists():
        path.unlink()


def _find_subsequence(haystack, needle, start=0):
    if not needle:
        return start
    max_i = len(haystack) - len(needle)
    for i in range(start, max_i + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None
