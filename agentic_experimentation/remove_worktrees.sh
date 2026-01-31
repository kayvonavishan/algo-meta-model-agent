#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./agentic_experimentation/remove_worktrees.sh [--dry-run]

Removes all Git worktrees registered under:
  agentic_experimentation/worktrees

Notes:
  - Skips the current worktree (the one you're running the script from).
  - Runs `git worktree prune` at the end.
EOF
}

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "error: git not found on PATH" >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
worktrees_base="${repo_root%/}/agentic_experimentation/worktrees"

canon_path() {
  local p="$1"

  if command -v cygpath >/dev/null 2>&1; then
    # Git-for-Windows / MSYS2: normalize to POSIX (/c/...) for stable comparisons.
    cygpath -u "$p" 2>/dev/null || printf '%s\n' "$p"
    return 0
  fi

  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$p"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$p" <<'PY'
import os
import sys
print(os.path.realpath(sys.argv[1]))
PY
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    python - "$p" <<'PY'
import os
import sys
print(os.path.realpath(sys.argv[1]))
PY
    return 0
  fi

  printf '%s\n' "$p"
}

base_canon="$(canon_path "$worktrees_base")"
current_canon="$(canon_path "$repo_root")"

run() {
  if [[ "$dry_run" -eq 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

if [[ ! -d "$worktrees_base" ]]; then
  echo "No worktrees directory found at: $worktrees_base"
  exit 0
fi

# 1) Remove worktrees that Git knows about and are under agentic_experimentation/worktrees.
while IFS= read -r line; do
  [[ "$line" == worktree\ * ]] || continue
  wt_path="${line#worktree }"

  wt_canon="$(canon_path "$wt_path")"
  if [[ "$wt_canon" == "$base_canon"* && "$wt_canon" != "$current_canon" ]]; then
    run git worktree remove --force "$wt_path"
  fi
done < <(git worktree list --porcelain)

# 2) Best-effort: also try removing any remaining directories under agentic_experimentation/worktrees.
if command -v find >/dev/null 2>&1; then
  while IFS= read -r -d '' dir; do
    dir_canon="$(canon_path "$dir")"
    if [[ "$dir_canon" == "$current_canon" ]]; then
      continue
    fi
    run git worktree remove --force "$dir" || true
  done < <(find "$worktrees_base" -mindepth 1 -maxdepth 1 -type d -print0)
else
  for dir in "$worktrees_base"/*; do
    [[ -d "$dir" ]] || continue
    dir_canon="$(canon_path "$dir")"
    if [[ "$dir_canon" == "$current_canon" ]]; then
      continue
    fi
    run git worktree remove --force "$dir" || true
  done
fi

# 3) Cleanup stale worktree metadata.
run git worktree prune

