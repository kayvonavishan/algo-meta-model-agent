from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_IDEA_FILE_RE = re.compile(r"^(?P<num>\d{3})_(?P<name>.+)\.md$", re.IGNORECASE)
_DEFAULT_HISTORY_MAX_CHARS = 20000


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def _read_json_file(path: Path) -> Any:
    raw = _read_text(path)
    # Be tolerant of UTF-8 BOM (common when files are edited on Windows).
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    return json.loads(raw)


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8", errors="replace")).hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8", errors="replace")
    os.replace(tmp, path)


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _to_jsonable(value: Any, *, depth: int = 0, max_depth: int = 8) -> Any:
    if depth >= max_depth:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, depth=depth + 1, max_depth=max_depth) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v, depth=depth + 1, max_depth=max_depth) for v in value]
    try:
        return _to_jsonable(getattr(value, "__dict__"), depth=depth + 1, max_depth=max_depth)
    except Exception:
        return str(value)


def _write_raw_llm_messages(
    *,
    idea_number: int,
    prompt_path: Path,
    model: Optional[str],
    node_id: Optional[str],
    conversation_id: Optional[str],
    turn_id: Optional[str],
    provider_session_id: Optional[str],
    provider_response_id: Optional[str],
    provider_metadata_debug: Optional[dict[str, Any]],
    raw_messages: Any,
) -> Path:
    raw_dir = _idea_log_root() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = _now_tag()
    name = f"claude_sdk_messages_{idea_number:03d}_{suffix}.jsonl"
    path = raw_dir / name
    record = {
        "ts": _utc_now_iso(),
        "provider": "claude_agent_sdk",
        "model": str(model) if model else None,
        "idea_number": int(idea_number),
        "prompt_path": str(prompt_path),
        "node_id": str(node_id) if node_id else None,
        "conversation_id": str(conversation_id) if conversation_id else None,
        "turn_id": str(turn_id) if turn_id else None,
        "provider_session_id": provider_session_id,
        "provider_response_id": provider_response_id,
        "provider_metadata_debug": provider_metadata_debug or {},
        "raw_messages": _to_jsonable(raw_messages),
    }
    _append_jsonl(path, record)
    return path


def _normalize_conversation_mode(raw: Optional[str]) -> str:
    value = str(raw or "off").strip().lower()
    if value in {"off", "auto", "native", "replay"}:
        return value
    return "off"


def _normalize_history_max_chars(raw: Any, *, default: int = _DEFAULT_HISTORY_MAX_CHARS) -> int:
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except Exception:
        return int(default)
    if value < -1:
        return int(default)
    return value


def _state_history_max_chars(state: dict[str, Any], *, default: int = _DEFAULT_HISTORY_MAX_CHARS) -> int:
    return _normalize_history_max_chars(state.get("history_max_chars"), default=default)


def _provider_supports_native_continuation(provider: Optional[dict[str, Any]]) -> bool:
    if not isinstance(provider, dict):
        return False
    provider_name = str(provider.get("name") or "").strip().lower()
    explicit = provider.get("supports_native_continuation")
    explicit_bool = bool(explicit) if isinstance(explicit, bool) else False
    inferred = provider_name in {"claude_agent_sdk", "claude", "openai_responses", "openai"}
    return bool(explicit_bool or inferred)


def _resolve_mode_used_for_turn(
    *,
    requested_mode: str,
    conversation_state: Optional[dict[str, Any]],
) -> str:
    if requested_mode == "off":
        return "off"
    if requested_mode == "replay":
        return "replay"
    provider = (conversation_state or {}).get("provider") if isinstance(conversation_state, dict) else None
    if _provider_supports_native_continuation(provider if isinstance(provider, dict) else None):
        return "native"
    return "replay"


def _claude_native_continuation_params(
    *,
    mode_used: str,
    conversation_state: Optional[dict[str, Any]],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "continue_conversation": False,
        "resume_session_id": None,
        "fork_session": False,
    }
    if mode_used != "native" or not isinstance(conversation_state, dict):
        return params

    provider = conversation_state.get("provider")
    if not _provider_supports_native_continuation(provider if isinstance(provider, dict) else None):
        return params

    provider_dict = provider if isinstance(provider, dict) else {}
    session_id = str(provider_dict.get("session_id") or "").strip() or None
    if not session_id:
        return params

    parent_conversation_id = str(conversation_state.get("parent_conversation_id") or "").strip() or None
    branch_session_initialized = bool(provider_dict.get("branch_session_initialized", False))
    fork_session = bool(parent_conversation_id and not branch_session_initialized)
    params.update(
        {
            "continue_conversation": True,
            "resume_session_id": session_id,
            "fork_session": fork_session,
        }
    )
    return params


def _default_conversation_state(*, requested_mode: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "conversation_id": None,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "requested_mode": requested_mode,
        "latest_turn_id": None,
        "next_turn_index": 1,
        "fork_from_turn_id": None,
        "history_window_turns": 12,
        "history_max_chars": _DEFAULT_HISTORY_MAX_CHARS,
        "compact_summary_text": "",
        "compact_summary_updated_at": None,
        "provider": {
            "name": "claude_agent_sdk",
            "supports_native_continuation": True,
            "session_id": None,
            "last_response_id": None,
            "branch_session_initialized": False,
        },
        "turns": [],
        "idea_file_turn_map": {},
        "operation_id_turn_map": {},
    }


def _load_conversation_state(path: Path, *, requested_mode: str) -> dict[str, Any]:
    if path.exists():
        try:
            parsed = _read_json_file(path)
        except Exception as exc:  # noqa: BLE001
            corrupt_path = path.with_suffix(path.suffix + f".corrupt_{_now_tag()}")
            try:
                os.replace(path, corrupt_path)
                print(
                    f"[conversation] warning: invalid state JSON recovered: {path} -> {corrupt_path} ({type(exc).__name__})",
                    file=sys.stderr,
                )
            except Exception:
                print(
                    f"[conversation] warning: invalid state JSON (unable to move): {path} ({type(exc).__name__})",
                    file=sys.stderr,
                )
            parsed = None
        if isinstance(parsed, dict):
            state = dict(parsed)
        else:
            state = _default_conversation_state(requested_mode=requested_mode)
    else:
        state = _default_conversation_state(requested_mode=requested_mode)

    state.setdefault("schema_version", 1)
    state.setdefault("created_at", _utc_now_iso())
    state["updated_at"] = _utc_now_iso()
    state["requested_mode"] = requested_mode
    state.setdefault("latest_turn_id", None)
    state.setdefault("next_turn_index", 1)
    state.setdefault("fork_from_turn_id", None)
    state.setdefault("history_window_turns", 12)
    state.setdefault("history_max_chars", _DEFAULT_HISTORY_MAX_CHARS)
    state["history_max_chars"] = _normalize_history_max_chars(state.get("history_max_chars"), default=_DEFAULT_HISTORY_MAX_CHARS)
    state.setdefault("compact_summary_text", "")
    state.setdefault("compact_summary_updated_at", None)
    state.setdefault("provider", {})
    if not isinstance(state["provider"], dict):
        state["provider"] = {}
    state["provider"].setdefault("name", "claude_agent_sdk")
    state["provider"]["supports_native_continuation"] = _provider_supports_native_continuation(state["provider"])
    state["provider"].setdefault("session_id", None)
    state["provider"].setdefault("last_response_id", None)
    state["provider"].setdefault("branch_session_initialized", False)
    state.setdefault("turns", [])
    if not isinstance(state["turns"], list):
        state["turns"] = []
    state.setdefault("idea_file_turn_map", {})
    if not isinstance(state["idea_file_turn_map"], dict):
        state["idea_file_turn_map"] = {}
    state.setdefault("operation_id_turn_map", {})
    if not isinstance(state["operation_id_turn_map"], dict):
        state["operation_id_turn_map"] = {}

    # Ensure next_turn_index stays monotonic when resuming.
    try:
        next_idx = int(state.get("next_turn_index") or 1)
    except Exception:
        next_idx = 1
    if next_idx < 1:
        next_idx = 1
    for t in state["turns"]:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("turn_id") or "")
        m = re.match(r"^turn_(\d+)$", tid)
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= next_idx:
            next_idx = idx + 1
        op_id = str(t.get("operation_id") or "").strip()
        if not op_id:
            op_id = _derive_operation_id_from_turn(t)
            t["operation_id"] = op_id
        if op_id and tid:
            state["operation_id_turn_map"][op_id] = tid
    state["next_turn_index"] = next_idx
    return state


def _conversation_next_turn_id(state: dict[str, Any]) -> str:
    idx = int(state.get("next_turn_index") or 1)
    turn_id = f"turn_{idx:04d}"
    state["next_turn_index"] = idx + 1
    return turn_id


def _derive_operation_id_from_turn(turn: dict[str, Any]) -> str:
    prompt_hash = str(turn.get("prompt_hash") or "").strip()
    out_rel = str(turn.get("output_idea_path_relative_to_repo") or "").strip()
    if not out_rel:
        out_rel = str(turn.get("output_idea_path_resolved") or "").strip()
    if not out_rel:
        out_rel = str(turn.get("output_idea_path") or "").strip()
    if prompt_hash and out_rel:
        return _sha256_text(f"{prompt_hash}|{out_rel}")
    out_hash = str(turn.get("output_hash") or "").strip()
    return _sha256_text(f"{out_hash}|{out_rel}")


def _clip_text(value: str, *, max_chars: int) -> str:
    txt = str(value or "")
    if max_chars < 0:
        return txt
    if max_chars == 0:
        return ""
    if len(txt) <= max_chars:
        return txt
    return txt[: max(0, max_chars - 17)] + "\n...[truncated]..."


def _turn_summary_line(turn: dict[str, Any]) -> str:
    tid = str(turn.get("turn_id") or "?")
    title = str(turn.get("output_title") or "").strip()
    out_path = str(turn.get("output_idea_path_relative_to_repo") or turn.get("output_idea_path") or "").strip()
    if not title:
        if out_path:
            title = Path(out_path).name
        else:
            title = "idea"
    output_hash = str(turn.get("output_hash") or "").strip()
    output_hash_short = output_hash[:10] if output_hash else "n/a"
    if out_path:
        return f"- {tid}: {title} ({out_path}) output_hash={output_hash_short}"
    return f"- {tid}: {title} output_hash={output_hash_short}"


def _build_compact_summary_from_turns(*, turns: list[dict[str, Any]], max_chars: int) -> str:
    if not turns:
        return ""
    if max_chars == 0:
        return ""
    header = "Earlier branch history summary (oldest -> newest):\n"
    lines = [_turn_summary_line(t) for t in turns if isinstance(t, dict)]
    if not lines:
        return ""
    if max_chars < 0:
        return header + "\n".join(lines)
    body_lines: list[str] = []
    current = header
    for line in reversed(lines):
        candidate = current + line + "\n"
        if len(candidate) > max_chars:
            continue
        body_lines.insert(0, line)
        current = candidate
    if not body_lines:
        fallback = _clip_text(lines[-1], max_chars=max(0, max_chars - len(header)))
        return header + fallback
    return header + "\n".join(body_lines)


def _merge_compact_summary(*, existing: str, new_chunk: str, max_chars: int) -> str:
    if not new_chunk:
        return _clip_text(existing or "", max_chars=max_chars)
    if not existing:
        return _clip_text(new_chunk, max_chars=max_chars)
    merged = (existing.rstrip() + "\n" + new_chunk.strip()).strip()
    return _clip_text(merged, max_chars=max_chars)


def _compact_conversation_state(
    *,
    state: dict[str, Any],
    keep_recent_turns: int,
    summary_max_chars: int,
) -> None:
    turns = state.get("turns") or []
    if not isinstance(turns, list):
        turns = []
    keep = max(int(keep_recent_turns), 1)
    if len(turns) <= keep:
        return

    dropped = [t for t in turns[:-keep] if isinstance(t, dict)]
    kept = [t for t in turns[-keep:] if isinstance(t, dict)]
    state["turns"] = kept
    if kept and isinstance(kept[-1], dict):
        state["latest_turn_id"] = kept[-1].get("turn_id")
    else:
        state["latest_turn_id"] = None

    existing_summary = str(state.get("compact_summary_text") or "")
    summary_half_budget = -1 if summary_max_chars < 0 else max(summary_max_chars // 2, 0)
    dropped_summary = _build_compact_summary_from_turns(turns=dropped, max_chars=summary_half_budget)
    state["compact_summary_text"] = _merge_compact_summary(
        existing=existing_summary,
        new_chunk=dropped_summary,
        max_chars=summary_max_chars,
    )
    state["compact_summary_updated_at"] = _utc_now_iso()


def _find_turn_index_by_operation_id(state: dict[str, Any], operation_id: str) -> Optional[int]:
    turns = state.get("turns") or []
    if not isinstance(turns, list):
        return None
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        if str(turn.get("operation_id") or "") == str(operation_id):
            return i
    return None


def _render_conversation_replay_block(
    *,
    conversation_state: dict[str, Any],
    max_turns: int,
    max_chars: int,
) -> str:
    turns = conversation_state.get("turns") or []
    if not isinstance(turns, list) or not turns:
        return ""

    try:
        max_turns_i = max(0, int(max_turns))
    except Exception:
        max_turns_i = 0
    if max_turns_i <= 0:
        return ""
    if max_chars == 0:
        return ""

    filtered_turns = [t for t in turns if isinstance(t, dict)]
    if not filtered_turns:
        return ""
    turns_slice = filtered_turns[-max_turns_i:]
    older_turns = filtered_turns[:-max_turns_i] if len(filtered_turns) > max_turns_i else []

    compact_summary = str(conversation_state.get("compact_summary_text") or "").strip()
    summary_half_budget = -1 if max_chars < 0 else max(max_chars // 2, 0)
    if older_turns:
        summary_from_older = _build_compact_summary_from_turns(
            turns=older_turns,
            max_chars=summary_half_budget,
        )
        compact_summary = _merge_compact_summary(
            existing=compact_summary,
            new_chunk=summary_from_older,
            max_chars=summary_half_budget,
        )
    conversation_state["compact_summary_text"] = compact_summary
    conversation_state["compact_summary_updated_at"] = _utc_now_iso()

    blocks: list[str] = []
    for turn in turns_slice:
        tid = str(turn.get("turn_id") or "")
        ts = str(turn.get("timestamp") or "")
        out_path = str(turn.get("output_idea_path") or "")
        out_text = str(turn.get("output_text") or "")
        blocks.append(
            "\n".join(
                [
                    f"[{tid}] {ts}".rstrip(),
                    (f"output_idea_path: {out_path}" if out_path else "output_idea_path: (unknown)"),
                    "assistant_output:",
                    out_text.strip(),
                ]
            ).strip()
        )

    header = "\n".join(
        [
            "===== IDEA CONVERSATION MEMORY (REPLAY) =====",
            "Use this for continuity with prior idea-generation turns in this branch.",
            "Prefer non-duplicate ideas and build on prior reasoning where useful.",
            "",
        ]
    )

    summary_block = ""
    if compact_summary:
        summary_block = "\n".join(["compact_summary:", compact_summary.strip()]).strip()

    base = header
    if summary_block:
        base = (header + summary_block + "\n\n")
    if max_chars > 0 and len(base) > max_chars:
        clipped = _clip_text(summary_block, max_chars=max(0, max_chars - len(header) - 32))
        base = header + clipped + "\n\n"

    # Keep newest content under budget; trim oldest first.
    selected: list[str] = []
    current = base
    for block in reversed(blocks):
        candidate = current + ("\n\n" if selected else "") + block
        if max_chars > 0 and len(candidate) > max_chars:
            continue
        selected.insert(0, block)
        current = candidate
    if not selected:
        # Always include at least one recent turn (truncated) for continuity.
        if max_chars > 0 and len(base) >= max_chars:
            # Prioritize one recent turn over summary text when budget is tight.
            base = header
        last_budget = -1 if max_chars < 0 else max(0, max_chars - len(base) - 64)
        last = _clip_text(blocks[-1], max_chars=last_budget)
        if not str(last).strip():
            first_line = str(blocks[-1]).splitlines()[0] if str(blocks[-1]).splitlines() else "[latest turn]"
            if max_chars < 0 or len(base + first_line) <= max_chars:
                last = first_line
        selected = [last]

    body = "\n\n".join(selected)
    return (base + body + "\n===== END IDEA CONVERSATION MEMORY =====\n").strip() + "\n"


def _write_debug_log(*, resolved_cli_path: str, cwd: Path, model: Optional[str], prompt_len: int, stderr_lines: List[str], exc: Exception) -> Path:
    log_dir = _idea_log_root() / "claude_debug"
    log_path = log_dir / f"claude_debug_{_now_tag()}.log"
    body = "\n".join(
        [
            "Claude Agent SDK debug log",
            f"time: {_now_tag()}",
            f"cli_path: {resolved_cli_path}",
            f"cwd: {cwd}",
            f"model: {model!r}",
            f"prompt_len: {prompt_len}",
            "",
            f"exception: {type(exc).__name__}: {exc}",
            "",
            "stderr (captured):",
            *(stderr_lines if stderr_lines else ["<empty>"]),
            "",
        ]
    )
    _write_text(log_path, body)
    return log_path


def _idea_log_root() -> Path:
    raw = os.getenv("AGENTIC_IDEA_LOG_DIR") or os.getenv("AGENTIC_LOG_DIR") or ""
    raw = str(raw).strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        p.mkdir(parents=True, exist_ok=True)
        return p
    default_dir = Path(__file__).resolve().parents[1] / "logs" / "idea_generation"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir


def _load_env_files(paths: Iterable[Path]) -> None:
    for p in paths:
        try:
            data = _read_text(p)
        except FileNotFoundError:
            continue
        for line in data.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val


def _resolve_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "META_MODEL_GUIDE.md").exists() and (p / "adaptive_vol_momentum.py").exists():
            return p
    return start.resolve()


def _resolve_context_source_root(*, repo_root: Path, baseline_ctx: Optional[dict[str, Any]]) -> Path:
    """
    Pick the location that represents the latest model state for context file paths:
    - root/original node -> repo root
    - expanded nodes -> node worktree path (or parent-node worktree fallback)
    """
    if not isinstance(baseline_ctx, dict):
        return repo_root

    depth = baseline_ctx.get("depth")
    try:
        depth_i = int(depth) if depth is not None else None
    except Exception:
        depth_i = None

    node_wt = str(baseline_ctx.get("node_worktree_path") or "").strip()
    parent_wt = str(baseline_ctx.get("parent_node_worktree_path") or "").strip()

    if depth_i == 0:
        return repo_root

    for raw in (node_wt, parent_wt):
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p.exists() and p.is_dir():
            return p

    # Fallback: derive from manifest if tree identifiers are present.
    tree_run_id = str(baseline_ctx.get("tree_run_id") or "").strip()
    node_id = str(baseline_ctx.get("node_id") or "").strip()
    if tree_run_id and node_id:
        manifest_path = (
            repo_root
            / "agentic_experimentation"
            / "worktrees"
            / "tree_runs"
            / tree_run_id
            / "manifest.json"
        )
        if manifest_path.exists():
            try:
                manifest = _read_json_file(manifest_path)
            except Exception:
                manifest = None
            if isinstance(manifest, dict):
                nodes = manifest.get("nodes") or {}
                if isinstance(nodes, dict):
                    nrec = nodes.get(node_id) or {}
                    if isinstance(nrec, dict):
                        p = Path(str(nrec.get("worktree_path") or "")).expanduser()
                        if p.exists() and p.is_dir():
                            return p

    return repo_root


def _render_meta_model_context_block(*, repo_root: Path, baseline_ctx: Optional[dict[str, Any]]) -> str:
    context_root = _resolve_context_source_root(repo_root=repo_root, baseline_ctx=baseline_ctx)
    files: list[tuple[str, str, Path]] = [
        (
            "META_MODEL_GUIDE.md",
            "High-level guide for the meta model design, assumptions, and workflow.",
            context_root / "META_MODEL_GUIDE.md",
        ),
        (
            "adaptive_vol_momentum.py",
            "Primary meta model implementation and sweep/backtest driver.",
            context_root / "adaptive_vol_momentum.py",
        ),
        (
            "scoring.py",
            "Performance scoring and summary metric computation utilities.",
            context_root / "scoring.py",
        ),
        (
            "selection.py",
            "Selection logic used to choose top strategies/models each period.",
            context_root / "selection.py",
        ),
    ]

    lines: list[str] = []
    lines.append("===== META MODEL CONTEXT =====")
    for name, desc, path in files:
        lines.append(name)
        lines.append(f" - description: {desc}")
        lines.append(f" - location: {path}")
    return "\n".join(lines).rstrip() + "\n"


def _collect_idea_files(ideas_dir: Path, completed_dir: Path) -> List[Path]:
    paths: List[Path] = []

    if ideas_dir.exists():
        for p in ideas_dir.glob("*.md"):
            # Include all idea markdowns in ideas/ (tested or not), even if they don't follow
            # the ddd_name.md naming convention yet.
            paths.append(p)

    if completed_dir.exists():
        for p in completed_dir.glob("*.md"):
            paths.append(p)

    def _sort_key(p: Path) -> Tuple[int, str]:
        m = _IDEA_FILE_RE.match(p.name)
        if not m:
            return (10**9, p.name.lower())
        return (int(m.group("num")), p.name.lower())

    # De-dupe (in case a file is reachable via both dirs in weird setups).
    uniq = sorted(set(paths), key=_sort_key)
    return uniq


def _collect_idea_files_from_dirs(idea_dirs: List[Path]) -> List[Path]:
    paths: List[Path] = []
    for d in idea_dirs:
        d = Path(d)
        if d.exists():
            paths.extend(d.glob("*.md"))
        completed = d / "completed"
        if completed.exists():
            paths.extend(completed.glob("*.md"))

    def _sort_key(p: Path) -> Tuple[int, str]:
        m = _IDEA_FILE_RE.match(p.name)
        if not m:
            return (10**9, p.name.lower())
        return (int(m.group("num")), p.name.lower())

    return sorted(set(paths), key=_sort_key)


def _next_idea_number(ideas_dir: Path, completed_dir: Path) -> int:
    max_num = 0
    for p in _collect_idea_files(ideas_dir, completed_dir):
        m = _IDEA_FILE_RE.match(p.name)
        if not m:
            continue
        max_num = max(max_num, int(m.group("num")))
    return max_num + 1


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate idea markdowns for the meta model.")
    parser.add_argument(
        "--ideas-dir",
        default=None,
        help="Directory to write new idea markdowns to (default: agentic_experimentation/ideas).",
    )
    parser.add_argument(
        "--completed-dir",
        default=None,
        help="Directory holding completed ideas for numbering (default: <ideas-dir>/completed).",
    )
    parser.add_argument(
        "--context-ideas-dir",
        action="append",
        default=[],
        help=(
            "Directory to include as prior-ideas context (repeatable). "
            "Each dir is scanned for *.md and <dir>/completed/*.md."
        ),
    )
    parser.add_argument(
        "--baseline-context-json",
        default=None,
        help=(
            "Optional JSON file describing baseline metrics + artifact paths for the current node. "
            "If provided, a short 'BASELINE ARTIFACTS' block is appended to the prompt so the LLM "
            "can explore artifacts as needed without loading everything into context."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override config.json count (number of ideas to generate).",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="Override config.json max_context_chars.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override config.json model (Claude Code model string).",
    )
    parser.add_argument(
        "--cli-path",
        default=None,
        help="Override config.json cli_path (path to Claude Code CLI).",
    )
    parser.add_argument(
        "--conversation-state-in",
        default=None,
        help="Optional JSON path for conversation state input (for branch continuation).",
    )
    parser.add_argument(
        "--conversation-state-out",
        default=None,
        help="Optional JSON path for conversation state output.",
    )
    parser.add_argument(
        "--conversation-mode",
        default="off",
        choices=["off", "auto", "native", "replay"],
        help="Conversation continuation mode.",
    )
    parser.add_argument(
        "--fork-from-turn-id",
        default=None,
        help="Optional parent checkpoint turn id used when this branch was forked.",
    )
    parser.add_argument(
        "--emit-turn-log",
        default=None,
        help="Optional JSONL path to append per-turn metadata records.",
    )
    parser.add_argument(
        "--conversation-history-window-turns",
        type=int,
        default=None,
        help="Max recent conversation turns to replay when continuation is enabled.",
    )
    parser.add_argument(
        "--conversation-history-max-chars",
        type=int,
        default=None,
        help="Replay memory char budget (-1 = unbounded, 0 = disable replay memory, None = default).",
    )
    parser.add_argument(
        "--idea-log-raw-messages",
        dest="idea_log_raw_messages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log raw Claude Agent SDK messages for each idea (default: on). Use --no-idea-log-raw-messages to disable.",
    )
    return parser.parse_args(argv)


def _resolve_cli_path(repo_root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _render_baseline_context_block(*, baseline_ctx: dict[str, Any], repo_root: Path) -> str:
    agentic_root = repo_root / "agentic_experimentation"
    docs_root = agentic_root / "artifact_docs"
    docs_root_exists = docs_root.exists()

    sweep_limit = baseline_ctx.get("sweep_config_limit")
    try:
        sweep_limit_i = int(sweep_limit) if sweep_limit is not None else None
    except Exception:
        sweep_limit_i = None

    baseline_metrics = baseline_ctx.get("baseline_metrics") or {}
    if not isinstance(baseline_metrics, dict):
        baseline_metrics = {}

    sweep_results_csv = str(baseline_ctx.get("sweep_results_csv") or "").strip()
    baseline_source_csv = str(baseline_ctx.get("baseline_source_csv") or "").strip()
    avg_plots_dir = str(baseline_ctx.get("avg_trade_return_plots_dir") or "").strip()

    lines: list[str] = []
    lines.append("-----------------------")
    lines.append("")

    def _display_path(raw: str) -> str:
        s = str(raw or "").strip()
        if not s:
            return ""
        p = Path(s).expanduser()
        if not p.is_absolute():
            return s
        try:
            rel = p.resolve().relative_to(repo_root.resolve())
            return str(rel).replace("/", "\\")
        except Exception:
            return str(p)

    tree_run_id = str(baseline_ctx.get("tree_run_id") or "").strip()
    node_id = str(baseline_ctx.get("node_id") or "").strip()
    parent_node_id = str(baseline_ctx.get("parent_node_id") or "").strip()
    depth = baseline_ctx.get("depth")
    try:
        depth_i = int(depth) if depth is not None else None
    except Exception:
        depth_i = None

    changes_applied = baseline_ctx.get("changes_applied_count")
    try:
        changes_applied_i = int(changes_applied) if changes_applied is not None else None
    except Exception:
        changes_applied_i = None

    lines.append("Current Status:")
    hdr = []
    if tree_run_id:
        hdr.append(f"tree_run_id={tree_run_id}")
    if node_id:
        hdr.append(f"node_id={node_id}")
    if depth_i is not None:
        hdr.append(f"depth={depth_i}")
    if parent_node_id:
        hdr.append(f"parent_node_id={parent_node_id}")
    if changes_applied_i is not None:
        hdr.append(f"changes_applied_to_original={changes_applied_i}")
    if hdr:
        lines.append("- " + " ".join(hdr))
    idea_chain = baseline_ctx.get("idea_chain") or []
    if isinstance(idea_chain, list) and idea_chain:
        lines.append("- applied_ideas (chronological file pointers):")
        for p in idea_chain:
            lines.append(f"  - {p}")
    else:
        lines.append("- applied_ideas: (none; original/root model)")
    lines.append("")

    lines.append("Artifacts & How To Interpret Them:")
    lines.append(f"- Static docs root: {docs_root} (exists={str(bool(docs_root_exists)).lower()})")
    lines.append("")

    col_defs = docs_root / "meta_config_sweep_results_columns.txt"
    overview_doc = docs_root / "avg_trade_return_plots" / "README.txt"

    # 1) Sweep results table (per node).
    agentic_output_root = str(baseline_ctx.get("agentic_output_root") or "").strip()
    is_initial_root = bool(depth_i == 0 and not parent_node_id)
    root_baseline_display = baseline_source_csv if baseline_source_csv else sweep_results_csv
    sweep_display = root_baseline_display if is_initial_root else sweep_results_csv
    sweep_exists = Path(sweep_display).expanduser().exists() if sweep_display else False
    lines.append("Output: meta_config_sweep_results.csv")
    lines.append("- What it stores: Per-config sweep results; each row is one meta-model backtest for a single parameter set (`config_id`).")
    lines.append("- Used for: Comparing parameter sets and computing the averaged metrics/deltas used to judge/promote ideas.")
    lines.append("- Location (current node): " + (_display_path(sweep_display) if sweep_display else "(unknown / not available)"))
    if not is_initial_root and agentic_output_root:
        lines.append(f"- Location (pattern): {Path(agentic_output_root) / 'run_0' / 'meta_config_sweep_results.csv'}")
    else:
        lines.append("- Location (pattern): <agentic_output_root>/run_0/meta_config_sweep_results.csv")
    lines.append(f"- Exists (current node path): {str(bool(sweep_exists)).lower()}")
    lines.append(f"- Column definitions: {col_defs} (exists={str(bool(col_defs.exists())).lower()})")
    lines.append("  - Format: `column_name: description` (search by column name).")
    lines.append("  - Note: metrics families include `core_*`, `rel_*`, `stab_*`, `trade_*`, `sig_*`.")
    lines.append("- Granularity: 1 CSV per node/run; rows are per parameter set tested (`config_id`).")
    lines.append("")

    # 2) Per-config diagnostics (per node, per config_id) — available only
    # after candidate idea sweeps have been executed.
    avg_exists = Path(avg_plots_dir).expanduser().exists() if avg_plots_dir else False
    lines.append("Output: avg_trade_return_plots/")
    if not avg_exists:
        lines.append("- Availability: not available yet for this node.")
        lines.append("- Why: these diagnostics are produced only after running a candidate idea sweep.")
        lines.append("- Location (current node): (not generated yet)")
        lines.append("- Location (pattern): <agentic_output_root>/run_0/avg_trade_return_plots/")
        lines.append(f"- Overview doc: {overview_doc} (exists={str(bool(overview_doc.exists())).lower()})")
        lines.append("- Note: once available, files are per `config_id` with suffixes `_config_000`, `_config_001`, ...")
        lines.append("")
    else:
        lines.append("- What it stores: Per-parameter-set diagnostics (plots + row-metric CSVs) for a node/run.")
        lines.append("- Used for: Deep-diving into *why* a specific config improved/regressed (distributions, drawdowns, stability, significance, trade quality).")
        lines.append("- Location (current node): " + (avg_plots_dir if avg_plots_dir else "(unknown / not available)"))
        if agentic_output_root:
            lines.append(f"- Location (pattern): {Path(agentic_output_root) / 'run_0' / 'avg_trade_return_plots'}")
        else:
            lines.append("- Location (pattern): <agentic_output_root>/run_0/avg_trade_return_plots/")
        lines.append(f"- Exists (current node path): {str(bool(avg_exists)).lower()}")
        lines.append(f"- Overview doc: {overview_doc} (exists={str(bool(overview_doc.exists())).lower()})")
        lines.append("- Naming convention: Files are per `config_id` and suffixed `_config_000`, `_config_001`, ... matching `meta_config_sweep_results.csv` rows.")
        lines.append("- Per-config artifacts (each is one file per `config_id`):")

        schema_by_prefix: Dict[str, Path] = {
            "core_metrics_config": docs_root / "avg_trade_return_plots" / "core_metrics_config.txt",
            "relative_metrics_config": docs_root / "avg_trade_return_plots" / "relative_metrics_config.txt",
            "trade_metrics_config": docs_root / "avg_trade_return_plots" / "trade_metrics_config.txt",
            "stability_metrics_config": docs_root / "avg_trade_return_plots" / "stability_metrics_config.txt",
            "significance_metrics_config": docs_root / "avg_trade_return_plots" / "significance_metrics_config.txt",
        }

        lines.append("  Artifact: avg_trade_return_config_XXX.png (PNG)")
        lines.append("  - Shows: average return per trade over time (all_models vs topN).")
        lines.append("  - Use it to: check whether improvements are consistent across periods or concentrated in a few regimes.")

        lines.append("  Artifact: trade_quality_hist_config_XXX.png (PNG)")
        lines.append("  - Shows: histogram of average return per trade across periods (all_models vs topN).")
        lines.append("  - Use it to: see distribution shifts and whether downside tail risk worsened.")

        lines.append("  Artifact: trade_quality_rollmean_config_XXX.png (PNG)")
        lines.append("  - Shows: rolling mean of average return per trade (all_models vs topN).")
        lines.append("  - Use it to: see stability of trade-quality improvements through time.")

        lines.append("  Artifact: equity_ratio_config_XXX.png (PNG)")
        lines.append("  - Shows: equity ratio over time (equity_topN / equity_all).")
        lines.append("  - Use it to: see whether topN compounds faster than the baseline universe.")

        lines.append("  Artifact: rolling_sharpe_sortino_config_XXX.png (PNG)")
        lines.append("  - Shows: rolling Sharpe and Sortino for period returns (all_models vs topN).")
        lines.append("  - Use it to: verify risk-adjusted improvements are not just point-estimate noise.")

        lines.append("  Artifact: rolling_outperformance_config_XXX.png (PNG)")
        lines.append("  - Shows: rolling outperformance rate (fraction of periods topN_return > all_models_return).")
        lines.append("  - Use it to: distinguish frequent small wins vs rare big wins.")

        lines.append("  Artifact: drawdown_curves_config_XXX.png (PNG)")
        lines.append("  - Shows: drawdown curves from equity peaks (all_models vs topN).")
        lines.append("  - Use it to: inspect max drawdown depth and duration behavior.")

        lines.append("  Artifact: return_hist_config_XXX.png (PNG)")
        lines.append("  - Shows: histogram (and optional KDE) of period returns (all_models vs topN).")
        lines.append("  - Use it to: compare central tendency and tails of per-period returns.")

        lines.append("  Artifact: return_delta_hist_config_XXX.png (PNG)")
        lines.append("  - Shows: histogram of deltas (topN_return - all_models_return).")
        lines.append("  - Use it to: see whether outperformance is broad-based vs driven by a few periods.")

        lines.append("  Artifact: return_scatter_config_XXX.png (PNG)")
        lines.append("  - Shows: scatter of (all_models_return, topN_return) with y=x line and quadrant axes.")
        lines.append("  - Use it to: identify regimes where meta-selection helps or hurts (e.g., baseline<0 while topN>0).")

        lines.append("  Artifact: core_metrics_config_XXX.csv (CSV)")
        lines.append("  - Stores: core performance metrics for all_models and topN (Sharpe/Sortino/Calmar/drawdowns/etc).")
        schema = schema_by_prefix["core_metrics_config"]
        lines.append(f"  - Column definitions: {schema} (exists={str(bool(schema.exists())).lower()})")
        lines.append("  - Use it to: get a compact numeric summary for one config_id.")

        lines.append("  Artifact: relative_metrics_config_XXX.csv (CSV)")
        lines.append("  - Stores: relative edge metrics (deltas, capture ratios, equity ratio) for topN vs all_models.")
        schema = schema_by_prefix["relative_metrics_config"]
        lines.append(f"  - Column definitions: {schema} (exists={str(bool(schema.exists())).lower()})")
        lines.append("  - Use it to: diagnose how the edge is achieved (frequency vs magnitude, capture, etc).")

        lines.append("  Artifact: stability_metrics_config_XXX.csv (CSV)")
        lines.append("  - Stores: rolling-window stability metrics for all_models, topN, and the delta series.")
        schema = schema_by_prefix["stability_metrics_config"]
        lines.append(f"  - Column definitions: {schema} (exists={str(bool(schema.exists())).lower()})")
        lines.append("  - Use it to: find weak stability (bad rolling mins, long losing streaks, etc).")

        lines.append("  Artifact: trade_metrics_config_XXX.csv (CSV)")
        lines.append("  - Stores: trade-quality metrics computed from avg-return-per-trade series (plus relative deltas).")
        schema = schema_by_prefix["trade_metrics_config"]
        lines.append(f"  - Column definitions: {schema} (exists={str(bool(schema.exists())).lower()})")
        lines.append("  - Use it to: check if improvements come from better per-period trade outcomes and consistency.")

        lines.append("  Artifact: significance_metrics_config_XXX.csv (CSV)")
        lines.append("  - Stores: t-test, sign test, and bootstrap metrics for the delta series (topN_return - all_models_return).")
        schema = schema_by_prefix["significance_metrics_config"]
        lines.append(f"  - Column definitions: {schema} (exists={str(bool(schema.exists())).lower()})")
        lines.append("  - Use it to: sanity-check whether deltas look statistically meaningful.")
        lines.append("- Granularity: 1 directory per node/run; inside it, many files per parameter set tested (`config_id`).")
        lines.append("")

    lines.append("- Guidance: consult docs first; open only the minimum files needed.")
    lines.append("")

    lines.append("Branch Timeline (chronological):")
    timeline = baseline_ctx.get("branch_timeline") or []
    if not isinstance(timeline, list) or not timeline:
        lines.append("- (no timeline available)")
        return "\n".join(lines).rstrip() + "\n"

    metrics_order = [
        "core_topN_sharpe",
        "mean_topN_avg_return_per_trade_pct_oos",
        "mean_topN_avg_return_per_trade_pct",
        "core_topN_sortino",
        "core_topN_calmar",
        "core_topN_max_drawdown",
    ]

    for step_idx, step in enumerate(timeline):
        if not isinstance(step, dict):
            continue

        sid = str(step.get("node_id") or "").strip()
        sdepth = step.get("depth")
        try:
            sdepth_i = int(sdepth) if sdepth is not None else None
        except Exception:
            sdepth_i = None

        applied = str(step.get("applied_idea_path") or "").strip()
        metrics = step.get("metrics") or {}
        deltas = step.get("metric_deltas_vs_parent") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        if not isinstance(deltas, dict):
            deltas = {}

        lines.append(f"{step_idx}. node_id={sid}" + (f" depth={sdepth_i}" if sdepth_i is not None else ""))
        lines.append(f"   applied_idea: {applied}" if applied else "   applied_idea: (root/original)")

        if sweep_limit_i is not None:
            lines.append(f"   sweep_config_limit: {sweep_limit_i} (uses config_id < {sweep_limit_i})")

        for k in metrics_order:
            if k not in metrics:
                continue
            v = metrics.get(k)
            if k in deltas:
                lines.append(f"   {k}: {v}   (delta_vs_parent={deltas.get(k)})")
            else:
                lines.append(f"   {k}: {v}")

        base_csv = str(step.get("baseline_results_csv_path") or "").strip()
        sweep_csv = str(step.get("sweep_results_csv") or "").strip()
        plots_dir = str(step.get("avg_trade_return_plots_dir") or "").strip()

        if base_csv:
            lines.append(f"   baseline_results_csv_path: {base_csv}")
        if sweep_csv:
            lines.append(f"   sweep_results_csv: {sweep_csv}")
        if plots_dir:
            lines.append(f"   avg_trade_return_plots_dir: {plots_dir}")

        lines.append("")

    rejected = baseline_ctx.get("rejected_branch_evals") or []
    if isinstance(rejected, list) and rejected:
        lines.append("Rejected Ideas (evaluated but not selected for this branch):")
        rej_idx = 0
        for ev in rejected:
            if not isinstance(ev, dict):
                continue
            eid = str(ev.get("eval_id") or "").strip()
            pid = str(ev.get("parent_node_id") or "").strip()
            edepth = ev.get("depth")
            try:
                edepth_i = int(edepth) if edepth is not None else None
            except Exception:
                edepth_i = None

            idea_path = str(ev.get("idea_path") or "").strip()
            metrics = ev.get("metrics") or {}
            deltas = ev.get("metric_deltas_vs_parent") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            if not isinstance(deltas, dict):
                deltas = {}

            lines.append(
                f"{rej_idx}. eval_id={eid} parent_node_id={pid}"
                + (f" depth={edepth_i}" if edepth_i is not None else "")
            )
            lines.append(f"   applied_idea: {idea_path}" if idea_path else "   applied_idea: (unknown)")
            if sweep_limit_i is not None:
                lines.append(f"   sweep_config_limit: {sweep_limit_i} (uses config_id < {sweep_limit_i})")

            for k in metrics_order:
                if k not in metrics:
                    continue
                v = metrics.get(k)
                if k in deltas:
                    lines.append(f"   {k}: {v}   (delta_vs_parent={deltas.get(k)})")
                else:
                    lines.append(f"   {k}: {v}")

            base_csv = str(ev.get("baseline_results_csv_path") or "").strip()
            sweep_csv = str(ev.get("sweep_results_csv") or "").strip()
            plots_dir = str(ev.get("avg_trade_return_plots_dir") or "").strip()
            if base_csv:
                lines.append(f"   baseline_results_csv_path: {base_csv}")
            if sweep_csv:
                lines.append(f"   sweep_results_csv: {sweep_csv}")
            if plots_dir:
                lines.append(f"   avg_trade_return_plots_dir: {plots_dir}")

            lines.append("")
            rej_idx += 1

    return "\n".join(lines).rstrip() + "\n"


def _slugify(title: str, *, max_len: int = 60) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    if not s:
        s = "idea"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def _parse_idea_title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.strip().startswith("IDEA:"):
            return line.split("IDEA:", 1)[1].strip()
    return ""


def _strip_agent_markup(text: str) -> str:
    cleaned = re.sub(r"<function_calls>.*?</function_calls>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"(?im)^\\s*</?function_calls>\\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\\s*</?thinking>\\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _trim_to_first_label(text: str) -> str:
    if not text:
        return ""
    match = re.search(
        r"(?im)^\s*(?:[-*\d]+[\).]?\s+)?(?:\*\*)?\s*IDEA\s*:",
        text,
    )
    if match:
        return text[match.start():].lstrip()
    return text


def _extract_idea_sections(markdown: str) -> Optional[str]:
    label_re = re.compile(
        r"^\s*(?:[-*\d]+[\).]?\s+)?(?:\*\*)?\s*(IDEA|RATIONALE|REQUIRED_CHANGES)\s*:\s*(?:\*\*)?\s*(.*)$",
        re.IGNORECASE,
    )
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in markdown.splitlines():
        match = label_re.match(line)
        if match:
            label = match.group(1).upper()
            current = label
            sections.setdefault(label, [])
            remainder = match.group(2).strip()
            if remainder:
                sections[label].append(remainder)
            continue
        if current:
            sections[current].append(line)
    required = ("IDEA", "RATIONALE", "REQUIRED_CHANGES")
    if not all(label in sections for label in required):
        return None

    def _format_section(label: str) -> str:
        lines = sections.get(label, [])
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return f"{label}:"
        head = lines[0].strip()
        tail = lines[1:]
        if tail:
            return f"{label}: {head}\n" + "\n".join(tail)
        return f"{label}: {head}"

    return "\n\n".join(_format_section(label) for label in required).strip() + "\n"


def _clean_idea_output(raw: str) -> str:
    cleaned = _strip_agent_markup(raw or "")
    trimmed = _trim_to_first_label(cleaned)
    extracted = _extract_idea_sections(trimmed)
    if extracted:
        return extracted
    extracted = _extract_idea_sections(cleaned)
    if extracted:
        return extracted
    return trimmed.strip() + "\n" if trimmed else ""


def _validate_idea_output(markdown: str) -> None:
    required = ("IDEA:", "RATIONALE:", "REQUIRED_CHANGES:")
    missing = [k for k in required if k not in markdown]
    if missing:
        raise ValueError(f"Claude output missing required fields: {missing}")


def _bundle_context(
    *,
    prompt_template: str,
    meta_model_context_block: str,
    max_context_chars: int,
) -> str:
    base_parts: List[str] = [
        prompt_template.strip(),
        "",
        meta_model_context_block.rstrip(),
    ]

    base_only = ("\n".join(base_parts).strip() + "\n")
    if len(base_only) <= max_context_chars:
        return base_only
    note = "\n[NOTE: Prompt truncated to fit context limit]\n"
    budget = max(0, max_context_chars - len(note))
    return base_only[:budget].rstrip() + note


@dataclass(frozen=True)
class IdeaGenConfig:
    count: int
    model: Optional[str]
    max_context_chars: int
    cli_path: Optional[str]


def _load_config(config_path: Path) -> IdeaGenConfig:
    try:
        raw = json.loads(_read_text(config_path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing config file at {config_path}. Create it (example: agentic_experimentation/idea_generation/config.json)."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")

    count = raw.get("count", 1)
    model = raw.get("model", None)
    max_context_chars = raw.get("max_context_chars", 200_000)
    cli_path = raw.get("cli_path", None)

    if not isinstance(count, int) or count <= 0:
        raise ValueError("config.count must be a positive integer.")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("config.model must be a non-empty string or null.")
    if not isinstance(max_context_chars, int) or max_context_chars <= 0:
        raise ValueError("config.max_context_chars must be a positive integer.")
    if cli_path is not None and (not isinstance(cli_path, str) or not cli_path.strip()):
        raise ValueError("config.cli_path must be a non-empty string or null.")
    if isinstance(cli_path, str) and Path(cli_path).suffix.lower() in (".cmd", ".bat", ".ps1"):
        raise ValueError(
            "config.cli_path must point to an executable (e.g., claude.exe), not a shell script (.cmd/.bat/.ps1). "
            "Set it to null to use the SDK-bundled claude.exe copy."
        )

    return IdeaGenConfig(
        count=count,
        model=(model.strip() if isinstance(model, str) else None),
        max_context_chars=max_context_chars,
        cli_path=(cli_path.strip() if isinstance(cli_path, str) else None),
    )


def _extract_final_text(messages: List[object]) -> str:
    """
    The Agent SDK yields a stream of message objects; the final response may appear on
    `.result` or inside `.content` blocks depending on SDK version. Fall back to a
    best-effort string conversion if message shapes change.
    """

    def _coerce_text(value: object, depth: int = 0) -> str:
        if depth > 4 or value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [_coerce_text(item, depth + 1) for item in value]
            return "".join(p for p in parts if p)
        if isinstance(value, dict):
            for key in ("text", "content", "message", "value"):
                if key in value:
                    return _coerce_text(value.get(key), depth + 1)
            return ""
        for attr in ("text", "content", "message", "value"):
            try:
                attr_val = getattr(value, attr, None)
            except Exception:  # noqa: BLE001
                attr_val = None
            if attr_val is not None:
                return _coerce_text(attr_val, depth + 1)
        return ""

    for msg in reversed(messages):
        try:
            result = getattr(msg, "result", None)
        except Exception:  # noqa: BLE001
            result = None
        if isinstance(result, str) and result.strip():
            return result

        if isinstance(msg, dict):
            result2 = msg.get("result")
            if isinstance(result2, str) and result2.strip():
                return result2
            content2 = _coerce_text(msg.get("content") or msg.get("message"))
            if content2.strip():
                return content2

        content = _coerce_text(getattr(msg, "content", None))
        if content.strip():
            return content
        message = _coerce_text(getattr(msg, "message", None))
        if message.strip():
            return message
        text = _coerce_text(getattr(msg, "text", None))
        if text.strip():
            return text

    # Last resort: stringify the last message.
    if messages:
        return str(messages[-1])
    return ""


def _extract_provider_metadata(messages: List[object]) -> dict[str, Optional[str]]:
    response_id: Optional[str] = None
    session_id: Optional[str] = None

    for msg in reversed(messages):
        if isinstance(msg, dict):
            if response_id is None:
                value = msg.get("response_id") or msg.get("id")
                if isinstance(value, str) and value.strip():
                    response_id = value.strip()
            if session_id is None:
                value = msg.get("session_id")
                if isinstance(value, str) and value.strip():
                    session_id = value.strip()
        else:
            if response_id is None:
                try:
                    value = getattr(msg, "response_id", None) or getattr(msg, "id", None)
                except Exception:
                    value = None
                if isinstance(value, str) and value.strip():
                    response_id = value.strip()
            if session_id is None:
                try:
                    value = getattr(msg, "session_id", None)
                except Exception:
                    value = None
                if isinstance(value, str) and value.strip():
                    session_id = value.strip()
        if response_id and session_id:
            break
    return {"response_id": response_id, "session_id": session_id}


def _extract_provider_metadata_debug(messages: List[object]) -> dict[str, Any]:
    response_id_candidates: List[str] = []
    session_id_candidates: List[str] = []
    message_types: List[str] = []

    def _add_unique(target: List[str], value: Optional[str]) -> None:
        if not value:
            return
        if value not in target:
            target.append(value)

    for msg in messages:
        if isinstance(msg, dict):
            message_types.append(str(msg.get("type") or "dict"))
            _add_unique(response_id_candidates, str(msg.get("response_id") or msg.get("id") or "").strip() or None)
            _add_unique(session_id_candidates, str(msg.get("session_id") or "").strip() or None)
        else:
            message_types.append(type(msg).__name__)
            try:
                _add_unique(
                    response_id_candidates,
                    str(getattr(msg, "response_id", None) or getattr(msg, "id", None) or "").strip() or None,
                )
            except Exception:
                pass
            try:
                _add_unique(
                    session_id_candidates,
                    str(getattr(msg, "session_id", None) or "").strip() or None,
                )
            except Exception:
                pass

    return {
        "response_id_candidates": response_id_candidates,
        "session_id_candidates": session_id_candidates,
        "message_types": message_types,
    }


async def _run_claude_agent_sdk_once(
    *,
    prompt: str,
    model: Optional[str],
    cwd: Path,
    cli_path: Optional[str],
    continue_conversation: bool = False,
    resume_session_id: Optional[str] = None,
    fork_session: bool = False,
    capture_raw_messages: bool = False,
) -> dict[str, Any]:
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Claude Agent SDK is not installed. Install it with: pip install claude-agent-sdk"
        ) from exc

    def _is_shell_script(p: str) -> bool:
        return Path(p).suffix.lower() in (".cmd", ".bat", ".ps1")

    def _maybe_copy_bundled_cli() -> str:
        """
        The SDK can bundle `claude.exe`, but its path may exceed Windows CreateProcess limits.
        Copy it into this repo (short path) and use that copy.
        """
        import platform
        import claude_agent_sdk  # type: ignore

        cli_name = "claude.exe" if platform.system() == "Windows" else "claude"
        bundled = (Path(claude_agent_sdk.__file__).resolve().parent / "_bundled" / cli_name)
        if not bundled.exists():
            raise RuntimeError(f"Claude Agent SDK bundled CLI not found at: {bundled}")

        dest_dir = Path(__file__).resolve().parent / ".claude_bin"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / cli_name

        # If we already copied it once, reuse it. (The bundled path can be locked by an in-flight process on Windows.)
        if dest.exists() and dest.is_file():
            return str(dest)

        # Best-effort copy. Avoid shutil.copy2 -> CopyFile2, which can fail on Windows if the file is in use.
        with open(bundled, "rb") as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)

        return str(dest)

    resolved_cli_path = cli_path
    if resolved_cli_path is None:
        # Prefer a real executable. On Windows, `shutil.which("claude")` often returns `claude.cmd`,
        # which cannot be launched by CreateProcess (and the SDK transport uses CreateProcess).
        resolved_cli_path = shutil.which("claude.exe") or shutil.which("claude")
        if resolved_cli_path and _is_shell_script(resolved_cli_path):
            resolved_cli_path = None

    if resolved_cli_path is None:
        resolved_cli_path = _maybe_copy_bundled_cli()

    stderr_lines: List[str] = []

    def _on_stderr(line: str) -> None:
        if line:
            stderr_lines.append(line.rstrip())

    options = ClaudeAgentOptions(
        model=model,
        cwd=str(cwd),
        max_turns=1,
        cli_path=resolved_cli_path,
        continue_conversation=bool(continue_conversation),
        resume=(str(resume_session_id) if resume_session_id else None),
        fork_session=bool(fork_session),
        stderr=_on_stderr,
        # For this use-case, we want pure text generation (no filesystem/shell/web tools).
        tools=[],
        allowed_tools=[],
        # Ensure we never block on interactive permission prompts.
        permission_mode="bypassPermissions",
        # Force CLI debug logs to stderr so we can capture root-cause when it exits early.
        extra_args={"debug-to-stderr": None},
    )

    async def _prompt_stream():
        # Use streaming mode so the prompt is sent over stdin (avoids Windows command-line length limits).
        yield {"type": "user", "message": {"role": "user", "content": prompt}}

    messages: List[object] = []
    try:
        async for message in query(prompt=_prompt_stream(), options=options):
            messages.append(message)
    except Exception as exc:  # noqa: BLE001
        log_path = _write_debug_log(
            resolved_cli_path=resolved_cli_path,
            cwd=cwd,
            model=model,
            prompt_len=len(prompt),
            stderr_lines=stderr_lines,
            exc=exc,
        )
        raise RuntimeError(
            "Claude Agent SDK run failed. "
            f"Debug log saved to: {log_path}"
        ) from exc

    meta = _extract_provider_metadata(messages)
    meta_debug = _extract_provider_metadata_debug(messages)
    return {
        "text": _extract_final_text(messages).strip(),
        "provider_response_id": meta.get("response_id"),
        "provider_session_id": meta.get("session_id"),
        "provider_metadata_debug": meta_debug,
        "raw_messages": messages if capture_raw_messages else None,
    }


def main() -> int:
    args = _parse_args(sys.argv[1:])
    repo_root = _resolve_repo_root(Path(__file__).resolve())
    agentic_root = repo_root / "agentic_experimentation"
    cfg = _load_config(Path(__file__).with_name("config.json"))
    run_cwd = Path(__file__).resolve().parent

    # Load .env from agentic_experimentation and repo root (if present).
    _load_env_files([agentic_root / ".env", repo_root / ".env"])

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in `agentic_experimentation/.env` or set it in your environment."
        )

    # Make `agentic_experimentation/*.py` importable even when this script is invoked as a file path
    # (e.g. `python agentic_experimentation/idea_generation/generate_ideas.py`).
    agentic_root_str = str(agentic_root.resolve())
    if agentic_root_str not in sys.path:
        sys.path.insert(0, agentic_root_str)

    phoenix_obs = None
    phoenix_tracer = None
    try:
        import phoenix_tracing as phoenix_obs  # type: ignore

        phoenix_tracer = phoenix_obs.init_phoenix_tracing(
            project_name=None,
            endpoint=None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[phoenix] tracing disabled: {exc}", file=sys.stderr)
        phoenix_obs = None
        phoenix_tracer = None

    prompt_path = agentic_root / "prompts" / "idea_generator" / "idea_generator_prompt.txt"
    ideas_dir = _resolve_cli_path(repo_root, args.ideas_dir) or (agentic_root / "ideas")
    completed_dir = _resolve_cli_path(repo_root, args.completed_dir) or (ideas_dir / "completed")

    # Context dirs: by default, match legacy behavior (ideas_dir + completed_dir).
    # If context dirs are provided, always include the output ideas_dir first so newly
    # generated ideas become context within the same run.
    context_dirs: List[Path] = [ideas_dir]
    for raw in (args.context_ideas_dir or []):
        p = _resolve_cli_path(repo_root, raw)
        if p is not None and p not in context_dirs:
            context_dirs.append(p)

    prompt_template = _read_text(prompt_path)
    baseline_raw: Optional[dict[str, Any]] = None

    if args.baseline_context_json:
        baseline_path = _resolve_cli_path(repo_root, args.baseline_context_json) or Path(str(args.baseline_context_json)).expanduser().resolve()
        parsed = _read_json_file(baseline_path)
        baseline_raw = parsed if isinstance(parsed, dict) else None
        if not isinstance(baseline_raw, dict):
            raise ValueError(f"--baseline-context-json must contain a JSON object: {baseline_path}")
        baseline_block = _render_baseline_context_block(baseline_ctx=baseline_raw, repo_root=repo_root)
        prompt_template = prompt_template.rstrip() + "\n\n" + baseline_block
    meta_model_context_block = _render_meta_model_context_block(repo_root=repo_root, baseline_ctx=baseline_raw)

    created: List[Path] = []
    next_num = _next_idea_number(ideas_dir, completed_dir)

    count = int(args.count) if args.count is not None else int(cfg.count)
    max_context_chars = int(args.max_context_chars) if args.max_context_chars is not None else int(cfg.max_context_chars)
    model = args.model if args.model is not None else cfg.model
    cli_path = args.cli_path if args.cli_path is not None else cfg.cli_path

    requested_conv_mode = _normalize_conversation_mode(args.conversation_mode)
    conv_state_in: Optional[Path] = _resolve_cli_path(repo_root, args.conversation_state_in) if args.conversation_state_in else None
    conv_state_out: Optional[Path] = _resolve_cli_path(repo_root, args.conversation_state_out) if args.conversation_state_out else None
    emit_turn_log: Optional[Path] = _resolve_cli_path(repo_root, args.emit_turn_log) if args.emit_turn_log else None
    log_raw_messages = bool(getattr(args, "idea_log_raw_messages", True))
    if conv_state_out is None and conv_state_in is not None:
        conv_state_out = conv_state_in

    conversation_enabled = requested_conv_mode != "off"
    conversation_state: Optional[dict[str, Any]] = None
    if conversation_enabled:
        if conv_state_out is None:
            raise ValueError("Conversation continuation mode requires --conversation-state-out (or --conversation-state-in).")
        load_path = conv_state_in or conv_state_out
        conversation_state = _load_conversation_state(load_path, requested_mode=requested_conv_mode)
        if args.fork_from_turn_id:
            conversation_state["fork_from_turn_id"] = str(args.fork_from_turn_id)
        if args.conversation_history_window_turns is not None:
            conversation_state["history_window_turns"] = max(0, int(args.conversation_history_window_turns))
        if args.conversation_history_max_chars is not None:
            history_max_chars = int(args.conversation_history_max_chars)
            if history_max_chars < -1:
                raise ValueError("--conversation-history-max-chars must be -1, 0, or a positive integer.")
            conversation_state["history_max_chars"] = history_max_chars
        _compact_conversation_state(
            state=conversation_state,
            keep_recent_turns=max(1, int(conversation_state.get("history_window_turns") or 12)),
            summary_max_chars=_state_history_max_chars(conversation_state),
        )
        # Persist early so interrupted runs still expose conversation metadata.
        _atomic_write_json(conv_state_out, conversation_state)

    run_span_cm = contextlib.nullcontext()
    if phoenix_tracer is not None:
                run_span_cm = phoenix_tracer.start_as_current_span(
                    "idea_generation.run",
                    attributes={
                        "count": int(count),
                        "max_context_chars": int(max_context_chars),
                        "ideas_dir": str(ideas_dir),
                        "context_ideas_dirs": json.dumps([str(p) for p in context_dirs], ensure_ascii=False),
                        "baseline_context_json": (str(args.baseline_context_json) if args.baseline_context_json else ""),
                        "repo_root": str(repo_root),
                        "conversation_mode": requested_conv_mode,
                    },
                )
    with run_span_cm as run_span:
        if phoenix_obs is not None and run_span is not None:
            phoenix_obs.set_openinference_kind(run_span, "CHAIN")

        for _ in range(count):
            prompt = _bundle_context(
                prompt_template=prompt_template,
                meta_model_context_block=meta_model_context_block,
                max_context_chars=int(max_context_chars),
            )
            mode_used = "off"
            if conversation_enabled and isinstance(conversation_state, dict):
                mode_used = _resolve_mode_used_for_turn(
                    requested_mode=requested_conv_mode,
                    conversation_state=conversation_state,
                )
                if mode_used == "replay":
                    replay_block = _render_conversation_replay_block(
                        conversation_state=conversation_state,
                        max_turns=int(conversation_state.get("history_window_turns") or 12),
                        max_chars=_state_history_max_chars(conversation_state),
                    )
                    if replay_block:
                        prompt = prompt.rstrip() + "\n\n" + replay_block
            prompt_hash = _sha256_text(prompt)
            native_params = _claude_native_continuation_params(
                mode_used=mode_used,
                conversation_state=conversation_state,
            )

            prompt_dump_dir = _idea_log_root() / "prompts"
            prompt_dump_path = prompt_dump_dir / f"prompt_{_now_tag()}_{next_num:03d}.txt"
            _write_text(prompt_dump_path, prompt)

            idea_span_cm = contextlib.nullcontext()
            if phoenix_tracer is not None:
                idea_span_cm = phoenix_tracer.start_as_current_span(
                    "idea_generation.idea",
                    attributes={
                        "idea_number": int(next_num),
                        "prompt_path": str(prompt_dump_path),
                        "model": str(model) if model else None,
                    },
                )
            with idea_span_cm as idea_span:
                if phoenix_obs is not None and idea_span is not None:
                    phoenix_obs.set_openinference_kind(idea_span, "CHAIN")
                    phoenix_obs.set_io(idea_span, input_text=prompt)

                llm_span_cm = contextlib.nullcontext()
                if phoenix_tracer is not None:
                    llm_span_cm = phoenix_tracer.start_as_current_span(
                        "llm.claude_agent_sdk",
                        attributes={
                            "llm.provider": "anthropic",
                            "llm.model_name": str(model) if model else None,
                            "prompt_path": str(prompt_dump_path),
                        },
                    )
                with llm_span_cm as llm_span:
                    if phoenix_obs is not None and llm_span is not None:
                        phoenix_obs.set_openinference_kind(llm_span, "LLM")
                        phoenix_obs.set_io(llm_span, input_text=prompt)

                    try:
                        llm_result = asyncio.run(
                            _run_claude_agent_sdk_once(
                                prompt=prompt,
                                model=model,
                                cwd=run_cwd,
                                cli_path=cli_path,
                                continue_conversation=bool(native_params.get("continue_conversation", False)),
                                resume_session_id=native_params.get("resume_session_id"),
                                fork_session=bool(native_params.get("fork_session", False)),
                                capture_raw_messages=log_raw_messages,
                            )
                        )
                    except RuntimeError:
                        # Common failure mode: unsupported/invalid model string for Claude Code.
                        # If a model was specified, retry once with default model to reduce friction.
                        if model is None:
                            raise
                        llm_result = asyncio.run(
                            _run_claude_agent_sdk_once(
                                prompt=prompt,
                                model=None,
                                cwd=run_cwd,
                                cli_path=cli_path,
                                continue_conversation=bool(native_params.get("continue_conversation", False)),
                                resume_session_id=native_params.get("resume_session_id"),
                                fork_session=bool(native_params.get("fork_session", False)),
                                capture_raw_messages=log_raw_messages,
                            )
                        )
                    idea_md = _clean_idea_output(str(llm_result.get("text") or ""))
                    provider_meta_debug = llm_result.get("provider_metadata_debug") or {}
                    raw_messages = llm_result.get("raw_messages")

                    if phoenix_obs is not None and llm_span is not None:
                        phoenix_obs.set_io(llm_span, output_text=idea_md)

                _validate_idea_output(idea_md)

                title = _parse_idea_title(idea_md)
                slug = _slugify(title)

                out_path = ideas_dir / f"{next_num:03d}_{slug}.md"
                # Avoid overwriting if numbers collide or slug repeats.
                if out_path.exists():
                    suffix = 2
                    while True:
                        cand = ideas_dir / f"{next_num:03d}_{slug}_{suffix}.md"
                        if not cand.exists():
                            out_path = cand
                            break
                        suffix += 1

                _write_text(out_path, idea_md.rstrip() + "\n")
                created.append(out_path)

                turn_id_for_log: Optional[str] = None
                if conversation_enabled and isinstance(conversation_state, dict):
                    out_abs = str(out_path.resolve())
                    try:
                        out_rel = str(out_path.resolve().relative_to(repo_root.resolve()))
                    except Exception:
                        out_rel = str(out_path)
                    operation_id = _sha256_text(f"{prompt_hash}|{out_rel}")
                    turns_list = conversation_state.setdefault("turns", [])
                    if not isinstance(turns_list, list):
                        turns_list = []
                        conversation_state["turns"] = turns_list
                    operation_map = conversation_state.setdefault("operation_id_turn_map", {})
                    if not isinstance(operation_map, dict):
                        operation_map = {}
                        conversation_state["operation_id_turn_map"] = operation_map
                    existing_turn_id = str(operation_map.get(operation_id) or "").strip()
                    existing_idx = _find_turn_index_by_operation_id(conversation_state, operation_id)
                    if existing_idx is not None:
                        existing_turn = conversation_state["turns"][existing_idx]
                        existing_turn_id_from_row = ""
                        if isinstance(existing_turn, dict):
                            existing_turn_id_from_row = str(existing_turn.get("turn_id") or "").strip()
                        turn_id = existing_turn_id_from_row or existing_turn_id or _conversation_next_turn_id(conversation_state)
                    else:
                        turn_id = existing_turn_id if existing_turn_id else _conversation_next_turn_id(conversation_state)
                    turn_id_for_log = turn_id
                    turn = {
                        "turn_id": turn_id,
                        "operation_id": operation_id,
                        "timestamp": _utc_now_iso(),
                        "mode_requested": requested_conv_mode,
                        "mode_used": mode_used,
                        "fork_from_turn_id": conversation_state.get("fork_from_turn_id"),
                        "prompt_path": str(prompt_dump_path),
                        "prompt_hash": prompt_hash,
                        "prompt_chars": len(prompt),
                        "model": str(model) if model else None,
                        "output_idea_path": str(out_path),
                        "output_idea_path_resolved": out_abs,
                        "output_idea_path_relative_to_repo": out_rel,
                        "output_hash": _sha256_text(idea_md),
                        "output_title": title or None,
                        "output_text": idea_md,
                        "provider_session_id": llm_result.get("provider_session_id"),
                        "provider_response_id": llm_result.get("provider_response_id"),
                        "provider_response_id_candidates": provider_meta_debug.get("response_id_candidates") if isinstance(provider_meta_debug, dict) else None,
                        "provider_session_id_candidates": provider_meta_debug.get("session_id_candidates") if isinstance(provider_meta_debug, dict) else None,
                        "provider_message_types": provider_meta_debug.get("message_types") if isinstance(provider_meta_debug, dict) else None,
                    }
                    turn_action = "appended"
                    if existing_idx is not None:
                        conversation_state["turns"][existing_idx] = turn
                        turn_action = "upserted"
                    elif existing_turn_id:
                        # Idempotent retry for an operation already compacted out of `turns`.
                        turn_action = "reused_compacted"
                    else:
                        conversation_state["turns"].append(turn)
                    operation_map[operation_id] = turn_id
                    conversation_state["latest_turn_id"] = turn_id
                    conversation_state["updated_at"] = _utc_now_iso()
                    idea_file_turn_map = conversation_state.setdefault("idea_file_turn_map", {})
                    if isinstance(idea_file_turn_map, dict):
                        idea_file_turn_map[str(out_path)] = turn_id
                        idea_file_turn_map[out_abs] = turn_id
                        idea_file_turn_map[out_rel] = turn_id
                    provider = conversation_state.setdefault("provider", {})
                    if isinstance(provider, dict):
                        provider["supports_native_continuation"] = _provider_supports_native_continuation(provider)
                        provider["last_response_id"] = llm_result.get("provider_response_id")
                        if llm_result.get("provider_session_id"):
                            provider["session_id"] = llm_result.get("provider_session_id")
                        if bool(native_params.get("fork_session", False)):
                            provider["branch_session_initialized"] = True
                    _compact_conversation_state(
                        state=conversation_state,
                        keep_recent_turns=max(1, int(conversation_state.get("history_window_turns") or 12)),
                        summary_max_chars=_state_history_max_chars(conversation_state),
                    )
                    if conv_state_out is not None:
                        _atomic_write_json(conv_state_out, conversation_state)
                    if emit_turn_log is not None:
                        _append_jsonl(
                            emit_turn_log,
                            {
                                "turn_id": turn_id,
                                "timestamp": turn["timestamp"],
                                "mode_used": mode_used,
                                "prompt_hash": prompt_hash,
                                "prompt_chars": len(prompt),
                                "output_idea_path": str(out_path),
                                "output_hash": turn["output_hash"],
                                "provider_session_id": llm_result.get("provider_session_id"),
                                "provider_response_id": llm_result.get("provider_response_id"),
                                "provider_response_id_candidates": provider_meta_debug.get("response_id_candidates") if isinstance(provider_meta_debug, dict) else None,
                                "provider_session_id_candidates": provider_meta_debug.get("session_id_candidates") if isinstance(provider_meta_debug, dict) else None,
                                "provider_message_types": provider_meta_debug.get("message_types") if isinstance(provider_meta_debug, dict) else None,
                                "turn_action": turn_action,
                                "operation_id": operation_id,
                            },
                        )

                if log_raw_messages and raw_messages is not None:
                    _write_raw_llm_messages(
                        idea_number=next_num,
                        prompt_path=prompt_dump_path,
                        model=model,
                        node_id=(str(baseline_raw.get("node_id")) if isinstance(baseline_raw, dict) else None),
                        conversation_id=(str(conversation_state.get("conversation_id")) if isinstance(conversation_state, dict) else None),
                        turn_id=turn_id_for_log,
                        provider_session_id=llm_result.get("provider_session_id"),
                        provider_response_id=llm_result.get("provider_response_id"),
                        provider_metadata_debug=(provider_meta_debug if isinstance(provider_meta_debug, dict) else None),
                        raw_messages=raw_messages,
                    )

                if phoenix_obs is not None and idea_span is not None:
                    phoenix_obs.set_attrs(idea_span, {"output_path": str(out_path)})
                    phoenix_obs.set_io(idea_span, output_text=idea_md)

                next_num += 1

    for p in created:
        print(str(p))
    if phoenix_obs is not None:
        try:
            phoenix_obs.force_flush()
        except Exception:  # noqa: BLE001
            pass
    if conversation_enabled and isinstance(conversation_state, dict) and conv_state_out is not None:
        _compact_conversation_state(
            state=conversation_state,
            keep_recent_turns=max(1, int(conversation_state.get("history_window_turns") or 12)),
            summary_max_chars=_state_history_max_chars(conversation_state),
        )
        conversation_state["updated_at"] = _utc_now_iso()
        _atomic_write_json(conv_state_out, conversation_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
