from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


_IDEA_FILE_RE = re.compile(r"^(?P<num>\d{3})_(?P<name>.+)\.md$", re.IGNORECASE)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _write_debug_log(*, resolved_cli_path: str, cwd: Path, model: Optional[str], prompt_len: int, stderr_lines: List[str], exc: Exception) -> Path:
    log_dir = Path(__file__).resolve().parent / ".idea_generation_logs"
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
    return parser.parse_args(argv)


def _resolve_cli_path(repo_root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


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


def _validate_idea_output(markdown: str) -> None:
    required = ("IDEA:", "RATIONALE:", "REQUIRED_CHANGES:")
    missing = [k for k in required if k not in markdown]
    if missing:
        raise ValueError(f"Claude output missing required fields: {missing}")


def _bundle_context(
    *,
    prompt_template: str,
    meta_model_guide: str,
    driver_code: str,
    idea_files: List[Path],
    max_context_chars: int,
) -> str:
    def _render_idea(p: Path) -> str:
        return "\n".join(["", f"### {p.name}", _read_text(p).rstrip()])

    guide_block = meta_model_guide.rstrip()
    driver_block = driver_code.rstrip()

    base_parts: List[str] = [
        prompt_template.strip(),
        "",
        "===== REPO CONTEXT (READ-ONLY) =====",
        "",
        "----- FILE: META_MODEL_GUIDE.md -----",
        guide_block,
        "",
        "----- FILE: adaptive_vol_momentum.py -----",
        "```python",
        driver_block,
        "```",
        "",
        "----- PRIOR IDEAS (DO NOT DUPLICATE) -----",
    ]

    idea_blocks = [_render_idea(p) for p in idea_files]

    def _assemble(ideas: List[str]) -> str:
        return ("\n".join(base_parts + ideas).strip() + "\n")

    # 1) Try with all prior ideas.
    prompt = _assemble(idea_blocks)
    if len(prompt) <= max_context_chars:
        return prompt

    # 2) Drop oldest prior ideas until it fits (possibly dropping all).
    trimmed_blocks = idea_blocks[:]
    while trimmed_blocks and len(_assemble(trimmed_blocks)) > max_context_chars:
        trimmed_blocks.pop(0)
    prompt = _assemble(trimmed_blocks)
    if len(prompt) <= max_context_chars:
        if trimmed_blocks != idea_blocks:
            prompt += "\n[NOTE: Dropped some older prior ideas to fit context limit]\n"
        return prompt

    # 3) Still too large: truncate the guide first, then the driver code if needed.
    def _truncate_to_budget(text: str, budget: int, note: str) -> str:
        if budget <= 0:
            return f"[TRUNCATED {note} TO FIT CONTEXT LIMIT]"
        if len(text) <= budget:
            return text
        return text[:budget].rstrip() + f"\n\n[TRUNCATED {note} TO FIT CONTEXT LIMIT]"

    # Reserve space for the non-guide/non-driver scaffolding.
    scaffolding = _assemble([])  # base_parts only (already includes full guide/driver right now)
    # Recompute scaffolding with empty guide/driver placeholders to estimate fixed overhead.
    overhead_parts = base_parts[:]
    overhead_parts[overhead_parts.index(guide_block)] = ""
    overhead_parts[overhead_parts.index(driver_block)] = ""
    overhead = ("\n".join(overhead_parts).strip() + "\n")
    budget_total = max_context_chars - len(overhead)

    guide_budget = max(0, int(budget_total * 0.7))
    driver_budget = max(0, budget_total - guide_budget)

    truncated_guide = _truncate_to_budget(guide_block, guide_budget, "META_MODEL_GUIDE.md")
    truncated_driver = _truncate_to_budget(driver_block, driver_budget, "adaptive_vol_momentum.py")

    base_parts_trunc = base_parts[:]
    base_parts_trunc[base_parts_trunc.index(guide_block)] = truncated_guide
    base_parts_trunc[base_parts_trunc.index(driver_block)] = truncated_driver

    return ("\n".join(base_parts_trunc).strip() + "\n")


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
    The Agent SDK yields a stream of message objects; in examples, the final response is available on `.result`.
    Fall back to a best-effort string conversion if the SDK changes message shapes.
    """
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

    # Last resort: stringify the last message.
    if messages:
        return str(messages[-1])
    return ""


async def _run_claude_agent_sdk_once(
    *,
    prompt: str,
    model: Optional[str],
    cwd: Path,
    cli_path: Optional[str],
) -> str:
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

    return _extract_final_text(messages).strip()


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
    guide_path = repo_root / "META_MODEL_GUIDE.md"
    driver_path = repo_root / "adaptive_vol_momentum.py"
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
    meta_model_guide = _read_text(guide_path)
    driver_code = _read_text(driver_path)

    created: List[Path] = []
    next_num = _next_idea_number(ideas_dir, completed_dir)

    count = int(args.count) if args.count is not None else int(cfg.count)
    max_context_chars = int(args.max_context_chars) if args.max_context_chars is not None else int(cfg.max_context_chars)
    model = args.model if args.model is not None else cfg.model
    cli_path = args.cli_path if args.cli_path is not None else cfg.cli_path

    run_span_cm = contextlib.nullcontext()
    if phoenix_tracer is not None:
        run_span_cm = phoenix_tracer.start_as_current_span(
            "idea_generation.run",
            attributes={
                "count": int(count),
                "max_context_chars": int(max_context_chars),
                "ideas_dir": str(ideas_dir),
                "context_ideas_dirs": json.dumps([str(p) for p in context_dirs], ensure_ascii=False),
                "repo_root": str(repo_root),
            },
        )
    with run_span_cm as run_span:
        if phoenix_obs is not None and run_span is not None:
            phoenix_obs.set_openinference_kind(run_span, "CHAIN")

        for _ in range(count):
            # Re-scan each loop so the newly written idea becomes context for the next call.
            prior_idea_files = _collect_idea_files_from_dirs(context_dirs)

            prompt = _bundle_context(
                prompt_template=prompt_template,
                meta_model_guide=meta_model_guide,
                driver_code=driver_code,
                idea_files=prior_idea_files,
                max_context_chars=int(max_context_chars),
            )

            prompt_dump_dir = Path(__file__).resolve().parent / ".idea_generation_logs"
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
                        idea_md = asyncio.run(
                            _run_claude_agent_sdk_once(
                                prompt=prompt,
                                model=model,
                                cwd=run_cwd,
                                cli_path=cli_path,
                            )
                        )
                    except RuntimeError:
                        # Common failure mode: unsupported/invalid model string for Claude Code.
                        # If a model was specified, retry once with default model to reduce friction.
                        if model is None:
                            raise
                        idea_md = asyncio.run(
                            _run_claude_agent_sdk_once(
                                prompt=prompt,
                                model=None,
                                cwd=run_cwd,
                                cli_path=cli_path,
                            )
                        )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
