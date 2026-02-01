from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


_IDEA_FILE_RE = re.compile(r"^(?P<num>\d{3})_(?P<name>.+)\.md$", re.IGNORECASE)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


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
            if _IDEA_FILE_RE.match(p.name):
                paths.append(p)

    if completed_dir.exists():
        for p in completed_dir.glob("*.md"):
            if _IDEA_FILE_RE.match(p.name):
                paths.append(p)

    def _sort_key(p: Path) -> Tuple[int, str]:
        m = _IDEA_FILE_RE.match(p.name)
        if not m:
            return (10**9, p.name.lower())
        return (int(m.group("num")), p.name.lower())

    return sorted(paths, key=_sort_key)


def _next_idea_number(ideas_dir: Path, completed_dir: Path) -> int:
    max_num = 0
    for p in _collect_idea_files(ideas_dir, completed_dir):
        m = _IDEA_FILE_RE.match(p.name)
        if not m:
            continue
        max_num = max(max_num, int(m.group("num")))
    return max_num + 1


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

    if not isinstance(count, int) or count <= 0:
        raise ValueError("config.count must be a positive integer.")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("config.model must be a non-empty string or null.")
    if not isinstance(max_context_chars, int) or max_context_chars <= 0:
        raise ValueError("config.max_context_chars must be a positive integer.")

    return IdeaGenConfig(
        count=count,
        model=(model.strip() if isinstance(model, str) else None),
        max_context_chars=max_context_chars,
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


async def _run_claude_agent_sdk_once(*, prompt: str, model: Optional[str], cwd: Path) -> str:
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Claude Agent SDK is not installed. Install it with: pip install claude-agent-sdk"
        ) from exc

    options = ClaudeAgentOptions(
        model=model,
        cwd=str(cwd),
        max_turns=1,
        # For this use-case, we want pure text generation (no filesystem/shell/web tools).
        tools=[],
        allowed_tools=[],
        # Mirror CLI behavior: plain text output.
        extra_args={"output-format": "text"},
    )

    messages: List[object] = []
    async for message in query(prompt=prompt, options=options):
        messages.append(message)

    return _extract_final_text(messages).strip()


def main() -> int:
    repo_root = _resolve_repo_root(Path(__file__).resolve())
    agentic_root = repo_root / "agentic_experimentation"
    cfg = _load_config(Path(__file__).with_name("config.json"))

    # Load .env from agentic_experimentation and repo root (if present).
    _load_env_files([agentic_root / ".env", repo_root / ".env"])

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in `agentic_experimentation/.env` or set it in your environment."
        )

    prompt_path = agentic_root / "prompts" / "idea_generator" / "idea_generator_prompt.txt"
    guide_path = repo_root / "META_MODEL_GUIDE.md"
    driver_path = repo_root / "adaptive_vol_momentum.py"
    ideas_dir = agentic_root / "ideas"
    completed_dir = ideas_dir / "completed"

    prompt_template = _read_text(prompt_path)
    meta_model_guide = _read_text(guide_path)
    driver_code = _read_text(driver_path)

    created: List[Path] = []
    next_num = _next_idea_number(ideas_dir, completed_dir)

    for _ in range(cfg.count):
        # Re-scan each loop so the newly written idea becomes context for the next call.
        prior_idea_files = _collect_idea_files(ideas_dir, completed_dir)

        prompt = _bundle_context(
            prompt_template=prompt_template,
            meta_model_guide=meta_model_guide,
            driver_code=driver_code,
            idea_files=prior_idea_files,
            max_context_chars=int(cfg.max_context_chars),
        )

        idea_md = asyncio.run(
            _run_claude_agent_sdk_once(prompt=prompt, model=cfg.model, cwd=repo_root)
        )
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
        next_num += 1

    for p in created:
        print(str(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
