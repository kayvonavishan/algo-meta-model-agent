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
from typing import Any, Iterable, List, Optional, Tuple


_IDEA_FILE_RE = re.compile(r"^(?P<num>\d{3})_(?P<name>.+)\.md$", re.IGNORECASE)


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
    avg_plots_dir = str(baseline_ctx.get("avg_trade_return_plots_dir") or "").strip()

    lines: list[str] = []
    lines.append("-----------------------")
    lines.append("")

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
    schema_files = [
        docs_root / "avg_trade_return_plots" / "core_metrics_config.txt",
        docs_root / "avg_trade_return_plots" / "relative_metrics_config.txt",
        docs_root / "avg_trade_return_plots" / "trade_metrics_config.txt",
        docs_root / "avg_trade_return_plots" / "stability_metrics_config.txt",
        docs_root / "avg_trade_return_plots" / "significance_metrics_config.txt",
    ]

    # 1) Sweep results table (per node).
    agentic_output_root = str(baseline_ctx.get("agentic_output_root") or "").strip()
    sweep_exists = Path(sweep_results_csv).expanduser().exists() if sweep_results_csv else False
    lines.append("Output: meta_config_sweep_results.csv")
    lines.append("- What it stores: Per-config sweep results; each row is one meta-model backtest for a single parameter set (`config_id`).")
    lines.append("- Used for: Comparing parameter sets and computing the averaged metrics/deltas used to judge/promote ideas.")
    lines.append("- Location (current node): " + (sweep_results_csv if sweep_results_csv else "(unknown / not available)"))
    if agentic_output_root:
        lines.append(f"- Location (pattern): {Path(agentic_output_root) / 'run_0' / 'meta_config_sweep_results.csv'}")
    else:
        lines.append("- Location (pattern): <agentic_output_root>/run_0/meta_config_sweep_results.csv")
    lines.append(f"- Exists (current node path): {str(bool(sweep_exists)).lower()}")
    lines.append(f"- Column definitions: {col_defs} (exists={str(bool(col_defs.exists())).lower()})")
    lines.append("  - Format: `column_name: description` (search by column name).")
    lines.append("  - Note: metrics families include `core_*`, `rel_*`, `stab_*`, `trade_*`, `sig_*`.")
    lines.append("- Granularity: 1 CSV per node/run; rows are per parameter set tested (`config_id`).")
    lines.append("")

    # 2) Per-config diagnostics (per node, per config_id).
    avg_exists = Path(avg_plots_dir).expanduser().exists() if avg_plots_dir else False
    lines.append("Output: avg_trade_return_plots/")
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

    if args.baseline_context_json:
        baseline_path = _resolve_cli_path(repo_root, args.baseline_context_json) or Path(str(args.baseline_context_json)).expanduser().resolve()
        baseline_raw = _read_json_file(baseline_path)
        if not isinstance(baseline_raw, dict):
            raise ValueError(f"--baseline-context-json must contain a JSON object: {baseline_path}")
        baseline_block = _render_baseline_context_block(baseline_ctx=baseline_raw, repo_root=repo_root)
        prompt_template = prompt_template.rstrip() + "\n\n" + baseline_block

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
                        "baseline_context_json": (str(args.baseline_context_json) if args.baseline_context_json else ""),
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
