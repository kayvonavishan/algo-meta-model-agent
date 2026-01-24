import argparse
import asyncio
import contextlib
import contextvars
import dataclasses
import datetime as _dt
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agent_runner import (
    _load_ideas,
    _load_json,
    _read_text,
    _render_prompt,
    _resolve_path,
    _resolve_repo_root,
    _run_sweep,
    _write_json,
    _write_text,
    _load_env_files,
)
from git_worktree import create_worktree, remove_worktree, sync_working_tree
from scoring_hooks import compute_score


SYSTEM_PROMPT = "You are a coding agent improving a meta model. Keep changes small and runnable."


_TRACE_CONTEXT = contextvars.ContextVar("_TRACE_CONTEXT", default=None)
_TRACE_PROCESSOR_REGISTERED = False


def _configure_logging():
    root = logging.getLogger()
    if getattr(root, "_agentic_configured", False):
        return

    root.setLevel(logging.INFO)
    root.handlers.clear()

    class _DropNoisyCodexNotifications(logging.Filter):
        def filter(self, record):  # noqa: ANN001
            msg = record.getMessage() or ""
            # Codex MCP server emits custom `codex/event` notifications that the MCP client may warn about.
            # They are typically safe to ignore and extremely verbose.
            if msg.startswith("Failed to validate notification:") and "codex/event" in msg:
                _write_runner_log(record)
                return False
            return True

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.addFilter(_DropNoisyCodexNotifications())
    root.addHandler(handler)

    root._agentic_configured = True  # type: ignore[attr-defined]


def _write_runner_log(record):  # noqa: ANN001
    trace_dir = os.environ.get("AGENTIC_TRACE_DIR")
    if not trace_dir:
        return
    try:
        path = Path(trace_dir) / "runner_warnings.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            ts = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
            f.write(f"{ts} {record.levelname} {record.name}: {record.getMessage()}\n")
    except Exception:  # pylint: disable=broad-except
        return


@contextlib.contextmanager
def _tracing_scope(*, exp_dir, run_id, step, round_idx=None):
    prev_env = {
        "AGENTIC_TRACE_DIR": os.environ.get("AGENTIC_TRACE_DIR"),
        "AGENTIC_TRACE_RUN_ID": os.environ.get("AGENTIC_TRACE_RUN_ID"),
        "AGENTIC_TRACE_STEP": os.environ.get("AGENTIC_TRACE_STEP"),
        "AGENTIC_TRACE_ROUND": os.environ.get("AGENTIC_TRACE_ROUND"),
    }
    os.environ["AGENTIC_TRACE_DIR"] = str(Path(exp_dir).resolve())
    os.environ["AGENTIC_TRACE_RUN_ID"] = str(run_id)
    os.environ["AGENTIC_TRACE_STEP"] = str(step)
    os.environ["AGENTIC_TRACE_ROUND"] = "" if round_idx is None else str(round_idx)
    token = _TRACE_CONTEXT.set(
        {
            "exp_dir": str(Path(exp_dir).resolve()),
            "run_id": str(run_id),
            "step": str(step),
            "round_idx": round_idx,
        }
    )
    try:
        yield
    finally:
        _TRACE_CONTEXT.reset(token)
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _ensure_trace_processor_registered():
    global _TRACE_PROCESSOR_REGISTERED  # pylint: disable=global-statement
    if _TRACE_PROCESSOR_REGISTERED:
        return
    try:
        from agents.tracing import add_trace_processor  # type: ignore
        from agents.tracing.processor_interface import TracingProcessor  # type: ignore
    except ImportError:
        # Tracing is optional. If the Agents SDK isn't installed, the caller will fail earlier
        # when trying to run the agents; keep this as best-effort.
        return

    class _JsonlTraceProcessor(TracingProcessor):
        def __init__(self):
            self._lock = None  # reserved for future threading lock if needed

        def on_trace_start(self, trace):  # noqa: ANN001
            self._write({"event": "trace_start", "trace": _summarize_trace(trace)})

        def on_trace_end(self, trace):  # noqa: ANN001
            self._write({"event": "trace_end", "trace": _summarize_trace(trace)})

        def on_span_start(self, span):  # noqa: ANN001
            payload = _summarize_span(span)
            self._write({"event": "span_start", "span": payload})
            if _is_codex_tool_span(payload):
                self._write_codex({"event": "tool_start", "span": payload})

        def on_span_end(self, span):  # noqa: ANN001
            payload = _summarize_span(span)
            self._write({"event": "span_end", "span": payload})
            if _is_codex_tool_span(payload):
                self._write_codex({"event": "tool_end", "span": payload})

        def shutdown(self) -> None:
            return

        def force_flush(self) -> None:
            return

        def _write(self, payload):  # noqa: ANN001
            ctx = _TRACE_CONTEXT.get() or {}
            exp_dir = Path(ctx.get("exp_dir") or os.environ.get("AGENTIC_TRACE_DIR") or ".")
            exp_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(payload)
            payload["ts"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
            payload["run_id"] = ctx.get("run_id") or os.environ.get("AGENTIC_TRACE_RUN_ID")
            payload["step"] = ctx.get("step") or os.environ.get("AGENTIC_TRACE_STEP")
            payload["round_idx"] = ctx.get("round_idx")
            if payload["round_idx"] is None:
                payload["round_idx"] = os.environ.get("AGENTIC_TRACE_ROUND") or None
            _append_jsonl(exp_dir / "agents_trace.jsonl", payload)

        def _write_codex(self, payload):  # noqa: ANN001
            ctx = _TRACE_CONTEXT.get() or {}
            exp_dir = Path(ctx.get("exp_dir") or os.environ.get("AGENTIC_TRACE_DIR") or ".")
            exp_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(payload)
            payload["ts"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
            payload["run_id"] = ctx.get("run_id") or os.environ.get("AGENTIC_TRACE_RUN_ID")
            payload["step"] = ctx.get("step") or os.environ.get("AGENTIC_TRACE_STEP")
            payload["round_idx"] = ctx.get("round_idx")
            if payload["round_idx"] is None:
                payload["round_idx"] = os.environ.get("AGENTIC_TRACE_ROUND") or None
            _append_jsonl(exp_dir / "codex_mcp_transcript.jsonl", payload)

    add_trace_processor(_JsonlTraceProcessor())
    _TRACE_PROCESSOR_REGISTERED = True


def _append_jsonl(path, payload):  # noqa: ANN001
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _to_jsonable(obj):  # noqa: ANN001
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        return _to_jsonable(dataclasses.asdict(obj))
    # Agents SDK tracing SpanData objects expose `export()` and frequently use `__slots__` (no `__dict__`).
    if hasattr(obj, "export") and callable(getattr(obj, "export")):
        try:
            return _to_jsonable(obj.export())
        except Exception:  # pylint: disable=broad-except
            pass
    # pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:  # pylint: disable=broad-except
            pass
    # pydantic v1
    if hasattr(obj, "dict"):
        try:
            return _to_jsonable(obj.dict())
        except Exception:  # pylint: disable=broad-except
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _to_jsonable({k: v for k, v in obj.__dict__.items() if not k.startswith("_")})
        except Exception:  # pylint: disable=broad-except
            pass
    return str(obj)


def _summarize_trace(trace):  # noqa: ANN001
    return {
        "trace_id": getattr(trace, "trace_id", None),
        "name": getattr(trace, "name", None),
        "group_id": getattr(trace, "group_id", None),
        "workflow_name": getattr(trace, "workflow_name", None),
        "metadata": _to_jsonable(getattr(trace, "metadata", None)),
    }


def _summarize_span(span):  # noqa: ANN001
    span_data = None
    try:
        span_data = getattr(span, "span_data", None)
    except Exception:  # pylint: disable=broad-except
        span_data = None
    span_data_payload = _to_jsonable(span_data)
    return {
        "trace_id": getattr(span, "trace_id", None),
        "span_id": getattr(span, "span_id", None),
        "parent_id": getattr(span, "parent_id", None),
        "started_at": _to_jsonable(getattr(span, "started_at", None)),
        "ended_at": _to_jsonable(getattr(span, "ended_at", None)),
        "error": _to_jsonable(getattr(span, "error", None)),
        "span_data_type": type(span_data).__name__ if span_data is not None else None,
        "span_data": span_data_payload,
    }


def _is_codex_tool_span(span_payload):  # noqa: ANN001
    # Tool spans typically include a function span payload with name "codex" / "codex-reply".
    span_data = (span_payload or {}).get("span_data") if isinstance(span_payload, dict) else None
    if isinstance(span_data, dict):
        name = (span_data.get("name") or "").lower().strip()
        typ = (span_data.get("type") or "").lower().strip()
        if typ == "function" and name in ("codex", "codex-reply"):
            return True
        # Some implementations may not normalize name; keep a fallback substring match.
        if "codex" in name:
            return True

    payload = span_payload

    def _contains_codex(x):  # noqa: ANN001
        if x is None:
            return False
        if isinstance(x, str):
            s = x.lower()
            return "codex" in s
        if isinstance(x, dict):
            return any(_contains_codex(k) or _contains_codex(v) for k, v in x.items())
        if isinstance(x, list):
            return any(_contains_codex(v) for v in x)
        return False

    return _contains_codex(payload)


def main():
    asyncio.run(_main_async())


async def _main_async():
    _configure_logging()
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = _load_json(config_path)

    # Load .env from config directory first, then repo root (if present)
    _load_env_files([config_path.parent / ".env"])

    repo_root = _resolve_repo_root(config, config_path)
    experiments_root = _resolve_path(repo_root, config.get("experiments_root"))
    worktree_root = _resolve_path(repo_root, config.get("worktree_root"))
    baseline_csv = _resolve_path(repo_root, config.get("baseline_csv"))
    results_csv = Path(config.get("results_csv")).expanduser()
    agentic_output_root = config.get("agentic_output_root")
    if agentic_output_root:
        agentic_output_root = Path(agentic_output_root).expanduser()
    ideas_path = _resolve_path(repo_root, config.get("ideas_file"))

    _load_env_files([repo_root / ".env"])

    # Ensure Agents SDK tracing is enabled and includes tool I/O, so we can log a full local transcript.
    # This only affects this process and can still be overridden externally if desired.
    os.environ.pop("OPENAI_AGENTS_DISABLE_TRACING", None)
    os.environ.setdefault("OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA", "1")

    if not ideas_path:
        raise RuntimeError("ideas_file must be set in the config.")
    if not baseline_csv.exists():
        raise RuntimeError(f"Baseline CSV not found at {baseline_csv}. Run with --refresh-baseline.")

    ideas = _load_ideas(ideas_path)
    if not ideas:
        raise RuntimeError(f"ideas_file provided but no ideas found at {ideas_path}")

    prompts_cfg = config.get("prompts", {})
    planner_prompt_path = _resolve_path(repo_root, prompts_cfg.get("planner"))
    planner_system_path = _resolve_path(repo_root, prompts_cfg.get("planner_system"))
    planner_files_path = _resolve_path(repo_root, prompts_cfg.get("planner_repo_files"))
    coder_prompt_path = _resolve_path(repo_root, prompts_cfg.get("coder"))
    coder_fix_prompt_path = _resolve_path(repo_root, prompts_cfg.get("coder_fix")) or coder_prompt_path
    coder_system_path = _resolve_path(repo_root, prompts_cfg.get("coder_system"))
    coder_files_path = _resolve_path(repo_root, prompts_cfg.get("coder_repo_files"))
    reviewer_prompt_path = _resolve_path(repo_root, prompts_cfg.get("reviewer"))
    reviewer_system_path = _resolve_path(repo_root, prompts_cfg.get("reviewer_system"))
    reviewer_files_path = _resolve_path(repo_root, prompts_cfg.get("reviewer_repo_files"))
    if not (planner_prompt_path and coder_prompt_path and reviewer_prompt_path):
        raise RuntimeError("Planner, coder, and reviewer prompts must be set in config['prompts'].")

    iterations_requested = args.iterations
    if iterations_requested is None:
        iterations_requested = config.get("iterations", len(ideas))
    iterations = int(iterations_requested)
    if iterations > len(ideas):
        raise RuntimeError(
            f"Requested {iterations} iterations but only {len(ideas)} ideas in {ideas_path}"
        )
    experiments_root.mkdir(parents=True, exist_ok=True)

    # Agents SDK / Codex MCP configuration
    codex_mcp_cfg = config.get("codex_mcp", {})
    agents_sdk_cfg = config.get("agents_sdk", {})
    max_turns = int(agents_sdk_cfg.get("max_turns", 30))

    for i in range(iterations):
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{i:02d}"
        exp_dir = experiments_root / run_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        worktree_path = create_worktree(repo_root, worktree_root, run_id)
        try:
            if config.get("base_on_working_tree", False):
                sync_working_tree(repo_root, worktree_path)

            planner_context = _build_repo_context_for_role(repo_root, "planner", planner_files_path)
            coder_context = _build_repo_context_for_role(repo_root, "coder", coder_files_path)
            reviewer_context = _build_repo_context_for_role(repo_root, "reviewer", reviewer_files_path)
            idea_text = ideas[i]
            _write_text(exp_dir / "idea.md", idea_text)

            # Planner: produce plan text
            planner_prompt = _render_prompt(
                _read_text(planner_prompt_path),
                idea_text=idea_text,
                repo_context=planner_context,
            )
            _write_text(exp_dir / "planner_prompt.txt", planner_prompt)
            planner_system = _read_text(planner_system_path) if planner_system_path else SYSTEM_PROMPT
            plan_text = ""
            if args.dry_run_llm:
                plan_text = "PLAN:\n1. (dry-run) No-op.\nRISKS:\n- dry-run\n"
            else:
                _ensure_trace_processor_registered()
                with _tracing_scope(exp_dir=exp_dir, run_id=run_id, step="planner"):
                    plan_text = await _agents_sdk_generate_text(
                        config.get("agents", {}).get("planner", {}),
                        system_prompt=planner_system,
                        user_prompt=planner_prompt,
                        max_turns=max_turns,
                    )
            _write_text(exp_dir / "plan.md", plan_text)

            # Coder <-> Reviewer repair loop (in-place edits in the same worktree via Codex MCP)
            coder_system = _read_text(coder_system_path) if coder_system_path else SYSTEM_PROMPT
            reviewer_system = _read_text(reviewer_system_path) if reviewer_system_path else SYSTEM_PROMPT
            max_review_rounds = args.max_review_rounds
            if max_review_rounds is None:
                max_review_rounds = config.get("max_review_rounds", 2)
            max_review_rounds = int(max_review_rounds)

            review_rounds = []
            diff_text = ""
            coder_output = ""
            review_text = ""
            review_issues = ""
            verdict = "UNKNOWN"

            for round_idx in range(max_review_rounds + 1):
                coder_template_path = coder_prompt_path if round_idx == 0 else coder_fix_prompt_path
                coder_prompt = _render_prompt(
                    _read_text(coder_template_path),
                    idea_text=idea_text,
                    plan_text=plan_text,
                    repo_context=coder_context,
                    review_text=review_text,
                    review_issues=review_issues,
                    prev_diff_text=diff_text,
                )
                _write_text(exp_dir / f"coder_prompt_round_{round_idx}.txt", coder_prompt)

                if args.dry_run_llm:
                    coder_output = "(dry-run) skipped Codex edits."
                else:
                    _ensure_trace_processor_registered()
                    with _tracing_scope(exp_dir=exp_dir, run_id=run_id, step="coder", round_idx=round_idx):
                        coder_output = await _codex_mcp_edit_repo(
                            worktree_path=worktree_path,
                            codex_mcp_cfg=codex_mcp_cfg,
                            agent_cfg=config.get("agents", {}).get("coder", {}),
                            system_prompt=coder_system,
                            user_prompt=coder_prompt,
                            max_turns=max_turns,
                        )
                _write_text(exp_dir / f"coder_output_round_{round_idx}.txt", coder_output)

                diff_text = _git_diff_with_untracked(worktree_path)
                _write_text(exp_dir / f"diff_round_{round_idx}.diff", diff_text)

                if not diff_text.strip():
                    verdict = "REJECT"
                    review_text = (
                        "VERDICT: REJECT\n"
                        "ISSUES:\n"
                        "- coder produced no changes (empty git diff)\n"
                        "NOTES:\n"
                        "- ensure the coder edits files in-place using Codex MCP\n"
                    )
                    review_issues = "coder produced no changes"
                    _write_text(exp_dir / f"review_round_{round_idx}.md", review_text)
                    review_rounds.append(
                        {
                            "round": round_idx,
                            "diff_present": False,
                            "review_verdict": verdict,
                            "review_issues": review_issues,
                        }
                    )
                    continue

                review_prompt = _render_prompt(
                    _read_text(reviewer_prompt_path),
                    idea_text=idea_text,
                    plan_text=plan_text,
                    patch_text=diff_text,
                    repo_context=reviewer_context,
                )
                _write_text(exp_dir / f"reviewer_prompt_round_{round_idx}.txt", review_prompt)
                if args.dry_run_llm:
                    review_text = "VERDICT: REJECT\nISSUES:\n- dry-run\nNOTES:\n- dry-run\n"
                else:
                    _ensure_trace_processor_registered()
                    with _tracing_scope(exp_dir=exp_dir, run_id=run_id, step="reviewer", round_idx=round_idx):
                        review_text = await _agents_sdk_generate_text(
                            config.get("agents", {}).get("reviewer", {}),
                            system_prompt=reviewer_system,
                            user_prompt=review_prompt,
                            max_turns=max_turns,
                        )
                _write_text(exp_dir / f"review_round_{round_idx}.md", review_text)
                verdict = _parse_reviewer_verdict(review_text)
                review_issues = _extract_reviewer_issues(review_text)
                review_rounds.append(
                    {
                        "round": round_idx,
                        "diff_present": True,
                        "review_verdict": verdict,
                        "review_issues": review_issues,
                    }
                )

                if verdict == "APPROVE":
                    break

            tests_exit = None
            sweep_exit = None
            candidate_csv = exp_dir / "meta_config_sweep_results.csv"
            score_result = None

            proceed = bool(diff_text.strip()) and verdict == "APPROVE"

            # Tests
            test_cmd = config.get("test_command")
            if proceed and test_cmd:
                tests_log = exp_dir / "tests.log"
                test_cwd_value = config.get("test_cwd", ".")
                test_cwd = Path(test_cwd_value)
                if not test_cwd.is_absolute():
                    test_cwd = Path(worktree_path) / test_cwd
                test_pattern = config.get("test_pattern")
                final_cmd = test_cmd
                if test_pattern and isinstance(test_cmd, str):
                    final_cmd = f"{test_cmd} {test_pattern}"
                elif test_pattern and isinstance(test_cmd, list):
                    final_cmd = test_cmd + [test_pattern]
                tests_exit = _run_command(final_cmd, test_cwd, tests_log)
                proceed = proceed and tests_exit == 0

            # Sweep
            if proceed:
                sweep_log = exp_dir / "sweep.log"
                run_results_csv = results_csv
                env_extra = None
                if agentic_output_root:
                    run_output_dir = agentic_output_root / f"run_{i}"
                    run_output_dir.mkdir(parents=True, exist_ok=True)
                    run_results_csv = run_output_dir / "meta_config_sweep_results.csv"
                    env_extra = {
                        "AGENTIC_OUTPUT_DIR": str(run_output_dir),
                        "AGENTIC_RESULTS_CSV": str(run_results_csv),
                    }
                sweep_exit = _run_sweep(config, config_path, worktree_path, sweep_log, env_extra=env_extra)

                if run_results_csv.exists():
                    shutil.copy2(run_results_csv, candidate_csv)

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
                "review_verdict": verdict,
                "review_rounds": review_rounds,
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
    parser.add_argument("--max-review-rounds", type=int, default=None)
    return parser.parse_args()


def _import_agents_sdk():
    try:
        from agents import Agent, Runner, RunConfig, set_default_openai_api  # type: ignore
        try:
            from agents import ModelSettings  # type: ignore
        except Exception:  # pylint: disable=broad-except
            ModelSettings = None  # type: ignore
        from agents.mcp import MCPServerStdio  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for Codex MCP workflow. Install:\n"
            "  pip install openai-agents openai\n"
            "And ensure Node/Codex CLI are available (e.g. `npx -y codex mcp-server`)."
        ) from exc
    return Agent, Runner, RunConfig, ModelSettings, set_default_openai_api, MCPServerStdio


def _build_reasoning(effort):
    if not effort:
        return None
    try:
        from openai.types.shared import Reasoning  # type: ignore

        return Reasoning(effort=str(effort))
    except Exception:  # pylint: disable=broad-except
        return None


def _openai_model_supports_temperature(model):
    m = (model or "").lower().strip()
    # Reasoning-first models generally do not accept temperature.
    return not (m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"))


def _build_model_settings(ModelSettings, cfg, model):
    if not ModelSettings:
        return None
    kwargs = {}
    temperature = cfg.get("temperature", None)
    if temperature is not None and _openai_model_supports_temperature(model):
        kwargs["temperature"] = temperature
    reasoning = _build_reasoning(cfg.get("reasoning"))
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    if not kwargs:
        return None
    try:
        return ModelSettings(**kwargs)
    except TypeError:
        return None


def _create_agent(Agent, *, name, model, instructions, mcp_servers=None, model_settings=None):
    kwargs = {
        "name": name,
        "model": model,
        "instructions": instructions,
    }
    if mcp_servers:
        kwargs["mcp_servers"] = mcp_servers
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    try:
        return Agent(**kwargs)
    except TypeError:
        kwargs.pop("model_settings", None)
        return Agent(**kwargs)


def _build_run_config(RunConfig):  # noqa: N802
    ctx = _TRACE_CONTEXT.get() or {}
    group_id = ctx.get("run_id")
    step = ctx.get("step")
    round_idx = ctx.get("round_idx")

    metadata = {}
    if group_id is not None:
        metadata["run_id"] = str(group_id)
    if step is not None:
        metadata["step"] = str(step)
    if round_idx is not None:
        metadata["round_idx"] = str(round_idx)

    kwargs = {
        "workflow_name": "algo-meta-model-agent/multi_agent_runner",
        "group_id": str(group_id) if group_id is not None else None,
        "trace_metadata": metadata,
    }
    # Prefer to include tool I/O in traces (which also improves what we can log locally).
    # If the installed Agents SDK doesn't support this arg, fall back silently.
    try:
        kwargs["trace_include_sensitive_data"] = True
        return RunConfig(**kwargs)
    except TypeError:
        kwargs.pop("trace_include_sensitive_data", None)
        return RunConfig(**kwargs)


async def _agents_sdk_generate_text(agent_cfg, *, system_prompt, user_prompt, max_turns):
    Agent, Runner, RunConfig, ModelSettings, set_default_openai_api, _ = _import_agents_sdk()
    set_default_openai_api(os.getenv("OPENAI_API_KEY"))

    model = agent_cfg.get("model") or "gpt-5"
    model_settings = _build_model_settings(ModelSettings, agent_cfg or {}, model)
    agent = _create_agent(Agent, name="Text Agent", model=model, instructions=system_prompt, model_settings=model_settings)
    result = await Runner.run(agent, user_prompt, max_turns=max_turns, run_config=_build_run_config(RunConfig))
    return (getattr(result, "final_output", None) or "").strip() + "\n"


async def _codex_mcp_edit_repo(*, worktree_path, codex_mcp_cfg, agent_cfg, system_prompt, user_prompt, max_turns):
    Agent, Runner, RunConfig, ModelSettings, set_default_openai_api, MCPServerStdio = _import_agents_sdk()
    set_default_openai_api(os.getenv("OPENAI_API_KEY"))

    command = (codex_mcp_cfg or {}).get("command") or "npx"
    args = (codex_mcp_cfg or {}).get("args") or ["-y", "codex", "mcp-server"]
    client_session_timeout_seconds = int((codex_mcp_cfg or {}).get("client_session_timeout_seconds", 360000))

    model = (agent_cfg or {}).get("model") or "gpt-5"
    model_settings = _build_model_settings(ModelSettings, agent_cfg or {}, model)

    prev = os.getcwd()
    os.chdir(worktree_path)
    try:
        async with MCPServerStdio(
            name="Codex CLI",
            params={"command": command, "args": args},
            client_session_timeout_seconds=client_session_timeout_seconds,
        ) as codex_mcp_server:
            agent = _create_agent(
                Agent,
                name="Coder",
                model=model,
                instructions=system_prompt,
                mcp_servers=[codex_mcp_server],
                model_settings=model_settings,
            )
            result = await Runner.run(agent, user_prompt, max_turns=max_turns, run_config=_build_run_config(RunConfig))
            return (getattr(result, "final_output", None) or "").strip() + "\n"
    finally:
        os.chdir(prev)


def _parse_reviewer_verdict(text):
    first_line = (text or "").strip().splitlines()[0] if text else ""
    upper = first_line.upper()
    if "APPROVE" in upper:
        return "APPROVE"
    if "REJECT" in upper:
        return "REJECT"
    return "UNKNOWN"


def _extract_reviewer_issues(text):
    if not text:
        return ""
    lines = text.splitlines()
    issues = []
    in_issues = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper() == "ISSUES:":
            in_issues = True
            continue
        if stripped.upper() == "NOTES:":
            break
        if in_issues:
            if stripped.startswith("-"):
                issues.append(stripped[1:].strip())
            elif stripped:
                issues.append(stripped)
    return "\n".join(issues).strip()


def _run_command(cmd, cwd, log_path):
    with open(log_path, "w", encoding="utf-8") as f:
        if isinstance(cmd, list):
            proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=False)
        else:
            proc = subprocess.run(cmd, cwd=cwd, shell=True, stdout=f, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def _git_diff_with_untracked(worktree_path):
    worktree_path = Path(worktree_path)
    base = _run_capture_git(["git", "diff", "--no-color"], cwd=worktree_path)
    untracked = _run_capture_git(["git", "ls-files", "--others", "--exclude-standard"], cwd=worktree_path).splitlines()
    chunks = [base.rstrip()]
    for rel in [p.strip() for p in untracked if p.strip()]:
        # `--no-index` emits a standard `diff --git ...` header; exit code is 1 when diffs exist.
        extra = _run_capture_git(
            ["git", "diff", "--no-color", "--no-index", "--", "/dev/null", rel],
            cwd=worktree_path,
            allow_exit_codes={0, 1},
        ).rstrip()
        if extra:
            chunks.append(extra)
    combined = "\n\n".join([c for c in chunks if c]).strip()
    return combined + ("\n" if combined else "")


def _run_capture_git(cmd, cwd, allow_exit_codes=None):
    if allow_exit_codes is None:
        allow_exit_codes = {0}
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode not in allow_exit_codes:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(detail or f"git command failed: {' '.join(cmd)}")
    return proc.stdout or ""


def _build_repo_context_for_role(repo_root, role, files_path):
    repo_root = Path(repo_root)
    keep = []
    if files_path and Path(files_path).exists():
        raw = _read_text(files_path)
        for line in raw.splitlines():
            rel = line.strip()
            if not rel:
                continue
            p = repo_root / rel
            if p.exists():
                keep.append(rel)
    if not keep:
        return f"Role: {role}\nRepo root: {repo_root}"
    lines = [
        f"Role: {role}",
        f"Repo root: {repo_root}",
        "Key files:",
        "\n".join(keep),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
