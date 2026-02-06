# Log Consolidation Implementation Plan

Goal: consolidate log outputs into a dedicated `logs/` tree under each `tree_run_id`, while keeping high-volume raw logs (sweep/tests/Codex CLI) separate and adding a small number of structured JSONL event streams.

## Phase 0 - Agree on Layout and Config
1. Adopt the log root: `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/logs/`.
2. Confirm naming for consolidated logs:
   - `logs/run_events.jsonl`
   - `logs/conversations/<conversation_id>.jsonl`
   - `logs/conversations/conversation_debug.jsonl`
   - `logs/evals/<eval_id>/eval_events.jsonl`
3. Decide whether to move existing artifacts or only new runs.

## Phase 1 - Add Central Log Path Helpers
1. Add helper(s) to resolve log paths from `run_root`, `eval_id`, `node_id`:
   - `logs_root(run_root)`
   - `eval_logs_root(run_root, eval_id)`
   - `idea_logs_root(run_root, node_id)`
2. Add helpers to create log directories idempotently.
3. Add helpers to append JSONL with a standard envelope:
   - `ts`, `level`, `component`, `run_id`, `eval_id`, `node_id`, `event`, `message`, `payload`.

## Phase 2 - Tree Runner Log Migration
1. Change `tree_runner.log` output to `logs/run_events.jsonl` with structured events.
2. Move conversation logs to `logs/conversations/`:
   - `<conversation_id>.jsonl`
   - `conversation_debug.jsonl`
3. Update manifest fields to point to the new log paths.

## Phase 3 - Eval and Subprocess Logs
1. Write `multi_agent_runner.subprocess.log` to:
   - `logs/evals/<eval_id>/multi_agent_runner.subprocess.log`
2. Write `sweep.log` and `tests.log` to:
   - `logs/evals/<eval_id>/sweep.log`
   - `logs/evals/<eval_id>/tests.log`
3. Keep raw sweep/test logs unmerged to avoid enormous run-level logs.

## Phase 4 - Agents SDK and Codex Logs
1. Move/duplicate:
   - `agents_trace.jsonl` to `logs/evals/<eval_id>/agents_trace.jsonl`
   - `codex_mcp_transcript.jsonl` to `logs/evals/<eval_id>/codex_mcp_transcript.jsonl`
2. Move/duplicate Codex CLI outputs under:
   - `logs/evals/<eval_id>/codex_cli/`
3. Update any references in summary or manifests to the new paths.

## Phase 5 - Idea Generation Logs
1. Redirect `generate_ideas.subprocess.log` to:
   - `logs/idea_generation/generate_ideas.subprocess.node_<node_id>.log`
2. Keep prompt dumps under:
   - `logs/idea_generation/prompts/`
3. Keep Claude debug logs under:
   - `logs/idea_generation/claude_debug/`

## Phase 6 - OpenAI Direct LLM Debug Logs
1. Default log path to:
   - `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/logs/llm/openai_llm_debug.log`
2. Ensure `LLM_DEBUG_FILE` is set for each run (so logs are run-scoped).

## Phase 7 - Documentation Updates
1. Update `TREE_RUNNER_GUIDE.md` with the new log layout.
2. Add a short "Log Map" section that lists all paths and contents.

## Phase 8 - Validation and Guardrails
1. Add a validation check that expected log files are created when their features are used.
2. Add unit tests for log path resolution and JSONL append formatting.
