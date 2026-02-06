# Tree Runner User Guide (`tree_runner.py`)

`agentic_experimentation/tree_runner.py` runs a **beam search over meta-model code states**. Each "node" is a specific git commit plus its sweep results CSV, and each "edge" is "apply one idea -> implement (planner/coder/reviewer) -> sweep -> score".

This is built on top of the existing multi-agent loop in `agentic_experimentation/multi_agent_runner.py` (planner -> coder -> reviewer -> tests -> sweep -> scoring).

---

## What It Does (Mental Model)

- **Node**: a meta-model code state (git commit) + a baseline sweep CSV for that state.
- **Expansion**: generate `K = --ideas-per-node` ideas for each node in the current frontier, then evaluate them.
- **Gate** (parent-relative): a candidate can be promoted only if it does **not regress** the primary metric (`core_topN_sharpe`) and is recommended to explore (or is `grade=="mixed"` with no primary regression).
- **Rank** (root-relative): promoted candidates are ranked globally using root-relative scoring, then the **top `B = --beam-width`** become the next frontier (global beam search).
- **Depth**: repeat until `--max-depth` or other stop conditions.

Important: **one invocation advances at most one depth**. Re-run the command to continue the same tree run.

---

## Prerequisites

### 1) Agent config

`tree_runner.py` reads `agentic_experimentation/agent_config.json` via `--config`. The config must include:

- `baseline_csv`: path to the baseline sweep CSV for the *root* node
- anything required by `multi_agent_runner.py` (planner/coder/reviewer backends, etc.)

### 2) Clean root working tree (for new runs)

Starting a **new** tree run requires a clean root repo working tree (so each node is defined by an immutable commit):

- commit or stash local changes before the first run

Resuming an existing run (manifest already exists) does not require the root tree to be clean.

---

## Quick Start

### Start a new run

From repo root:

```powershell
python agentic_experimentation/tree_runner.py `
  --tree-run-id 20260204_090000 `
  --ideas-per-node 7 `
  --beam-width 1 `
  --max-depth 3 `
  --sweep-config-limit 25
```

This creates a run directory:

- `agentic_experimentation/worktrees/tree_runs/20260204_090000/`

### Continue / resume the same run

Re-run with the same `--tree-run-id`:

```powershell
python agentic_experimentation/tree_runner.py --tree-run-id 20260204_090000
```

Stop when the printed JSON output shows `stop_reason` is set (or `current_depth >= max_depth`).

---

## Key CLI Options

### Search shape

- `--ideas-per-node K`: number of ideas to generate/evaluate per node expansion
- `--beam-width B`: number of promoted nodes to keep globally per depth (global beam)
- `--max-depth D`: number of promotion rounds to run

### Determinism / speed (recommended)

- `--sweep-config-limit N`: evaluate only `config_id < N` in sweeps and scoring (deterministic subset)

`tree_runner.py` uses strict completeness checks when `N` is set (prevents "improving by failing").

### Budget / stopping

- `--max-total-idea-evals M`: hard cap on evaluations across the entire run
- `--stop-on-empty-frontier` / `--no-stop-on-empty-frontier`: stop if no candidates pass the gate

### Dedupe (repeat-avoidance)

- `--dedupe-scope node_plus_ancestors` (default): don't re-try identical idea text already present in this node's ancestor idea sets
- `--dedupe-scope global`: dedupe against all node idea sets in the run
- `--dedupe-scope none`: disable dedupe

Skipped duplicates are recorded under `nodes[<node_id>].dedupe.skipped` in `manifest.json`.

### Debug retention

- `--keep-rejected-worktrees`: keep candidate worktrees (otherwise they're pruned)
- `--keep-failed-artifacts`: keep failed eval artifacts (otherwise they're pruned)

### Idea conversation continuation

These flags control branch-aware idea-generation memory:

- `--idea-conversation-mode off|auto|native|replay`
  - `off`: no conversation continuity.
  - `auto`: prefer provider-native continuation if available, otherwise replay branch memory.
  - `native`: force provider-native continuation.
  - `replay`: force prompt replay of compact summary + recent turns.
- `--idea-history-window-turns N`: max recent turns to replay (`replay`/`auto` fallback).
- `--idea-history-max-chars C`: replay memory char budget.
  - `-1`: unbounded replay memory block.
  - `0`: disable replay memory block.
  - `>0`: bounded replay memory block size.

Conversation lineage model:

- One conversation per node (`conversation_id` usually `node_<node_id>`).
- Child conversations fork from the parent node checkpoint turn (`fork_from_turn_id`).
- Sibling branches are isolated; each branch continues from its own conversation file.

Where conversation artifacts live:

- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/conversations/<conversation_id>.json`
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/conversations/<conversation_id>.jsonl`
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/conversations/<conversation_id>_summary.md`
- Optional debug event log:
  - `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/conversations/conversation_debug.jsonl`
  - path also recorded in `manifest.json` at `conversation_config.debug_log_jsonl_path`.

### Locking / crash safety

- `--lock-stale-seconds S`: treat a run lock as stale after S seconds without heartbeat
- `--force`: take over an existing lock (records a lock takeover event)

### Mode examples

Disable continuation:

```powershell
python agentic_experimentation/tree_runner.py --tree-run-id 20260206_090000 --idea-conversation-mode off
```

Default branch continuation (`auto`):

```powershell
python agentic_experimentation/tree_runner.py --tree-run-id 20260206_090000 --idea-conversation-mode auto
```

Force replay memory:

```powershell
python agentic_experimentation/tree_runner.py `
  --tree-run-id 20260206_090000 `
  --idea-conversation-mode replay `
  --idea-history-window-turns 16 `
  --idea-history-max-chars 24000
```

---

## Outputs and Where To Look

Inside `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/`:

- `manifest.json`: source-of-truth state for resume/audit
- `TREE_SUMMARY.md`: human-readable summary (best path + per-node table)
- `TREE_GRAPH.md`: Mermaid graph of nodes/edges + frontier (for Markdown viewers with Mermaid support)
- `VALIDATION_REPORT.md`: integrity checks (paths, sha256, state consistency)
- `artifacts/`: copied sweep CSV artifacts (baseline + candidates), with sha256 provenance
- `node_ideas/`: per-node idea files (see "Idea Management" below)
- `eval/<eval_id>/experiment/`: per-eval multi-agent logs and outputs
- `wt/<node_id>/`: worktrees for promoted nodes (the beam)
- `cand/<eval_id>/`: temporary candidate worktrees (usually pruned)

---

## Logs and Traces (Where To Debug)

There is no single "app.log" for `tree_runner.py`. Instead, debugging information is spread across a few locations.

### 1) Tree-run level state + reports

Check these first to understand what the tree thinks happened:

- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/manifest.json`: full state, eval records, promotions, errors, and events
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/TREE_SUMMARY.md`: high-level status + best path
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/VALIDATION_REPORT.md`: integrity checks (missing paths, sha256 mismatches, frontier/expanded consistency, etc.)
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/run.lock.json`: lock + heartbeat metadata (useful if you hit a stale/active lock error)
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/tree_runner.log`: append-only orchestration log for `tree_runner.py` itself
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/conversations/conversation_debug.jsonl`: conversation recovery/attach/fork debug events (if configured)

Tip: `tree_runner.py` prints a small JSON summary to stdout at the end of each invocation. If you want a persistent record of that, capture stdout/stderr when you run it.

### 2) Per-evaluation multi-agent logs (most useful)

Each evaluated idea gets its own experiment folder at:

- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/eval/<eval_id>/experiment/<eval_id>/`

Common files to inspect:

- `summary.json`: approved status, exit codes, result paths, scoring summary
- `planner_prompt.txt`, `plan.md`
- `review_round_*.md` (and optionally `_sanitized.md`, `_dropped_new_issues.txt`)
- `tests.log` (if tests are enabled)
- `sweep.log` (sweep stdout/stderr)
- `meta_config_sweep_results.csv` (candidate sweep results copied into the experiment folder)

Additionally, `tree_runner.py` tees the stdout/stderr of the multi-agent subprocess into:

- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/eval/<eval_id>/multi_agent_runner.subprocess.log`

If the coder backend uses Codex / MCP, also check:

- `codex_mcp_transcript.jsonl`
- `codex_cli_events_round_*.jsonl`, `codex_cli_stderr_round_*.log`, `codex_cli_cmd_round_*.txt`
- `coder_prompt_round_*.txt`, `coder_output_round_*.txt`, `diff_round_*.diff`

If Agents SDK tracing is enabled, a local JSONL trace is also written:

- `agents_trace.jsonl`

### 3) Idea generation logs (when ideas look low-quality or duplicated)

The ideas generated for each node are stored under:

- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/node_ideas/<node_id>/`

If `tree_runner.py` had to generate missing ideas for a node, it also tees the stdout/stderr of the idea generation subprocess into:

- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/node_ideas/<node_id>/generate_ideas.subprocess.log`

If you want to inspect the raw prompt/context used during idea generation, check:

- `agentic_experimentation/idea_generation/.idea_generation_logs/` (e.g. `prompt_*.txt`, `claude_debug_*.log`)

### 4) Sweep output outside the tree run (optional)

If `agent_config.json` sets `agentic_output_root`, sweeps may also write to an external per-run directory (in addition to what is copied into the tree run's `eval/...` folders). Use `summary.json` to locate the exact output path.

### 5) Phoenix / Arize traces (external)

If Phoenix tracing is enabled (via `PHOENIX_*` env vars and/or agent config), LLM/tool spans are emitted by:

- `agentic_experimentation/multi_agent_runner.py` (planner/reviewer and tool steps; coder depends on backend)
- `agentic_experimentation/idea_generation/generate_ideas.py` (idea generation calls)

These traces appear in Phoenix/Arize, not as local log files.

---

## Idea Management (Storage, Tracking, and Branch Membership)

### Where ideas are stored (per node)

For each node, the runner stores/generated ideas under:

- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/node_ideas/<node_id>/`

These are ordinary `*.md` files (for example `idea_000.md`, `idea_001.md`, ...). They are not stored in git; they're run artifacts.

### Do we keep track of ideas? Yes (in `manifest.json`)

The runner keeps an auditable record of:

- **Per node**
  - `nodes[<node_id>].node_ideas_dir`: directory where the node's idea files live
  - `nodes[<node_id>].generated_idea_files`: the `K` idea files that were selected for evaluation this depth (stable order)
  - `nodes[<node_id>].context_ideas_dirs`: the idea directories provided as context to idea generation for this node
  - `nodes[<node_id>].context_idea_files`: a snapshot of the idea files (and sha256) that existed in the context dirs when expanding the node
- **Per evaluation**
  - `evaluations[<eval_id>].idea_path`: the specific idea file evaluated
  - `evaluations[<eval_id>].parent_node_id`: which node generated/owned the evaluation
  - `evaluations[<eval_id>].experiment_dir`: where the multi-agent runner logs/artifacts were written for that evaluation

### Do child nodes "remember" parent ideas? Yes (two different ways)

There are two separate concepts:

1) **Generation context (broad, includes rejected ideas)**  
When generating new ideas for a node, the runner passes `--context-ideas-dir` pointing at the node's ancestor `node_ideas_dir` directories (including the node itself). This means the generator can "see" *all* prior ideas from the branch, including ideas that were tested but not promoted.

2) **Branch membership / accepted path (narrow, promoted ideas only)**  
When an evaluation is promoted into a new node, the new node records an **accepted chain**:

- `nodes[<node_id>].idea_chain`: ordered list of `idea_path` values that define the branch (only promoted ideas)

This is the canonical answer to "which ideas are now part of this branch?"

### How to find "the idea that created this node"

For a promoted node:

- Look at `nodes[<node_id>].artifacts.promoted_from_eval_id` to get the eval id.
- Then read `evaluations[<eval_id>].idea_path` (the actual idea file).

Or just look at the last element of `nodes[<node_id>].idea_chain`.

### Promotion bookkeeping (what gets written where)

On promotion:

- `evaluations[<eval_id>].decision.promoted_to_node_id` is set to the new node id
- `nodes[<new_node_id>].artifacts.promoted_from_eval_id` points back to the originating evaluation
- `nodes[<new_node_id>].baseline_results_csv_path` becomes the promoted candidate's sweep results CSV (this is the baseline for any children)

---

## Report / Validate Without Running Anything

These operate on an existing run (manifest already exists):

```powershell
python agentic_experimentation/tree_runner.py --tree-run-id 20260204_090000 --report-only
python agentic_experimentation/tree_runner.py --tree-run-id 20260204_090000 --validate-only
```

---

## Dry Run (No LLM, No Sweep)

Use `--dry-run` to simulate eval records and promotions deterministically (useful for testing tree mechanics).

```powershell
python agentic_experimentation/tree_runner.py `
  --tree-run-id 20260204_dry `
  --ideas-per-node 3 `
  --beam-width 1 `
  --max-depth 2 `
  --sweep-config-limit 25 `
  --dry-run
```

Note: starting a new dry-run still requires a clean root working tree.

---

## Common Troubleshooting

### "Root working tree must be clean..."

You're starting a new run and have uncommitted changes. Commit or stash, then retry.

### "Run lock is active (not stale)..."

Another `tree_runner` process is running, or it crashed recently. Wait for the lock to become stale, or use `--force`.

### Validation issues

Open `VALIDATION_REPORT.md` in the run folder. It can flag:

- missing experiment dirs / summary.json pointers
- artifacts not under `artifacts/`
- sha256 mismatches for copied artifacts
- frontier/expanded inconsistencies in the manifest state

### Conversation issues

- If a conversation JSON is partially written/corrupt, runner/idea generation recover it to `*.corrupt_<timestamp>` and continue from a fresh state.
- Recovery events are recorded in:
  - `manifest.json -> events[]` (`type=conversation_json_recovery`)
  - `conversations/conversation_debug.jsonl`
- `--idea-conversation-mode auto` currently falls back to replay in most runs unless provider-native continuation capability is explicitly available in conversation state.
- If you need deterministic behavior, use `--idea-conversation-mode replay`.
- If you need no history injection at all, use `--idea-conversation-mode off`.
