# Tree Runner User Guide (`tree_runner.py`)

`agentic_experimentation/tree_runner.py` runs a **beam-search over code states** of the meta model. Each “node” is a specific git commit plus its sweep results CSV, and each “edge” is “apply one idea → implement (planner/coder/reviewer) → sweep → score”.

This is built on top of the existing multi-agent loop in `agentic_experimentation/multi_agent_runner.py` (planner → coder → reviewer → tests → sweep → scoring).

---

## What It Does (Mental Model)

- **Node**: a meta-model code state (git commit) + a baseline sweep CSV for that state.
- **Expansion**: generate `K = --ideas-per-node` ideas for each node in the current frontier, then evaluate them.
- **Gate** (parent-relative): a candidate can be promoted only if it *does not regress* the primary metric and is recommended to explore (or is `grade=="mixed"` with no primary regression).
- **Rank** (root-relative): promoted candidates are ranked globally using root-relative scoring, then the **top `B = --beam-width`** become the next frontier (global beam search).
- **Depth**: repeat until `--max-depth` or other stop conditions.

Important: **one invocation advances at most one depth**. Re-run the command to continue the same tree run.

---

## Prerequisites

### 1) Agent config

`tree_runner.py` reads `agentic_experimentation/agent_config.json` via `--config` (default shown below). The config must include:

- `baseline_csv`: path to the baseline sweep CSV for the *root* node
- `scoring.score_column`: optional (scoring also computes per-column deltas either way)
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

`tree_runner.py` uses strict completeness checks when `N` is set (prevents “improving by failing”).

### Budget / stopping

- `--max-total-idea-evals M`: hard cap on evaluations across the entire run
- `--stop-on-empty-frontier` / `--no-stop-on-empty-frontier`: stop if no candidates pass the gate

### Dedupe (repeat-avoidance)

- `--dedupe-scope node_plus_ancestors` (default): don’t re-try identical idea text already present in this node’s ancestor idea sets
- `--dedupe-scope global`: dedupe against all node idea sets in the run
- `--dedupe-scope none`: disable dedupe

Skipped duplicates are recorded under `nodes[<node_id>].dedupe.skipped` in `manifest.json`.

### Debug retention

- `--keep-rejected-worktrees`: keep candidate worktrees (otherwise they’re pruned)
- `--keep-failed-artifacts`: keep failed eval artifacts (otherwise pruned)

### Locking / crash safety

- `--lock-stale-seconds S`: treat a run lock as stale after S seconds without heartbeat
- `--force`: take over an existing lock (records a lock takeover event)

---

## Outputs and Where To Look

Inside `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/`:

- `manifest.json`: source-of-truth state for resume/audit
- `TREE_SUMMARY.md`: human-readable summary (best path + per-node table)
- `VALIDATION_REPORT.md`: integrity checks (paths, sha256, state consistency)
- `artifacts/`: copied sweep CSV artifacts (baseline + candidates), with sha256 provenance
- `eval/<eval_id>/experiment/`: per-eval multi-agent logs and outputs
- `wt/<node_id>/`: worktrees for promoted nodes (the beam)
- `cand/<eval_id>/`: temporary candidate worktrees (usually pruned)

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

### “Root working tree must be clean…”

You’re starting a new run and have uncommitted changes. Commit or stash, then retry.

### “Run lock is active (not stale)…”

Another `tree_runner` process is running, or it crashed recently. Wait for the lock to become stale, or use `--force`.

### Validation issues

Open `VALIDATION_REPORT.md` in the run folder. It can flag:

- missing experiment dirs / summary.json pointers
- artifacts not under `artifacts/`
- sha256 mismatches for copied artifacts
- frontier/expanded inconsistencies in the manifest state

