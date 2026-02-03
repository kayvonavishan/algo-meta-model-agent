# Tree / Beam-Search Runner Plan (Git Worktrees)

This document describes a plan to extend the current agentic workflow into a higher-level loop that **builds a tree of meta-model states** (nodes) and explores improvements via:

**idea -> implement -> sweep -> score**

using **git worktrees** to isolate each node.

User decisions (locked in):
- Promising criteria: allow `grade == "mixed"` (i.e., results do not need to be perfect to continue).
- Selection strategy: **global beam** (keep the best `beam_width` nodes per depth across the whole frontier).
- Baselines: each node baseline is **its own sweep results** (parent-relative comparisons), while still keeping a root-relative view.
- Beam ranking basis: for global-beam selection, rank using **root-relative** performance so candidates from different parents are comparable.
- Idea context strategy: `node_plus_ancestors` (node-specific ideas, but context includes ancestor ideas).
- Root working tree: require a clean root working tree at tree-run start (each node is defined by a commit).

---

## 1) Reframe the Goal as Search

### Node
A **node** is a specific meta-model code state plus its evaluation artifacts:
- `git_commit`: commit hash for the state
- `worktree_path`: working directory for that state
- `baseline_results_csv`: the sweep results CSV for this node (used as baseline when expanding this node)
- `summary.json` and associated logs
- `idea_chain`: the sequence of ideas applied from root -> this node

### Edge
An **edge** is "apply idea X to node A -> produce node B", including:
- idea source text/path
- implementation loop logs (coder/reviewer rounds)
- sweep/scoring artifacts

### Objective signal
Use the existing scoring output:
- `summary.json["scoring_summary"]["recommendation"]`
  - `should_explore` (boolean)
  - `grade` (`strong` / `promising` / `mixed` / `weak`)
  - `score` (float)
  - `reasons` and metric-level details

---

## 2) Search Strategy: Global Beam Search

Rather than an unbounded tree, use **beam search**:
- Depth 0: start from the root node (current baseline state).
- Expand each node in the current frontier by evaluating `ideas_per_node` ideas.
- Pool all "promising" children across the frontier and keep only the best `beam_width` children as the next frontier.
- Repeat for `max_depth` levels.

This controls growth while still allowing branching when multiple promising directions appear.

### Promising criteria (promotion gate)
A candidate is considered "promising enough to continue" if either:
- `recommendation.should_explore == true`, OR
- `recommendation.grade == "mixed"` **AND** the primary metric has not regressed.

Guardrail for allowing `grade == "mixed"`:
- Do not allow promotion if the recommendation indicates primary regression (e.g. `reasons` contains `primary_metric_regressed`).

### Ranking for global beam (important)
To make global-beam selection meaningful with per-node baselines:
- Gate candidates based on **parent-relative** scoring (did it improve the parent?).
- Rank/select the beam based on **root-relative** scoring (comparable across parents).

---

## 3) Baseline Semantics (Parent-relative + Root-relative)

Each evaluation should compute and record two comparisons:

1) **Parent-relative** (used for local decisions)
- baseline: parent node's `baseline_results_csv`
- candidate: candidate's sweep results CSV
- purpose: "did this idea improve the current state?"

2) **Root-relative** (used for global comparability / sanity)
- baseline: the fixed root baseline CSV (e.g. `agentic_experimentation/baselines/meta_config_sweep_results_baseline.csv`)
- candidate: candidate's sweep results CSV
- purpose: "is this branch actually better than the original baseline?"

When a candidate becomes a node, its own sweep results CSV becomes:
- that node's `baseline_results_csv` for future expansions

---

## 4) Determinism and Fair Comparisons

To ensure apples-to-apples comparisons across the tree:
- Use a consistent `--sweep-config-limit N` for the entire tree-run.
- Align baseline/candidate using the same rule (`config_id < N`) when computing deltas.
- Persist `sweep_config_limit` into the run manifest (and per-node metadata).
- Gate promotions only if the evaluation used the full expected set of rows (strict):
  - if `sweep_config_limit` is set, require `candidate_rows_used == sweep_config_limit`.

---

## 5) Run Manifest (Resume-Friendly State Machine)

Introduce a single JSON manifest that acts as the index for the whole tree run. It enables:
- crash-safe resumption
- reproducibility
- traceability from a node back to the exact logs/commits that produced it

### Suggested manifest fields

**Run config**
- `tree_run_id`
- `created_at`
- `ideas_per_node`
- `max_depth`
- `beam_width`
- `sweep_config_limit`
- `max_total_idea_evals` (hard budget)
- any existing agent config that impacts behavior (max review rounds, proceed-on-max, etc.)

**Root baseline**
- `root_baseline_csv_path`
- `root_commit`

**Nodes**
- `nodes[node_id] = { parent_node_id, depth, commit, worktree_path, baseline_results_csv_path, idea_chain, status, created_at }`

**Evaluations**
- each evaluation record contains:
  - `parent_node_id`, `idea_id`, `idea_text_path`
  - `status` (`pending`/`running`/`completed`/`failed`)
  - `experiment_dir`
  - `candidate_commit`
  - `candidate_results_csv_path`
  - extracted parent-relative and root-relative `recommendation`
  - error information if failed

### Atomic writes
Write manifest updates atomically (write temp -> rename) so partial writes do not corrupt state.

---

## 6) Git Worktree Layout and Lifecycle

Treat git worktrees as the physical representation of nodes.

Suggested filesystem layout (single run root; keep paths short on Windows):
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/wt/<node_id>/` (node worktrees)
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/cand/<eval_id>/` (candidate worktrees; temporary)
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/node_ideas/<node_id>/` (generated ideas per node)
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/eval/<eval_id>/` (per-eval logs/output roots)
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/artifacts/` (copied CSVs, checksums, provenance)
- `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/manifest.json`

Worktree policy:
- Keep worktrees for nodes that survive pruning (beam).
- Delete worktrees (and branches) for discarded candidates to control disk usage.

Commit policy (reachability):
- Promoted nodes must have a reachable commit (keep a branch or tag pointing at it).

Promotion policy (recommended):
- Do not rename/move worktree folders.
- On promotion, create a fresh node worktree at the candidate commit and delete the candidate worktree.

---

## 7) Node Expansion: Evaluate K Ideas from This State

Given a node:

1) Generate `ideas_per_node` ideas
- Run `agentic_experimentation/idea_generation/generate_ideas.py`.
- Ensure it supports node-specific output (e.g. `--output-dir` / `--ideas-dir`) so each node's ideas are isolated.
- Provide node-specific context so ideas are relevant:
  - short `git diff --stat` vs root (or parent)
  - last node summary deltas / recommendation reasons
  - constraints (single-change, implementable, etc.)

2) For each idea
- Create a candidate branch/worktree from the node's commit.
- Run the existing coder/reviewer loop to implement the idea.
- Run the sweep (`--sweep-config-limit N` if configured).
- Compute scoring summaries (parent-relative and root-relative).

3) Filter + rank + select
- Gate: allow if `should_explore==true` OR (`grade=="mixed"` and no primary regression).
- Rank (global beam): by root-relative `recommendation.score`, with tie-breakers if needed.
- Enforce strict completeness (required):
  - if `sweep_config_limit` is set, require all `config_id < N` rows have `status == "ok"` before allowing promotion.

---

## 8) Budgeting and Stopping Conditions

To prevent runaway compute:
- `max_total_idea_evals`: stop when reached (even mid-depth).
- Stop early if a depth produces zero candidates that pass the gate.

Recommended starting values:
- `beam_width = 1`
- `ideas_per_node = 5..7`
- `max_depth = 2..3`

---

## 9) Traceability

Each evaluation should have a stable trail:
- idea file used
- `review_round_*.md` logs
- `sweep.log`
- `summary.json`
- baseline/candidate CSV paths
- commit hashes for parent and candidate

Manifest should link all of these.

---

## 10) Suggested Delivery Milestones

1) MVP (single-threaded beam search)
- manifest + resume
- depth-limited, global beam, per-node baselines

2) Node-specific idea generation context
- generator prompt includes current-node deltas / diff context

3) Parallelism
- parallelize within a node (ideas) and/or across nodes (frontier), respecting rate limits

4) Better promotion policy
- refine how `mixed` is treated (e.g. require primary metric non-negative)

5) Reporting and validation (recommended)
- generate `TREE_SUMMARY.md` from `manifest.json` (report-only mode)
- validate artifacts (paths + sha256 + reachable refs) so promoted nodes are reproducible

6) Parallelism and scaling (recommended after MVP)
- add controlled parallel eval execution with a single-writer manifest coordinator
- enforce budgets/timeouts and avoid output collisions
