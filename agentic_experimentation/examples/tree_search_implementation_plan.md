# Tree / Beam-Search Runner - Implementation Plan

This is a detailed, phased implementation plan for building a "tree of model states" runner that:
- Uses git worktrees to represent nodes (code states).
- Uses global beam search to control branching.
- Treats each node baseline as its own sweep results (parent-relative baseline), while also tracking root-relative performance.
- Promotes candidates if `recommendation.should_explore == true` OR (`recommendation.grade == "mixed"` AND primary metric did not regress).
- Ranks the global beam using root-relative performance (so candidates from different parents are comparable).

Related overview doc: `agentic_experimentation/examples/tree_search_plan.md`.

---

## Phase 0 - Lock Interfaces and Definitions (No Behavioral Changes Yet)

1) Pick the worktree ownership model (resolve the "double-worktree" problem)
- Recommended: introduce a new `tree_runner.py` that owns node/candidate worktrees, and add an "in-place" mode to the existing runners so they do not create nested worktrees.
  - Add `--no-worktree` (or `--in-place`) to `agent_runner.py` and/or `multi_agent_runner.py` so they can operate directly in an already-created worktree directory.
- Alternative (less ideal): reuse `multi_agent_runner.py` worktrees as the only worktree layer and re-architect tree search around "keeping/promoting certain worktrees". This tends to be awkward because you need to start runs from arbitrary node commits and control cleanup precisely.

2) Define CLI / config parameters for the tree runner
- `--ideas-per-node K` (e.g. 7)
- `--max-depth D` (e.g. 3)
- `--beam-width B` (e.g. 1)
- `--sweep-config-limit N` (reuse existing behavior; must be consistent across the tree run)
- `--max-total-idea-evals` (hard cap; prevents explosion)
- `--stop-on-empty-frontier` (default true)
- `--keep-rejected-worktrees` / `--keep-failed-artifacts` (default false)
- `--resume` (default true if manifest exists)
- `--force-regenerate-ideas` (default false)

3) Define promotion gate + ranking policy (make global beam meaningful)

Promotion gate (parent-relative):
- Pass gate if:
  - parent-relative `recommendation.should_explore == true`, OR
  - parent-relative `recommendation.grade == "mixed"` AND primary metric did not regress (numeric check)
- Primary regression rule:
  - do not allow promotion if either:
    - `reasons` contains `primary_metric_regressed`, OR
    - `parent_relative.primary_delta < 0` (optionally allow a tiny tolerance like `>= -1e-6`)
- Strict data-completeness rule (required; prevents "improving by failing"):
  - if `sweep_config_limit` is set, require `candidate_rows_used == sweep_config_limit`
  - if `sweep_config_limit` is not set, define and require a minimum row count (e.g. `>= 100`) before allowing promotion

Beam ranking (root-relative):
- Rank candidates by root-relative `recommendation.score` descending.
- Tie-breakers:
  - root-relative primary metric improvement: `mean_topN_avg_return_per_trade_pct_oos` (candidate - baseline)
  - prefer no risk-regression flags in `reasons` if present
  - optionally: fewer files changed (proxy for complexity)
- Ranking fallback rule (required):
  - if `root_relative.recommendation.score` is missing/NaN, rank by `root_relative.primary_delta`
  - if both are missing, treat as worst (do not promote)

4) Confirm baseline semantics
- Root baseline CSV is fixed (your existing baseline file).
- Node baseline CSV is the node's own sweep results CSV.
- Every evaluation records:
  - parent-relative comparison (parent baseline vs candidate)
  - root-relative comparison (root baseline vs candidate)

5) Add explicit support for node-specific idea output (required)
- Update `agentic_experimentation/idea_generation/generate_ideas.py` to accept an output directory:
  - e.g. `--ideas-dir <path>` or `--output-dir <path>`
- Ensure generated ideas can be isolated per node.
- Add explicit context input control (required to support `node_plus_ancestors`):
  - separate "write output here" from "use these dirs as prior-idea context"
  - example flags:
    - `--ideas-dir <node_output_dir>`
    - `--context-ideas-dir <dir>` (repeatable) or `--context-ideas-dirs <dir1,dir2,...>`
  - default `ideas_context_strategy = "node_plus_ancestors"`

Deliverable for Phase 0:
- A short README section and/or doc updates describing:
  - gating vs ranking (parent-relative gate, root-relative beam rank)
  - worktree ownership model
  - idea output directory support requirement

---

## Phase 1 - Run Manifest + Resume (Control Plane)

1) Introduce a run directory and a run lock
- Create a stable run root:
  - `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/`
- Add a single-run lock file to prevent two processes from mutating the same manifest:
  - `run.lock.json` (PID, hostname, created_at, last_heartbeat_at)
  - Lock liveness policy (Windows-friendly; no new dependencies required):
    - Treat the lock as valid until it is stale.
    - Record `last_heartbeat_at` and update it periodically (e.g., every N seconds or after each eval completes).
    - On startup: if lock exists and `now - last_heartbeat_at < stale_timeout_seconds`, abort (or require `--force`).
    - If lock is stale, allow takeover and record a `lock_takeover` event in the manifest.
  - On clean exit: remove lock (best-effort).
  - Add CLI knobs:
    - `--lock-stale-seconds` (default e.g. 600)
    - `--force` (ignore lock and proceed)
  - Heartbeat schedule (required):
    - update `last_heartbeat_at` at least every 30 seconds while running
    - also update after each eval transitions state (e.g., `running -> completed/failed`)

2) Design manifest schema (must be resumable and decision-auditable)
- `manifest.json` should contain:
  - `manifest_version` (so you can evolve the schema)
  - run config: `tree_run_id`, timestamps, K/D/B/N, budgets, and copies of key agent settings
  - artifact policy: how/where baseline+candidate CSVs are stored (see below)
  - root: `root_commit`, `root_baseline_csv_path`
  - state: `current_depth` and the current frontier (see below)
  - nodes: map `node_id -> node_record`
  - evaluations: list/map of eval records

Node record (minimum):
- `node_id`, `parent_node_id`, `depth`
- `commit` and `ref_name` (reachable: keep a branch/tag pointing at the commit)
- `worktree_path`
- `baseline_results_csv_path`
- `idea_chain` (ordered list)
- pointers to key artifacts (node summary, experiments)
- status fields: `created_at`, `status`

Evaluation record (minimum):
- identifiers: `eval_id`, `parent_node_id`, `depth`, `idea_path`
- artifacts: `experiment_dir`, `candidate_commit`, `candidate_results_csv_path`
- extracted scoring (store both so you can reproduce the decision):
  - Store a compact summary in the manifest for quick ranking/audit:
    - `parent_relative.recommendation_summary = { should_explore, grade, score, reasons }`
    - `root_relative.recommendation_summary = { should_explore, grade, score, reasons }`
  - Store numeric primary deltas explicitly (do not rely only on `reasons` strings):
    - `parent_relative.primary_delta` (e.g. delta in `mean_topN_avg_return_per_trade_pct_oos`)
    - `root_relative.primary_delta`
  - Store rows-used counters (required for strict completeness gating):
    - `parent_relative.baseline_rows_used`
    - `parent_relative.candidate_rows_used`
    - `root_relative.baseline_rows_used`
    - `root_relative.candidate_rows_used`
  - Store pointers to the canonical full details:
    - `parent_relative.summary_json_path` (or a path to a small per-eval scoring JSON)
    - `root_relative.summary_json_path`
- decision fields (make "gate vs rank" explicit):
  - `decision.gate_basis = "parent_relative"`
  - `decision.rank_basis = "root_relative"`
  - `decision.passed_gate` (bool)
  - `decision.rank_score` (root-relative score used for beam ranking)
  - `decision.promotion_reason` (short string for traceability)
  - `decision.primary_regressed` (bool; derived from reasons and/or numeric delta)
- status/error fields
- output routing fields (avoid artifact collisions):
  - `eval_output_root` (a unique folder for this eval's sweep/test outputs)
  - output routing mechanism (must be explicit so the sweep actually writes there):
    - plumb `eval_output_root` into sweep via env vars (already used elsewhere):
      - `AGENTIC_OUTPUT_DIR=<eval_output_root>`
      - `AGENTIC_RESULTS_CSV=<eval_output_root>/meta_config_sweep_results.csv`

State block (required for correct resume without accidental re-expansion):
- `state.current_depth`
- `state.frontier_node_ids` (the nodes to expand at `current_depth`)
- `state.expanded_node_ids_by_depth` (or per-node `expanded_depths`; depth-aware so resume cannot re-expand nodes accidentally)
- `state.completed_depths` (optional)
- `state.next_node_id` / `state.next_eval_id` (or a UUID policy) to ensure deterministic ID allocation on resume

Artifact policy (required to keep baselines stable over time):
- Decide and record one policy in `run_config`:
  - `artifact_policy = "copy_to_run_root"` (recommended): copy the baseline/candidate sweep CSVs into
    `agentic_experimentation/worktrees/tree_runs/<tree_run_id>/artifacts/` and point the manifest to those copies.
  - `artifact_policy = "external_paths"`: store absolute paths to sweep outputs outside the repo and treat them as immutable.
- Reason: node baselines must remain readable for child expansion and for later audit.
- Provenance requirement (recommended):
  - When copying into the run root, record:
    - `source_path`
    - `copied_to_path`
    - a checksum (e.g. sha256) so integrity can be verified later
  - Copy scope (required):
    - copy the *exact* baseline CSV used for parent-relative scoring into the run artifacts dir
    - copy the candidate results CSV used for scoring into the run artifacts dir
    - ensure the manifest points at the copied versions for future child expansions

Idea generation output + context dirs (required for node isolation without losing global context):
- In `run_config`, record:
  - `node_ideas_root_dir` (per-node output location)
  - `ideas_context_strategy` (e.g., "global_prior_ideas", "node_only", "node_plus_ancestors")
  - any additional input dirs used to build prompt context (so runs are reproducible)
  - Additionally record the context inputs actually used per node:
    - list of idea file paths + hashes included in the prompt context

3) Implement atomic manifest writes
- Write to `manifest.json.tmp` then rename to `manifest.json`.
- Never partially update in-place.

4) Implement resume semantics
- If `manifest.json` exists:
  - load it
  - acquire/validate the run lock (see step 1)
  - rebuild the frontier deterministically from `state.frontier_node_ids` (preferred),
    or from node status markers if you choose not to store an explicit frontier
  - detect in-progress evals and decide:
    - mark failed, or
    - retry (optional flag)
- Ensure idempotency:
  - if node ideas already exist, do not regenerate unless forced
  - do not re-run completed evals unless a `--rerun-evals` flag is provided
 - Ensure cleanup robustness on Windows:
   - worktree deletion can fail due to file locks
   - implement retry/backoff and a deferred cleanup queue recorded in the manifest:
     - `state.deferred_cleanup = [{ path, kind: "worktree|branch|dir", reason, attempts, next_retry_at }]`

Deliverable for Phase 1:
- "Hello tree run" that:
  - creates a manifest
  - records a root node
  - can be resumed safely after interruption

---

## Phase 2 - Git Worktrees as Nodes (State Isolation)

0) Preconditions (required for correctness)
- Require a clean root working tree at tree-run start:
  - no uncommitted changes, no untracked files that matter
  - reason: each node must correspond to a well-defined git commit, not an implicit synced workspace state

1) Create standard paths under the run root (keep them short on Windows)
- Node worktrees: `.../wt/<node_id>/`
- Candidate worktrees: `.../cand/<eval_id>/`
- Node ideas: `.../node_ideas/<node_id>/`
- Artifacts: `.../artifacts/` (copied CSVs, checksums)
- Notes:
  - keep `tree_run_id`, `node_id`, and `eval_id` short to avoid Windows path-length issues

2) Implement worktree helpers (thin wrappers)
- Create worktrees from explicit commits (not from syncing the current workspace):
  - `create_node_worktree(node_id, commit, ref_name) -> path`
  - `create_candidate_worktree(eval_id, parent_commit, ref_name_or_detached) -> path`
- `cleanup_worktree(path)`
- Explicitly forbid nested worktrees:
  - the tree runner owns worktrees; inner runners operate in-place in the provided worktree

3) Ref naming policy (avoid collisions across worktrees)
- Git does not allow the same branch to be checked out in multiple worktrees.
- Require globally unique ref names per worktree:
  - Node ref: `tree/<tree_run_id>/n<node_id>` (short)
  - Candidate ref: `tree/<tree_run_id>/e<eval_id>` (short)
- Persist `ref_name` in the manifest so resume reuses the same refs.

4) Promotion policy (recommended; Windows-friendly)
- Do not rename/move worktree folders.
- On promotion:
  - create a new node worktree at the candidate commit under the final node path (`.../wt/<node_id>/`)
  - delete the candidate worktree (`.../cand/<eval_id>/`)
- Reason: moving worktrees on Windows is fragile (locks, long paths) and makes resume harder.

5) Artifact and log location policy (avoid polluting commits)
- Ensure evaluation logs and sweep outputs are written outside the git-tracked worktree whenever possible:
  - use per-eval `experiment_dir` under the tree-run root
  - route sweep output using `AGENTIC_OUTPUT_DIR` / `AGENTIC_RESULTS_CSV` into `eval_output_root`
- Enforce a "no artifacts committed" rule:
  - candidate commits should include only the code changes required by the idea.

---

## Phase 3 - Node Expansion: Evaluate K Ideas From a Node

For each node in the current frontier:

0) Derive node-specific paths and context inputs (deterministic)
- `node_ideas_dir = <run_root>/node_ideas/<node_id>/` (write new ideas here)
- `node_worktree_dir = <run_root>/wt/<node_id>/`
- `node_baseline_csv = nodes[node_id].baseline_results_csv_path` (should point to a copied artifact under `<run_root>/artifacts/`)
- Build `context_ideas_dirs` for `node_plus_ancestors`:
  - include the node's ancestors' `node_ideas_dir` folders (root -> parent -> node), if they exist
  - record the final ordered list of dirs in the manifest (reproducibility)

1) Generate K ideas into a node-specific folder (with node_plus_ancestors context)
- Run `generate_ideas.py` with:
  - `--ideas-dir <node_ideas_dir>`
  - `--context-ideas-dir <dir>` (repeatable) for each dir in `context_ideas_dirs`
- Record in the manifest:
  - the output directory
  - the exact prompt/context inputs used (paths + hashes of idea markdowns included)
  - the list of generated idea files

2) Evaluate each idea (deterministic order, reproducible IDs)
- Sort idea files by filename and evaluate in that order.
- For each idea file:
  - enforce `max_total_idea_evals` mid-depth (required):
    - before allocating `eval_id`, check if the run has reached the budget
    - if reached, stop scheduling new evals immediately (even if there are remaining ideas for this node)
    - record a clear stop reason in the manifest (e.g., `state.stop_reason = "max_total_idea_evals_reached"`)
  - allocate `eval_id = state.next_eval_id` and increment the counter
  - set `eval_output_root = <run_root>/eval/<eval_id>/` (unique per eval; no collisions)
  - set `experiment_dir = <eval_output_root>/experiment/`
  - create a candidate worktree at the node commit:
    - `candidate_worktree_dir = <run_root>/cand/<eval_id>/`
    - `candidate_ref_name = tree/<tree_run_id>/e<eval_id>`
  - run the existing implementation loop in-place (no nested worktree):
    - invoke the runner with `--no-worktree/--in-place`
    - ensure logs (`review_round_*.md`, transcripts, etc.) are written under `experiment_dir` (outside git checkout)
  - produce a candidate commit (required for node identity):
    - after the idea is approved (and tests if configured), commit code changes
    - enforce "no artifacts committed" (logs and outputs must live outside the repo worktree)
    - record `candidate_commit` in the manifest
  - run sweep with strict output routing:
    - set env vars so the sweep writes into `eval_output_root`:
      - `AGENTIC_OUTPUT_DIR=<eval_output_root>`
      - `AGENTIC_RESULTS_CSV=<eval_output_root>/meta_config_sweep_results.csv`
    - always run with `--sweep-config-limit N` (if configured for the tree run)
  - copy artifacts + record provenance/checksums (required for future expansions):
    - copy the *exact* parent baseline CSV used for parent-relative scoring into `<run_root>/artifacts/` (if not already there)
    - copy the candidate results CSV into `<run_root>/artifacts/`
    - compute sha256 for each copied file and record `{source_path, copied_to_path, sha256}`
    - update the eval record to point at the copied artifact paths
  - compute scoring (both views) and extract compact fields into the manifest:
    - parent-relative: baseline = `node_baseline_csv`, candidate = copied candidate CSV
    - root-relative: baseline = root baseline CSV artifact, candidate = copied candidate CSV
    - record:
      - `recommendation_summary` (should_explore, grade, score, reasons)
      - `primary_delta` (numeric)
      - `baseline_rows_used` / `candidate_rows_used`
  - compute strict completeness counters (required; prevents "improving by failing"):
    - for `config_id < N`, require `status == "ok"` for all rows
    - record `ok_count`, `error_count`, and `expected_count` (= N) in the eval record
    - if incomplete, mark `decision.passed_gate = false` (and capture a reason)
  - mark eval status `completed` or `failed` and write the manifest atomically

---

## Phase 4 - Global Beam Selection + Promotion

At the end of each depth:

1) Apply promotion gate (parent-relative; explicit checklist)
- For each eval at this depth, compute and store:
  - `decision.passed_gate` (bool)
  - `decision.promotion_reason` (string)
- An eval passes the gate only if all of the following are true:
  - parent-relative recommendation gate:
    - `parent_relative.recommendation_summary.should_explore == true`, OR
    - `parent_relative.recommendation_summary.grade == "mixed"` AND primary did not regress
  - primary did not regress (strict numeric):
    - `parent_relative.primary_delta >= 0` (optionally allow tiny tolerance)
  - strict completeness (prevents "improving by failing"):
    - if `sweep_config_limit` is set:
      - `parent_relative.candidate_rows_used == sweep_config_limit`
      - `ok_count == sweep_config_limit` for `config_id < sweep_config_limit`
    - if `sweep_config_limit` is not set:
      - require a minimum expected row count and an ok-rate threshold (define in Phase 0)
  - required artifacts exist and are stable:
    - candidate results CSV path points at `<run_root>/artifacts/...` (not a temp output dir)
    - parent baseline CSV path points at `<run_root>/artifacts/...`
- Any eval that fails the gate is excluded from ranking and must have a clear `promotion_reason` recorded.

2) Rank for global beam (root-relative; deterministic)
- For each eval that passed the gate:
  - compute and store `decision.rank_score` using root-relative scoring:
    - preferred: `root_relative.recommendation_summary.score`
    - fallback: `root_relative.primary_delta` if score is missing/NaN
    - if both missing: treat as worst
- Sort candidates deterministically by:
  1) `decision.rank_score` (desc)
  2) `root_relative.primary_delta` (desc)
  3) `eval_id` (asc) for stability across resume

3) Select next frontier (global beam)
- Keep top `beam_width` candidates as the next frontier.
- For each selected candidate:
  - allocate `node_id = state.next_node_id` and increment
  - create a promoted node record:
    - `commit = candidate_commit`
    - create and record a reachable `ref_name` for the node (e.g., `tree/<tree_run_id>/n<node_id>`)
    - `baseline_results_csv_path = <run_root>/artifacts/...` (the copied candidate results CSV)
    - `idea_chain = parent.idea_chain + [idea_id]`
  - create a fresh node worktree at the candidate commit:
    - `.../wt/<node_id>/`
  - delete the candidate worktree:
    - `.../cand/<eval_id>/`
  - optionally delete the candidate ref (branch) after the node ref has been created (no dangling commits)

4) Update run state for resume (must be atomic)
- Mark the current depth as completed:
  - add to `state.completed_depths`
- Mark all expanded frontier nodes for this depth:
  - append to `state.expanded_node_ids_by_depth[current_depth]`
- Advance depth:
  - `state.current_depth += 1`
- Set next frontier:
  - `state.frontier_node_ids = [promoted_node_ids...]`
- Persist the manifest atomically after each state transition block.

5) Prune rejected candidates (with deferred cleanup)
- For candidates not selected:
  - delete candidate worktrees (`.../cand/<eval_id>/`) unless configured to keep
  - delete candidate refs (`tree/<tree_run_id>/e<eval_id>`) unless configured to keep
- On Windows failures (locked files / git worktree remove failures):
  - enqueue an item in `state.deferred_cleanup`:
    - `{ path, kind: "worktree|branch|dir", reason, attempts, next_retry_at }`
  - retry cleanup later (best-effort), and record cleanup outcomes in the manifest

Stop conditions:
- stop if frontier empty, `max_depth` reached, or `max_total_idea_evals` reached.

---

## Phase 5 - Quality-of-Life (Optional but High Value)

1) Node-aware idea prompt context pack (make `node_plus_ancestors` effective)
- Build a deterministic context pack per node that includes:
  - code context:
    - `git diff --stat` (root -> node) and (parent -> node)
    - optional: a short list of the top changed files along the path
  - performance context:
    - last node's root-relative recommendation summary (score/grade/reasons)
    - last node's top positive/negative deltas (compact)
    - strict gating failures observed in sibling evals (to steer away from brittle ideas)
  - idea history:
    - list of ancestor ideas (titles/ids) and their outcomes
- Persist:
  - the exact context pack used (as a text file under `<run_root>/eval/.../prompt_context.txt` or similar)
  - the exact set of idea markdowns included as context (paths + sha256) for reproducibility

2) Deterministic deduping / "avoid repeats" policy
- Define and implement a repeat-avoidance mechanism:
  - compute a normalized hash of idea text (strip whitespace, normalize bullets)
  - dedupe within a node and across `node_plus_ancestors` context (do not re-try an identical idea)
  - optionally dedupe globally across the entire tree run (configurable)
- Record:
  - `dedupe.policy` and the hashes of skipped ideas in the manifest

3) End-of-run reporting: `TREE_SUMMARY.md` (and report-only mode)
- Add a report generator that reads only `manifest.json` and produces:
  - `<run_root>/TREE_SUMMARY.md`
- Define "best path" deterministically:
  - best node is the node with the maximum root-relative `decision.rank_score`
  - tie-breakers: root-relative primary delta, then `node_id`
  - path is that node's ancestors up to root
- `TREE_SUMMARY.md` should include (minimum):
  - run config (K/D/B/N, strict completeness rule, context strategy)
  - per-depth table:
    - node_id, parent_node_id, commit/ref_name
    - root-relative rank_score + grade + should_explore
    - parent-relative gate outcome summary
    - ok_count/expected_count + candidate_rows_used (so you can trust comparability)
    - artifact paths for baseline + candidate CSVs
    - experiment_dir links
  - best-path section:
    - ordered nodes from root -> best
    - the ideas applied along the path
- Add a `--report-only` mode:
  - read manifest, validate artifacts (see below), and regenerate `TREE_SUMMARY.md` without running any evals

4) Run validation (manifest + artifact integrity)
- Add a validator that checks:
  - every referenced file path exists (manifest, artifacts, summary pointers)
  - sha256 checksums match for copied artifacts
  - every promoted node commit is reachable via its `ref_name`
  - every node baseline CSV path points into `<run_root>/artifacts/`
  - frontier + expanded markers are consistent (no re-expansion holes)
- Emit:
  - `<run_root>/VALIDATION_REPORT.md` (or JSON) summarizing any issues

5) Dry-run mode (fast, deterministic integration tests for tree mechanics)
- Add `--dry-run` mode that does not call LLMs or sweep:
  - generate a fixed set of fake eval records with a mix of:
    - pass gate vs fail gate
    - strict completeness failures (ok_count < N)
    - missing/NaN scores to test fallback ranking
    - ties to test deterministic tie-breaking
    - simulated Windows cleanup failures to populate deferred cleanup queue
  - run through multiple depths and resume mid-depth
- Dry-run acceptance criteria:
  - identical outputs across repeated runs
  - stable `TREE_SUMMARY.md`
  - correct frontier advancement and pruning

---

## Phase 6 - Parallelism and Scaling (Optional; Recommended Once MVP Is Stable)

This phase adds controlled parallel execution without corrupting `manifest.json` and without output collisions.

1) Concurrency model: single-writer manifest (required)
- Keep `manifest.json` as a single-writer file.
- Introduce a coordinator/driver process that:
  - owns the run lock (`run.lock.json`)
  - creates worktrees, schedules evals, and merges eval results into the manifest
- Introduce worker processes that:
  - do not write to `manifest.json` directly
  - write results to a per-eval file under `eval_output_root`:
    - `<eval_output_root>/eval_result.json`
  - write logs under `<eval_output_root>/experiment/`
- Coordinator polls for completion and merges results atomically into the manifest.

2) Execution controls
- Add CLI knobs:
  - `--max-parallel-evals` (default 1)
  - `--eval-timeout-seconds` (per eval hard timeout)
  - `--llm-rate-limit` / `--llm-max-inflight` (optional, if needed)
  - `--sweep-max-parallel` (optional; typically keep low because sweep is CPU heavy)
- Enforce `max_total_idea_evals` strictly:
  - do not schedule more evals once the budget is reached

3) Output collision guarantees (required)
- Each worker must have a unique `eval_output_root` and unique candidate worktree path:
  - `.../eval/<eval_id>/` and `.../cand/<eval_id>/`
- Each worker must set:
  - `AGENTIC_OUTPUT_DIR=<eval_output_root>`
  - `AGENTIC_RESULTS_CSV=<eval_output_root>/meta_config_sweep_results.csv`

4) Resume semantics in parallel mode
- On resume, coordinator:
  - scans `eval/<eval_id>/eval_result.json` for completed evals not yet merged
  - marks "running" evals as failed if they exceeded timeout or if their worker process is gone (best-effort)
  - merges any completed eval_result files into the manifest deterministically

5) Windows-specific cleanup under concurrency
- Route all cleanup (worktrees/branches/dirs) through the coordinator.
- If cleanup fails due to locks:
  - enqueue deferred cleanup items in `state.deferred_cleanup`
  - retry after a delay; do not block the whole run indefinitely

6) Parallel dry-run scenarios (extend Phase 5)
- Add dry-run cases that simulate:
  - multiple workers completing out of order
  - coordinator merging results deterministically
  - crash mid-merge and resume without duplication
