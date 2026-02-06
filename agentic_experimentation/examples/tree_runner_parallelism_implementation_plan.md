# Tree Runner Parallelism Implementation Plan

## Scope
Introduce parallel evaluation in `agentic_experimentation/tree_runner.py` while preserving deterministic selection/promotion and safe resume behavior.

## What Will Be Parallel
- Parallel unit: one eval task = `(parent_node_id, idea_path, eval_id)`.
- Each eval task runs the full candidate pipeline (implement/review/sweep/score) via `multi_agent_runner.py`.
- Multiple eval tasks can run concurrently at the same tree depth.

## What Stays Serial
- Manifest coordination/writes (single coordinator).
- Promotion and beam selection (after all evals at a depth finish).
- Node creation for promoted candidates.

---

## Phase 0 — Behavior Lock
1. Keep breadth-first expansion by depth.
2. Add a per-depth barrier: all eval tasks complete before promotion.
3. Preserve deterministic ordering and results with fixed inputs.

## Phase 1 — Task Model
1. Define task record fields:
   - `eval_id`, `parent_node_id`, `idea_path`, `depth`, `task_state`.
2. Precompute all tasks for the depth in deterministic order:
   - sort frontier nodes,
   - sort ideas per node,
   - allocate `eval_id` up front.
3. Persist queued tasks into manifest before dispatch.

## Phase 2 — CLI and Runtime Controls
1. Add flags:
   - `--max-parallel-evals` (global worker cap, default `1`),
   - `--max-parallel-per-node` (optional fairness cap),
   - `--parallel-backend` (`threadpool` initially).
2. Validate values and keep `1` as current behavior parity mode.

## Phase 3 — Coordinator + Workers
1. Coordinator owns:
   - task queue,
   - manifest updates,
   - dispatch lifecycle.
2. Worker execution:
   - run one eval subprocess,
   - write per-eval logs/artifacts under unique paths.
3. Coordinator consumes worker completions and updates manifest state.

## Phase 4 — Manifest State Machine and Resume
1. Add task/eval lifecycle fields:
   - `queued_at`, `started_at`, `finished_at`, `worker_pid`, `attempt`, `task_state`.
2. On resume:
   - recover queued tasks,
   - requeue stale `running` tasks,
   - never duplicate `completed` tasks.
3. Use idempotency key `(parent_node_id, idea_path)` for dedupe.

## Phase 5 — Isolation and File Safety
1. Ensure each eval uses unique:
   - output root,
   - subprocess log path,
   - candidate/worktree refs.
2. Prevent shared temporary file collisions.
3. Run cleanup/pruning only after depth barrier completes.

## Phase 6 — Deterministic Promotion Barrier
1. After all depth tasks complete, run existing ranking/promotion logic serially.
2. Apply beam/global selection exactly once from completed eval set.
3. Promote in deterministic order.

## Phase 7 — Failure Policy
1. Add configurable retries/timeouts:
   - `--eval-retries`,
   - `--eval-timeout-seconds`.
2. Record structured error payloads for failed tasks.
3. Continue with partial completion by default; optional strict-fail flag can stop depth.

## Phase 8 — Observability
1. Emit scheduler events:
   - `queued`, `started`, `completed`, `failed`, `requeued`.
2. Add depth progress counters to summary/reporting.
3. Include per-eval timing and attempt counts in `TREE_SUMMARY.md`.

## Phase 9 — Test Coverage
1. Unit tests:
   - deterministic task generation/eval_id allocation,
   - lifecycle transitions and retries,
   - resume requeue behavior.
2. Integration dry-run:
   - `beam_width=3`, `max_depth=2`, `max_parallel_evals>1`,
   - verify promotions equal serial baseline.
3. Failure/resume tests:
   - forced subprocess failures,
   - interrupted run recovery.

## Phase 10 — Rollout
1. Ship with default `--max-parallel-evals 1`.
2. Validate parity against serial runs.
3. Increase parallelism gradually based on resource usage and stability.

---

## Acceptance Criteria
- Multiple ideas are evaluated concurrently at a depth when `--max-parallel-evals > 1`.
- Promotion decisions remain deterministic and match serial baseline.
- Resume remains safe: no duplicated completed evals, stale running tasks requeued cleanly.
- Run artifacts/logs remain traceable and non-colliding.
