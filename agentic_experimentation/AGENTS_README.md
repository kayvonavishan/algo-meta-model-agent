# Multi-Agent Flow

This describes the current multi-agent loop driven by `multi_agent_runner.py`.

## High-level flow

```mermaid
graph TD
    A[Idea from ideas.txt] --> B[Coordinator LLM<br/>notes handoff]
    B --> C[Planner LLM<br/>PLAN / RISKS / TESTS]
    C --> D[Coder LLM<br/>unified diff]
    D --> E[git apply patch]
    E --> F[Reviewer LLM<br/>APPROVE/REJECT]
    F --> G{Proceed?}
    G -- no --> H[Stop<br/>record summary]
    G -- yes --> I[Tests (optional)]
    I --> J{Tests pass?}
    J -- no --> H
    J -- yes --> K[Sweep command<br/>(adaptive_vol_momentum.py)]
    K --> L[Copy results<br/>score vs baseline]
    L --> M[Summary<br/>cleanup worktree]
```

## Artifact pipeline (per iteration)

- `idea.md`: selected idea text.
- `coordinator_prompt.txt` / `coordinator.md`: prompt and notes from coordinator.
- `planner_prompt.txt` / `plan.md`: prompt and detailed plan.
- `coder_prompt.txt` / `patch.diff`: prompt and generated diff.
- `reviewer_prompt.txt` / `review.md`: prompt and review verdict.
- `tests.log`: optional test run output.
- `sweep.log`: sweep command output.
- `meta_config_sweep_results.csv`: copied sweep results (if produced).
- `summary.json`: run metadata (idea, notes, plan, patch status, review verdict, tests/sweep exit codes, score).

## Control logic

1. Create git worktree; optionally sync working tree changes if `base_on_working_tree` is true.
2. Coordinator produces notes; planner produces the plan.
3. Coder produces and applies patch. If apply fails, stop.
4. Reviewer renders verdict. If not APPROVE, stop.
5. If `test_command` is set, run it; require exit code 0 to proceed.
6. Run sweep (`sweep_command`) and log output.
7. Copy sweep `results_csv` into the run dir and score vs `baseline_csv`.
8. Write `summary.json`; delete worktree unless keep flags are set.

## Configuration hooks

- `agents`: per-role LLM config for coordinator, planner, coder, reviewer.
- `prompts`: file paths for each agent.
- `test_command` / `test_cwd`: optional test gate.
- `sweep_command` / `sweep_cwd`: sweep execution.
- `baseline_csv` / `results_csv`: scoring inputs/outputs.
- `worktree_root` / `experiments_root`: isolation of runs.
