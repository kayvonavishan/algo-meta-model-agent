# Example Idea Generation Prompt (Initial / Root Node)

This example shows the prompt shape for the **very first idea generation call** in a new tree run (root node, depth 0).

Notes:
- Values/paths are illustrative.
- This is for node `0000` before any idea has been promoted.
- A replay-memory section is appended only when conversation continuation uses replay mode (`--conversation-mode replay`, or `auto` fallback).

---

You are a quantitative trading researcher proposing ONE self-contained improvement to the meta model.

Experiment Description:
- We are iteratively improving the meta model using a beam-search style process: generate multiple candidate ideas, implement/test them, then promote the most promising changes as new "nodes" in an improvement tree/branch.
- Below you will find information about the current status of this experiment (where we are in the branch), which ideas have been applied, and how performance has changed.
- You may also be given a "BASELINE ARTIFACTS (READ-ONLY, OPTIONAL)" section with baseline metrics + file paths to sweep artifacts; use it to inspect results as needed, but do not load everything unnecessarily.

Model Description:
- The meta model selects top-performing base strategies for the next period using training-free signals (see `adaptive_vol_momentum.py`).
- You will be given additional read-only repo context below (`META_MODEL_GUIDE.md`, `adaptive_vol_momentum.py`, `scoring.py`, `selection.py`).
- You will also be given branch timeline and rejected-idea summaries (idea file paths + metrics) so you can avoid duplicating ideas already tested.

Output Guidelines:
- Ideas must be implementable as a single coding change (idea -> implement).
- No multi-step experiments inside one idea. If you think multiple variants could help, emit them as separate ideas, not "try A then B".
- Each idea must include all context needed for a quant developer to implement it, even if that means repeating information across ideas.
- Provide specific details on what needs to change in the context of the current meta model implementation.
- Exact code changes are not necessary, but helpful toy examples are fine.
- Pick the idea you think could be most positively impactful.

Return exactly:
IDEA: <one concise, self-contained change>
RATIONALE: <why it might help, in plain terms>
REQUIRED_CHANGES: <elaborate on changes required to the meta model. Detailed code changes are not necessary>

-----------------------

Current Status:
- tree_run_id=20260206 node_id=0000 depth=0 changes_applied_to_original=0
- applied_ideas: (none; original/root model)

Artifacts & How To Interpret Them:
- Static docs root: `agentic_experimentation/artifact_docs/` (exists=true)

Output: meta_config_sweep_results.csv
- What it stores: Per-config sweep results; each row is one meta-model backtest for a single parameter set (`config_id`).
- Used for: Comparing parameter sets and computing the averaged metrics/deltas used to judge/promote ideas.
- Location (current node): `agentic_experimentation/baselines/meta_config_sweep_results_baseline.csv`
- Location (pattern): `<agentic_output_root>/run_0/meta_config_sweep_results.csv`
- Exists (current node path): true
- Column definitions: `agentic_experimentation/artifact_docs/meta_config_sweep_results_columns.txt` (exists=true)
  - Format: `column_name: description` (search by column name).
  - Note: metrics families include `core_*`, `rel_*`, `stab_*`, `trade_*`, `sig_*`.
- Granularity: 1 CSV per node/run; rows are per parameter set tested (`config_id`).

Output: avg_trade_return_plots/
- Availability: not available yet for this node.
- Why: these diagnostics are produced only after running a candidate idea sweep.
- Location (current node): (not generated yet)
- Location (pattern): `<agentic_output_root>/run_0/avg_trade_return_plots/`
- Overview doc: `agentic_experimentation/artifact_docs/avg_trade_return_plots/README.txt` (exists=true)
- Note: once available, files are per `config_id` with suffixes `_config_000`, `_config_001`, ...

- Guidance: consult docs first; open only the minimum files needed.

Branch Timeline (chronological):
0. node_id=0000 depth=0
   applied_idea: (root/original)
   sweep_config_limit: 25 (uses config_id < 25)
   core_topN_sharpe: 1.60
   mean_topN_avg_return_per_trade_pct_oos: 0.030
   mean_topN_avg_return_per_trade_pct: 0.135
   core_topN_sortino: 3.70
   core_topN_calmar: 2.50
   core_topN_max_drawdown: -0.250
   baseline_results_csv_path: `agentic_experimentation/baselines/meta_config_sweep_results_baseline.csv`

===== META MODEL CONTEXT =====
META_MODEL_GUIDE.md
 - description: High-level guide for the meta model design, assumptions, and workflow.
 - location: C:\Users\micha\myhome\git\algo-meta-model-agent\META_MODEL_GUIDE.md
adaptive_vol_momentum.py
 - description: Primary meta model implementation and sweep/backtest driver.
 - location: C:\Users\micha\myhome\git\algo-meta-model-agent\adaptive_vol_momentum.py
scoring.py
 - description: Performance scoring and summary metric computation utilities.
 - location: C:\Users\micha\myhome\git\algo-meta-model-agent\scoring.py
selection.py
 - description: Selection logic used to choose top strategies/models each period.
 - location: C:\Users\micha\myhome\git\algo-meta-model-agent\selection.py

===== IDEA CONVERSATION MEMORY (REPLAY) =====
Use this for continuity with prior idea-generation turns in this branch.
Prefer non-duplicate ideas and build on prior reasoning where useful.

compact_summary:
(none yet)

[turn_0001] 2026-02-06T09:01:00Z
output_idea_path: agentic_experimentation/worktrees/tree_runs/20260206/node_ideas/0000/0001_example.md
assistant_output:
IDEA: ...
RATIONALE: ...
REQUIRED_CHANGES: ...

===== END IDEA CONVERSATION MEMORY =====
