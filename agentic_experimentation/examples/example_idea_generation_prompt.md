# Example Idea Generation Prompt (Structure)

This is an example of the *general structure* of the idea generation prompt assembled by `agentic_experimentation/idea_generation/generate_ideas.py`.

Notes:
- Sections may be truncated if `max_context_chars` is reached.
- The "Current Status / Artifacts / Branch Timeline" block is included only when `--baseline-context-json` is provided.
- A replay-memory section is appended when conversation continuation uses replay mode (`--conversation-mode replay`, or `auto` fallback).
- Values/paths below are illustrative; treat them as placeholders.

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
- tree_run_id=20260205 node_id=0007 depth=2 parent_node_id=0003 changes_applied_to_original=2
- applied_ideas (chronological file pointers):
  - `agentic_experimentation/worktrees/tree_runs/20260205/node_ideas/0001/0001_first_idea.md`
  - `agentic_experimentation/worktrees/tree_runs/20260205/node_ideas/0003/0002_second_idea.md`

Artifacts & How To Interpret Them:
- Static docs root: `agentic_experimentation/artifact_docs/` (exists=true)

Output: meta_config_sweep_results.csv
- What it stores: Per-config sweep results; each row is one meta-model backtest for a single parameter set (`config_id`).
- Used for: Comparing parameter sets and computing the averaged metrics/deltas used to judge/promote ideas.
- Location (current node): `C:\Users\micha\myhome\algo\artifacts\period_returns\agentic_20260205_015323\run_0\meta_config_sweep_results.csv`
- Location (pattern): `<agentic_output_root>/run_0/meta_config_sweep_results.csv`
- Exists (current node path): true
- Column definitions: `agentic_experimentation/artifact_docs/meta_config_sweep_results_columns.txt` (exists=true)
  - Format: `column_name: description` (search by column name).
  - Note: metrics families include `core_*`, `rel_*`, `stab_*`, `trade_*`, `sig_*`.
- Granularity: 1 CSV per node/run; rows are per parameter set tested (`config_id`).

Output: avg_trade_return_plots/
- What it stores: Per-parameter-set diagnostics (plots + row-metric CSVs) for a node/run.
- Used for: Deep-diving into *why* a specific config improved/regressed (distributions, drawdowns, stability, significance, trade quality).
- Location (current node): `C:\Users\micha\myhome\algo\artifacts\period_returns\agentic_20260205_015323\run_0\avg_trade_return_plots`
- Location (pattern): `<agentic_output_root>/run_0/avg_trade_return_plots/`
- Exists (current node path): true
- Overview doc: `agentic_experimentation/artifact_docs/avg_trade_return_plots/README.txt`
- Naming convention: Files are per `config_id` and suffixed `_config_000`, `_config_001`, ... matching `meta_config_sweep_results.csv` rows.
- Per-config artifacts (each is one file per `config_id`):

  Artifact: avg_trade_return_config_XXX.png (PNG)
  - Shows: average return per trade over time (all_models vs topN).
  - Use it to: check whether improvements are consistent across periods or concentrated in a few regimes.

  Artifact: trade_quality_hist_config_XXX.png (PNG)
  - Shows: histogram of average return per trade across periods (all_models vs topN).
  - Use it to: see distribution shifts and whether downside tail risk worsened.

  Artifact: trade_quality_rollmean_config_XXX.png (PNG)
  - Shows: rolling mean of average return per trade (all_models vs topN).
  - Use it to: see stability of trade-quality improvements through time.

  Artifact: equity_ratio_config_XXX.png (PNG)
  - Shows: equity ratio over time (equity_topN / equity_all).
  - Use it to: see whether topN compounds faster than the baseline universe.

  Artifact: rolling_sharpe_sortino_config_XXX.png (PNG)
  - Shows: rolling Sharpe and Sortino for period returns (all_models vs topN).
  - Use it to: verify risk-adjusted improvements are not just point-estimate noise.

  Artifact: rolling_outperformance_config_XXX.png (PNG)
  - Shows: rolling outperformance rate (fraction of periods topN_return > all_models_return).
  - Use it to: distinguish frequent small wins vs rare big wins.

  Artifact: drawdown_curves_config_XXX.png (PNG)
  - Shows: drawdown curves from equity peaks (all_models vs topN).
  - Use it to: inspect max drawdown depth and duration behavior.

  Artifact: return_hist_config_XXX.png (PNG)
  - Shows: histogram (and optional KDE) of period returns (all_models vs topN).
  - Use it to: compare central tendency and tails of per-period returns.

  Artifact: return_delta_hist_config_XXX.png (PNG)
  - Shows: histogram of deltas (topN_return - all_models_return).
  - Use it to: see whether outperformance is broad-based vs driven by a few periods.

  Artifact: return_scatter_config_XXX.png (PNG)
  - Shows: scatter of (all_models_return, topN_return) with y=x line and quadrant axes.
  - Use it to: identify regimes where meta-selection helps or hurts (e.g., baseline<0 while topN>0).

  Artifact: core_metrics_config_XXX.csv (CSV)
  - Stores: core performance metrics for all_models and topN (Sharpe/Sortino/Calmar/drawdowns/etc).
  - Column definitions: `agentic_experimentation/artifact_docs/avg_trade_return_plots/core_metrics_config.txt` (exists=true)
  - Use it to: get a compact numeric summary for one config_id.

  Artifact: relative_metrics_config_XXX.csv (CSV)
  - Stores: relative edge metrics (deltas, capture ratios, equity ratio) for topN vs all_models.
  - Column definitions: `agentic_experimentation/artifact_docs/avg_trade_return_plots/relative_metrics_config.txt` (exists=true)
  - Use it to: diagnose how the edge is achieved (frequency vs magnitude, capture, etc).

  Artifact: stability_metrics_config_XXX.csv (CSV)
  - Stores: rolling-window stability metrics for all_models, topN, and the delta series.
  - Column definitions: `agentic_experimentation/artifact_docs/avg_trade_return_plots/stability_metrics_config.txt` (exists=true)
  - Use it to: find weak stability (bad rolling mins, long losing streaks, etc).

  Artifact: trade_metrics_config_XXX.csv (CSV)
  - Stores: trade-quality metrics computed from avg-return-per-trade series (plus relative deltas).
  - Column definitions: `agentic_experimentation/artifact_docs/avg_trade_return_plots/trade_metrics_config.txt` (exists=true)
  - Use it to: check if improvements come from better per-period trade outcomes and consistency.

  Artifact: significance_metrics_config_XXX.csv (CSV)
  - Stores: t-test, sign test, and bootstrap metrics for the delta series (topN_return - all_models_return).
  - Column definitions: `agentic_experimentation/artifact_docs/avg_trade_return_plots/significance_metrics_config.txt` (exists=true)
  - Use it to: sanity-check whether deltas look statistically meaningful.
- Granularity: 1 directory per node/run; inside it, many files per parameter set tested (`config_id`).

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
   baseline_results_csv_path: `agentic_experimentation/worktrees/tree_runs/20260205/artifacts/root_baseline.csv`

1. node_id=0003 depth=1
   applied_idea: `agentic_experimentation/worktrees/tree_runs/20260205/node_ideas/0001/0001_first_idea.md`
   sweep_config_limit: 25 (uses config_id < 25)
   core_topN_sharpe: 1.64   (delta_vs_parent=0.04)
   mean_topN_avg_return_per_trade_pct_oos: 0.045   (delta_vs_parent=0.015)
   mean_topN_avg_return_per_trade_pct: 0.139
   core_topN_sortino: 3.78
   core_topN_calmar: 2.58
   core_topN_max_drawdown: -0.248
   baseline_results_csv_path: `agentic_experimentation/worktrees/tree_runs/20260205/artifacts/candidate_eval_0009.csv`
   sweep_results_csv: `C:\Users\micha\myhome\algo\artifacts\period_returns\agentic_20260205_000000\run_0\meta_config_sweep_results.csv`
   avg_trade_return_plots_dir: `C:\Users\micha\myhome\algo\artifacts\period_returns\agentic_20260205_000000\run_0\avg_trade_return_plots`

2. node_id=0007 depth=2
   applied_idea: `agentic_experimentation/worktrees/tree_runs/20260205/node_ideas/0003/0002_second_idea.md`
   sweep_config_limit: 25 (uses config_id < 25)
   core_topN_sharpe: 1.68   (delta_vs_parent=0.04)
   mean_topN_avg_return_per_trade_pct_oos: 0.063   (delta_vs_parent=0.018)
   mean_topN_avg_return_per_trade_pct: 0.143
   core_topN_sortino: 3.84
   core_topN_calmar: 2.65
   core_topN_max_drawdown: -0.243
   baseline_results_csv_path: `agentic_experimentation/worktrees/tree_runs/20260205/artifacts/candidate_eval_0015.csv`
   sweep_results_csv: `C:\Users\micha\myhome\algo\artifacts\period_returns\agentic_20260205_015323\run_0\meta_config_sweep_results.csv`
   avg_trade_return_plots_dir: `C:\Users\micha\myhome\algo\artifacts\period_returns\agentic_20260205_015323\run_0\avg_trade_return_plots`

Rejected Ideas (evaluated but not selected for this branch):
0. eval_id=0016 parent_node_id=0003 depth=1
   applied_idea: `agentic_experimentation/worktrees/tree_runs/20260205/node_ideas/0003/0003_rejected_idea.md`
   sweep_config_limit: 25 (uses config_id < 25)
   core_topN_sharpe: 1.61   (delta_vs_parent=-0.03)
   mean_topN_avg_return_per_trade_pct_oos: 0.028   (delta_vs_parent=-0.017)
   mean_topN_avg_return_per_trade_pct: 0.132
   core_topN_sortino: 3.66
   core_topN_calmar: 2.41
   core_topN_max_drawdown: -0.262
   baseline_results_csv_path: `agentic_experimentation/worktrees/tree_runs/20260205/artifacts/candidate_eval_0016.csv`
   sweep_results_csv: `C:\Users\micha\myhome\algo\artifacts\period_returns\agentic_20260205_015900\run_0\meta_config_sweep_results.csv`
   avg_trade_return_plots_dir: `C:\Users\micha\myhome\algo\artifacts\period_returns\agentic_20260205_015900\run_0\avg_trade_return_plots`

===== META MODEL CONTEXT =====
META_MODEL_GUIDE.md
 - description: High-level guide for the meta model design, assumptions, and workflow.
 - location: C:\Users\micha\myhome\git\algo-meta-model-agent\agentic_experimentation\worktrees\tree_runs\20260205\wt\0007\META_MODEL_GUIDE.md
adaptive_vol_momentum.py
 - description: Primary meta model implementation and sweep/backtest driver.
 - location: C:\Users\micha\myhome\git\algo-meta-model-agent\agentic_experimentation\worktrees\tree_runs\20260205\wt\0007\adaptive_vol_momentum.py
scoring.py
 - description: Performance scoring and summary metric computation utilities.
 - location: C:\Users\micha\myhome\git\algo-meta-model-agent\agentic_experimentation\worktrees\tree_runs\20260205\wt\0007\scoring.py
selection.py
 - description: Selection logic used to choose top strategies/models each period.
 - location: C:\Users\micha\myhome\git\algo-meta-model-agent\agentic_experimentation\worktrees\tree_runs\20260205\wt\0007\selection.py

===== IDEA CONVERSATION MEMORY (REPLAY) =====
Use this for continuity with prior idea-generation turns in this branch.
Prefer non-duplicate ideas and build on prior reasoning where useful.

compact_summary:
Earlier branch history summary (oldest -> newest):
- turn_0001: ... output_hash=...
- turn_0002: ... output_hash=...

[turn_0007] 2026-02-05T01:23:45Z
output_idea_path: agentic_experimentation/worktrees/tree_runs/20260205/node_ideas/0007/0005_example.md
assistant_output:
IDEA: ...
RATIONALE: ...
REQUIRED_CHANGES: ...

===== END IDEA CONVERSATION MEMORY =====
