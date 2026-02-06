# Artifact Docs (static, repo-committed)

These files are intended to be referenced by LLM prompts so the model can:
- Locate sweep artifacts on disk (e.g. `.../agentic_YYYYMMDD_HHMMSS/run_0/...`)
- Interpret the CSV/plot outputs without loading everything into prompt context

Docs in this folder are intentionally static and hand-curated because the artifact schemas
and column sets are expected to be stable over time.

Key docs:
- `meta_config_sweep_results_columns.txt`
  Column definitions for `meta_config_sweep_results.csv` (the main sweep results table; 1 row per config).
- `avg_trade_return_plots/README.txt`
  Overview of the per-config plots/CSVs found in `avg_trade_return_plots/`.
- `avg_trade_return_plots/*.txt`
  Per-CSV-type structure + row-key definitions for:
  `core_metrics_config_*.csv`, `relative_metrics_config_*.csv`, `trade_metrics_config_*.csv`,
  `stability_metrics_config_*.csv`, `significance_metrics_config_*.csv`.

