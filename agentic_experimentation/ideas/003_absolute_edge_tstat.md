IDEA: Add an "absolute edge" term to `compute_scores_for_ticker_v2` by tilting scores with a rolling t-stat (or Sharpe-like) of each model's *raw returns*, so models that are consistently negative get pushed down even if they rank well cross-sectionally.

RATIONALE: The current meta model is largely rank-based (`Q` percentile ranks -> adaptive momentum `M`), so a model can look strong simply by being "least bad" within a ticker/period (ranks ignore absolute sign/level when the whole cross-section is weak). Adding a small, bounded absolute-return significance term (computed only from each model's own past returns) makes the score reflect "is this model actually making money recently?" in addition to "is it beating peers?", improving cross-ticker comparability and reducing selection of stable-but-losing strategies -- without introducing training or non-causal features (the existing final `shift(1)` keeps it causal).

REQUIRED_CHANGES:
- In `scoring.py` inside `compute_scores_for_ticker_v2` (after `R` is built, before `scores_df = scores_df.shift(axis=1)`), compute a per-model rolling mean/std over the last `L` periods on *raw returns* `R` (axis=time), then form a t-stat-like signal:
  - `mu_{i,t} = mean(R_{i,t-L+1:t})`
  - `sd_{i,t} = std(R_{i,t-L+1:t}) + eps`
  - `t_{i,t} = sqrt(n_eff) * mu_{i,t} / sd_{i,t}`
  - squash to a stable bounded scale, e.g. `ABS_{i,t} = tanh(t_{i,t} / t_scale)` so it can't dominate.
- Add new `MetaConfig` params in `config.py` (defaults shown as reasonable starting points):
  - `abs_lookback: int = 12` (or reuse `conf_lookback` if you want zero new knobs)
  - `abs_weight: float = 0.15` (small, because the main signal stays rank-based)
  - `abs_eps: float = 1e-8`
  - `abs_t_scale: float = 2.0` (so `t~=+/-2` maps near saturation)
- Modify the final score combination in `compute_scores_for_ticker_v2` from:
  - `SCORE = (rel * CONF) - risk_pen`
  to:
  - `SCORE = (rel * CONF) - risk_pen + cfg.abs_weight * ABS`
  (ticker gating automatically benefits because `ticker_score` is computed from `scores_df` before the causal shift).
