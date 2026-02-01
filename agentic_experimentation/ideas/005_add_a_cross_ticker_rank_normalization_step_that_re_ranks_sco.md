IDEA: Add a "cross-ticker rank normalization" step that re-ranks scores across all tickers within each period before final selection, so that models from tickers with naturally tighter score distributions compete fairly against models from tickers with wider score spreads.

RATIONALE: The current pipeline computes scores independently per ticker (Section 6), then pools them globally for selection (Section 8). However, different tickers can have very different score distributions: a "high-scoring" model in a low-dispersion ticker might have an absolute score of 0.15, while a mediocre model in a high-dispersion ticker might score 0.40. The global selection then favors models from high-variance tickers simply because their scores are numerically larger, not because they're actually better picks. By adding a cross-ticker normalization step (converting absolute scores to percentile ranks across all ticker-model pairs within each period), we ensure the global selection is comparing apples to apples. This is training-free, causal (uses only current-period scores which are already shifted), and directly addresses the heterogeneity problem without changing the within-ticker scoring logic.

REQUIRED_CHANGES:

**1. Add new config parameters to `MetaConfig` (in `config.py`):**
```python
enable_cross_ticker_normalization: bool = True
cross_ticker_norm_method: str = "percentile"  # options: "percentile", "zscore", "minmax"
```

**2. Add cross-ticker normalization function in `scoring.py`:**

```python
def normalize_scores_across_tickers(scores_by_ticker: dict, method: str = "percentile") -> dict:
    """
    Normalize scores across all tickers for each period to ensure fair cross-ticker comparison.
    
    Parameters:
    - scores_by_ticker: dict mapping ticker -> DataFrame (models x periods) of scores
    - method: normalization method
        - "percentile": convert to percentile ranks across all ticker-models per period
        - "zscore": standardize to zero mean, unit variance per period
        - "minmax": scale to [0, 1] per period
    
    Returns:
    - normalized_scores_by_ticker: dict with same structure, normalized scores
    """
    # Stack all scores into a single DataFrame with MultiIndex (ticker, model_id)
    all_scores = []
    for ticker, scores_df in scores_by_ticker.items():
        scores_df = scores_df.copy()
        scores_df['ticker'] = ticker
        scores_df = scores_df.reset_index()  # model_id becomes column
        all_scores.append(scores_df)
    
    combined = pd.concat(all_scores, ignore_index=True)
    combined = combined.set_index(['ticker', 'model_id'])
    
    # Get period columns (exclude any metadata)
    period_cols = [c for c in combined.columns if c not in ['ticker', 'model_id']]
    
    # Normalize each period column
    normalized = combined.copy()
    for col in period_cols:
        values = combined[col].values
        valid_mask = ~np.isnan(values)
        
        if valid_mask.sum() < 2:
            continue
            
        if method == "percentile":
            # Rank all valid scores, scale to [0, 1]
            ranks = np.full_like(values, np.nan)
            ranks[valid_mask] = scipy.stats.rankdata(values[valid_mask], method='average')
            ranks[valid_mask] = (ranks[valid_mask] - 1) / (valid_mask.sum() - 1)
            normalized[col] = ranks
            
        elif method == "zscore":
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values) + 1e-8
            normalized[col] = (values - mean_val) / std_val
            
        elif method == "minmax":
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)
            range_val = max_val - min_val + 1e-8
            normalized[col] = (values - min_val) / range_val
    
    # Unstack back to per-ticker DataFrames
    normalized_by_ticker = {}
    for ticker in scores_by_ticker.keys():
        ticker_df = normalized.loc[ticker].copy()
        normalized_by_ticker[ticker] = ticker_df
    
    return normalized_by_ticker
```

**3. Integrate into `select_models_universal_v2` in `selection.py`:**

After computing scores per ticker but before building the long selection table:

```python
def select_models_universal_v2(aligned_returns, cfg, ...):
    # Compute scores per ticker (existing logic)
    scores_by_ticker = {}
    for ticker, returns_df in aligned_returns.items():
        scores_df = compute_scores_for_ticker_v2(returns_df, cfg)
        scores_by_ticker[ticker] = scores_df
    
    # NEW: Cross-ticker normalization
    if cfg.enable_cross_ticker_normalization:
        scores_by_ticker = normalize_scores_across_tickers(
            scores_by_ticker, 
            method=cfg.cross_ticker_norm_method
        )
    
    # Continue with existing selection logic (build long table, rank, select top_n_global)
    ...
```

**4. Toy example showing the problem and fix:**

Before normalization (raw scores, one period):
```
Ticker AAPL (tight distribution):
  Model A: 0.12
  Model B: 0.10
  Model C: 0.08

Ticker TSLA (wide distribution):
  Model X: 0.45
  Model Y: 0.25
  Model Z: 0.05
```

Global selection with top_n_global=3 would pick: X (0.45), Y (0.25), A (0.12)
→ TSLA gets 2 picks despite Model Y possibly being mediocre within TSLA

After percentile normalization:
```
Combined pool of 6 models, percentile ranks:
  Model X: 1.00 (highest)
  Model Y: 0.60
  Model A: 0.80 (second-highest after cross-ticker ranking)
  Model B: 0.40
  Model C: 0.20
  Model Z: 0.00

Global selection with top_n_global=3 would pick: X (1.00), A (0.80), Y (0.60)
→ More balanced: AAPL's best model competes fairly with TSLA's
```

**5. Why this helps:**

- **Fair competition**: Models are judged by their relative standing among *all* models across tickers, not inflated by ticker-specific score variance
- **Reduces concentration**: Prevents high-volatility tickers from dominating selections purely due to score magnitude
- **Complements existing per-ticker cap**: The cap limits models *per ticker*, but this addresses the more subtle issue of score comparability *across* tickers
- **Training-free and causal**: Uses only the already-computed (and shifted) scores, no new lookback or future data
