IDEA: Add an "adaptive confidence weighting" mechanism that scales the confidence multiplier based on the current dispersion regime, so that confidence matters more during stable (low-dispersion) periods when rank stability is a reliable signal, and matters less during volatile (high-dispersion) periods when even good models can have unstable ranks.

RATIONALE: The current scoring formula applies confidence uniformly: `SCORE = (rel * CONF) - risk_pen`. However, the informativeness of rank stability (what confidence measures) varies with market regime. During low-dispersion periods, models are tightly clustered and small differences persist—here, confidence (rank stability) is highly predictive because stable models genuinely have an edge. During high-dispersion periods, even fundamentally strong models can see wild rank swings due to regime changes, outlier events, or temporary dislocations—here, penalizing rank volatility via confidence may unfairly punish adaptive models that are simply responding appropriately to a chaotic environment. By scaling the effective confidence weight with the same dispersion z-scores already computed (reusing Step 6.2 infrastructure), we make the meta model more regime-aware without adding new lookback windows or features. This is a single, localized change to the score combination step.

REQUIRED_CHANGES:

**1. Add new config parameters to `MetaConfig` (in `config.py`):**
```python
enable_adaptive_conf_weight: bool = True
conf_weight_base: float = 1.0  # baseline confidence multiplier
conf_weight_low_disp: float = 1.3  # confidence weight when dispersion is low (stable regime)
conf_weight_high_disp: float = 0.7  # confidence weight when dispersion is high (volatile regime)
```

**2. Modify `compute_scores_for_ticker_v2` in `scoring.py`:**

Currently, confidence is computed and applied uniformly. After computing `disp_z` (dispersion z-scores per period, already done in Step 6.2.2 for adaptive alpha), add logic to compute a regime-dependent confidence weight:

```python
# After computing disp_z (already exists for adaptive alpha)
# and after computing CONF (confidence percentile ranks)

if cfg.enable_adaptive_conf_weight:
    # Map dispersion z-score to confidence weight
    # Low dispersion (z < 0) -> higher confidence weight (more trust in stability signal)
    # High dispersion (z > 0) -> lower confidence weight (discount stability signal)
    
    z_clipped = np.clip(disp_z, cfg.z_low, cfg.z_high)
    z_norm = (z_clipped - cfg.z_low) / (cfg.z_high - cfg.z_low)  # [0,1], 0=low disp, 1=high disp
    
    # Linear interpolation: low dispersion -> conf_weight_low_disp, high -> conf_weight_high_disp
    conf_weight_per_period = (
        cfg.conf_weight_low_disp + 
        z_norm * (cfg.conf_weight_high_disp - cfg.conf_weight_low_disp)
    )
else:
    conf_weight_per_period = cfg.conf_weight_base  # scalar, applied uniformly
```

**3. Update final score combination:**

Change from:
```python
SCORE = (rel * CONF) - risk_pen
```

To:
```python
# conf_weight_per_period is either a Series (per-period weights) or scalar
SCORE = (rel * (CONF * conf_weight_per_period)) - risk_pen
```

If `conf_weight_per_period` is a Series aligned to periods, broadcast it across models when multiplying with the CONF DataFrame.

**Toy example (with defaults `z_low=-1`, `z_high=1`, `conf_weight_low_disp=1.3`, `conf_weight_high_disp=0.7`):**

```
Period with disp_z = -1.0 (very stable regime):
  z_norm = 0.0
  conf_weight = 1.3 + 0.0 * (0.7 - 1.3) = 1.3
  -> Confidence differences amplified by 30%

Period with disp_z = 0.0 (typical regime):
  z_norm = 0.5
  conf_weight = 1.3 + 0.5 * (-0.6) = 1.0
  -> Confidence at baseline weight

Period with disp_z = 1.0 (volatile regime):
  z_norm = 1.0
  conf_weight = 1.3 + 1.0 * (-0.6) = 0.7
  -> Confidence differences dampened by 30%
```

**Why this helps:**

- **Reuses existing infrastructure**: No new lookback windows or dispersion calculations needed—just piggybacking on the already-computed `disp_z` from adaptive alpha.
- **Principled regime adaptation**: In stable regimes, rank stability genuinely differentiates models; in volatile regimes, it's noisier and should matter less.
- **Single localized change**: Only modifies the score combination step, leaving all upstream computations (momentum, delta, baseline, confidence calculation itself, risk) unchanged.
- **Avoids redundancy with prior ideas**: Prior ideas adjust momentum decay (002), add absolute edge (003), streak bonuses (004), cross-ticker normalization (005), and momentum acceleration (006). This idea specifically targets how confidence is *applied*, not how it's computed, making it orthogonal to existing proposals.
