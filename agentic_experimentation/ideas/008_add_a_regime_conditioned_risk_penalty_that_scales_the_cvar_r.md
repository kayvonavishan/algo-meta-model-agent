IDEA: Add a "regime-conditioned risk penalty" that scales the CVaR risk aversion coefficient based on the current dispersion regime, applying stronger risk penalties during stable (low-dispersion) periods when downside rank shocks are more anomalous and concerning, and weaker penalties during volatile (high-dispersion) periods when rank fluctuations are expected and less informative about true model quality.

RATIONALE: The current risk penalty (Step 6.8) applies a fixed `cvar_risk_aversion` coefficient uniformly across all periods. However, the same magnitude of downside CVaR has different implications depending on the regime. During low-dispersion periods, models are tightly clustered—a model that experiences a rank shock in this environment is showing genuine weakness that stands out from the pack. During high-dispersion periods, even strong models can experience temporary rank drops due to the chaotic environment—penalizing these equally harsh may unfairly punish models that are simply caught in market-wide turbulence rather than exhibiting fundamental problems. This mirrors the logic of idea 007 (adaptive confidence weighting) but applied to the risk penalty side: just as we trust rank stability more in stable regimes, we should also treat rank shocks more seriously in stable regimes. The implementation reuses the existing `disp_z` infrastructure, making it a clean, localized change.

REQUIRED_CHANGES:

**1. Add new config parameters to `MetaConfig` (in `config.py`):**
```python
enable_adaptive_risk_penalty: bool = True
risk_aversion_low_disp: float = 1.0  # risk aversion multiplier when dispersion is low (stable regime)
risk_aversion_high_disp: float = 0.5  # risk aversion multiplier when dispersion is high (volatile regime)
```

These multiply the base `cvar_risk_aversion` parameter, so the effective risk aversion for a period becomes `cvar_risk_aversion * regime_multiplier`.

**2. Modify `compute_scores_for_ticker_v2` in `scoring.py`:**

After computing `disp_z` (dispersion z-scores per period, already computed in Step 6.2.2 for adaptive alpha) and after computing the raw downside CVaR values (`CVaR_{i,t}`), add logic to compute a regime-dependent risk aversion multiplier:

```python
# After computing disp_z (already exists for adaptive alpha)
# and after computing raw CVaR values (risk_lookback rolling window)

if cfg.enable_adaptive_risk_penalty:
    # Map dispersion z-score to risk aversion multiplier
    # Low dispersion (z < 0) -> higher risk aversion (penalize rank shocks more)
    # High dispersion (z > 0) -> lower risk aversion (expect some volatility)
    
    z_clipped = np.clip(disp_z, cfg.z_low, cfg.z_high)
    z_norm = (z_clipped - cfg.z_low) / (cfg.z_high - cfg.z_low)  # [0,1], 0=low disp, 1=high disp
    
    # Linear interpolation: low dispersion -> risk_aversion_low_disp, high -> risk_aversion_high_disp
    risk_multiplier_per_period = (
        cfg.risk_aversion_low_disp + 
        z_norm * (cfg.risk_aversion_high_disp - cfg.risk_aversion_low_disp)
    )
    
    # Apply to base risk aversion
    effective_risk_aversion = cfg.cvar_risk_aversion * risk_multiplier_per_period
else:
    effective_risk_aversion = cfg.cvar_risk_aversion  # scalar, applied uniformly
```

**3. Update risk penalty calculation:**

Change from:
```python
risk_pen_{i,t} = cfg.cvar_risk_aversion * CVaR_{i,t}
```

To:
```python
# effective_risk_aversion is either a Series (per-period) or scalar
risk_pen_{i,t} = effective_risk_aversion[t] * CVaR_{i,t}
```

When `effective_risk_aversion` is a Series aligned to periods, broadcast it appropriately when computing the penalty DataFrame.

**Toy example (with defaults `z_low=-1`, `z_high=1`, `risk_aversion_low_disp=1.0`, `risk_aversion_high_disp=0.5`, `cvar_risk_aversion=0.75`):**

```
Period with disp_z = -1.0 (very stable regime):
  z_norm = 0.0
  risk_multiplier = 1.0 + 0.0 * (0.5 - 1.0) = 1.0
  effective_risk_aversion = 0.75 * 1.0 = 0.75
  -> Full risk penalty applied (rank shocks are meaningful here)

Period with disp_z = 0.0 (typical regime):
  z_norm = 0.5
  risk_multiplier = 1.0 + 0.5 * (-0.5) = 0.75
  effective_risk_aversion = 0.75 * 0.75 = 0.5625
  -> Moderately reduced risk penalty

Period with disp_z = 1.0 (volatile regime):
  z_norm = 1.0
  risk_multiplier = 1.0 + 1.0 * (-0.5) = 0.5
  effective_risk_aversion = 0.75 * 0.5 = 0.375
  -> Risk penalty halved (rank volatility is expected, don't over-penalize)
```

**Why this helps:**

- **Symmetric regime adaptation**: Pairs naturally with idea 007 (adaptive confidence weighting)—together they make both the "reward" side (confidence) and "penalty" side (risk) regime-aware.
- **Principled interpretation**: A model dropping from rank 0.8 to 0.3 during a calm period is a red flag; the same drop during market chaos might just be noise.
- **Reuses existing infrastructure**: No new lookback windows or dispersion calculations—just piggybacking on `disp_z` from adaptive alpha.
- **Single localized change**: Only modifies how `risk_pen` is scaled, leaving CVaR calculation, confidence, momentum, and all other components unchanged.
- **Avoids over-penalizing adaptive models**: Models that respond appropriately to volatile regimes (and thus have some rank volatility) won't be unfairly penalized compared to "stale" models that happen to have stable but outdated positioning.
