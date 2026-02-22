IDEA: Add a "Regime-Adaptive Risk Scaling" signal that dynamically increases the CVaR risk penalty weight during periods when the cross-sectional median return is negative (adverse regimes), and decreases it during favorable regimes

RATIONALE: The current CVaR risk penalty (`cvar_risk_aversion * CVaR`) uses a fixed weight regardless of market conditions. However, downside risk becomes much more important during adverse regimes (when most models are underperforming) because in these periods, even "good" models tend to decline, and selecting models with excessive downside exposure leads to amplified losses. Conversely, during favorable regimes, being overly risk-averse can cause the model to miss high-momentum opportunities. The baseline shows a significant OOS performance degradation (0.037% vs 0.14% in-sample), and the rolling min Sharpe can be quite negative (-0.87), suggesting the model is not adequately protecting against drawdowns during regime shifts. By scaling the risk penalty based on a rolling indicator of regime favorability (derived from cross-sectional median returns), we can make the selection more defensive precisely when defensiveness matters most, while remaining opportunistic during favorable periods. This is training-free, uses only lookback data, and directly targets the regime-transition weakness in current scoring.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `regime_risk_scaling_weight: float = 0.5` — how much to scale up risk penalty in adverse regimes (1.0 = double the penalty when fully adverse; 0 = no regime adaptation)
   - `regime_risk_lookback: int = 6` — lookback for computing the rolling regime indicator

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing the raw returns R but before final scoring):**
   - Compute a rolling regime indicator based on cross-sectional median returns:
     ```python
     cs_median_return = np.nanmedian(R, axis=0)  # shape (T,)
     regime_indicator = pd.Series(cs_median_return).rolling(window=cfg.regime_risk_lookback, min_periods=2).mean()
     regime_indicator = regime_indicator.to_numpy()
     ```
   - Normalize to a scaling factor: when regime_indicator is negative (adverse), scale up risk penalty; when positive, scale down:
     ```python
     regime_scaling = 1.0 + cfg.regime_risk_scaling_weight * np.clip(-regime_indicator / (np.nanstd(regime_indicator) + 1e-8), -1.5, 1.5)
     regime_scaling = np.where(np.isfinite(regime_scaling), regime_scaling, 1.0)
     ```
   - This produces a multiplier > 1.0 during adverse regimes (negative median returns) and < 1.0 during favorable regimes

3. **Modify the risk penalty application (around line ~549-550 in scoring.py):**
   - Currently: `risk_pen = cfg.cvar_risk_aversion * risk`
   - Change to: `risk_pen = cfg.cvar_risk_aversion * risk * regime_scaling[None, :]` (broadcast the (T,) regime scaling across all models)

4. **Implementation specifics:**
   - The `regime_scaling` is a 1D array of shape (T,), computed once per ticker, then broadcast to all models
   - When the rolling average of cross-sectional median returns is below zero (bad regime), the z-score is positive after negation, and `regime_scaling > 1.0` increases the risk penalty
   - When the rolling average is above zero (good regime), `regime_scaling < 1.0` decreases the risk penalty
   - The clip to [-1.5, 1.5] prevents extreme scaling factors from outlier regime states
   - This signal is causal and uses only lookback data available at selection time
