from dataclasses import dataclass
from typing import Optional


@dataclass
class MetaConfig:
    # Alignment / parsing
    min_models_per_ticker: int = 5   # skip tickers with too few models
    require_common_periods: int = 8  # skip tickers with too few common periods after alignment
    
    # Vol -> alpha (adaptive memory)
    vol_window: int = 4              # 3-5 typical
    alpha_low: float = 0.30          # 0.25-0.35 typical
    alpha_high: float = 0.70         # 0.60-0.75 typical
    z_low: float = -1.0              # clip mapping range
    z_high: float = 1.0
    alpha_smooth: float = 0.30       # EMA smoothing for alpha series (0.1-0.4 typical)
    
    # Momentum / delta
    momentum_lookback: int = 12
    enable_momentum_lookback: bool = True  # when True, truncate adaptive momentum to last N periods
    delta_weight: float = 0.20       # small (0.05-0.3)
    efficiency_weight: float = 0.10  # momentum smoothness bonus
    win_rate_weight: float = 0.08    # fraction of positive-return periods bonus
    momentum_sharpe_weight: float = 0.10  # volatility-adjusted momentum weight
    momentum_sharpe_lookback: int = 8     # lookback in Q periods (percentile-rank series), not raw returns
    rank_persistence_weight: float = 0.06  # weight for rank persistence signal
    rank_persistence_lookback: int = 8     # lookback window for rank persistence
    hit_asymmetry_weight: float = 0.06  # weight for hit rate asymmetry signal
    hit_asymmetry_lookback: int = 10  # lookback window for asymmetric hit rates
    hit_asymmetry_threshold: float = 0.75  # percentile threshold for "top" performance
    breakout_weight: float = 0.08  # weight for top-tier breakout signal
    breakout_threshold: float = 0.80  # rank percentile threshold for top tier
    breakout_lookback: int = 4  # lookback window for breakout stability
    breakout_oscillation_penalty: float = 0.5  # penalty for oscillators around threshold
    rank_durability_weight: float = 0.08  # weight for rank durability signal
    rank_durability_cap: int = 12  # max consecutive above-median tenure counted
    velocity_confirmation_weight: float = 0.06  # weight for velocity confirmation signal
    velocity_lookback: int = 3  # lookback window for velocity confirmation
    velocity_threshold: float = 0.02  # minimum delta magnitude to count as moving
    
    # Confidence (training-free)
    conf_lookback: int = 12
    conf_eps: float = 1e-8

    # Score spread gating (training-free)
    score_spread_boost_weight: float = 0.10
    score_spread_lookback: int = 4
    score_spread_z_threshold: float = 0.5
    
    # Risk penalty (training-free)
    risk_lookback: int = 20
    cvar_alpha: float = 0.10         # tail depth
    cvar_risk_aversion: float = 0.75 # penalty strength
    cvar_window_stride: int = 1      # downsample within CVaR lookback window (1 = exact)

    # Downside volatility cap
    downside_vol_cap_weight: float = 0.12  # 0 disables downside volatility cap
    downside_vol_lookback: int = 8
    downside_vol_threshold_z: float = 1.0
    
    # Baseline
    baseline_method: str = "median"  # "median" or "mean"
    regime_baseline_adjust: float = 0.10  # 0 disables regime adjustment
    regime_dispersion_lookback: int = 6  # lookback for regime dispersion z-score
    
    # Redundancy control (optional)
    enable_uniqueness_weighting: bool = True
    corr_cluster_threshold: float = 0.95  # high correlation => duplicates
    uniqueness_floor: float = 0.25        # prevent weights from going too small
    
    # Selection
    top_n_global: int = 20           # total selected per period across all tickers
    top_m_for_ticker_gate: int = 5   # use top M per ticker to compute ticker score
    include_n_top_tickers: Optional[int] = None  # None = include all tickers; else take top N tickers per period by ticker_score
    per_ticker_cap: Optional[int] = 10  # cap selected models per ticker (None = no cap)
    per_symbol_outer_trial_cap: Optional[int] = 3  # cap selected models per (ticker, outer_trial_number) per period
    min_ticker_score: Optional[float] = None  # abstain tickers below this score (None = no abstain)

    # Sweep / reporting
    scorecard_every: Optional[int] = 10  # build scorecard every N configs during sweep (None = disable)

    def __post_init__(self) -> None:
        if self.downside_vol_cap_weight < 0:
            self.downside_vol_cap_weight = 0.0
        if self.downside_vol_lookback < 2:
            self.downside_vol_lookback = 2
        if self.downside_vol_threshold_z < 0:
            self.downside_vol_threshold_z = 0.0
        if self.rank_durability_cap < 1:
            self.rank_durability_cap = 1
        if self.velocity_lookback < 2:
            self.velocity_lookback = 2
        if self.velocity_threshold < 0:
            self.velocity_threshold = 0.0
