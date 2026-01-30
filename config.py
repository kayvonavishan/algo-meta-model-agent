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
    
    # Confidence (training-free)
    conf_lookback: int = 12
    conf_eps: float = 1e-8
    
    # Risk penalty (training-free)
    risk_lookback: int = 20
    cvar_alpha: float = 0.10         # tail depth
    cvar_risk_aversion: float = 0.75 # penalty strength
    cvar_window_stride: int = 1      # downsample within CVaR lookback window (1 = exact)
    
    # Baseline
    baseline_method: str = "median"  # "median" or "mean"
    
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
