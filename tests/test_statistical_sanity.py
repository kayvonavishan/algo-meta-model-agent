import numpy as np
import pandas as pd

from evaluation import compute_equity_curves
from selection import select_models_universal_v2
from testing_utils import make_synthetic_aligned_returns, make_test_config


def _mean_edge(curves: pd.DataFrame) -> float:
    all_ret = pd.to_numeric(curves["all_models_return"], errors="coerce")
    top_ret = pd.to_numeric(curves["topN_return"], errors="coerce")
    return float((top_ret - all_ret).mean())


def _shuffle_cross_section(
    aligned_returns: dict,
    seed: int = 99,
) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for ticker, mat in aligned_returns.items():
        arr = mat.to_numpy(copy=True)
        for t in range(arr.shape[1]):
            rng.shuffle(arr[:, t])
        out[ticker] = pd.DataFrame(arr, index=mat.index, columns=mat.columns)
    return out


def test_baseline_outperformance_and_label_shuffle() -> None:
    cfg = make_test_config()
    aligned_returns, long_df, _ = make_synthetic_aligned_returns()
    selections = select_models_universal_v2(aligned_returns, cfg)
    curves = compute_equity_curves(
        aligned_returns,
        selections,
        long_df,
        warmup_periods=5,
        top_n=cfg.top_n_global,
    )
    edge_signal = _mean_edge(curves)

    shuffled_returns = _shuffle_cross_section(aligned_returns, seed=123)
    shuffled_selections = select_models_universal_v2(shuffled_returns, cfg)
    shuffled_curves = compute_equity_curves(
        shuffled_returns,
        shuffled_selections,
        long_df,
        warmup_periods=5,
        top_n=cfg.top_n_global,
    )
    edge_shuffled = _mean_edge(shuffled_curves)

    assert edge_signal > 0.001
    assert edge_shuffled < edge_signal * 0.5
