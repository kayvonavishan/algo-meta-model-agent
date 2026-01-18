import numpy as np
import pandas as pd

from io_periods import load_aligned_periods_from_csv
from scoring import compute_scores_for_ticker_v2
from selection import select_models_universal_v2
from testing_utils import (
    aligned_returns_to_wide_df,
    make_synthetic_aligned_returns,
    make_test_config,
)


def test_scores_are_finite_and_shifted() -> None:
    cfg = make_test_config()
    aligned_returns, _, period_keys = make_synthetic_aligned_returns()
    ticker, mat = next(iter(aligned_returns.items()))
    scores_df, ticker_score = compute_scores_for_ticker_v2(mat, cfg)

    assert list(scores_df.columns) == period_keys[1:], "scores should be shifted by one period"
    assert list(ticker_score.index) == period_keys[1:], "ticker score should be shifted by one period"
    assert np.isfinite(scores_df.to_numpy()).all()
    assert np.isfinite(ticker_score.to_numpy()).all()


def test_scores_with_strict_momentum_window() -> None:
    cfg = make_test_config()
    cfg.enable_momentum_lookback = True
    cfg.momentum_lookback = 5
    aligned_returns, _, period_keys = make_synthetic_aligned_returns()
    ticker, mat = next(iter(aligned_returns.items()))
    scores_df, ticker_score = compute_scores_for_ticker_v2(mat, cfg)

    assert list(scores_df.columns) == period_keys[1:], "scores should be shifted by one period"
    assert list(ticker_score.index) == period_keys[1:], "ticker score should be shifted by one period"
    assert np.isfinite(scores_df.to_numpy()).all()
    assert np.isfinite(ticker_score.to_numpy()).all()


def test_selection_is_deterministic() -> None:
    cfg = make_test_config()
    aligned_returns, _, _ = make_synthetic_aligned_returns()
    sel_a = select_models_universal_v2(aligned_returns, cfg).reset_index(drop=True)
    sel_b = select_models_universal_v2(aligned_returns, cfg).reset_index(drop=True)
    pd.testing.assert_frame_equal(sel_a, sel_b)


def test_selection_invariant_to_model_order() -> None:
    cfg = make_test_config()
    aligned_returns, _, _ = make_synthetic_aligned_returns()
    rng = np.random.default_rng(123)

    shuffled = {}
    for ticker, mat in aligned_returns.items():
        order = rng.permutation(mat.index)
        shuffled[ticker] = mat.loc[order]

    base = select_models_universal_v2(aligned_returns, cfg).sort_values(
        ["period_end", "rank_global", "ticker", "model_id"]
    ).reset_index(drop=True)
    alt = select_models_universal_v2(shuffled, cfg).sort_values(
        ["period_end", "rank_global", "ticker", "model_id"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(base, alt)


def test_no_nan_or_inf_in_selection_scores() -> None:
    cfg = make_test_config()
    aligned_returns, _, _ = make_synthetic_aligned_returns()
    selections = select_models_universal_v2(aligned_returns, cfg)
    assert np.isfinite(selections["score"]).all()
    assert np.isfinite(selections["ticker_score"]).all()


def test_load_aligned_periods_round_trip() -> None:
    cfg = make_test_config()
    aligned_returns, _, _ = make_synthetic_aligned_returns()
    aligned_df = aligned_returns_to_wide_df(aligned_returns)
    loaded_returns, _ = load_aligned_periods_from_csv(aligned_df, cfg)

    assert set(loaded_returns.keys()) == set(aligned_returns.keys())
    for ticker, mat in aligned_returns.items():
        loaded = loaded_returns[ticker]
        pd.testing.assert_frame_equal(mat, loaded)
