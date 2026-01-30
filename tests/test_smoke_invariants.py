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


def test_per_symbol_outer_trial_cap_enforced() -> None:
    cfg = make_test_config()
    cfg.per_ticker_cap = None
    cfg.top_n_global = 4
    cfg.per_symbol_outer_trial_cap = 1

    aligned_returns, _, _ = make_synthetic_aligned_returns(n_tickers=1, n_models=6)
    # Encode outer+inner trial numbers into model_id so selection can infer outer_trial_number.
    ticker = next(iter(aligned_returns.keys()))
    mat = aligned_returns[ticker]
    new_ids = []
    for model_id in mat.index:
        m = int(str(model_id).split("_m", 1)[1])
        outer = m // 3
        inner = m
        new_ids.append(f"{ticker}|{outer}|{inner}|{model_id}")
    aligned_returns[ticker] = mat.copy()
    aligned_returns[ticker].index = new_ids

    selections = select_models_universal_v2(aligned_returns, cfg)

    model_parts = selections["model_id"].astype(str).str.split("|")
    outer = model_parts.str[1].astype(int)
    sel_with_outer = selections.assign(outer_trial_number=outer)

    counts = sel_with_outer.groupby(["period_end", "ticker", "outer_trial_number"]).size()
    assert (counts <= cfg.per_symbol_outer_trial_cap).all()
