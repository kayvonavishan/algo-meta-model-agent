import json
from pathlib import Path

import pandas as pd

from evaluation import compute_equity_curves
from selection import select_models_universal_v2
from testing_utils import make_synthetic_aligned_returns, make_test_config


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_golden_selection_snapshot() -> None:
    cfg = make_test_config()
    aligned_returns, long_df, _ = make_synthetic_aligned_returns()
    selections = select_models_universal_v2(aligned_returns, cfg).reset_index(drop=True)
    selections["score"] = selections["score"].round(6)
    selections["ticker_score"] = selections["ticker_score"].round(6)

    expected = pd.read_csv(FIXTURES / "golden_selections.csv")
    pd.testing.assert_frame_equal(selections, expected)


def test_golden_curve_summary_snapshot() -> None:
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
    summary = {
        "mean_all_models_return": round(float(curves["all_models_return"].mean()), 6),
        "mean_topN_return": round(float(curves["topN_return"].mean()), 6),
        "mean_edge": round(float((curves["topN_return"] - curves["all_models_return"]).mean()), 6),
        "final_equity_all": round(float(curves["equity_all"].iloc[-1]), 6),
        "final_equity_topN": round(float(curves["equity_topN"].iloc[-1]), 6),
    }
    expected = json.loads((FIXTURES / "golden_curve_summary.json").read_text())
    assert summary == expected
