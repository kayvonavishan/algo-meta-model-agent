from __future__ import annotations

from pathlib import Path

from agentic_experimentation.scoring_hooks import compute_score


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_compute_score_returns_per_column_deltas(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        baseline,
        """
config_id,a,b,non_numeric
0,1,10,x
1,3,30,y
""",
    )
    _write_csv(
        candidate,
        """
config_id,a,b,non_numeric
0,2,5,x
1,4,15,y
""",
    )

    result = compute_score(baseline, candidate, score_column=None, higher_is_better=True)
    assert result["score"] is None
    assert result["n_numeric_columns_compared"] == 3  # config_id, a, b

    deltas = result["column_deltas"]
    assert set(deltas.keys()) == {"config_id", "a", "b"}
    assert deltas["a"]["baseline_mean"] == 2.0
    assert deltas["a"]["candidate_mean"] == 3.0
    assert deltas["a"]["delta"] == 1.0
    assert deltas["b"]["baseline_mean"] == 20.0
    assert deltas["b"]["candidate_mean"] == 10.0
    assert deltas["b"]["delta"] == -10.0


def test_compute_score_keeps_legacy_score_column_behavior(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        baseline,
        """
config_id,metric
0,1
1,3
""",
    )
    _write_csv(
        candidate,
        """
config_id,metric
0,2
1,4
""",
    )

    result = compute_score(baseline, candidate, score_column="metric", higher_is_better=True)
    assert result["baseline_mean"] == 2.0
    assert result["candidate_mean"] == 3.0
    assert result["delta"] == 1.0
    assert result["score"] == 1.0


def test_compute_score_filters_by_config_id_limit(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        baseline,
        """
config_id,metric
0,0
1,10
2,20
""",
    )
    _write_csv(
        candidate,
        """
config_id,metric
0,1
1,11
2,999
""",
    )

    # Only compare config_id 0..1
    result = compute_score(baseline, candidate, score_column="metric", higher_is_better=True, config_id_limit=2)
    assert result["baseline_rows_used"] == 2
    assert result["candidate_rows_used"] == 2
    assert result["baseline_mean"] == 5.0
    assert result["candidate_mean"] == 6.0
    assert result["delta"] == 1.0


def test_compute_score_recommendation_should_explore_true_on_clear_improvement(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        baseline,
        """
config_id,mean_topN_avg_return_per_trade_pct_oos,core_topN_cagr,core_topN_sharpe,core_topN_max_drawdown,core_topN_cvar_05
0,0.02,0.60,1.50,-0.25,-0.09
""",
    )
    _write_csv(
        candidate,
        """
config_id,mean_topN_avg_return_per_trade_pct_oos,core_topN_cagr,core_topN_sharpe,core_topN_max_drawdown,core_topN_cvar_05
0,0.022,0.63,1.55,-0.24,-0.088
""",
    )

    result = compute_score(baseline, candidate, score_column=None, higher_is_better=True)
    rec = result["recommendation"]
    assert rec["should_explore"] is True
    assert rec["grade"] in {"promising", "strong"}
    assert rec["score"] > 0


def test_compute_score_recommendation_blocks_on_primary_metric_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        baseline,
        """
config_id,mean_topN_avg_return_per_trade_pct_oos,core_topN_cagr,core_topN_sharpe,core_topN_max_drawdown,core_topN_cvar_05
0,0.02,0.60,1.50,-0.25,-0.09
""",
    )
    _write_csv(
        candidate,
        """
config_id,mean_topN_avg_return_per_trade_pct_oos,core_topN_cagr,core_topN_sharpe,core_topN_max_drawdown,core_topN_cvar_05
0,0.018,0.62,1.55,-0.25,-0.09
""",
    )

    result = compute_score(baseline, candidate, score_column=None, higher_is_better=True)
    rec = result["recommendation"]
    assert rec["should_explore"] is False
    assert "primary_metric_regressed" in rec["reasons"]


def test_compute_score_recommendation_blocks_on_risk_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        baseline,
        """
config_id,mean_topN_avg_return_per_trade_pct_oos,core_topN_max_drawdown,core_topN_cvar_05
0,0.02,-0.25,-0.09
""",
    )
    _write_csv(
        candidate,
        """
config_id,mean_topN_avg_return_per_trade_pct_oos,core_topN_max_drawdown,core_topN_cvar_05
0,0.022,-0.35,-0.12
""",
    )

    result = compute_score(baseline, candidate, score_column=None, higher_is_better=True)
    rec = result["recommendation"]
    assert rec["should_explore"] is False
    assert any(r.startswith("risk_regressed:") for r in rec["reasons"])
