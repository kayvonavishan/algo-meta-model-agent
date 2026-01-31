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

