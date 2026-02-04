from __future__ import annotations

from pathlib import Path

from agentic_experimentation.tree_runner import (
    _compute_gate_and_reason,
    _rank_score,
    _select_global_beam,
)


def _minimal_eval(
    *,
    eval_id: str = "0001",
    status: str = "completed",
    candidate_commit: str = "abc123",
    candidate_results_csv_path: str,
    parent_baseline_csv_path: str,
    should_explore: bool = True,
    grade: str | None = "good",
    parent_primary_delta: float | None = 0.1,
    root_score: float | None = 0.5,
    root_primary_delta: float | None = 0.2,
) -> dict:
    return {
        "eval_id": eval_id,
        "status": status,
        "candidate_commit": candidate_commit,
        "candidate_results_csv_path": candidate_results_csv_path,
        "parent_baseline_provenance": {"copied_to_path": parent_baseline_csv_path, "sha256": "x"},
        "parent_relative": {
            "primary_delta": parent_primary_delta,
            "recommendation_summary": {
                "should_explore": should_explore,
                "grade": grade,
                "score": 0.0,
            },
        },
        "root_relative": {
            "primary_delta": root_primary_delta,
            "recommendation_summary": {
                "should_explore": True,
                "grade": "good",
                "score": root_score,
            },
        },
    }


def test_gate_passes_on_should_explore_true(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    baseline_path = artifacts / "baseline.csv"
    baseline_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")
    csv_path = artifacts / "candidate.csv"
    csv_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")

    eval_rec = _minimal_eval(
        candidate_results_csv_path=str(csv_path),
        parent_baseline_csv_path=str(baseline_path),
        should_explore=True,
        grade="bad",
        parent_primary_delta=0.0,
    )
    passed, reason, primary_regressed = _compute_gate_and_reason(eval_rec, artifacts_root=artifacts, sweep_config_limit=None)
    assert passed is True
    assert reason is None
    assert primary_regressed is False


def test_gate_passes_on_mixed_grade_when_primary_not_negative(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    baseline_path = artifacts / "baseline.csv"
    baseline_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")
    csv_path = artifacts / "candidate.csv"
    csv_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")

    eval_rec = _minimal_eval(
        candidate_results_csv_path=str(csv_path),
        parent_baseline_csv_path=str(baseline_path),
        should_explore=False,
        grade="mixed",
        parent_primary_delta=0.0,
    )
    passed, reason, primary_regressed = _compute_gate_and_reason(eval_rec, artifacts_root=artifacts, sweep_config_limit=None)
    assert passed is True
    assert reason is None
    assert primary_regressed is False


def test_gate_fails_when_primary_regresses(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    baseline_path = artifacts / "baseline.csv"
    baseline_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")
    csv_path = artifacts / "candidate.csv"
    csv_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")

    eval_rec = _minimal_eval(
        candidate_results_csv_path=str(csv_path),
        parent_baseline_csv_path=str(baseline_path),
        should_explore=False,
        grade="mixed",
        parent_primary_delta=-1e-9,
    )
    passed, reason, primary_regressed = _compute_gate_and_reason(eval_rec, artifacts_root=artifacts, sweep_config_limit=None)
    assert passed is False
    assert reason == "primary_regressed"
    assert primary_regressed is True


def test_gate_fails_when_candidate_csv_not_in_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    baseline_path = artifacts / "baseline.csv"
    baseline_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")
    outside = tmp_path / "outside.csv"
    outside.write_text("config_id,status\n0,ok\n", encoding="utf-8")

    eval_rec = _minimal_eval(
        candidate_results_csv_path=str(outside),
        parent_baseline_csv_path=str(baseline_path),
        should_explore=True,
        grade="good",
        parent_primary_delta=0.1,
    )
    passed, reason, primary_regressed = _compute_gate_and_reason(eval_rec, artifacts_root=artifacts, sweep_config_limit=None)
    assert passed is False
    assert reason == "candidate_results_csv_not_in_artifacts"
    assert primary_regressed is None


def test_gate_requires_strict_completeness_when_limit_set(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    baseline_path = artifacts / "baseline.csv"
    baseline_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")
    csv_path = artifacts / "candidate.csv"
    csv_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")

    eval_rec = _minimal_eval(
        candidate_results_csv_path=str(csv_path),
        parent_baseline_csv_path=str(baseline_path),
        should_explore=True,
        grade="good",
        parent_primary_delta=0.1,
    )
    passed, reason, primary_regressed = _compute_gate_and_reason(eval_rec, artifacts_root=artifacts, sweep_config_limit=25)
    assert passed is False
    assert reason == "incomplete_or_failed_rows"
    assert primary_regressed is None


def test_rank_score_uses_root_recommendation_score_then_fallbacks(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    baseline_path = artifacts / "baseline.csv"
    baseline_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")
    csv_path = artifacts / "candidate.csv"
    csv_path.write_text("config_id,status\n0,ok\n", encoding="utf-8")

    rec = _minimal_eval(candidate_results_csv_path=str(csv_path), parent_baseline_csv_path=str(baseline_path), root_score=0.75, root_primary_delta=0.11)
    assert _rank_score(rec) == 0.75

    rec2 = _minimal_eval(candidate_results_csv_path=str(csv_path), parent_baseline_csv_path=str(baseline_path), root_score=None, root_primary_delta=0.33)
    assert _rank_score(rec2) == 0.33

    rec3 = _minimal_eval(candidate_results_csv_path=str(csv_path), parent_baseline_csv_path=str(baseline_path))
    rec3["root_relative"] = {}
    assert _rank_score(rec3) == float("-inf")


def test_select_global_beam_is_deterministic() -> None:
    a = {"eval_id": "0001", "root_relative": {"recommendation_summary": {"score": 1.0}, "primary_delta": 0.2}}
    b = {"eval_id": "0002", "root_relative": {"recommendation_summary": {"score": 1.0}, "primary_delta": 0.2}}
    c = {"eval_id": "0003", "root_relative": {"recommendation_summary": {"score": 0.9}, "primary_delta": 0.9}}

    # Deterministic: rank_score desc, then root primary delta desc, then eval_id asc.
    selected = _select_global_beam([a, b, c], beam_width=2)
    assert [r["eval_id"] for r in selected] == ["0001", "0002"]
