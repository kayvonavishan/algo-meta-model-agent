import csv
import math
from statistics import mean


def compute_score(baseline_csv, candidate_csv, score_column, higher_is_better=True):
    baseline_data = _load_csv(baseline_csv)
    candidate_data = _load_csv(candidate_csv)

    baseline_means = _numeric_means(baseline_data["rows"], baseline_data["columns"])
    candidate_means = _numeric_means(candidate_data["rows"], candidate_data["columns"])

    common_cols = sorted(set(baseline_means.keys()) & set(candidate_means.keys()))
    column_deltas = {
        col: {
            "baseline_mean": baseline_means[col],
            "candidate_mean": candidate_means[col],
            "delta": candidate_means[col] - baseline_means[col],
        }
        for col in common_cols
    }

    # Keep the legacy single-column score for backwards compatibility, but also provide a full per-column summary.
    score = None
    base_mean = None
    cand_mean = None
    delta = None
    if score_column and score_column in baseline_means and score_column in candidate_means:
        base_mean = baseline_means[score_column]
        cand_mean = candidate_means[score_column]
        delta = cand_mean - base_mean
        score = delta if higher_is_better else -delta

    sorted_cols = sorted(common_cols, key=lambda c: column_deltas[c]["delta"])
    top_k = 25
    top_negative = [
        {"column": c, **column_deltas[c]}
        for c in sorted_cols[:top_k]
        if column_deltas[c]["delta"] < 0
    ]
    top_positive = [
        {"column": c, **column_deltas[c]}
        for c in reversed(sorted_cols[-top_k:])
        if column_deltas[c]["delta"] > 0
    ]

    note = None
    if not common_cols:
        note = "No shared numeric columns found between baseline and candidate."
    elif score_column and score is None:
        note = f"score_column={score_column!r} not found (or non-numeric); per-column deltas still computed."

    return {
        "score": score,
        "score_column": score_column,
        "higher_is_better": bool(higher_is_better),
        "baseline_mean": base_mean,
        "candidate_mean": cand_mean,
        "delta": delta,
        "n_numeric_columns_compared": len(common_cols),
        "column_deltas": column_deltas,
        "top_positive_deltas": top_positive,
        "top_negative_deltas": top_negative,
        "baseline_columns": baseline_data["columns"],
        "candidate_columns": candidate_data["columns"],
        "note": note,
    }


def _load_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return {"columns": [], "rows": []}
    columns = rows[0]
    return {"columns": columns, "rows": rows[1:]}


def _numeric_column(rows, columns, col_name):
    try:
        idx = columns.index(col_name)
    except ValueError:
        return []
    values = []
    for row in rows:
        if idx >= len(row):
            continue
        try:
            val = float(row[idx])
        except ValueError:
            continue
        if math.isfinite(val):
            values.append(val)
    return values


def _numeric_means(rows, columns):
    means = {}
    for col in columns:
        vals = _numeric_column(rows, columns, col)
        if vals:
            means[col] = mean(vals)
    return means
