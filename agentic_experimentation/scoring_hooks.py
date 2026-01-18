import csv
from statistics import mean


def compute_score(baseline_csv, candidate_csv, score_column, higher_is_better=True):
    baseline_data = _load_csv(baseline_csv)
    candidate_data = _load_csv(candidate_csv)

    if score_column and score_column in baseline_data["columns"] and score_column in candidate_data["columns"]:
        base_vals = _numeric_column(baseline_data["rows"], baseline_data["columns"], score_column)
        cand_vals = _numeric_column(candidate_data["rows"], candidate_data["columns"], score_column)
        if base_vals and cand_vals:
            base_mean = mean(base_vals)
            cand_mean = mean(cand_vals)
            delta = cand_mean - base_mean
            score = delta if higher_is_better else -delta
            return {
                "score": score,
                "baseline_mean": base_mean,
                "candidate_mean": cand_mean,
                "delta": delta,
                "score_column": score_column,
            }

    return {
        "score": None,
        "score_column": score_column,
        "baseline_columns": baseline_data["columns"],
        "candidate_columns": candidate_data["columns"],
        "note": "Score column not found or non-numeric. Update scoring_hooks.py.",
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
            values.append(float(row[idx]))
        except ValueError:
            continue
    return values
