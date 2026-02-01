import csv
import math
from statistics import mean


def compute_score(baseline_csv, candidate_csv, score_column, higher_is_better=True, config_id_limit=None):
    baseline_data = _load_csv(baseline_csv)
    candidate_data = _load_csv(candidate_csv)

    baseline_rows = baseline_data["rows"]
    candidate_rows = candidate_data["rows"]
    applied_limit = None
    if config_id_limit is not None:
        try:
            applied_limit = int(config_id_limit)
        except (TypeError, ValueError):
            applied_limit = None
    if applied_limit is not None:
        if applied_limit < 0:
            raise ValueError(f"config_id_limit must be >= 0 (got {config_id_limit})")
        baseline_rows = _filter_rows_by_config_id_limit(baseline_rows, baseline_data["columns"], applied_limit)
        candidate_rows = _filter_rows_by_config_id_limit(candidate_rows, candidate_data["columns"], applied_limit)

    baseline_means = _numeric_means(baseline_rows, baseline_data["columns"])
    candidate_means = _numeric_means(candidate_rows, candidate_data["columns"])

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

    recommendation = _recommend_follow_up(column_deltas)

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
        "config_id_limit": applied_limit,
        "baseline_rows_used": len(baseline_rows),
        "candidate_rows_used": len(candidate_rows),
        "n_numeric_columns_compared": len(common_cols),
        "column_deltas": column_deltas,
        "top_positive_deltas": top_positive,
        "top_negative_deltas": top_negative,
        "recommendation": recommendation,
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


def _filter_rows_by_config_id_limit(rows, columns, limit):
    """
    Keep only rows whose `config_id` is < limit.
    Falls back to taking the first N rows if `config_id` is missing or non-parseable.
    """
    if limit is None:
        return rows
    try:
        idx = columns.index("config_id")
    except ValueError:
        return rows[:limit]

    kept = []
    for row in rows:
        if idx >= len(row):
            continue
        raw = (row[idx] or "").strip()
        if not raw:
            continue
        try:
            cid = int(raw)
        except ValueError:
            try:
                cid = int(float(raw))
            except ValueError:
                continue
        if cid < limit:
            kept.append(row)
    # If parsing failed and we kept nothing, fall back to deterministic "first N rows".
    return kept if kept else rows[:limit]


def _recommend_follow_up(column_deltas):
    """
    Heuristic "should we explore this idea further?" gate.

    Interprets a small set of key metrics with directionality. Returns a JSONable dict.
    """
    # (column, direction, weight)
    # direction: "higher" means higher is better; "lower" means lower is better.
    metrics = [
        ("mean_topN_avg_return_per_trade_pct_oos", "higher", 3.0),
        ("core_topN_cagr", "higher", 2.0),
        ("core_topN_sharpe", "higher", 1.5),
        ("core_topN_sortino", "higher", 1.5),
        ("core_topN_calmar", "higher", 1.5),
        ("rel_equity_ratio_end", "higher", 1.5),
        ("rel_equity_ratio_cagr", "higher", 1.0),
        ("rel_pct_outperform", "higher", 1.0),
        ("rel_pct_underperform", "lower", 1.0),
        ("core_topN_volatility", "lower", 0.75),
        # These are usually negative; "higher" means less negative tail/drawdown.
        ("core_topN_max_drawdown", "higher", 1.0),
        ("core_topN_cvar_05", "higher", 0.75),
        # Significance (optional; can be noisy in this framework)
        ("sig_t_stat", "higher", 0.5),
        ("sig_t_p_value", "lower", 0.5),
        ("sig_bootstrap_prob_mean_gt0", "higher", 0.5),
    ]

    eps = 1e-12
    per_metric = {}
    weighted_score = 0.0
    weight_sum = 0.0
    positive_signals = 0
    negative_signals = 0

    for col, direction, weight in metrics:
        rec = column_deltas.get(col)
        if not rec:
            continue
        base = rec.get("baseline_mean")
        cand = rec.get("candidate_mean")
        try:
            base_f = float(base)
            cand_f = float(cand)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(base_f) and math.isfinite(cand_f)):
            continue

        if direction == "higher":
            raw_delta = cand_f - base_f
        else:
            raw_delta = base_f - cand_f

        denom = abs(base_f) if abs(base_f) > eps else 1.0
        rel = raw_delta / denom
        # Clamp so a single column can't dominate.
        rel_clamped = max(-2.0, min(2.0, rel))

        per_metric[col] = {
            "direction": direction,
            "weight": weight,
            "baseline_mean": base_f,
            "candidate_mean": cand_f,
            "raw_improvement": raw_delta,
            "relative_improvement": rel,
        }

        weighted_score += weight * rel_clamped
        weight_sum += weight
        if raw_delta > 0:
            positive_signals += 1
        elif raw_delta < 0:
            negative_signals += 1

    score_norm = weighted_score / weight_sum if weight_sum > 0 else 0.0

    # Hard guardrails: block if key risk metrics regress materially.
    # (use relative scale if baseline non-zero; treat drawdown/tail as "higher is better" as above)
    risk_regressions = []
    for col, max_rel_drop in [
        ("core_topN_max_drawdown", 0.10),
        ("core_topN_cvar_05", 0.10),
    ]:
        m = per_metric.get(col)
        if not m:
            continue
        base = abs(m["baseline_mean"])
        if base <= eps:
            continue
        # raw_improvement < 0 means got worse (in the "better" direction)
        if m["raw_improvement"] < 0 and abs(m["raw_improvement"]) / base > max_rel_drop:
            risk_regressions.append(col)

    # Primary metric must not be meaningfully worse.
    primary = per_metric.get("mean_topN_avg_return_per_trade_pct_oos")
    primary_ok = True
    if primary:
        base = abs(primary["baseline_mean"])
        if primary["raw_improvement"] < 0 and (base <= eps or abs(primary["raw_improvement"]) / base > 0.05):
            primary_ok = False

    should_explore = (
        primary_ok
        and not risk_regressions
        and (score_norm >= 0.03 or positive_signals >= max(2, negative_signals + 2))
    )

    if score_norm >= 0.10:
        grade = "strong"
    elif score_norm >= 0.03:
        grade = "promising"
    elif score_norm >= -0.02:
        grade = "mixed"
    else:
        grade = "weak"

    reasons = []
    if not primary_ok:
        reasons.append("primary_metric_regressed")
    if risk_regressions:
        reasons.append("risk_regressed: " + ", ".join(risk_regressions))
    if should_explore:
        reasons.append("net_improvement")
    else:
        reasons.append("insufficient_signal")

    return {
        "should_explore": bool(should_explore),
        "grade": grade,
        "score": score_norm,
        "positive_signals": positive_signals,
        "negative_signals": negative_signals,
        "metrics_used": per_metric,
        "reasons": reasons,
    }
