import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from io_periods import (
    aggregate_daily_periods_to_windows,
    find_first_common_monday,
    wide_to_long_periods,
)


DEFAULT_INPUT = r"C:\Users\micha\myhome\algo\artifacts\period_returns\period_returns_days_1.csv"


def longest_continuous_block(periods: pd.DataFrame) -> List[str]:
    """
    periods columns: [period_key, period_start, period_end], unique and sorted/unsorted.
    Returns period_key list in the longest contiguous block.
    Continuity uses the most common step between consecutive period_end values.
    """
    if periods.empty:
        return []

    periods = periods.sort_values("period_end").reset_index(drop=True)
    ends = periods["period_end"]
    if len(ends) == 1:
        return periods["period_key"].tolist()

    diffs = ends.diff().dropna()
    mode = diffs.mode()
    step = mode.iloc[0] if not mode.empty else diffs.median()

    best_start = 0
    best_len = 1
    cur_start = 0
    cur_len = 1

    for i in range(1, len(ends)):
        if (ends.iloc[i] - ends.iloc[i - 1]) == step:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
            cur_start = i
            cur_len = 1

    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start

    block = periods.iloc[best_start : best_start + best_len]
    return block["period_key"].tolist()


def missing_models_for_period(
    period_key: str,
    models_by_period: Dict[str, Set[str]],
    all_models: Set[str],
) -> List[str]:
    present = models_by_period.get(period_key, set())
    missing = all_models - present
    return sorted(missing)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose which models break global common-period continuity.",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to period_returns_days_1.csv")
    parser.add_argument("--window-weeks", type=int, default=2, help="Aggregation window size in weeks")
    parser.add_argument("--top-models", type=int, default=50, help="Rows to keep in top-missing-models output")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for diagnostic CSVs (defaults to input file directory)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    outdir = Path(args.outdir) if args.outdir else input_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    window_days = args.window_weeks * 7

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path).drop_duplicates()
    print(f"Loaded df: {df.shape} rows/cols")

    anchor_monday = find_first_common_monday(df)
    print(f"Anchor Monday: {anchor_monday.date()}")

    daily_long_df = wide_to_long_periods(df)
    print(f"Daily long df: {daily_long_df.shape} rows")

    agg_df = aggregate_daily_periods_to_windows(
        daily_long_df,
        anchor_monday,
        window_days=window_days,
    )
    print(f"Aggregated {args.window_weeks}-week df: {agg_df.shape} rows")

    agg_df["period_key"] = (
        agg_df["period_start"].dt.strftime("%Y-%m-%d")
        + " to "
        + agg_df["period_end"].dt.strftime("%Y-%m-%d")
    )

    presence = agg_df[["model_id", "period_key", "period_start", "period_end"]].drop_duplicates()
    all_models = sorted(presence["model_id"].unique())
    all_model_set = set(all_models)
    total_models = len(all_models)

    period_stats = (
        presence.groupby(["period_key", "period_start", "period_end"], as_index=False)["model_id"]
        .nunique()
        .rename(columns={"model_id": "model_count"})
        .sort_values("period_end")
        .reset_index(drop=True)
    )
    total_periods = len(period_stats)

    models_by_period = {
        pk: set(g["model_id"].astype(str).tolist())
        for pk, g in presence.groupby("period_key", sort=False)
    }

    period_stats["missing_models"] = total_models - period_stats["model_count"]
    period_stats["coverage_ratio"] = period_stats["model_count"] / max(total_models, 1)

    common_periods = period_stats[period_stats["model_count"] == total_models][
        ["period_key", "period_start", "period_end"]
    ].drop_duplicates()
    keep_keys = longest_continuous_block(common_periods)

    print(f"Global models: {total_models}")
    print(f"Total aggregated periods: {total_periods}")
    print(f"Globally common periods: {len(common_periods)}")
    print(f"Longest common continuous block: {len(keep_keys)}")

    period_rank = {pk: i for i, pk in enumerate(period_stats["period_key"].tolist())}
    boundary_rows: List[dict] = []

    if keep_keys:
        block_sorted = (
            period_stats[period_stats["period_key"].isin(keep_keys)]
            .sort_values("period_end")
            .reset_index(drop=True)
        )
        start_key = block_sorted.iloc[0]["period_key"]
        end_key = block_sorted.iloc[-1]["period_key"]
        start_idx = period_rank[start_key]
        end_idx = period_rank[end_key]

        prev_key: Optional[str] = None
        next_key: Optional[str] = None
        if start_idx > 0:
            prev_key = period_stats.iloc[start_idx - 1]["period_key"]
        if end_idx < (total_periods - 1):
            next_key = period_stats.iloc[end_idx + 1]["period_key"]

        if prev_key is not None:
            missing_prev = missing_models_for_period(prev_key, models_by_period, all_model_set)
            print(f"Boundary before kept block: {prev_key} missing {len(missing_prev)} models")
            for m in missing_prev:
                boundary_rows.append(
                    {
                        "side": "before_kept_block",
                        "period_key": prev_key,
                        "missing_model_id": m,
                    }
                )
        if next_key is not None:
            missing_next = missing_models_for_period(next_key, models_by_period, all_model_set)
            print(f"Boundary after kept block: {next_key} missing {len(missing_next)} models")
            for m in missing_next:
                boundary_rows.append(
                    {
                        "side": "after_kept_block",
                        "period_key": next_key,
                        "missing_model_id": m,
                    }
                )

    present_counts = presence.groupby("model_id")["period_key"].nunique()
    model_stats = (
        pd.DataFrame({"model_id": all_models})
        .set_index("model_id")
        .join(present_counts.rename("present_periods"))
        .fillna(0)
        .astype({"present_periods": int})
    )
    model_stats["missing_periods"] = total_periods - model_stats["present_periods"]
    model_stats["missing_pct"] = model_stats["missing_periods"] / max(total_periods, 1)
    model_stats = model_stats.sort_values(
        ["missing_periods", "present_periods"],
        ascending=[False, True],
    ).reset_index()

    top_model_stats = model_stats.head(max(args.top_models, 1))

    coverage_out = outdir / f"period_coverage_counts_weeks_{args.window_weeks}.csv"
    boundary_out = outdir / f"break_boundary_models_weeks_{args.window_weeks}.csv"
    model_out = outdir / f"top_missing_models_weeks_{args.window_weeks}.csv"

    period_stats.to_csv(coverage_out, index=False)
    pd.DataFrame(boundary_rows).to_csv(boundary_out, index=False)
    top_model_stats.to_csv(model_out, index=False)

    print(f"Wrote: {coverage_out}")
    print(f"Wrote: {boundary_out}")
    print(f"Wrote: {model_out}")
    print("\nTop models by missing periods:")
    print(top_model_stats.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
