import pandas as pd

from config import MetaConfig
from io_periods import (
    find_first_common_monday,
    wide_to_long_periods,
    aggregate_daily_periods_to_windows,
    align_to_common_periods_per_ticker,
    write_aligned_periods_to_csv,
)


def main() -> None:
    cfg = MetaConfig()
    window_weeks = 2
    window_days = window_weeks * 7

    file_path = r"C:\Users\micha\myhome\algo\artifacts\period_returns\period_returns_days_1.csv"
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()

    print(f"Loaded df: {df.shape} rows/cols")

    anchor_monday = find_first_common_monday(df)
    print(f"Anchoring aggregated periods to Monday: {anchor_monday.date()}")

    daily_long_df = wide_to_long_periods(df)
    print(f"Daily long df: {daily_long_df.shape} rows (model-period records)")

    long_df = aggregate_daily_periods_to_windows(daily_long_df, anchor_monday, window_days=window_days)
    print(f"Aggregated {window_weeks}-week df: {long_df.shape} rows (model-period records)")

    aligned_returns, aligned_meta = align_to_common_periods_per_ticker(long_df, cfg)
    print(f"Tickers kept after alignment: {len(aligned_returns)}")
    for tkr, mat in aligned_returns.items():
        print(f"  {tkr}: models={mat.shape[0]}, common_periods={mat.shape[1]}")

    if not aligned_returns:
        raise RuntimeError("No tickers survived alignment filters. Lower min_models_per_ticker or require_common_periods.")

    aligned_out_path = file_path.replace(
        "period_returns_days_1.csv",
        f"period_returns_weeks_{window_weeks}_aligned.csv",
    )
    write_aligned_periods_to_csv(aligned_returns, aligned_meta, df, aligned_out_path)


if __name__ == "__main__":
    main()
