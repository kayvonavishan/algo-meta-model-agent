from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import MetaConfig
from sweep import run_config_sweep


def test_run_config_sweep_respects_config_limit(tmp_path: Path) -> None:
    out_path = tmp_path / "out.csv"
    cfg = MetaConfig()

    # Use empty inputs; run_config_sweep will still write one row per config (typically as status=error),
    # which is sufficient to verify slicing behavior without running the full computation.
    run_config_sweep(
        aligned_returns={},
        long_df=pd.DataFrame(),
        base_cfg=cfg,
        out_path=str(out_path),
        n_configs=100,
        config_limit=5,
        seed=42,
        enable_plots=False,
    )

    df = pd.read_csv(out_path)
    assert df.shape[0] == 5
    assert list(df["config_id"]) == [0, 1, 2, 3, 4]


def test_run_config_sweep_zero_limit_writes_nothing(tmp_path: Path) -> None:
    out_path = tmp_path / "out.csv"
    cfg = MetaConfig()

    run_config_sweep(
        aligned_returns={},
        long_df=pd.DataFrame(),
        base_cfg=cfg,
        out_path=str(out_path),
        n_configs=100,
        config_limit=0,
        seed=42,
        enable_plots=False,
    )

    assert not out_path.exists()

