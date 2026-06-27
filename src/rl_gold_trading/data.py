"""Data loading & splitting for the PPO RAW baseline (Paper Section IV.A, PDF p.5-6).

XAU/USD -> filter 2017-01..2025-01 -> resample HOURLY OHLCV -> drop weekends
(Mon-Fri only) -> forward-fill (<0.1%). Splits are purely temporal (calendar
dates), so there is no leakage.

DEVIATIONS FROM PAPER (user-directed):
  1. Frequency: HOURLY bars (paper resamples to daily). Source is 1-min -> 1h.
  2. Sessions: gold is open 5 days/week, so weekend bars are removed. The source
     produces a spurious Sunday session (weekday 6); keeping only Mon-Fri
     (weekday < 5) yields exactly 5 sessions/week.
"""
from typing import Tuple

import pandas as pd

from rl_gold_trading.config import DataConfig


def _load_raw(cfg: DataConfig) -> pd.DataFrame:
    if cfg.csv_path:
        return pd.read_csv(cfg.csv_path)
    from datasets import load_dataset

    ds = load_dataset(cfg.hf_dataset)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return ds[split].to_pandas()


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    date_col = next((cols[k] for k in ("date", "datetime", "time", "timestamp") if k in cols), None)
    if date_col is None:
        raise ValueError("No datetime column found in dataset.")
    rename = {date_col: "datetime"}
    for c in ("open", "high", "low", "close", "volume"):
        if c in cols:
            rename[cols[c]] = c
    df = df.rename(columns=rename)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
    return df[["open", "high", "low", "close", "volume"]]


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule).agg(agg)
    # Drop empty (non-trading) bars (all-OHLC NaN). Gold is open 5 days/week, so
    # keep Mon-Fri only (weekday < 5); this removes the spurious Sunday session
    # (weekday 6) and stray Saturday bars.
    out = out.dropna(subset=["open", "high", "low", "close"], how="all")
    out = out[out.index.weekday < 5]
    # Forward-fill residual sparse gaps ("forward-fill interpolation", p.6).
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].ffill()
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def load_data(cfg: DataConfig) -> pd.DataFrame:
    df = _standardize(_load_raw(cfg))
    lo = pd.Timestamp(cfg.start, tz="UTC")
    hi = pd.Timestamp(cfg.end, tz="UTC")
    df = df.loc[(df.index >= lo) & (df.index < hi)]
    if cfg.skip_resample:
        return df
    return _resample(df, cfg.resample_rule)


def split_train_test(df: pd.DataFrame, cfg: DataConfig):
    """Calendar split (p.6): train <= train_end ; test >= test_start.

    `df` is the post-normalization (post-warmup) feature frame, so the
    1-year (6048-bar) z-score window has already consumed the earliest rows causally.
    """
    train_end = pd.Timestamp(cfg.train_end, tz="UTC")
    test_start = pd.Timestamp(cfg.test_start, tz="UTC")
    train = df.loc[df.index <= train_end].copy()
    test = df.loc[df.index >= test_start].copy()
    return train, test


def eval_window(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """621-day reported evaluation window Jan 2 2023 -> Sep 12 2024 (p.6, p.9)."""
    lo = pd.Timestamp(cfg.eval_start, tz="UTC")
    hi = pd.Timestamp(cfg.eval_end, tz="UTC")
    return df.loc[(df.index >= lo) & (df.index <= hi)].copy()
