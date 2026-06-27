#!/usr/bin/env python3
"""Download native XAU/USD H1 and D1 OHLCV from Dukascopy (with volume)."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import dukascopy_python
import pandas as pd
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "data" / "dukascopy"

_INTERVALS = {
    "1h": (dukascopy_python.INTERVAL_HOUR_1, "xauusd_1h.csv"),
    "1d": (dukascopy_python.INTERVAL_DAY_1, "xauusd_1d.csv"),
}


def _fetch(interval_key: str, start: datetime, end: datetime) -> pd.DataFrame:
    interval, _ = _INTERVALS[interval_key]
    df = dukascopy_python.fetch(
        INSTRUMENT_FX_METALS_XAU_USD,
        interval,
        dukascopy_python.OFFER_SIDE_BID,
        start,
        end,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {interval_key}")
    return df


def _to_pipeline_csv(df: pd.DataFrame, out_path: Path) -> None:
    out = df.reset_index()
    ts_col = "timestamp" if "timestamp" in out.columns else out.columns[0]
    out = out.rename(columns={ts_col: "datetime"})
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    out = out[cols].sort_values("datetime").drop_duplicates("datetime")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    vol_frac = (out["volume"] > 0).mean()
    print(
        f"  {out_path.name}: {len(out):,} bars | "
        f"{out['datetime'].min()} → {out['datetime'].max()} | "
        f"non-zero volume: {vol_frac:.1%}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Download XAU/USD from Dukascopy")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2025-02-01")
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--timeframes",
        nargs="+",
        choices=list(_INTERVALS),
        default=list(_INTERVALS),
    )
    args = ap.parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    for tf in args.timeframes:
        print(f"Fetching {tf.upper()} …")
        df = _fetch(tf, start, end)
        _, filename = _INTERVALS[tf]
        _to_pipeline_csv(df, args.out_dir / filename)


if __name__ == "__main__":
    main()
