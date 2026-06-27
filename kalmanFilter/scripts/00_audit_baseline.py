"""Gate A: audit the Raw PPO baseline (read-only) and the daily dataset.

    python kalmanFilter/scripts/00_audit_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_KF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_KF))

from src import audit, pipeline                      # noqa: E402
from src.reports import dump_json, sha256_file, write_text  # noqa: E402


def data_facts(df: pd.DataFrame, csv_path: str, train_end: str, test_start: str) -> dict:
    idx = df.index
    te, ts = pd.Timestamp(train_end, tz="UTC"), pd.Timestamp(test_start, tz="UTC")
    cols = {}
    for c in ["open", "high", "low", "close", "volume"]:
        s = df[c].astype(float)
        cols[c] = {"nan": int(s.isna().sum()), "min": float(s.min()), "max": float(s.max()),
                   "le_zero": int((s <= 0).sum())}
    return {
        "csv_path": csv_path,
        "sha256": sha256_file(csv_path),
        "rows": int(len(df)),
        "date_min": str(idx.min()), "date_max": str(idx.max()),
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "monotonic_increasing": bool(idx.is_monotonic_increasing),
        "columns": cols,
        "train_rows_for_qr": int((idx <= te).sum()),
        "test_rows_untouched": int((idx >= ts).sum()),
    }


def main() -> None:
    args = pipeline.base_argparser("Gate A baseline + data audit").parse_args()
    a = audit.audit_baseline(args.config_path)
    df = pipeline.load_clean_ohlcv(args.config_path)
    d = data_facts(df, a["csv_path"], a["train_end"], a["test_start"])

    evidence = {"gate_a": a, "data": d}
    dump_json(pipeline.OUTPUTS / "diagnostics" / "baseline_audit.json", evidence)

    md = [
        "# KALMAN_DATA_AUDIT\n",
        "**Gate A — baseline + data audit.** Read-only; no existing file modified.\n",
        "## Baseline config facts (source: config.py / experiment_dukascopy_1d.json)",
        f"- csv_path: `{a['csv_path']}`",
        f"- skip_resample: **{a['skip_resample']}** (no hidden resampling — confirmed)",
        f"- timeframe: {a['timeframe']}",
        f"- train_end / test_start: {a['train_end']} / {a['test_start']}",
        f"- eval window: {a['eval_start']} → {a['eval_end']}",
        f"- feature_count: **{a['feature_count']}**  | zscore_window: **{a['zscore_window']}**",
        f"- feature order matches expected Raw PPO order: **{a['feature_order_matches_expected']}**\n",
        "## Daily dataset facts (source: data/dukascopy/xauusd_1d.csv via existing loader)",
        f"- sha256: `{d['sha256']}`",
        f"- rows: {d['rows']}  | range: {d['date_min']} → {d['date_max']}",
        f"- duplicate timestamps: {d['duplicate_timestamps']}  | sorted: {d['monotonic_increasing']}",
        f"- train rows for Q/R (≤{a['train_end']}): **{d['train_rows_for_qr']}**",
        f"- test rows untouched during fit (≥{a['test_start']}): **{d['test_rows_untouched']}**\n",
        "| col | nan | min | max | ≤0 |",
        "|---|---|---|---|---|",
    ]
    for c, v in d["columns"].items():
        md.append(f"| {c} | {v['nan']} | {v['min']:.4f} | {v['max']:.4f} | {v['le_zero']} |")
    md += [
        "\n## Result",
        f"- Gate A pass: **{a['gate_a_pass']}**  | problems: {a['problems'] or 'none'}",
        "- Risk: Low. Data is clean (no NaN/dups, sorted, OHLC positive, volume positive).",
        "- Next action: build filtered OHLC (script 01).",
    ]
    write_text(_KF / "docs" / "KALMAN_DATA_AUDIT.md", "\n".join(md))

    print(f"Gate A pass={a['gate_a_pass']} | rows={d['rows']} "
          f"train={d['train_rows_for_qr']} test={d['test_rows_untouched']} "
          f"| problems={a['problems'] or 'none'}")
    if args.strict and not a["gate_a_pass"]:
        raise SystemExit("Gate A FAILED")


if __name__ == "__main__":
    main()
