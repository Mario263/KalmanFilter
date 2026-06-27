#!/usr/bin/env python3
"""Compare PPO metrics across timeframe experiment runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_METRICS = [
    ("cumulative_return", "Cumulative return", "{:.2%}"),
    ("cagr", "CAGR", "{:.2%}"),
    ("sharpe", "Sharpe", "{:.2f}"),
    ("max_drawdown", "Max drawdown", "{:.2%}"),
    ("trade_win_rate", "Trade win rate", "{:.2%}"),
    ("round_trips", "Round trips", "{:d}"),
    ("n_periods", "Eval bars", "{:d}"),
]


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "—"
    if fmt == "{:d}":
        return fmt.format(int(value))
    return fmt.format(value)


def _best_val(path: Path) -> dict | None:
    """Return the validation record with highest cumulative_return."""
    if not path.is_file():
        return None
    best: dict | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            metrics = rec.get("metrics", rec)
            ret = metrics.get("cumulative_return")
            if ret is None:
                continue
            if best is None or ret > best["metrics"]["cumulative_return"]:
                best = {
                    "iteration": rec.get("iteration"),
                    "timesteps": rec.get("timesteps"),
                    "metrics": metrics,
                }
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--1h",
        dest="metrics_1h",
        type=Path,
        default=_REPO_ROOT / "models/dukascopy_1h/ppo_raw_metrics.json",
    )
    ap.add_argument(
        "--1d",
        dest="metrics_1d",
        type=Path,
        default=_REPO_ROOT / "models/dukascopy_1d/ppo_raw_metrics.json",
    )
    ap.add_argument(
        "--1h-val",
        dest="val_1h",
        type=Path,
        default=_REPO_ROOT / "logs/dukascopy_1h/validation_history.jsonl",
    )
    ap.add_argument(
        "--1d-val",
        dest="val_1d",
        type=Path,
        default=_REPO_ROOT / "logs/dukascopy_1d/validation_history.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "results/dukascopy_1h_vs_1d.json",
    )
    args = ap.parse_args()

    m1h = _load(args.metrics_1h)
    m1d = _load(args.metrics_1d)
    paper = m1h.get("paper_target") or m1d.get("paper_target") or {}
    bv1h = _best_val(args.val_1h)
    bv1d = _best_val(args.val_1d)
    bm1h = bv1h["metrics"] if bv1h else {}
    bm1d = bv1d["metrics"] if bv1d else {}

    rows = []
    header = (
        f"\n{'Metric':<22} {'1H final':>12} {'1H best val':>12} "
        f"{'1D final':>12} {'1D best val':>12} {'Paper':>10}"
    )
    print(header)
    print("-" * 84)
    for key, label, fmt in _METRICS:
        v1h = m1h["reproduced"].get(key)
        v1d = m1d["reproduced"].get(key)
        vb1h = bm1h.get(key)
        vb1d = bm1d.get(key)
        vp = paper.get(key) if key != "round_trips" and key != "n_periods" else None
        if key == "win_rate" and vp is not None:
            vp = paper.get("win_rate")
        print(
            f"{label:<22} {_fmt(v1h, fmt):>12} {_fmt(vb1h, fmt):>12} "
            f"{_fmt(v1d, fmt):>12} {_fmt(vb1d, fmt):>12} "
            f"{_fmt(vp, fmt) if vp is not None else '—':>10}"
        )
        rows.append({
            "metric": key,
            "label": label,
            "1h_final": v1h,
            "1h_best_val": vb1h,
            "1d_final": v1d,
            "1d_best_val": vb1d,
            "paper": vp,
        })

    if bv1h:
        print(
            f"\n1H best val: iter {bv1h['iteration']} "
            f"({bv1h['timesteps']:,} steps)"
        )
    if bv1d:
        print(
            f"1D best val: iter {bv1d['iteration']} "
            f"({bv1d['timesteps']:,} steps)"
        )

    report = {
        "1h": {
            "path": str(args.metrics_1h),
            "timesteps": m1h.get("timesteps"),
            "eval_window": m1h.get("eval_window"),
            "reproduced": m1h["reproduced"],
            "best_val": bv1h,
        },
        "1d": {
            "path": str(args.metrics_1d),
            "timesteps": m1d.get("timesteps"),
            "eval_window": m1d.get("eval_window"),
            "reproduced": m1d["reproduced"],
            "best_val": bv1d,
        },
        "paper_target": paper,
        "comparison": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
