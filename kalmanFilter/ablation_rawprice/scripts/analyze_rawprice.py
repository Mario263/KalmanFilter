"""Aggregate raw-price eval metrics into a seed-aware scoreboard + paired-seed stats.

Honest basis only (price_basis == raw_close). Paired vs rawctl_corrected: mean/median
deltas, bootstrap 95% CI of the Sharpe & return delta, P(Kalman>Raw) per seed, Wilcoxon.
Writes RAWPRICE_SCOREBOARD.md, RAWPRICE_SEED_ROBUSTNESS_REPORT.md, scoreboard.json.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

import numpy as np

_ABLR = Path(__file__).resolve().parents[1]
OUT = _ABLR / "outputs"
DOCS = _ABLR / "docs"
REG = json.loads((_ABLR / "configs" / "experiments_registry.json").read_text())
ORDER = [e["id"] for e in REG["experiments"]]
BASE = "rawctl_corrected"
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def collect(eid):
    rows = {}
    md = OUT / eid / "metrics"
    if md.exists():
        for f in sorted(md.glob("eval_s*.json")) + sorted(md.glob("eval.json")):
            d = json.loads(f.read_text())
            if d.get("price_basis") != "raw_close":
                continue
            seed = d.get("seed") or 42
            rows[int(seed)] = d["kalman_env"]
    return rows


def _boot(deltas, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, float)
    means = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means > 0).mean())


def main():
    data = {eid: collect(eid) for eid in ORDER}
    base = data.get(BASE, {})
    base_sharpe = {s: m["sharpe"] for s, m in base.items()}
    base_ret = {s: m["cumulative_return"] for s, m in base.items()}

    sb_rows, stats_rows = [], []
    for eid in ORDER:
        rows = data[eid]
        if not rows:
            continue
        rets = [m["cumulative_return"] for m in rows.values()]
        shp = [m["sharpe"] for m in rows.values()]
        dd = [m["max_drawdown"] for m in rows.values()]
        rt = [m["round_trips"] for m in rows.values()]
        neg = sum(1 for r in rets if r < 0)
        sb_rows.append({
            "exp": eid, "n": len(rows),
            "ret_mean": sum(rets) / len(rets), "ret_median": st.median(rets),
            "ret_std": st.pstdev(rets) if len(rets) > 1 else 0.0, "ret_min": min(rets), "ret_max": max(rets),
            "sharpe_mean": sum(shp) / len(shp), "sharpe_median": st.median(shp),
            "sharpe_std": st.pstdev(shp) if len(shp) > 1 else 0.0,
            "maxdd_mean": sum(dd) / len(dd), "neg_seeds": neg, "rt_mean": round(sum(rt) / len(rt), 1),
        })
        if eid != BASE and base:
            common = sorted(set(rows) & set(base))
            ds = [rows[s]["sharpe"] - base_sharpe[s] for s in common]
            dr = [rows[s]["cumulative_return"] - base_ret[s] for s in common]
            win = sum(1 for x in ds if x > 0) / len(ds) if ds else 0.0
            slo, shi, sp = _boot(ds) if len(ds) > 1 else (ds[0], ds[0], float(ds[0] > 0))
            rlo, rhi, rp = _boot(dr) if len(dr) > 1 else (dr[0], dr[0], float(dr[0] > 0))
            w = None
            if wilcoxon and len(ds) >= 6 and any(x != 0 for x in ds):
                try:
                    w = float(wilcoxon(ds).pvalue)
                except Exception:
                    w = None
            stats_rows.append({
                "exp": eid, "n_paired": len(common),
                "mean_sharpe_delta": sum(ds) / len(ds), "sharpe_delta_CI95": [slo, shi],
                "P_mean_sharpe_delta_gt0": sp, "P_kalman_beats_raw_per_seed": win,
                "mean_return_delta": sum(dr) / len(dr), "return_delta_CI95": [rlo, rhi],
                "P_mean_return_delta_gt0": rp, "wilcoxon_p_sharpe": w,
            })

    # scoreboard md
    L = ["# RAWPRICE_SCOREBOARD (honest raw_close basis, seed-aware)", "",
         "price_basis = raw_close for ALL cells (PnL/reward/equity/fills on the real tradeable close).",
         "Filtered features used as observation only. PPO/reward/costs/split/z-score frozen.", "",
         "| Exp | n | Ret mean | Ret median | Ret [min,max] | Sharpe mean | Sharpe median | mean MaxDD | neg seeds | trades |",
         "|---|--:|--:|--:|---|--:|--:|--:|--:|--:|"]
    for r in sb_rows:
        L.append(f"| {r['exp']} | {r['n']} | {r['ret_mean']:+.2%} | {r['ret_median']:+.2%} | "
                 f"[{r['ret_min']:+.1%},{r['ret_max']:+.1%}] | {r['sharpe_mean']:.2f} | {r['sharpe_median']:.2f} | "
                 f"{r['maxdd_mean']:.2%} | {r['neg_seeds']} | {r['rt_mean']} |")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "RAWPRICE_SCOREBOARD.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # robustness md
    R = ["# RAWPRICE_SEED_ROBUSTNESS_REPORT", "",
         f"Paired vs **{BASE}** (same seeds). Bootstrap B=20000 on paired deltas.",
         "Promotion needs: mean & median Sharpe > raw, P(per-seed win) >= 70%, Sharpe-delta CI95 lower > 0.", "",
         "| Exp | n | ΔSharpe mean | ΔSharpe CI95 | P(Δμ>0) | P(win/seed) | ΔReturn mean | ΔRet CI95 | Wilcoxon p |",
         "|---|--:|--:|---|--:|--:|--:|---|--:|"]
    for s in stats_rows:
        ci = s["sharpe_delta_CI95"]; rci = s["return_delta_CI95"]
        wp = f"{s['wilcoxon_p_sharpe']:.3f}" if s["wilcoxon_p_sharpe"] is not None else "—"
        R.append(f"| {s['exp']} | {s['n_paired']} | {s['mean_sharpe_delta']:+.2f} | "
                 f"[{ci[0]:+.2f},{ci[1]:+.2f}] | {s['P_mean_sharpe_delta_gt0']:.0%} | {s['P_kalman_beats_raw_per_seed']:.0%} | "
                 f"{s['mean_return_delta']:+.2%} | [{rci[0]:+.1%},{rci[1]:+.1%}] | {wp} |")
    (DOCS / "RAWPRICE_SEED_ROBUSTNESS_REPORT.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    (DOCS / "scoreboard.json").write_text(json.dumps({"scoreboard": sb_rows, "paired_stats": stats_rows}, indent=2), encoding="utf-8")
    print("\n".join(L)); print(); print("\n".join(R))


if __name__ == "__main__":
    main()
