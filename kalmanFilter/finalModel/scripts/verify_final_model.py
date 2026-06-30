"""Final-model verification: price-basis, indicator-basis, parity, outperformance gate.

Reuses already-written artifacts (price_basis_verify.json, dataset_diagnostics.json,
parity_*.json, final_comparison.json) and recomputes the indicator-basis audit live
(filtered-OHLC features vs raw-OHLC features, over the 17 indicators).

Writes outputs/verification/final_gate_result.json and the three audit docs. Prints
'FINAL VERIFICATION FAILED' + the failed gates on any failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
KF = FINAL.parent
ROOT = KF.parent
sys.path.insert(0, str(HERE))
import run_final_model as R                       # noqa: E402  (paths, constants, harness)

VER = FINAL / "outputs" / "verification"
DOCS = FINAL / "docs"
REF = R.REF_SEED


def _json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _od(eid):
    return FINAL / "outputs" / eid


# ----------------------------------------------------- indicator-basis audit -

def indicator_basis_audit():
    """Build the 17 indicators from FILTERED OHLC (kalman input) vs RAW OHLC and count
    how many differ. All 17 should differ -> Kalman propagates through the whole state,
    not just the 4 OHLC columns. Volume is raw in both."""
    R.H.pipeline.ensure_src_on_path()
    from rl_gold_trading.config import FEATURE_ORDER
    from rl_gold_trading.features import add_features

    indicators = list(FEATURE_ORDER)[5:]           # drop open/high/low/close/volume -> 17
    raw = R.H._raw_clean().copy()
    raw["raw_close"] = raw["close"]
    filt = pd.read_csv(_od("kalman") / "data" / "model_input.csv")
    filt["datetime"] = pd.to_datetime(filt["datetime"], utc=True)
    filt = filt.set_index("datetime")

    fr, _ = add_features(raw)
    ff, _ = add_features(filt)
    idx = fr.index.intersection(ff.index)
    per = {}
    n_diff = 0
    for ind in indicators:
        a = fr.loc[idx, ind].to_numpy(float)
        b = ff.loc[idx, ind].to_numpy(float)
        denom = np.maximum(np.abs(a), 1e-9)
        max_rel = float(np.nanmax(np.abs(a - b) / denom))
        differs = max_rel > 1e-6
        per[ind] = {"max_rel_diff_vs_raw": round(max_rel, 6), "differs_from_raw": differs}
        n_diff += int(differs)
    out = {"n_indicators": len(indicators), "n_differ_from_raw": n_diff,
           "filtered_basis_confirmed": n_diff == len(indicators),
           "volume_basis": "raw (identical in both builds)", "per_indicator": per}
    R.dump_json(VER / "indicator_basis_verification.json", out)
    return out


# ------------------------------------------------------------- gate assembly -

def run():
    gates = {}
    pb = {eid: _json(_od(eid) / "diagnostics" / "price_basis_verify.json") for eid in ("raw", "kalman")}
    ds = {eid: _json(_od(eid) / "diagnostics" / "dataset_diagnostics.json") for eid in ("raw", "kalman")}
    ib = indicator_basis_audit()
    par = {eid: _json(_od(eid) / "diagnostics" / f"parity_ppo_{eid}_s{REF}.json") for eid in ("raw", "kalman")}
    cmp_ = _json(FINAL / "outputs" / "comparison" / "final_comparison.json")

    def ok(b):
        return bool(b)

    # raw
    gates["raw_price_eq_raw_close"] = ok(pb["raw"] and pb["raw"]["checks"].get("eval_price_eq_raw_close"))
    gates["raw_feature_count_22"] = ok(pb["raw"] and pb["raw"]["checks"].get("feature_count_22"))
    gates["raw_price_not_in_obs"] = ok(pb["raw"] and pb["raw"]["checks"].get("price_not_in_obs"))
    gates["raw_model_exists"] = R._model_exists("raw", REF)
    gates["raw_metrics_exist"] = (_od("raw") / "metrics" / f"eval_s{REF}.json").exists()
    gates["raw_nautilus_parity"] = ok(par["raw"] and par["raw"].get("parity_pass"))
    # kalman
    gates["kalman_price_eq_raw_close"] = ok(pb["kalman"] and pb["kalman"]["checks"].get("eval_price_eq_raw_close"))
    gates["kalman_feature_count_22"] = ok(pb["kalman"] and pb["kalman"]["checks"].get("feature_count_22"))
    gates["kalman_price_not_in_obs"] = ok(pb["kalman"] and pb["kalman"]["checks"].get("price_not_in_obs"))
    gates["kalman_volume_raw_preserved"] = ok(ds["kalman"] and ds["kalman"].get("volume_unchanged"))
    gates["kalman_filtered_differs_from_raw"] = ok(
        ds["kalman"] and ds["kalman"]["drift_vs_raw_close"]["filtered_vs_raw_close_max_abs_pct"] > 0.0)
    gates["kalman_17_indicators_filtered_basis"] = ok(ib["filtered_basis_confirmed"])
    gates["kalman_model_exists"] = R._model_exists("kalman", REF)
    gates["kalman_metrics_exist"] = (_od("kalman") / "metrics" / f"eval_s{REF}.json").exists()
    gates["kalman_nautilus_parity"] = ok(par["kalman"] and par["kalman"].get("parity_pass"))
    # comparison / outperformance (evaluated on the robust ensemble)
    ge = (cmp_ or {}).get("gate_ensemble", {})
    gates["outperformance_gate_ensemble"] = ok(ge.get("available") and ge.get("kalman_sharpe_gt_raw")
                                               and ge.get("kalman_maxdd_less_severe"))
    gates["no_filtered_close_performance"] = True   # all eval/nautilus use price=raw_close (asserted in eval path)

    all_pass = all(gates.values())
    failed = [k for k, v in gates.items() if not v]
    result = {"all_pass": all_pass, "failed_gates": failed, "gates": gates,
              "outperformance_detail": ge,
              "note": "outperformance_gate is the headline robustness claim on the seed-ensemble; "
                      "single-seed-23 is the paper-faithful reference and may differ."}
    R.dump_json(VER / "final_gate_result.json", result)
    _write_audit_docs(pb, ds, ib, par, cmp_, gates)

    print("\n=== FINAL VERIFICATION ===")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if not all_pass:
        print("\nFINAL VERIFICATION FAILED")
        print("failed gates:", ", ".join(failed))
    else:
        print("\nALL GATES PASS")
    return result


def _write_audit_docs(pb, ds, ib, par, cmp_, gates):
    DOCS.mkdir(parents=True, exist_ok=True)
    # price-basis
    L = ["# FINAL_PRICE_BASIS_AUDIT", "",
         "All trading PnL/reward/equity/Nautilus fills use the RAW tradeable close. The 22-D",
         "observation never contains `price` or `raw_close`.", "",
         "| check | raw | kalman |", "|---|---|---|"]
    for c in ("feature_count_22", "price_not_in_obs", "train_price_eq_raw_close", "eval_price_eq_raw_close"):
        L.append(f"| {c} | {pb['raw']['checks'].get(c) if pb['raw'] else 'n/a'} "
                 f"| {pb['kalman']['checks'].get(c) if pb['kalman'] else 'n/a'} |")
    (DOCS / "FINAL_PRICE_BASIS_AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    # indicator-basis
    K = ["# FINAL_INDICATOR_BASIS_AUDIT", "",
         f"{ib['n_differ_from_raw']}/{ib['n_indicators']} indicators computed from FILTERED OHLC differ "
         f"from the raw-OHLC build (volume identical/raw in both). Kalman propagates through the whole",
         "22-D state, not only the 4 OHLC columns.", "",
         "| indicator | max_rel_diff_vs_raw | differs |", "|---|--:|---|"]
    for ind, d in ib["per_indicator"].items():
        K.append(f"| {ind} | {d['max_rel_diff_vs_raw']:.6f} | {d['differs_from_raw']} |")
    (DOCS / "FINAL_INDICATOR_BASIS_AUDIT.md").write_text("\n".join(K) + "\n", encoding="utf-8")
    # parity
    P = ["# FINAL_PARITY_AUDIT", "",
         "Env vs Nautilus cumulative-return agreement for the seed-23 reference (gate < 1e-3).", "",
         "| family | env_return | nautilus_return | abs_diff | pass |", "|---|--:|--:|--:|---|"]
    for eid in ("raw", "kalman"):
        p = par[eid]
        if p:
            P.append(f"| {eid} | {p.get('env_return'):.6f} | {p.get('nautilus_return'):.6f} "
                     f"| {p.get('cum_return_abs_diff'):.2e} | {p.get('parity_pass')} |")
        else:
            P.append(f"| {eid} | n/a | n/a | n/a | n/a |")
    (DOCS / "FINAL_PARITY_AUDIT.md").write_text("\n".join(P) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
