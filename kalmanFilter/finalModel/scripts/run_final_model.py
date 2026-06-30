"""Final model orchestrator: Raw PPO vs Kalman PPO, single seed-23 + seed-ensemble.

This is an ORCHESTRATION WRAPPER. It reuses the frozen entry points unchanged:
  - dataset/verify/train/eval/nautilus : kalmanFilter/ablation_rawprice/scripts/run_rawprice_ablation.py
  - PPO training + reward             : rl_gold_trading.run.main
  - metric formulas                   : rl_gold_trading.metrics.evaluate_model  (reused as-is, even for the ensemble)
  - Nautilus execution                : nautilus/run_backtest.py

Why an ensemble: a single PPO seed is variance-dominated on this ~1500-row daily task
(~25 trades). The honest fix for "I don't want it seed-dependent" is to deploy a fixed
N-seed ensemble (soft vote = argmax of summed action-probabilities). Its behaviour no
longer hinges on a lucky seed. Seed 23 is kept as the paper-faithful single reference.

Trading PnL/reward/equity/fills use the RAW close for BOTH families (honest). Kalman uses
filtered OHLC + raw Volume only as the 22-D observation. Nothing in PPO/reward/costs/
action/split/z-score or Nautilus logic changes.

Stages: setup train_raw eval_raw nautilus_raw train_kalman eval_kalman nautilus_kalman
        compare verify diagnose excel all

    python run_final_model.py --stage all --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # finalModel/scripts
FINAL = HERE.parent                             # finalModel
KF = FINAL.parent                               # kalmanFilter
ROOT = KF.parent                                # repo root
ABLR = KF / "ablation_rawprice"

# Reuse the ablation harness, redirect its output root into finalModel/outputs/<eid>.
sys.path.insert(0, str(ABLR / "scripts"))
import run_rawprice_ablation as H               # noqa: E402
H._ABLR = FINAL                                 # _od(eid) -> finalModel/outputs/<eid>

sys.path.insert(0, str(KF))
from src.reports import dump_json               # noqa: E402

# --- fixed final configuration (do NOT change; see configs/final_model_spec.json) ---
ENSEMBLE_SEEDS = [1, 7, 13, 23, 42]             # predeclared; seed 23 included as the reference
REF_SEED = 23
EXPS = {
    "raw":    {"id": "raw", "source": "raw",
               "transform": "none", "covariance": "bypass"},
    "kalman": {"id": "kalman", "source": "kalmanFilter/outputs/data/xauusd_1d_kalman_input.csv",
               "transform": "none", "covariance": "full"},
}
# existing honest 10-seed families (for seed-dependence root-cause; no retraining)
DIAG_RAW, DIAG_KAL = "rawctl_corrected", "exp_002_corrected"
DIAG_SEEDS = [1, 2, 3, 7, 11, 13, 17, 19, 23, 42]


def _od(eid):
    return FINAL / "outputs" / eid


def _model_stub(eid, seed):
    return _od(eid) / "models" / f"ppo_{eid}_s{seed}"


def _model_exists(eid, seed):
    return Path(str(_model_stub(eid, seed)) + ".zip").exists()


# ------------------------------------------------------------------ setup ----

def setup():
    for eid in EXPS:
        for sub in ("data", "models", "metrics", "diagnostics", "nautilus", "logs"):
            (_od(eid) / sub).mkdir(parents=True, exist_ok=True)
    (FINAL / "outputs" / "comparison").mkdir(parents=True, exist_ok=True)
    (FINAL / "outputs" / "verification").mkdir(parents=True, exist_ok=True)
    _write_spec()
    for exp in EXPS.values():
        H.prepare_dataset(exp)                  # writes model_input.csv + runtime_config.json
        H.verify(exp, None)                     # price-basis invariants (hard stop on fail)
    # publish the two effective runtime configs under configs/ (deliverable names)
    for eid, name in (("raw", "final_raw_config.json"), ("kalman", "final_kalman_config.json")):
        src = _od(eid) / "diagnostics" / "runtime_config.json"
        (FINAL / "configs" / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print("[setup] datasets prepared + price-basis verified; configs published")


def _write_spec():
    spec = {
        "final_model_name": "ppo_kalman_fullcov_rawprice_seed23",
        "robust_production_model": "ppo_kalman_fullcov_rawprice_ensemble5",
        "final_experiment_id": "exp_002_final",
        "source_experiment_family": "exp_002_corrected",
        "reference_seed": REF_SEED,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "dataset": "dukascopy_1d", "asset": "XAUUSD", "frequency": "1d",
        "eval_window": ["2023-01-02", "2024-09-12"],
        "ppo_timesteps": 500000, "state_dimension": 22, "device": "cpu",
        "kalman": {"enabled": True, "state": "OHLC_4D", "volume_policy": "raw_passthrough",
                   "covariance_type": "full", "pre_kalman_transform": "none",
                   "q_r_estimation": "EM_train_only", "filter_type": "causal_filter_not_smoother"},
        "features": {"ohlc_basis": "kalman_filtered_ohlc", "volume_basis": "raw_volume",
                     "indicator_basis": "filtered_ohlc_plus_raw_volume",
                     "price_column_in_observation": False, "raw_close_column_in_observation": False},
        "trading_price": {"price_basis": "raw_close", "reward_price_basis": "raw_close",
                          "equity_price_basis": "raw_close", "nautilus_fill_price_basis": "raw_close"},
        "frozen_components": {k: False for k in (
            "ppo_hyperparameters_changed", "reward_changed", "transaction_costs_changed",
            "action_mapping_changed", "position_sizing_changed", "zscore_window_changed",
            "train_test_split_changed")},
        "seed_dependence_fix": {
            "method": "soft_vote_seed_ensemble",
            "aggregation": "argmax of summed action-probabilities over members",
            "rationale": "single-seed PPO is variance-dominated; ensemble removes seed-dependence",
            "members": len(ENSEMBLE_SEEDS)},
        "selection": {"rule": "fixed_representative_median_sharpe_seed_from_promoted_family",
                      "not_best_seed": True},
    }
    dump_json(FINAL / "configs" / "final_model_spec.json", spec)


# ----------------------------------------------------------- train / eval ----

def train_family(eid, device, skip_if_exists, force):
    exp = EXPS[eid]
    if not (_od(eid) / "data" / "model_input.csv").exists():
        H.prepare_dataset(exp)
        H.verify(exp, None)
    for s in ENSEMBLE_SEEDS:
        if _model_exists(eid, s) and skip_if_exists and not force:
            print(f"[train_{eid}] seed {s} exists -> skip")
            continue
        print(f"[train_{eid}] seed {s} ...")
        H.train(exp, s, device)


def eval_family(eid):
    """Per-seed eval (reuses harness) + ensemble eval (soft vote)."""
    exp = EXPS[eid]
    for s in ENSEMBLE_SEEDS:
        if _model_exists(eid, s):
            H.evaluate(exp, s, device="cpu")
    _eval_ensemble(eid)


class _SoftVote:
    """Duck-typed 'model' exposing .predict so metrics.evaluate_model can score the ensemble."""
    def __init__(self, models):
        self.models = models

    def predict(self, obs, deterministic=True):
        import torch
        probs = None
        for m in self.models:
            obs_t, _ = m.policy.obs_to_tensor(np.asarray(obs))
            with torch.no_grad():
                p = m.policy.get_distribution(obs_t).distribution.probs.cpu().numpy().reshape(-1)
            probs = p if probs is None else probs + p
        return int(np.argmax(probs)), None


def _eval_ensemble(eid):
    H.pipeline.ensure_src_on_path()
    from rl_gold_trading.config import data_config, env_config, set_config_path
    from rl_gold_trading.envs import XAUUSDTradingEnv
    from rl_gold_trading.metrics import evaluate_model
    from rl_gold_trading.run import prepare
    from stable_baselines3 import PPO

    members = [s for s in ENSEMBLE_SEEDS if _model_exists(eid, s)]
    if len(members) < 2:
        print(f"[eval_{eid}] <2 members trained; skip ensemble")
        return
    rcfg = _od(eid) / "diagnostics" / f"runtime_config_s{members[0]}.json"
    set_config_path(str(rcfg))
    cols, _t, eval_df, _d = prepare(data_config())
    rc = H._raw_clean()["close"].astype(float).reindex(eval_df.index).to_numpy(float)
    assert np.nanmax(np.abs(eval_df["price"].to_numpy(float) - rc)) == 0.0, "ensemble price != raw_close"
    co = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.2}
    models = [PPO.load(str(_model_stub(eid, s)), device="cpu", custom_objects=co) for s in members]
    metrics = evaluate_model(_SoftVote(models), XAUUSDTradingEnv(eval_df, cols, env_config(), random_reset=False))
    out = {"experiment": eid, "kind": "ensemble_soft_vote", "members": members,
           "price_basis": "raw_close", "is_tradeable_performance": True,
           "kalman_env": metrics, "obs_dim": len(cols), "n_eval_bars": int(len(eval_df))}
    dump_json(_od(eid) / "metrics" / "eval_ensemble.json", out)
    print(f"[eval_{eid}] ENSEMBLE n={len(members)} | ret={metrics['cumulative_return']:+.4f} "
          f"sharpe={metrics['sharpe']:.3f} maxDD={metrics['max_drawdown']:.4f} trades={metrics['round_trips']}")


def nautilus_ref(eid):
    """Env<->Nautilus parity for the paper-faithful single reference (seed 23)."""
    if not _model_exists(eid, REF_SEED):
        print(f"[nautilus_{eid}] seed {REF_SEED} missing; skip")
        return
    H.nautilus(EXPS[eid], REF_SEED)


# ------------------------------------------------------- compare / gate ------

def _load(eid, fn):
    p = _od(eid) / "metrics" / fn
    return json.loads(p.read_text())["kalman_env"] if p.exists() else None


def _row(metric, raw, kal, less_is_better=False):
    if raw is None or kal is None:
        return {"metric": metric, "raw_value": None, "kalman_value": None, "delta": None, "winner": "n/a"}
    delta = kal - raw
    better_kal = (kal < raw) if less_is_better else (kal > raw)
    return {"metric": metric, "raw_value": round(raw, 6), "kalman_value": round(kal, 6),
            "delta": round(delta, 6), "winner": "Kalman" if better_kal else ("Raw" if delta != 0 else "tie")}


def _table(raw, kal):
    return [
        _row("cumulative_return", raw["cumulative_return"], kal["cumulative_return"]),
        _row("sharpe", raw["sharpe"], kal["sharpe"]),
        _row("sortino", raw["sortino"], kal["sortino"]),
        _row("max_drawdown", raw["max_drawdown"], kal["max_drawdown"], less_is_better=True),
        _row("trade_win_rate", raw["trade_win_rate"], kal["trade_win_rate"]),
        _row("round_trips", raw["round_trips"], kal["round_trips"]),
        _row("final_equity", raw["final_equity"], kal["final_equity"]),
    ]


def compare():
    out = {"reference_seed": REF_SEED, "ensemble_seeds": ENSEMBLE_SEEDS, "price_basis": "raw_close"}
    single = {"raw": _load("raw", f"eval_s{REF_SEED}.json"), "kalman": _load("kalman", f"eval_s{REF_SEED}.json")}
    ens = {"raw": _load("raw", "eval_ensemble.json"), "kalman": _load("kalman", "eval_ensemble.json")}

    if single["raw"] and single["kalman"]:
        out["single_seed23"] = _table(single["raw"], single["kalman"])
    if ens["raw"] and ens["kalman"]:
        out["ensemble"] = _table(ens["raw"], ens["kalman"])

    # outperformance gate (evaluated on the robust ensemble; single also reported)
    def gate(raw, kal):
        if not raw or not kal:
            return {"available": False}
        return {"available": True,
                "kalman_sharpe_gt_raw": bool(kal["sharpe"] > raw["sharpe"]),
                "kalman_maxdd_less_severe": bool(kal["max_drawdown"] > raw["max_drawdown"]),
                "kalman_sharpe": kal["sharpe"], "raw_sharpe": raw["sharpe"],
                "kalman_maxdd": kal["max_drawdown"], "raw_maxdd": raw["max_drawdown"]}
    out["gate_ensemble"] = gate(ens["raw"], ens["kalman"])
    out["gate_single_seed23"] = gate(single["raw"], single["kalman"])
    g = out["gate_ensemble"]
    out["primary_gate_pass"] = bool(g.get("available") and g["kalman_sharpe_gt_raw"] and g["kalman_maxdd_less_severe"])

    dump_json(FINAL / "outputs" / "comparison" / "final_comparison.json", out)
    # csv (ensemble table preferred, else single)
    tbl = out.get("ensemble") or out.get("single_seed23") or []
    lines = ["metric,raw_value,kalman_value,delta,winner"]
    for r in tbl:
        lines.append(f"{r['metric']},{r['raw_value']},{r['kalman_value']},{r['delta']},{r['winner']}")
    (FINAL / "outputs" / "comparison" / "final_comparison.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare] primary_gate_pass={out['primary_gate_pass']} "
          f"(ensemble Kalman Sharpe {g.get('kalman_sharpe')} vs Raw {g.get('raw_sharpe')})")
    return out


# -------------------------------------------------- seed-dependence root cause

def _collect_diag(eid):
    rows = {}
    md = ABLR / "outputs" / eid / "metrics"
    for f in sorted(md.glob("eval_s*.json")):
        d = json.loads(f.read_text())
        if d.get("price_basis") == "raw_close":
            rows[int(d["seed"])] = d["kalman_env"]
    return rows


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _boot(deltas, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, float)
    means = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means > 0).mean())


def diagnose():
    raw, kal = _collect_diag(DIAG_RAW), _collect_diag(DIAG_KAL)
    seeds = sorted(set(raw) & set(kal))
    csv = ["family,seed,cumulative_return,sharpe,max_drawdown,round_trips,exposure,final_equity"]
    per = {DIAG_RAW: raw, DIAG_KAL: kal}
    for fam, rows in per.items():
        for s in sorted(rows):
            m = rows[s]
            expo = round(1.0 - m.get("flat_frac", 0.0), 4)
            csv.append(f"{fam},{s},{m['cumulative_return']:.6f},{m['sharpe']:.6f},"
                       f"{m['max_drawdown']:.6f},{m['round_trips']},{expo},{m['final_equity']:.2f}")
    (FINAL / "outputs" / "verification" / "seed_dependence_diagnostics.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")

    def summarize(rows):
        rets = [rows[s]["cumulative_return"] for s in sorted(rows)]
        shp = [rows[s]["sharpe"] for s in sorted(rows)]
        dd = [rows[s]["max_drawdown"] for s in sorted(rows)]
        rt = [rows[s]["round_trips"] for s in sorted(rows)]
        expo = [1.0 - rows[s].get("flat_frac", 0.0) for s in sorted(rows)]
        order = sorted(rows, key=lambda s: rows[s]["sharpe"])
        return {"n": len(rows),
                "ret_mean": st.mean(rets), "ret_std": st.pstdev(rets), "ret_min": min(rets), "ret_max": max(rets),
                "sharpe_mean": st.mean(shp), "sharpe_median": st.median(shp), "sharpe_std": st.pstdev(shp),
                "maxdd_mean": st.mean(dd), "neg_seeds": sum(1 for r in rets if r < 0),
                "best_seed": order[-1], "median_seed": order[len(order) // 2], "worst_seed": order[0],
                "corr_trades_sharpe": _corr(rt, shp), "corr_exposure_return": _corr(expo, rets),
                "corr_maxdd_sharpe": _corr(dd, shp)}

    ds = [kal[s]["sharpe"] - raw[s]["sharpe"] for s in seeds]
    dr = [kal[s]["cumulative_return"] - raw[s]["cumulative_return"] for s in seeds]
    slo, shi, sp = _boot(ds)
    rlo, rhi, rp = _boot(dr)

    # ensemble variance-reduction evidence (does the fix work?)
    ens_fix = {}
    for label, eid, fam in (("raw", "raw", DIAG_RAW), ("kalman", "kalman", DIAG_KAL)):
        ep = _od(eid) / "metrics" / "eval_ensemble.json"
        if ep.exists():
            em = json.loads(ep.read_text())["kalman_env"]
            spread = per[fam]
            shp = [spread[s]["sharpe"] for s in sorted(spread)]
            ens_fix[label] = {"ensemble_sharpe": em["sharpe"], "ensemble_return": em["cumulative_return"],
                              "member_sharpe_min": min(shp), "member_sharpe_max": max(shp),
                              "member_sharpe_std": st.pstdev(shp)}

    summ = {"basis": "existing honest 10-seed raw_close artifacts (no retraining)",
            "raw_family": DIAG_RAW, "kalman_family": DIAG_KAL, "paired_seeds": seeds,
            DIAG_RAW: summarize(raw), DIAG_KAL: summarize(kal),
            "paired_delta": {"mean_sharpe_delta": st.mean(ds), "sharpe_delta_CI95": [slo, shi],
                             "P_sharpe_delta_gt0": sp, "P_kalman_beats_raw_per_seed": sum(1 for x in ds if x > 0) / len(ds),
                             "mean_return_delta": st.mean(dr), "return_delta_CI95": [rlo, rhi], "P_return_delta_gt0": rp},
            "ensemble_fix": ens_fix,
            "validation_checkpointing": "DISABLED in study (registry note): final-iteration checkpoint only; "
                                        "no best-validation checkpoint exists to compare (symmetric Option B not run)."}
    dump_json(FINAL / "outputs" / "verification" / "seed_dependence_summary.json", summ)
    print(f"[diagnose] raw Sharpe std={summ[DIAG_RAW]['sharpe_std']:.3f} neg={summ[DIAG_RAW]['neg_seeds']} | "
          f"kalman Sharpe std={summ[DIAG_KAL]['sharpe_std']:.3f} neg={summ[DIAG_KAL]['neg_seeds']} | "
          f"corr(trades,sharpe) raw={summ[DIAG_RAW]['corr_trades_sharpe']} kal={summ[DIAG_KAL]['corr_trades_sharpe']}")
    return summ


def verify():
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    subprocess.run([str(py), str(HERE / "verify_final_model.py")], cwd=str(ROOT),
                   env={**os.environ, "PYTHONUTF8": "1"}, check=False)


def excel():
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    subprocess.run([str(py), str(HERE / "make_final_workbook.py")], cwd=str(ROOT),
                   env={**os.environ, "PYTHONUTF8": "1"}, check=False)


# ------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["setup", "train_raw", "eval_raw", "nautilus_raw", "train_kalman",
                             "eval_kalman", "nautilus_kalman", "compare", "verify", "diagnose",
                             "excel", "all"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-train-if-model-exists", default="true")
    ap.add_argument("--strict-outperformance", default="false")
    a = ap.parse_args()
    skip = str(a.skip_train_if_model_exists).lower() != "false"

    if a.stage in ("setup", "all"):
        setup()
    if a.stage in ("train_raw", "all"):
        train_family("raw", a.device, skip, a.force)
    if a.stage in ("train_kalman", "all"):
        train_family("kalman", a.device, skip, a.force)
    if a.stage in ("eval_raw", "all"):
        eval_family("raw")
    if a.stage in ("eval_kalman", "all"):
        eval_family("kalman")
    if a.stage in ("nautilus_raw", "all"):
        nautilus_ref("raw")
    if a.stage in ("nautilus_kalman", "all"):
        nautilus_ref("kalman")
    if a.stage in ("diagnose", "all"):
        diagnose()
    if a.stage in ("compare", "all"):
        res = compare()
        if str(a.strict_outperformance).lower() == "true" and not res.get("primary_gate_pass"):
            raise SystemExit("STRICT: final outperformance gate FAILED")
    if a.stage in ("verify", "all"):
        verify()
    if a.stage in ("excel", "all"):
        excel()


if __name__ == "__main__":
    main()
