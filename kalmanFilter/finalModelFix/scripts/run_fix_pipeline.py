"""Forensic fix pipeline: reproducibility audit + symmetric ensemble-aggregation fix.

Reuses everything frozen (harness train/eval, evaluate_model, finalModel models). Adds:
  - reproducibility: train the SAME seed twice (identical config/device) and compare ->
    proves whether 'seed-dependence' is CPU nondeterminism or intrinsic seed variance.
  - aggregation: the 5-seed soft-vote made the Kalman ensemble trade only 4x (killed return).
    Test SYMMETRIC aggregation rules on the EXISTING finalModel members (no retraining),
    select the rule by a predeclared VALIDATION metric (train-tail, never test), report test.

Stages: reproducibility aggregation compare excel all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIX = HERE.parent
KF = FIX.parent
ROOT = KF.parent
sys.path.insert(0, str(KF / "finalModel" / "scripts"))
import run_final_model as R                 # harness (H), EXPS, ENSEMBLE_SEEDS, _SoftVote, dump_json
H = R.H
dump_json = R.dump_json

FINAL = KF / "finalModel"
ABLR = KF / "ablation_rawprice"
POS = {0: -1, 1: 0, 2: 1}                   # action index -> position
VAL_BARS = 252                              # train-tail validation window (never test)
SEEDS = R.ENSEMBLE_SEEDS


# ----------------------------------------------------- helpers (models/probs) -

def _load(stub):
    from stable_baselines3 import PPO
    co = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.2}
    return PPO.load(str(stub), device="cpu", custom_objects=co)


def _param_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.policy.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def _probs(model, obs):
    import torch
    obs_t, _ = model.policy.obs_to_tensor(np.asarray(obs))
    with torch.no_grad():
        return model.policy.get_distribution(obs_t).distribution.probs.cpu().numpy().reshape(-1)


# ----------------------------------------------- reproducibility (determinism) -

def reproducibility(device="cpu"):
    """Train kalman & raw seed-23 twice (smoke) under identical config; compare. Also hash the
    existing finalModel vs ablation 500k seed-23 models (same seed, same config)."""
    out = {"device": device, "smoke_twice": {}, "existing_500k_same_seed": {}}
    H._ABLR = FIX                           # train repro models under finalModelFix
    rows = [["family", "run", "param_hash", "first10_actions", "ret", "sharpe", "maxdd"]]

    for fam, src in (("kalman", R.EXPS["kalman"]["source"]), ("raw", R.EXPS["raw"]["source"])):
        hashes, metr = [], []
        for run in ("A", "B"):
            eid = f"repro_{fam}_{run}"
            exp = {"id": eid, "source": src, "transform": "none",
                   "covariance": "full" if fam == "kalman" else "bypass"}
            if not (FIX / "outputs" / eid / "data" / "model_input.csv").exists():
                H.prepare_dataset(exp)
            H.train(exp, 23, device, smoke=True)
            m = _load(FIX / "outputs" / eid / "models" / f"ppo_{eid}_s23")
            ph = _param_hash(m)
            cols, _t, eval_df, _d = _prep(FIX / "outputs" / eid / "diagnostics" / "runtime_config_s23.json")
            acts, met = _eval_actions(m, eval_df, cols, n=10)
            hashes.append(ph); metr.append(met)
            rows.append([fam, run, ph, acts, round(met["cumulative_return"], 6),
                         round(met["sharpe"], 4), round(met["max_drawdown"], 4)])
        out["smoke_twice"][fam] = {
            "param_hash_A": hashes[0], "param_hash_B": hashes[1],
            "identical_weights": hashes[0] == hashes[1],
            "ret_A": metr[0]["cumulative_return"], "ret_B": metr[1]["cumulative_return"],
            "deterministic": hashes[0] == hashes[1]}

    # existing 500k seed-23: finalModel vs ablation (same seed, same substance)
    for fam, fm_stub, ab_stub in (
        ("kalman", FINAL / "outputs/kalman/models/ppo_kalman_s23", ABLR / "outputs/exp_002_corrected/models/ppo_exp_002_corrected_s23"),
        ("raw", FINAL / "outputs/raw/models/ppo_raw_s23", ABLR / "outputs/rawctl_corrected/models/ppo_rawctl_corrected_s23")):
        try:
            hf, ha = _param_hash(_load(fm_stub)), _param_hash(_load(ab_stub))
            out["existing_500k_same_seed"][fam] = {"finalModel_hash": hf, "ablation_hash": ha, "identical": hf == ha}
        except Exception as e:
            out["existing_500k_same_seed"][fam] = {"error": str(e)}

    det = all(v["deterministic"] for v in out["smoke_twice"].values())
    out["verdict"] = ("DETERMINISTIC: same seed+config reproduces identical weights; cross-run "
                      "differences are config/path drift, not RNG." if det else
                      "NONDETERMINISTIC: identical seed+config yields different weights on CPU "
                      "(multithreaded float reductions). Single-seed results are not run-to-run "
                      "reproducible -> this is the mechanism behind 'seed-dependence'.")
    dump_json(FIX / "outputs/reproducibility/reproducibility_audit.json", out)
    (FIX / "outputs/reproducibility/tiny_same_seed_comparison.csv").write_text(
        "\n".join(",".join(map(str, r)) for r in rows) + "\n", encoding="utf-8")
    print("[reproducibility]", out["verdict"])
    for fam, v in out["smoke_twice"].items():
        print(f"  {fam}: identical_weights={v['identical_weights']} hashA={v['param_hash_A']} hashB={v['param_hash_B']}")
    for fam, v in out["existing_500k_same_seed"].items():
        print(f"  500k {fam}: finalModel==ablation? {v.get('identical')}")
    return out


def _prep(rcfg):
    H.pipeline.ensure_src_on_path()
    from rl_gold_trading.config import data_config, set_config_path
    from rl_gold_trading.run import prepare
    set_config_path(str(rcfg))
    return prepare(data_config())


def _eval_actions(model_like, df, cols, n=10):
    from rl_gold_trading.config import env_config
    from rl_gold_trading.envs import XAUUSDTradingEnv
    from rl_gold_trading.metrics import evaluate_model
    env = XAUUSDTradingEnv(df, cols, env_config(), random_reset=False)
    # first n actions
    obs, _ = env.reset()
    acts = []
    for _ in range(n):
        a, _ = model_like.predict(obs, deterministic=True)
        acts.append(int(a))
        obs, _r, term, trunc, _i = env.step(int(a))
        if term or trunc:
            break
    env2 = XAUUSDTradingEnv(df, cols, env_config(), random_reset=False)
    met = evaluate_model(model_like, env2)
    return "".join(map(str, acts)), met


# ------------------------------------------- ensemble aggregation (the fix) ---

class _Agg:
    def __init__(self, models, rule, th=0.0):
        self.models, self.rule, self.th = models, rule, th

    def predict(self, obs, deterministic=True):
        if self.rule == "avg_prob_argmax":
            p = sum(_probs(m, obs) for m in self.models)
            return int(np.argmax(p)), None
        idxs = [int(m.predict(obs, deterministic=True)[0]) for m in self.models]
        if self.rule == "majority_vote":
            counts = [idxs.count(0), idxs.count(1), idxs.count(2)]
            mx = max(counts)
            winners = [i for i, c in enumerate(counts) if c == mx]
            return (winners[0] if len(winners) == 1 else 1), None  # tie -> hold
        # avg_signed_threshold
        mpos = float(np.mean([POS[i] for i in idxs]))
        idx = 2 if mpos > self.th else (0 if mpos < -self.th else 1)
        return idx, None


RULES = [("avg_prob_argmax", 0.0), ("majority_vote", 0.0),
         ("avg_signed_thr0", 0.0), ("avg_signed_thr0.2", 0.2)]


def _agg_metrics(models, df, cols, rule, th):
    from rl_gold_trading.config import env_config
    from rl_gold_trading.envs import XAUUSDTradingEnv
    from rl_gold_trading.metrics import evaluate_model
    return evaluate_model(_Agg(models, rule, th), XAUUSDTradingEnv(df, cols, env_config(), random_reset=False))


def aggregation():
    """Symmetric aggregation-rule selection on VALIDATION (train tail), reported on TEST."""
    results = {"rules": [r[0] for r in RULES], "val_window_bars": VAL_BARS, "families": {}}
    members = {}
    for fam in ("raw", "kalman"):
        rcfg = FINAL / "outputs" / fam / "diagnostics" / f"runtime_config_s{SEEDS[0]}.json"
        cols, train_df, eval_df, _d = _prep(rcfg)
        val_df = train_df.iloc[-VAL_BARS:]
        ms = [_load(FINAL / "outputs" / fam / "models" / f"ppo_{fam}_s{s}") for s in SEEDS]
        members[fam] = (ms, cols, val_df, eval_df)

    # validation score per rule, averaged across families (predeclared, symmetric)
    val_score = {}
    per_rule = {fam: {} for fam in ("raw", "kalman")}
    for name, th in RULES:
        scores = []
        for fam in ("raw", "kalman"):
            ms, cols, val_df, eval_df = members[fam]
            vm = _agg_metrics(ms, val_df, cols, name, th)
            tm = _agg_metrics(ms, eval_df, cols, name, th)
            per_rule[fam][name] = {"validation": vm, "test": tm}
            scores.append(vm["sharpe"] - 0.25 * abs(vm["max_drawdown"]))
        val_score[name] = float(np.mean(scores))
    chosen = max(val_score, key=val_score.get)
    results["validation_score_by_rule"] = val_score
    results["chosen_rule"] = chosen
    results["selection_metric"] = "mean over {raw,kalman} of (val_sharpe - 0.25*|val_maxdd|), train-tail validation"
    results["families"] = per_rule
    dump_json(FIX / "outputs/fixed/aggregation_rules.json", results)
    print(f"[aggregation] chosen rule (by validation, symmetric): {chosen}")
    for fam in ("raw", "kalman"):
        t = per_rule[fam][chosen]["test"]
        print(f"  {fam} TEST @ {chosen}: ret={t['cumulative_return']:+.4f} sharpe={t['sharpe']:.3f} "
              f"maxDD={t['max_drawdown']:.4f} trades={t['round_trips']}")
    return results


# ----------------------------------------------------------- compare + excel --

def compare():
    agg = json.loads((FIX / "outputs/fixed/aggregation_rules.json").read_text())
    rule = agg["chosen_rule"]
    raw = agg["families"]["raw"][rule]["test"]
    kal = agg["families"]["kalman"][rule]["test"]
    # also keep old soft-vote for reference
    raw_sv = agg["families"]["raw"]["avg_prob_argmax"]["test"]
    kal_sv = agg["families"]["kalman"]["avg_prob_argmax"]["test"]

    def tbl(r, k):
        keys = [("cumulative_return", False), ("sharpe", False), ("sortino", False),
                ("calmar", False), ("max_drawdown", True), ("trade_win_rate", False),
                ("round_trips", False), ("final_equity", False)]
        rows = []
        for m, less in keys:
            rv, kv = r[m], k[m]
            win = "Kalman" if ((kv < rv) if less else (kv > rv)) else ("Raw" if rv != kv else "tie")
            rows.append({"metric": m, "raw": round(rv, 6), "kalman": round(kv, 6),
                         "delta": round(kv - rv, 6), "winner": win})
        return rows

    out = {"chosen_rule": rule, "price_basis": "raw_close", "is_tradeable_performance": True,
           "fixed_ensemble": tbl(raw, kal),
           "old_softvote_ensemble": tbl(raw_sv, kal_sv),
           "gate": {"kalman_sharpe_gt_raw": bool(kal["sharpe"] > raw["sharpe"]),
                    "kalman_maxdd_less_severe": bool(kal["max_drawdown"] > raw["max_drawdown"]),
                    "kalman_return_ge_raw": bool(kal["cumulative_return"] >= raw["cumulative_return"]),
                    "kalman_sharpe": kal["sharpe"], "raw_sharpe": raw["sharpe"],
                    "kalman_return": kal["cumulative_return"], "raw_return": raw["cumulative_return"]}}
    dump_json(FIX / "outputs/fixed/fixed_comparison.json", out)
    lines = ["metric,raw,kalman,delta,winner"]
    for r in out["fixed_ensemble"]:
        lines.append(f"{r['metric']},{r['raw']},{r['kalman']},{r['delta']},{r['winner']}")
    (FIX / "outputs/fixed/fixed_comparison.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    g = out["gate"]
    print(f"[compare] rule={rule} | Kalman Sharpe {g['kalman_sharpe']:.3f} vs Raw {g['raw_sharpe']:.3f} "
          f"| Kalman ret {g['kalman_return']:+.4f} vs Raw {g['raw_return']:+.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["reproducibility", "aggregation", "compare", "excel", "all"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.stage in ("reproducibility", "all"):
        reproducibility(a.device)
    if a.stage in ("aggregation", "all"):
        aggregation()
    if a.stage in ("compare", "all"):
        compare()
    if a.stage in ("excel", "all"):
        import subprocess, os
        subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), str(HERE / "make_fix_workbook.py")],
                       cwd=str(ROOT), env={**os.environ, "PYTHONUTF8": "1"}, check=False)


if __name__ == "__main__":
    main()
