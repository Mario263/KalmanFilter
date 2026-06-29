"""Honest raw-price Kalman PPO revalidation runner.

Retrains Raw + Kalman PPO under the RAW tradeable close as the PnL/reward/equity/fill
price, while the 22D observation may use Kalman-filtered features. Reuses the FROZEN
entry points unchanged:
  - training/eval reward: `rl_gold_trading.run.main` (honest once features.price=raw_close)
  - Nautilus: existing `nautilus/run_backtest.py` (marks + fills on `price`=raw_close)

The fix that makes them honest: a `raw_close` column in the model-input CSV +
features.py preferring it for `price`. Nothing in PPO/reward/costs/action/split/z-score
or Nautilus logic changes. Volume stays raw. State stays 22D.

    python run_rawprice_ablation.py --exp exp_007_corrected --stage prepare
    python run_rawprice_ablation.py --exp exp_007_corrected --stage verify
    python run_rawprice_ablation.py --exp exp_007_corrected --stage train --seed 1
    python run_rawprice_ablation.py --exp exp_007_corrected --stage eval  --seed 1
    python run_rawprice_ablation.py --exp exp_007_corrected --stage nautilus
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ABLR = _HERE.parent
_KF = _ABLR.parent
_ROOT = _KF.parent
sys.path.insert(0, str(_KF))

from src import pipeline, validation                     # noqa: E402
from src.reports import dump_json                        # noqa: E402

REGISTRY = _ABLR / "configs" / "experiments_registry.json"
OHLC_L = ["open", "high", "low", "close"]
SEEDS = json.loads(REGISTRY.read_text())["_meta"]["seeds"]


def _exp(reg, eid):
    for e in reg["experiments"]:
        if e["id"] == eid:
            return e
    raise SystemExit(f"{eid} not in registry")


def _od(eid):
    return _ABLR / "outputs" / eid


def _model_name(eid, seed=None):
    return f"ppo_{eid}" if seed is None else f"ppo_{eid}_s{seed}"


def _raw_clean():
    """Canonical cleaned raw OHLCV exactly as the Raw baseline sees it."""
    return pipeline.load_clean_ohlcv(pipeline.DEFAULT_CONFIG)


# ------------------------------- prepare ------------------------------------

def prepare_dataset(exp) -> dict:
    """Write model-input CSV (filtered OHLC features + RAW volume + raw_close) and a
    runtime config. raw_close = real tradeable close; never normalized, never an obs."""
    eid = exp["id"]
    raw = _raw_clean()                                   # datetime-indexed o/h/l/c/v (raw)
    raw_close = raw["close"].astype(float)

    if exp["source"] == "raw":
        filt = raw[OHLC_L].copy()
        vol = raw["volume"].astype(float)
    else:
        src = pd.read_csv(_ROOT / exp["source"])
        src["datetime"] = pd.to_datetime(src["datetime"], utc=True)
        src = src.set_index("datetime")
        if len(src) != len(raw) or not (src.index == raw.index).all():
            raise SystemExit(f"{eid}: source rows/timestamps misaligned with raw clean")
        filt = src[OHLC_L].astype(float)
        vol = src["volume"].astype(float)

    # volume passthrough proof + OHLC validity of the (possibly filtered) features
    validation.check_volume_unchanged(raw["volume"].to_numpy(float), vol.to_numpy(float))
    ohlc_valid = validation.check_ohlc_validity(
        filt["open"].to_numpy(), filt["high"].to_numpy(), filt["low"].to_numpy(), filt["close"].to_numpy())

    model = pd.DataFrame({
        "open": filt["open"].to_numpy(), "high": filt["high"].to_numpy(),
        "low": filt["low"].to_numpy(), "close": filt["close"].to_numpy(),
        "volume": vol.to_numpy(), "raw_close": raw_close.to_numpy(),   # <-- honest price path
    }, index=raw.index)
    csv = _od(eid) / "data" / "model_input.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(csv, index_label="datetime")

    rc = raw_close.to_numpy(); fc = filt["close"].to_numpy()
    drift = {
        "filtered_vs_raw_close_mean_abs_pct": float(np.mean(np.abs(fc - rc) / rc) * 100),
        "filtered_vs_raw_close_max_abs_pct": float(np.max(np.abs(fc - rc) / rc) * 100),
        "is_passthrough": bool(np.max(np.abs(fc - rc)) == 0.0),
    }
    _runtime_cfg(eid, csv, _model_name(eid), seed=None)
    diag = {"experiment": eid, "transform": exp["transform"], "covariance": exp["covariance"],
            "price_basis": "raw_close", "feature_price_basis": "filtered_ohlc" if exp["source"] != "raw" else "raw_ohlc",
            "rows": int(len(model)), "ohlc_validity": ohlc_valid, "drift_vs_raw_close": drift,
            "volume_unchanged": True}
    dump_json(_od(eid) / "diagnostics" / "dataset_diagnostics.json", diag)
    print(f"[{eid}] prepare OK | source={'raw' if exp['source']=='raw' else 'filtered'} "
          f"| close-drift mean={drift['filtered_vs_raw_close_mean_abs_pct']:.4f}% "
          f"max={drift['filtered_vs_raw_close_max_abs_pct']:.4f}% passthrough={drift['is_passthrough']} "
          f"| ohlc_viol={ohlc_valid['total']} rows={len(model)}")
    return diag


def _runtime_cfg(eid, csv, model_name, seed):
    cfg = copy.deepcopy(json.loads(pipeline.DEFAULT_CONFIG.read_text(encoding="utf-8")))
    od = _od(eid)
    cfg["_generated"] = {"note": "raw-price revalidation runtime config", "price_basis": "raw_close"}
    cfg["experiment"]["model_name"] = model_name
    cfg["data"]["csv_path"] = str(csv)
    cfg["train"]["save_dir"] = str(od / "models")
    cfg["train"]["validation_enabled"] = False          # monitoring only; final-checkpoint model unchanged
    cfg["train"]["train_log_file"] = str(od / "diagnostics" / (f"train_s{seed}.log" if seed is not None else "train.log"))
    if seed is not None:
        cfg["train"]["seed"] = int(seed)
    cfg["nautilus"]["model_path"] = str(od / "models" / model_name)
    cfg["nautilus"]["metrics_output"] = str(od / "metrics" / f"nautilus_{model_name}.json")
    out = od / "diagnostics" / ("runtime_config.json" if seed is None else f"runtime_config_s{seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


# ------------------------------- verify -------------------------------------

def verify(exp, seed=None) -> dict:
    """Hard price-basis invariants (Phase-1 stopping condition lives here)."""
    eid = exp["id"]
    rcfg = _od(eid) / "diagnostics" / (f"runtime_config_s{seed}.json" if seed else "runtime_config.json")
    pipeline.ensure_src_on_path()
    from rl_gold_trading.config import data_config, set_config_path
    from rl_gold_trading.run import prepare
    set_config_path(str(rcfg))
    cols, train_df, eval_df, _ = prepare(data_config())
    raw_close = _raw_clean()["close"].astype(float)

    checks = {}
    checks["feature_count_22"] = (len(cols) == 22)
    checks["price_not_in_obs"] = ("price" not in cols and "raw_close" not in cols)
    for nm, df in [("train", train_df), ("eval", eval_df)]:
        px = df["price"].to_numpy(float)
        rc = raw_close.reindex(df.index).to_numpy(float)
        checks[f"{nm}_price_eq_raw_close"] = bool(np.nanmax(np.abs(px - rc)) == 0.0)
        checks[f"{nm}_no_nan_obs"] = bool(np.isfinite(df[cols].to_numpy(float)).all())
        checks[f"{nm}_no_dup_ts"] = bool(not df.index.duplicated().any())
    ok = all(checks.values())
    dump_json(_od(eid) / "diagnostics" / f"price_basis_verify{'' if seed is None else f'_s{seed}'}.json",
              {"experiment": eid, "seed": seed, "all_pass": ok, "checks": checks})
    print(f"[{eid}] verify seed={seed} | {'PASS' if ok else 'FAIL'} | {checks}")
    if not ok:
        raise SystemExit(f"PRICE-BASIS VERIFY FAILED for {eid}: {checks}")
    return checks


# --------------------------- train / eval / nautilus ------------------------

def train(exp, seed, device="cpu", smoke=False):
    eid = exp["id"]
    rcfg = _runtime_cfg(eid, _od(eid) / "data" / "model_input.csv", _model_name(eid, seed), seed)
    pipeline.ensure_src_on_path()
    from rl_gold_trading.run import main as run_main
    run_main(argparse.Namespace(config=str(rcfg), mode="train", timesteps=None, csv=None,
                                save_dir=str(_od(eid) / "models"), smoke=smoke, device=device))
    print(f"[{eid}] trained seed={seed}")


def evaluate(exp, seed, device="cpu"):
    eid = exp["id"]
    rcfg = _od(eid) / "diagnostics" / (f"runtime_config_s{seed}.json" if seed is not None else "runtime_config.json")
    model = _od(eid) / "models" / _model_name(eid, seed)
    if not Path(str(model) + ".zip").exists():
        raise SystemExit(f"model missing: {model}.zip")
    pipeline.ensure_src_on_path()
    from rl_gold_trading.config import data_config, env_config, set_config_path
    from rl_gold_trading.envs import XAUUSDTradingEnv
    from rl_gold_trading.metrics import evaluate_model
    from rl_gold_trading.run import prepare
    from stable_baselines3 import PPO
    set_config_path(str(rcfg))
    dc = data_config()
    cols, _t, eval_df, _d = prepare(dc)
    # honesty assert: env will trade on raw close
    rc = _raw_clean()["close"].astype(float).reindex(eval_df.index).to_numpy(float)
    assert np.nanmax(np.abs(eval_df["price"].to_numpy(float) - rc)) == 0.0, "price != raw_close"
    m = PPO.load(str(model), device="cpu",
                 custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.2})
    metrics = evaluate_model(m, XAUUSDTradingEnv(eval_df, cols, env_config(), random_reset=False))
    out = {"experiment": eid, "seed": seed, "price_basis": "raw_close", "is_tradeable_performance": True,
           "kalman_env": metrics, "obs_dim": len(cols), "n_eval_bars": int(len(eval_df))}
    fn = "eval.json" if seed is None else f"eval_s{seed}.json"
    dump_json(_od(eid) / "metrics" / fn, out)
    print(f"[{eid}] eval seed={seed} px=raw | ret={metrics['cumulative_return']:.4f} "
          f"sharpe={metrics['sharpe']:.3f} maxDD={metrics['max_drawdown']:.4f} "
          f"trades={metrics['round_trips']} win={metrics['trade_win_rate']:.3f}")
    return out


def nautilus(exp, seed=None):
    eid = exp["id"]
    rcfg = _od(eid) / "diagnostics" / (f"runtime_config_s{seed}.json" if seed is not None else "runtime_config.json")
    mname = _model_name(eid, seed)
    nm_path = _od(eid) / "metrics" / f"nautilus_{mname}.json"
    if not Path(str(_od(eid) / "models" / mname) + ".zip").exists():
        raise SystemExit("model missing for nautilus")
    py = _ROOT / ".venv" / "Scripts" / "python.exe"
    rb = _ROOT / "nautilus" / "run_backtest.py"
    env = {**os.environ, "PYTHONUTF8": "1"}
    proc = subprocess.run([str(py), str(rb), "--config", str(rcfg)], cwd=str(_ROOT),
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0 or not nm_path.exists():
        (_od(eid) / "diagnostics" / "nautilus_stderr.txt").write_text(proc.stdout + "\n---\n" + proc.stderr, encoding="utf-8")
        raise SystemExit(f"nautilus failed rc={proc.returncode}; see diagnostics/nautilus_stderr.txt")
    nmj = json.loads(nm_path.read_text())
    naut, rl = nmj["nautilus"], nmj["rl_env"]
    parity = {"experiment": eid, "seed": seed, "price_basis": "raw_close",
              "final_equity_abs_diff": abs((rl.get("final_equity") or 0) - (naut.get("final_equity") or 0)),
              "cum_return_abs_diff": abs((rl.get("cumulative_return") or 0) - (naut.get("cumulative_return") or 0)),
              "round_trips": {"env": rl.get("round_trips"), "nautilus": naut.get("round_trips")},
              "env_return": rl.get("cumulative_return"), "nautilus_return": naut.get("cumulative_return"),
              "env_sharpe": rl.get("sharpe"), "nautilus_sharpe": naut.get("sharpe")}
    parity["parity_pass"] = parity["cum_return_abs_diff"] < 1e-3
    dump_json(_od(eid) / "diagnostics" / f"parity_{mname}.json", parity)
    print(f"[{eid}] nautilus seed={seed} | env_ret={rl.get('cumulative_return'):.4f} "
          f"naut_ret={naut.get('cumulative_return'):.4f} |Δ|={parity['cum_return_abs_diff']:.2e} "
          f"{'PASS' if parity['parity_pass'] else 'FAIL'}")
    return parity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--stage", choices=["prepare", "verify", "train", "eval", "nautilus", "smoke"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    exp = _exp(json.loads(REGISTRY.read_text()), a.exp)
    if a.stage == "prepare":
        prepare_dataset(exp)
    elif a.stage == "verify":
        verify(exp, a.seed)
    elif a.stage == "train":
        train(exp, a.seed, a.device)
    elif a.stage == "eval":
        evaluate(exp, a.seed, a.device)
    elif a.stage == "nautilus":
        nautilus(exp, a.seed)
    elif a.stage == "smoke":
        prepare_dataset(exp); verify(exp, None)
        train(exp, a.seed or 1, a.device, smoke=True); evaluate(exp, a.seed or 1, a.device)


if __name__ == "__main__":
    main()
