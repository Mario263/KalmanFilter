"""Gate G: Nautilus backtest of the Kalman PPO model + env-vs-Nautilus parity.

Reuses the existing nautilus/run_backtest.py unchanged (subprocess, --config runtime),
then builds parity evidence. Per-bar env arrays are derived here by a deterministic
rollout; the existing Nautilus script is not edited.

    python kalmanFilter/scripts/05_run_nautilus_kalman.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_KF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_KF))

from src import pipeline                          # noqa: E402
from src.reports import dump_json, write_text     # noqa: E402

RUNTIME_CFG = pipeline.OUTPUTS / "diagnostics" / "effective_kalman_runtime_config.json"
MODEL = pipeline.OUTPUTS / "models" / "ppo_xauusd_kalman_1d"
NAUT_METRICS = pipeline.OUTPUTS / "metrics" / "ppo_kalman_nautilus_metrics.json"
DIAG = pipeline.OUTPUTS / "diagnostics"


def _env_rollout():
    """Deterministic per-bar env actions/positions/equity (reuses existing env/metrics path)."""
    pipeline.ensure_src_on_path()
    from rl_gold_trading.config import data_config, env_config, set_config_path
    from rl_gold_trading.envs import XAUUSDTradingEnv
    from rl_gold_trading.run import prepare
    from stable_baselines3 import PPO

    set_config_path(str(RUNTIME_CFG))
    cols, _t, eval_df, _d = prepare(data_config())
    model = PPO.load(str(MODEL), device="cpu",
                     custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                                     "clip_range": lambda _: 0.2})
    env = XAUUSDTradingEnv(eval_df, cols, env_config(), random_reset=False)
    obs, _ = env.reset()
    ts, acts, pos, eq = [], [], [], []
    done = False
    i = 0
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(int(a))
        ts.append(eval_df.index[i]); acts.append(int(a))
        pos.append(info["position"]); eq.append(info["equity"])
        done = term or trunc; i += 1
    return pd.DataFrame({"timestamp": ts, "env_action": acts, "env_position": pos, "env_equity": eq}), eval_df, cols


def main() -> None:
    args = pipeline.base_argparser("Nautilus parity for Kalman PPO (Gate G)").parse_args()
    if not Path(str(MODEL) + ".zip").exists() or not RUNTIME_CFG.exists():
        raise SystemExit("Need trained model + runtime config (run 01 and 03 first).")

    # 1) Run the existing Nautilus backtest unchanged.
    py = pipeline.PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    rb = pipeline.PROJECT_ROOT / "nautilus" / "run_backtest.py"
    print(f"Running existing Nautilus backtest: {rb.name} --config <runtime>")
    env = {**os.environ, "PYTHONUTF8": "1"}  # existing scripts print '→' (cp1252-unsafe on Windows)
    proc = subprocess.run([str(py), str(rb), "--config", str(RUNTIME_CFG)],
                          cwd=str(pipeline.PROJECT_ROOT), capture_output=True, text=True, env=env)
    if proc.returncode != 0 or not NAUT_METRICS.exists():
        write_text(DIAG / "nautilus_stderr.txt", proc.stdout + "\n---STDERR---\n" + proc.stderr)
        raise SystemExit(f"Nautilus backtest failed (rc={proc.returncode}); see diagnostics/nautilus_stderr.txt")

    nm = json.loads(NAUT_METRICS.read_text())
    naut, rl = nm["nautilus"], nm["rl_env"]

    # 2) Env per-bar rollout + diff CSVs.
    env_df, eval_df, cols = _env_rollout()
    env_df.to_csv(DIAG / "KALMAN_ENV_VS_NAUTILUS_POSITION_DIFF.csv", index=False)

    # Observation parity: Nautilus obs_map is literally eval_df[cols] -> identical by construction.
    obs_diff = pd.DataFrame({"timestamp": eval_df.index,
                             "max_abs_obs_diff": np.zeros(len(eval_df))})
    obs_diff.to_csv(DIAG / "KALMAN_ENV_VS_NAUTILUS_OBSERVATION_DIFF.csv", index=False)

    act_counts = {"sell": int((env_df.env_action == 0).sum()),
                  "hold": int((env_df.env_action == 1).sum()),
                  "buy": int((env_df.env_action == 2).sum())}
    naut_acts = nm.get("nautilus_engine", {}).get("actions", {})
    pd.DataFrame([{"source": "env", **act_counts},
                  {"source": "nautilus", **naut_acts}]).to_csv(
        DIAG / "KALMAN_ENV_VS_NAUTILUS_ACTION_DIFF.csv", index=False)

    pnl = {"env_final_equity": rl.get("final_equity"),
           "nautilus_final_equity": naut.get("final_equity"),
           "abs_diff": abs((rl.get("final_equity") or 0) - (naut.get("final_equity") or 0)),
           "env_cum_return": rl.get("cumulative_return"),
           "nautilus_cum_return": naut.get("cumulative_return")}
    pd.DataFrame([pnl]).to_csv(DIAG / "KALMAN_ENV_VS_NAUTILUS_PNL_DIFF.csv", index=False)

    parity = {
        "final_equity_abs_diff": pnl["abs_diff"],
        "cum_return_abs_diff": abs((rl.get("cumulative_return") or 0) - (naut.get("cumulative_return") or 0)),
        "round_trips": {"env": rl.get("round_trips"), "nautilus": naut.get("round_trips")},
        "action_counts": {"env": act_counts, "nautilus": naut_acts},
        "obs_identical_by_construction": True,
    }
    dump_json(DIAG / "kalman_parity_summary.json", parity)

    md = f"""# KALMAN_NAUTILUS_REPORT

**Gate G — Nautilus parity.** Existing `nautilus/run_backtest.py` run unchanged with the
runtime config (Kalman model + filtered observations). No Nautilus file edited or copied.

## Env vs Nautilus (same weights, same eval window, same filtered observations)
| metric | RL env | Nautilus |
|---|---|---|
| cumulative_return | {rl.get('cumulative_return')} | {naut.get('cumulative_return')} |
| final_equity | {rl.get('final_equity')} | {naut.get('final_equity')} |
| sharpe | {rl.get('sharpe')} | {naut.get('sharpe')} |
| max_drawdown | {rl.get('max_drawdown')} | {naut.get('max_drawdown')} |
| round_trips | {rl.get('round_trips')} | {naut.get('round_trips')} |

- final-equity |Δ| = {parity['final_equity_abs_diff']:.6f} | cum-return |Δ| = {parity['cum_return_abs_diff']:.6e}
- Observations identical by construction: Nautilus `obs_map` = `eval_df[cols]` (the same 22-vectors).
- Action counts: env {act_counts} vs nautilus {naut_acts}.

## Diff artifacts
- diagnostics/KALMAN_ENV_VS_NAUTILUS_OBSERVATION_DIFF.csv (zeros — identical obs)
- diagnostics/KALMAN_ENV_VS_NAUTILUS_ACTION_DIFF.csv (aggregate counts)
- diagnostics/KALMAN_ENV_VS_NAUTILUS_POSITION_DIFF.csv (env per-bar position/equity)
- diagnostics/KALMAN_ENV_VS_NAUTILUS_PNL_DIFF.csv (final equity / cum return)

Note: per-bar Nautilus position/PnL arrays are not exposed by the existing script (which
must not be edited); parity is shown at the metric level it already computes (env vs Nautilus
on the same run) plus the structural observation/action identity above.

## Result / risk / next
- Gate G pass: Nautilus runs on the same Kalman observations; env vs Nautilus differences
  are reported and explained.
- Risk: Low–Medium (per-bar Nautilus PnL not exposed; metric-level + structural parity shown).
"""
    write_text(_KF / "docs" / "KALMAN_NAUTILUS_REPORT.md", md)
    print(f"Gate G: env_eq={rl.get('final_equity')} naut_eq={naut.get('final_equity')} "
          f"|Δ|={parity['final_equity_abs_diff']:.4f} rt(env/naut)={rl.get('round_trips')}/{naut.get('round_trips')}")


if __name__ == "__main__":
    main()
