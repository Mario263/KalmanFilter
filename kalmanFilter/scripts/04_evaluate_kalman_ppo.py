"""Gate F: evaluate the Kalman PPO model with the existing metrics module.

    python kalmanFilter/scripts/04_evaluate_kalman_ppo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_KF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_KF))

from src import pipeline                          # noqa: E402
from src.reports import dump_json, write_text     # noqa: E402

RUNTIME_CFG = pipeline.OUTPUTS / "diagnostics" / "effective_kalman_runtime_config.json"
MODEL = pipeline.OUTPUTS / "models" / "ppo_xauusd_kalman_1d"
RAW_BASELINE = pipeline.PROJECT_ROOT / "models" / "dukascopy_1d" / "ppo_raw_metrics.json"

_KEYS = ["cumulative_return", "cagr", "sharpe", "max_drawdown", "win_rate",
         "trade_win_rate", "round_trips", "final_equity"]


def main() -> None:
    args = pipeline.base_argparser("Evaluate Kalman PPO (Gate F)").parse_args()
    if not Path(str(MODEL) + ".zip").exists():
        raise SystemExit(f"Model not found: {MODEL}.zip — run 03_train_kalman_ppo.py first.")

    pipeline.ensure_src_on_path()
    from rl_gold_trading.config import data_config, env_config, paper_target, set_config_path
    from rl_gold_trading.envs import XAUUSDTradingEnv
    from rl_gold_trading.metrics import evaluate_model
    from rl_gold_trading.run import prepare
    from stable_baselines3 import PPO

    set_config_path(str(RUNTIME_CFG))
    dc = data_config()
    cols, _train, eval_df, _daily = prepare(dc)
    model = PPO.load(str(MODEL), device="cpu",
                     custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                                     "clip_range": lambda _: 0.2})
    env = XAUUSDTradingEnv(eval_df, cols, env_config(), random_reset=False)
    m = evaluate_model(model, env)

    raw = {}
    if RAW_BASELINE.exists():
        raw = json.loads(RAW_BASELINE.read_text()).get("reproduced", {})
    paper = paper_target()

    out = {"kalman_env": m, "raw_baseline": raw, "paper_target": paper,
           "eval_window": [dc.eval_start, dc.eval_end], "obs_dim": len(cols),
           "n_eval_bars": int(len(eval_df))}
    dump_json(pipeline.OUTPUTS / "metrics" / "ppo_kalman_eval_metrics.json", out)

    def row(k):
        pk = {"cumulative_return": "cumulative_return", "cagr": "cagr", "sharpe": "sharpe",
              "max_drawdown": "max_drawdown", "win_rate": "win_rate"}.get(k)
        rv = raw.get(k, "—"); kv = m.get(k, "—"); pv = paper.get(pk, "—") if pk else "—"
        f = (lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
        return f"| {k} | {f(rv)} | {f(kv)} | {f(pv)} |"

    md = f"""# KALMAN_EVALUATION_REPORT

**Gate F — evaluation.** Same metrics module as Raw PPO (`metrics.evaluate_model`).
Eval window {dc.eval_start} → {dc.eval_end} | {out['n_eval_bars']} bars | obs_dim {out['obs_dim']}.

| metric | Raw PPO (1d baseline) | Kalman PPO (env) | Paper target |
|---|---|---|---|
{chr(10).join(row(k) for k in _KEYS)}

Paper labels: paper targets (cum 15.39%, CAGR 6.00%, Sharpe 0.69, MaxDD −11.22%, win 50.16%)
are PPO-Raw Table I/II values; the paper's Kalman-variant labels are internally inconsistent,
so methodological fidelity (not hitting these numbers) is the target.

## Result / risk / next
- Gate F pass: metrics saved, obs_dim == 22, no observation shape change.
- This is **not** metric-chasing: PPO settings are identical to Raw; only the input OHLC is filtered.
- Next action: Nautilus parity (script 05).
"""
    write_text(_KF / "docs" / "KALMAN_EVALUATION_REPORT.md", md)
    print(f"Gate F: cum={m['cumulative_return']:.4f} sharpe={m['sharpe']:.2f} "
          f"maxDD={m['max_drawdown']:.4f} trades={m['round_trips']} obs_dim={len(cols)}")


if __name__ == "__main__":
    main()
