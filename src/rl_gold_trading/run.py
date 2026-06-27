"""PPO RAW baseline reproduction orchestrator (Kili et al., IJACSA 16(11), 2025).

Pipeline (Paper Section IV, with user-directed hourly/5-day deviations):
  hourly XAU/USD 2017-2025 (Mon-Fri) -> 22 features -> 1-year (6048-bar) z-score
  -> calendar split (train 2017-2022) -> PPO RAW (SB3, paper arch/hparams, 500k)
  -> evaluate on the window (Jan 2 2023 -> Sep 12 2024).

NO Kalman / DQN / RPPO. PPO Raw only.
"""
import argparse
import json
import os
from typing import Optional

# Import data stack (datasets/pyarrow) BEFORE torch to avoid an OpenMP init clash.
from rl_gold_trading.config import (
    data_config,
    env_config,
    experiment_meta,
    paper_target,
    set_config_path,
    smoke_timesteps,
    train_config,
)
from rl_gold_trading.data import eval_window, load_data, split_train_test
from rl_gold_trading.features import add_features
from rl_gold_trading.logging_utils import TrainingLog, silence_sb3_stdout
from rl_gold_trading.normalize import rolling_zscore

# Paper PPO Raw reference (Table I/II) — loaded from config/experiment.json.
PAPER_TARGET = paper_target()


def prepare(data_cfg):
    daily = load_data(data_cfg)
    feat, cols = add_features(daily)
    feat_z = rolling_zscore(feat, cols)        # 6048-bar (1-year) causal z-score (drops warmup)
    train, _test = split_train_test(feat_z, data_cfg)
    eval_df = eval_window(feat_z, data_cfg)    # 621-day reported window
    return cols, train, eval_df, daily


def _parse(argv):
    ap = argparse.ArgumentParser(description="PPO RAW baseline (no Kalman).")
    ap.add_argument("--config", default=None, help="Path to experiment JSON (default: config/experiment.json).")
    ap.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--csv", default=os.environ.get("XAUUSD_CSV"))
    ap.add_argument("--save-dir", default="models")
    ap.add_argument("--smoke", action="store_true", help="Fast 20k-step sanity run.")
    ap.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="PPO policy device (cuda default; use --device cpu to avoid SB3 MlpPolicy GPU warning).",
    )
    return ap.parse_args(argv)


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args if args is not None else _parse(None)
    if args.config:
        set_config_path(args.config)
    global PAPER_TARGET
    PAPER_TARGET = paper_target()

    overrides = {}
    if args.csv:
        overrides["csv_path"] = args.csv
    data_cfg = data_config(**overrides)
    env_cfg = env_config()
    train_cfg = train_config(save_dir=args.save_dir)
    if args.smoke:
        train_cfg.total_timesteps = smoke_timesteps()
        # Smoke is a fast sanity check — skip mid-training eval (10k+ bars × every
        # PPO epoch makes a ~20k run ~100× slower than training alone).
        train_cfg.validation_enabled = False
    if args.timesteps is not None:
        train_cfg.total_timesteps = args.timesteps
    train_cfg.device = args.device

    training = args.mode in ("train", "train_eval")
    tlog = TrainingLog(train_cfg.train_log_file) if training else None
    if training:
        silence_sb3_stdout()

    try:
        _run(args, data_cfg, env_cfg, train_cfg, tlog)
    finally:
        if tlog is not None:
            tlog.close()


def _run(args, data_cfg, env_cfg, train_cfg, tlog: Optional[TrainingLog]) -> None:
    cols, train_df, eval_df, daily = prepare(data_cfg)
    if tlog is not None:
        tf = getattr(data_cfg, "timeframe", None) or data_cfg.resample_rule
        tlog.info(
            f"data: {len(daily):,} {tf} bars | train {len(train_df):,} "
            f"({train_df.index.min().date()} → {train_df.index.max().date()}) | "
            f"eval {len(eval_df):,} ({eval_df.index.min().date()} → {eval_df.index.max().date()})"
        )

    from rl_gold_trading.callbacks import ValidationRunner
    from rl_gold_trading.envs import XAUUSDTradingEnv
    from rl_gold_trading.metrics import evaluate_model
    from rl_gold_trading.train import build_model, train_model
    from rl_gold_trading.vec_env import make_env
    from stable_baselines3 import PPO

    model_name = experiment_meta()["model_name"]
    model_path = os.path.join(train_cfg.save_dir, model_name)

    validation_runner = None
    if train_cfg.validation_enabled and args.mode in ("train", "train_eval"):
        eval_env = XAUUSDTradingEnv(eval_df, cols, env_cfg, random_reset=False)
        validation_runner = ValidationRunner(
            eval_env,
            log_dir=train_cfg.validation_log_dir,
            training_log=tlog,
        )
        if tlog is not None:
            when = "every PPO epoch" if train_cfg.validation_every_epoch else "every rollout iteration"
            tlog.info(f"Validation: {when} → {train_cfg.validation_log_dir}/")

    if args.mode in ("train", "train_eval"):
        if tlog is not None:
            tlog.info(
                f"train: {train_cfg.total_timesteps:,} steps on {train_cfg.device} "
                f"| log {train_cfg.train_log_file}"
            )
        train_env = make_env(train_df, cols, env_cfg, random_reset=True)
        model = build_model(train_env, train_cfg, validation_runner=validation_runner, training_log=tlog)
        model = train_model(model, train_cfg, validation_runner=validation_runner)
        os.makedirs(train_cfg.save_dir, exist_ok=True)
        if hasattr(model, "detach_runtime_hooks"):
            model.detach_runtime_hooks()
        model.save(model_path)
        if tlog is not None:
            tlog.info(f"Saved model → {model_path}.zip")
    else:
        if tlog is not None:
            tlog.info(f"load model: {model_path}")
        model = PPO.load(
            model_path, device="cpu",
            custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                            "clip_range": lambda _: 0.2},
        )

    if args.mode == "train":
        return

    if tlog is not None:
        tlog.info("final evaluation")
    eval_env = XAUUSDTradingEnv(eval_df, cols, env_cfg, random_reset=False)
    m = evaluate_model(model, eval_env)

    extra = [
        f"per trade win rate = {m['trade_win_rate']:.2%} ({m['round_trips']} round trips)",
        f"in-market win rate = {m['active_win_rate']:.2%}",
        f"exposure = {1 - m['flat_frac']:.0%}  |  turnover = {m['total_turnover']}",
        f"Sortino {m['sortino']:.2f}  Calmar {m['calmar']:.2f}  Recovery {m['recovery_factor']:.2f}",
    ]
    if tlog is not None:
        tlog.metrics_report("Reproduction report (Eq.22 env vs paper)", m, PAPER_TARGET, extra)

    os.makedirs(train_cfg.save_dir, exist_ok=True)
    metrics_path = os.path.join(train_cfg.save_dir, "ppo_raw_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"reproduced": m, "paper_target": PAPER_TARGET,
                   "timesteps": train_cfg.total_timesteps,
                   "metrics_source": "Eq.22 env",
                   "eval_window": [data_cfg.eval_start, data_cfg.eval_end]}, f, indent=2)
    if tlog is not None:
        tlog.info(f"Saved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
