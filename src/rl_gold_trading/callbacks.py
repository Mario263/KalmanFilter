"""Training-time validation on the held-out eval window."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from stable_baselines3 import PPO

from rl_gold_trading.envs import XAUUSDTradingEnv
from rl_gold_trading.logging_utils import TrainingLog
from rl_gold_trading.metrics import evaluate_model


class ValidationRunner:
    """Run deterministic eval-window backtests during PPO training."""

    def __init__(
        self,
        eval_env: XAUUSDTradingEnv,
        log_dir: str,
        training_log: Optional[TrainingLog] = None,
    ) -> None:
        self.eval_env = eval_env
        self.log_dir = log_dir
        self._log = training_log
        self.history: List[Dict[str, Any]] = []
        os.makedirs(log_dir, exist_ok=True)
        self._jsonl_path = os.path.join(log_dir, "validation_history.jsonl")
        self._summary_path = os.path.join(log_dir, "validation_history.json")

    def __call__(
        self,
        model: PPO,
        *,
        iteration: int,
        epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        metrics = evaluate_model(model, self.eval_env)
        record = {
            "iteration": iteration,
            "epoch": epoch,
            "timesteps": int(model.num_timesteps),
            "metrics": metrics,
        }
        self.history.append(record)
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        if self._log is not None:
            self._log.validation(
                iteration=iteration,
                epoch=epoch,
                timesteps=int(model.num_timesteps),
                metrics=metrics,
            )
        return metrics
