"""PPO RAW model builder (Paper Section IV.G.2 / IV.H, PDF p.9).

Stable-Baselines3 PPO (mature library; NO custom PPO). Architecture:
actor & critic = [512, 512, 256, 128] with Tanh, softmax actor / linear critic.
Learning rate 3e-4 with LINEAR decay to zero.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv

# pyrefly: ignore [missing-import]
from rl_gold_trading.callbacks import ValidationRunner
from rl_gold_trading.config import TrainConfig
from rl_gold_trading.logging_utils import TrainingLog


def linear_schedule(initial: float):
    """SB3 schedule: progress_remaining goes 1 -> 0 over training (p.9)."""
    def f(progress_remaining: float) -> float:
        return initial * progress_remaining
    return f


class PPOWithValidation(PPO):
    """PPO with Rich logging and optional eval-window validation."""

    def __init__(
        self,
        *args,
        validation_runner: Optional[ValidationRunner] = None,
        validation_every_epoch: bool = True,
        training_log: Optional[TrainingLog] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._validation_runner = validation_runner
        self._validation_every_epoch = validation_every_epoch
        self._training_log = training_log
        self._iteration = 0
        self._rollout_metrics: Dict[str, Any] = {}

    def dump_logs(self, iteration: int = 0) -> None:
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))

        if iteration > 0:
            self._rollout_metrics = dict(self.logger.name_to_value)
        self.logger.dump(step=self.num_timesteps)

    def _train_once(self) -> Dict[str, Any]:
        captured: Dict[str, Any] = {}
        orig_dump = self.logger.dump

        def dump_and_capture(step: int = 0) -> None:
            captured.update(self.logger.name_to_value)
            orig_dump(step)

        self.logger.dump = dump_and_capture
        try:
            super().train()
        finally:
            self.logger.dump = orig_dump
        return captured

    def _log_iteration(self, train_metrics: Dict[str, Any]) -> None:
        if self._training_log is None:
            return
        merged = {**self._rollout_metrics, **train_metrics}
        self._training_log.ppo_iteration(self._iteration + 1, merged, self.num_timesteps)
        self._rollout_metrics = {}

    def train(self) -> None:
        runner = self._validation_runner
        train_metrics: Dict[str, Any] = {}

        if runner is not None and self._validation_every_epoch:
            saved_epochs = self.n_epochs
            for epoch in range(saved_epochs):
                self.n_epochs = 1
                train_metrics = self._train_once()
                runner(self, iteration=self._iteration, epoch=epoch)
            self.n_epochs = saved_epochs
        else:
            train_metrics = self._train_once()
            if runner is not None:
                runner(self, iteration=self._iteration, epoch=None)

        self._log_iteration(train_metrics)
        self._iteration += 1

    def detach_runtime_hooks(self) -> None:
        """Drop non-serializable training hooks before model.save()."""
        self._training_log = None
        self._validation_runner = None


def build_model(
    train_env: DummyVecEnv,
    cfg: TrainConfig,
    validation_runner: Optional[ValidationRunner] = None,
    training_log: Optional[TrainingLog] = None,
) -> PPO:
    policy_kwargs = dict(
        net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch)),
        activation_fn=nn.Tanh,
    )
    lr = linear_schedule(cfg.learning_rate) if cfg.lr_linear_decay else cfg.learning_rate
    use_hooks = validation_runner is not None or training_log is not None
    cls = PPOWithValidation if use_hooks else PPO
    return cls(
        "MlpPolicy",
        train_env,
        learning_rate=lr,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=cfg.seed,
        device=cfg.device,
        verbose=0,
        validation_runner=validation_runner,
        validation_every_epoch=cfg.validation_every_epoch,
        training_log=training_log,
    )


def train_model(
    model: PPO,
    cfg: TrainConfig,
    validation_runner: Optional[ValidationRunner] = None,
) -> PPO:
    if isinstance(model, PPOWithValidation) and validation_runner is not None:
        model._validation_runner = validation_runner
        model._validation_every_epoch = cfg.validation_every_epoch
    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True)
    return model
