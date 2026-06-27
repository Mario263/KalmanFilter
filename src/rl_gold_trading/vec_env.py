"""Vectorized env helpers for the PPO RAW baseline.

NOTE: NO VecNormalize. Observations are already normalized by the rolling
z-score (Paper Section IV.B, Eq.13; 6048-bar / 1-year window on hourly bars).
Adding VecNormalize would double-normalize with global running statistics,
contradicting the paper.
"""
from typing import List

import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_gold_trading.config import EnvConfig
from rl_gold_trading.envs import XAUUSDTradingEnv


def make_env(
    df: pd.DataFrame, feature_cols: List[str], config: EnvConfig, random_reset: bool
) -> DummyVecEnv:
    def _fn():
        return Monitor(XAUUSDTradingEnv(df, feature_cols, config, random_reset))

    return DummyVecEnv([_fn])
