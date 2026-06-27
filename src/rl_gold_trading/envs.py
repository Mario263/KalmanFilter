"""Trading environment for PPO training (Paper Section IV.E/F reward shape).

State: exactly 22 z-scored features (5 OHLCV + 17 indicators).
Action: Discrete(3) -> {sell:-1, hold:0, buy:+1}.
Sizing: fixed lots_per_trade (default 0.1 lot = 10 oz on XAU/USD).
Reward (Eq.22): r = alpha*R_port - beta*DD - gamma*Cost + delta*Stability.
Costs: commission_per_lot_round_trip charged per fill leg ($7/lot RT -> $3.50/lot/leg).

NO Kalman. PPO Raw baseline only.
"""
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl_gold_trading.config import EnvConfig

# Discrete action index -> position (paper A = {-1, 0, +1} = sell, hold, buy).
ACTION_TO_POSITION = {0: -1, 1: 0, 2: 1}


class XAUUSDTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        config: EnvConfig,
        random_reset: bool = True,
    ) -> None:
        super().__init__()
        if len(feature_cols) != 22:
            raise ValueError("State must be exactly 22 dimensions (paper p.7).")
        self.feature_cols = feature_cols
        self.config = config
        self.random_reset = random_reset

        self.feat = df[feature_cols].to_numpy(dtype=np.float32)
        # Use the RAW price path (real units), NOT the z-scored `close` feature.
        price_col = "price" if "price" in df.columns else "close"
        self.prices = df[price_col].to_numpy(dtype=np.float64)
        self.n = len(self.prices)
        if self.n != len(self.feat):
            raise ValueError("features and prices must align.")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._reset_state()

    def _reset_state(self) -> None:
        self.t = 0
        self.position = 0
        self.equity = float(self.config.initial_capital)
        self.peak = self.equity

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_state()
        # Episode = forward pass over the series. Random start during training
        # decorrelates rollouts (still strictly forward in time -> no leakage).
        if self.random_reset and self.n > 64:
            self.t = int(self.np_random.integers(0, max(1, self.n // 4)))
        return self.feat[self.t].copy(), {}

    def _trade_oz(self) -> float:
        """Notional size in troy oz for one directional position (e.g. 0.1 lot -> 10 oz)."""
        return self.config.lots_per_trade * self.config.contract_oz_per_lot

    def _commission_dollars(self, legs: int) -> float:
        """$7/lot round-trip -> $3.50/lot per fill leg; scaled by lots_per_trade."""
        cfg = self.config
        per_leg = cfg.commission_per_lot_round_trip / 2.0
        return legs * per_leg * cfg.lots_per_trade

    def step(self, action: int):
        cfg = self.config
        target = ACTION_TO_POSITION[int(action)]
        equity_before = self.equity

        turnover = abs(target - self.position)                 # in {0,1,2}
        legs = turnover

        p0 = self.prices[self.t]
        p1 = self.prices[self.t + 1] if self.t + 1 < self.n else p0

        oz = self._trade_oz()
        dollar_pnl = target * oz * (p1 - p0)
        commission = self._commission_dollars(legs)
        spread_cost = legs * cfg.spread * oz * p0 if p0 > 0 else 0.0
        total_cost = commission + spread_cost

        self.equity += dollar_pnl - total_cost
        self.peak = max(self.peak, self.equity)

        r_port = dollar_pnl / equity_before if equity_before > 0 else 0.0
        cost_frac = total_cost / equity_before if equity_before > 0 else 0.0

        dd = max(0.0, (self.peak - self.equity) / self.peak) if self.peak > 0 else 0.0
        stability = -float(turnover)

        reward = (
            cfg.alpha * r_port
            - cfg.beta * dd
            - cfg.gamma * cost_frac
            + cfg.delta * stability
        )

        net_ret = r_port - cost_frac
        self.position = target
        self.t += 1
        terminated = False
        truncated = self.t >= (self.n - 1)

        obs_idx = min(self.t, self.n - 1)
        info = {
            "equity": float(self.equity),
            "net_ret": float(net_ret),
            "position": int(self.position),
            "drawdown": float(dd),
            "cost": float(cost_frac),
            "cost_dollars": float(total_cost),
            "pnl_dollars": float(dollar_pnl),
        }
        return self.feat[obs_idx].copy(), float(reward), terminated, truncated, info
