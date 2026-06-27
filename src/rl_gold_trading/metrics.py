"""Evaluation metrics (Paper Section V, PDF p.15).

Computes the paper's PPO Raw metrics from the strategy equity curve over the
evaluation window: cumulative return, CAGR, Sharpe, max drawdown, win rate
(+ Sortino / Calmar / Recovery / VaR diagnostics). Deterministic single pass;
no random reset.

NOTE: the paper does not state its Sharpe annualization factor. On HOURLY bars
(user-directed deviation), annualization = 252 trading days x 24h = 6048 periods
per year (consistent with the paper's 252-trading-day basis). Documented assumption.
"""
from typing import Dict

import numpy as np
from stable_baselines3 import PPO

from rl_gold_trading.config import periods_per_year as config_periods_per_year
from rl_gold_trading.envs import XAUUSDTradingEnv


def evaluate_model(
    model: PPO, env: XAUUSDTradingEnv, periods_per_year: int | None = None
) -> Dict[str, float]:
    if periods_per_year is None:
        periods_per_year = config_periods_per_year()
    obs, _ = env.reset()
    net_rets, equity, positions = [], [env.equity], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(int(action))
        net_rets.append(info["net_ret"])
        equity.append(info["equity"])
        positions.append(info["position"])
        done = terminated or truncated

    net_rets = np.asarray(net_rets, dtype=np.float64)
    equity = np.asarray(equity, dtype=np.float64)
    n = len(net_rets)

    cum_return = equity[-1] / equity[0] - 1.0
    years = max(n / periods_per_year, 1e-9)
    cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0

    std = net_rets.std(ddof=0)
    sharpe = (net_rets.mean() / std) * np.sqrt(periods_per_year) if std > 1e-12 else 0.0

    downside = net_rets[net_rets < 0]
    dstd = downside.std(ddof=0) if downside.size else 0.0
    sortino = (net_rets.mean() / dstd) * np.sqrt(periods_per_year) if dstd > 1e-12 else 0.0

    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    max_dd = float(drawdowns.min())

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0
    recovery = (cum_return / abs(max_dd)) if max_dd < 0 else 0.0
    var95 = float(np.percentile(net_rets, 5)) if n else 0.0

    pos = np.asarray(positions)
    in_market = pos != 0
    turnover = int(np.sum(np.abs(np.diff(np.concatenate([[0], pos])))))

    # Win-rate definitions (all share the same numerator of profitable periods;
    # only the denominator differs). The paper's "win rate" (50.16%) is a
    # per-trade / in-market figure (its raw agent is ~always in the market);
    # the all-period figure is NOT comparable for a flat-heavy policy.
    win_rate = float((net_rets > 0).mean()) if n else 0.0                       # all periods
    active_win_rate = float((net_rets[in_market] > 0).mean()) if in_market.any() else 0.0  # in-market bars

    # Per-trade win rate: segment maximal constant non-zero position runs and
    # measure each round trip's equity-based return (incl. costs). This is the
    # standard trading "win rate" (profitable trades / total trades).
    trade_rets = []
    cur_dir, entry_idx = 0, None
    for t in range(n):
        d = int(pos[t])
        if d != cur_dir:
            if cur_dir != 0 and entry_idx is not None:
                trade_rets.append(equity[t] / equity[entry_idx] - 1.0)
            entry_idx = t if d != 0 else None
            cur_dir = d
    if cur_dir != 0 and entry_idx is not None:
        trade_rets.append(equity[n] / equity[entry_idx] - 1.0)
    trade_rets = np.asarray(trade_rets, dtype=np.float64)
    round_trips = int(len(trade_rets))
    trade_win_rate = float((trade_rets > 0).mean()) if round_trips else 0.0

    return {
        "n_periods": int(n),
        "cumulative_return": float(cum_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "recovery_factor": float(recovery),
        "max_drawdown": max_dd,
        "var_95": var95,
        "win_rate": win_rate,
        "active_win_rate": active_win_rate,
        "trade_win_rate": trade_win_rate,
        "round_trips": round_trips,
        "final_equity": float(equity[-1]),
        "total_turnover": turnover,
        "long_frac": float((pos == 1).mean()) if n else 0.0,
        "flat_frac": float((pos == 0).mean()) if n else 0.0,
        "short_frac": float((pos == -1).mean()) if n else 0.0,
    }
