"""Nautilus Trader backtest of the trained PPO Raw policy (inference only).

Event-driven bar-by-bar re-execution on the eval window. Observations are the
precomputed eval-pipeline 22-vectors (bitwise-consistent with RL training).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "nautilus"))

from rl_gold_trading.config import (  # type: ignore
    data_config,
    env_config,
    nautilus_config,
    paper_target,
    periods_per_year,
    set_config_path,
)
from rl_gold_trading.envs import XAUUSDTradingEnv  # type: ignore
from rl_gold_trading.metrics import evaluate_model  # type: ignore
from rl_gold_trading.run import prepare  # type: ignore
from sim import STARTING_USD, build_data, build_instrument, signed_qty  # type: ignore


def metrics_from_series(
    equity: np.ndarray,
    positions: np.ndarray,
    *,
    periods_per_year: int,
) -> dict:
    """Mirror rl_gold_trading.metrics.evaluate_model outputs from equity/position paths."""
    equity = np.asarray(equity, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.int64)
    if len(equity) < 2:
        return {"n_periods": 0, "cumulative_return": 0.0, "final_equity": float(equity[0]) if len(equity) else 0.0}

    net_rets = equity[1:] / equity[:-1] - 1.0
    n = len(net_rets)
    pos = pos[:n]
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

    in_market = pos != 0
    turnover = int(np.sum(np.abs(np.diff(np.concatenate([[0], pos])))))
    win_rate = float((net_rets > 0).mean()) if n else 0.0
    active_win_rate = float((net_rets[in_market] > 0).mean()) if in_market.any() else 0.0

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
        "min_equity": float(equity.min()),
        "nonpositive_periods": int((equity <= 0).sum()),
    }


def _reconstruct_equity(
    price: np.ndarray,
    fills_by_i: dict,
    *,
    trade_oz: float,
    commission_per_leg: float,
    lots_per_trade: float,
    starting_usd: float,
) -> tuple[np.ndarray, np.ndarray]:
    cash, pos_units = starting_usd, 0.0
    equity, positions = [], []
    max_abs_oz = 0.0
    sign_oz = int(trade_oz)
    for i in range(len(price)):
        for (_fpx, sq) in fills_by_i.get(i, []):
            cash -= sq * _fpx
            cash -= commission_per_leg * lots_per_trade
            pos_units += sq
        max_abs_oz = max(max_abs_oz, abs(pos_units))
        positions.append(
            1 if pos_units >= sign_oz else (-1 if pos_units <= -sign_oz else 0)
        )
        equity.append(cash + pos_units * price[i])
    return np.asarray(equity, dtype=np.float64), np.asarray(positions, dtype=np.int64), max_abs_oz


def _print_metrics(title: str, metrics: dict) -> None:
    pct_keys = {
        "cumulative_return", "cagr", "max_drawdown", "var_95",
        "win_rate", "active_win_rate", "trade_win_rate",
        "long_frac", "flat_frac", "short_frac",
    }
    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        if isinstance(value, float) and key in pct_keys:
            print(f"  {key:<22} {value:+.4%}" if key in {"max_drawdown", "var_95"} else f"  {key:<22} {value:.4%}")
        elif isinstance(value, float):
            print(f"  {key:<22} {value:.4f}")
        else:
            print(f"  {key:<22} {value}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Nautilus backtest of trained PPO policy.")
    ap.add_argument(
        "--config",
        default=None,
        help="Experiment JSON (default: config/experiment.json or RL_EXPERIMENT_CONFIG).",
    )
    args = ap.parse_args()
    if args.config:
        set_config_path(args.config)

    nt_cfg = nautilus_config()
    data_cfg = data_config()
    env_cfg = env_config()
    ppy = periods_per_year()
    trade_oz = env_cfg.lots_per_trade * env_cfg.contract_oz_per_lot
    commission_per_leg = env_cfg.commission_per_lot_round_trip / 2.0

    cols, _train, eval_df, _daily = prepare(data_cfg)

    from stable_baselines3 import PPO
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig  # type: ignore
    from nautilus_trader.backtest.models import MakerTakerFeeModel  # type: ignore
    from nautilus_trader.config import LoggingConfig  # type: ignore
    from nautilus_trader.model.currencies import USD  # type: ignore
    from nautilus_trader.model.enums import AccountType, OmsType  # type: ignore
    from nautilus_trader.model.identifiers import Venue  # type: ignore
    from nautilus_trader.model.objects import Money  # type: ignore
    from decimal import Decimal
    from strategy import RLConfig, RLPolicyStrategy  # type: ignore

    model_path = ROOT / nt_cfg.model_path
    print(f"Loading weights: {model_path}.zip")
    model = PPO.load(
        str(model_path),
        device="cpu",
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.2,
        },
    )

    obs_map = {
        int(pd.Timestamp(idx).value): eval_df.loc[idx, cols].to_numpy(dtype=np.float32)
        for idx in eval_df.index
    }

    instrument = build_instrument()
    bar_type, bars, quotes = build_data(instrument, eval_df, timeframe=data_cfg.timeframe)

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="BACKTESTER-001",
            logging=LoggingConfig(log_level="ERROR"),
        )
    )
    engine.add_venue(
        venue=Venue(nt_cfg.venue),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(STARTING_USD, USD)],
        default_leverage=Decimal(nt_cfg.default_leverage),
        fee_model=MakerTakerFeeModel(),
        bar_execution=False,
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_data(quotes)

    strat = RLPolicyStrategy(RLConfig(instrument_id=str(instrument.id), bar_type=str(bar_type)))
    strat.attach(
        model,
        obs_map,
        trade_oz=trade_oz,
        commission_per_leg=commission_per_leg,
        lots_per_trade=env_cfg.lots_per_trade,
    )
    engine.add_strategy(strat)

    print(
        f"Running Nautilus backtest: {len(bars)} {data_cfg.timeframe} bars "
        f"({eval_df.index.min().date()} → {eval_df.index.max().date()}) | "
        f"trade size {trade_oz:.0f} oz ({env_cfg.lots_per_trade} lot)"
    )
    engine.run()

    price = eval_df["price"].to_numpy(float)
    ns2i = {int(pd.Timestamp(i).value): k for k, i in enumerate(eval_df.index)}
    fills_by_i: dict[int, list] = {}
    for (fts, fpx, side, qty, dec_ns, _dc) in strat.fills_log:
        fi = ns2i.get(int(dec_ns)) if isinstance(dec_ns, int) else None
        if fi is None:
            fi = ns2i.get(int(fts))
        if fi is not None:
            fills_by_i.setdefault(fi, []).append((fpx, signed_qty(side, qty)))

    equity, positions, max_abs_oz = _reconstruct_equity(
        price,
        fills_by_i,
        trade_oz=trade_oz,
        commission_per_leg=commission_per_leg,
        lots_per_trade=env_cfg.lots_per_trade,
        starting_usd=STARTING_USD,
    )
    nm = metrics_from_series(equity, positions, periods_per_year=ppy)

    try:
        pos_rep = engine.trader.generate_positions_report()
    except Exception:
        pos_rep = pd.DataFrame()
    fills_rep = engine.trader.generate_order_fills_report()
    n_fills = int(len(fills_rep)) if fills_rep is not None else 0

    trade_stats = {"round_trips": 0, "win_rate": 0.0, "profit_factor": float("nan")}
    if pos_rep is not None and len(pos_rep) and "realized_pnl" in pos_rep.columns:
        pnl = pd.to_numeric(
            pos_rep["realized_pnl"].astype(str).str.replace(r"[^0-9eE.\-]", "", regex=True),
            errors="coerce",
        ).dropna()
        closed = pnl[pnl != 0]
        if len(closed):
            wins = closed[closed > 0]
            losses = closed[closed < 0]
            trade_stats = {
                "round_trips": int(len(closed)),
                "win_rate": float((closed > 0).mean()),
                "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf"),
            }

    # RL env reference (same weights, same eval window)
    eval_env = XAUUSDTradingEnv(eval_df, cols, env_cfg, random_reset=False)
    rl_metrics = evaluate_model(model, eval_env, periods_per_year=ppy)

    out = {
        "config": str(args.config or "default"),
        "model_path": str(model_path),
        "eval_window": [data_cfg.eval_start, data_cfg.eval_end],
        "timeframe": data_cfg.timeframe,
        "lots_per_trade": env_cfg.lots_per_trade,
        "trade_oz": trade_oz,
        "nautilus": nm,
        "rl_env": rl_metrics,
        "paper_target": paper_target(),
        "nautilus_engine": {
            "trade_stats": trade_stats,
            "n_fills": n_fills,
            "n_bars": len(bars),
            "bars_processed": strat.dbg.get("bars", 0),
            "obs_hit": strat.dbg.get("obs_hit", 0),
            "obs_miss": strat.dbg.get("obs_miss", 0),
            "orders_submitted": strat.dbg.get("orders", 0),
            "actions": {
                "sell": strat.dbg.get("act0", 0),
                "hold": strat.dbg.get("act1", 0),
                "buy": strat.dbg.get("act2", 0),
            },
            "max_abs_position_oz": max_abs_oz,
        },
        "starting_usd": STARTING_USD,
        "final_usd": nm["final_equity"],
    }
    out_path = ROOT / nt_cfg.metrics_output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))

    _print_metrics("Nautilus (bar-by-bar)", nm)
    _print_metrics("RL env (reference)", rl_metrics)
    _print_metrics("Paper target", paper_target())

    print("\n=== Engine diagnostics ===")
    for key, value in out["nautilus_engine"].items():
        print(f"  {key}: {value}")
    print(f"\nSaved → {out_path}")
    engine.dispose()


if __name__ == "__main__":
    main()
