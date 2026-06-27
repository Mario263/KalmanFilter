"""RLPolicyStrategy — drives the trained PPO Raw policy inside Nautilus Trader.

Inference only (PPO.load + predict). Observations are the SAME precomputed 22-vectors
used by the evaluation pipeline (looked up by exact hourly UTC-ns timestamp) — NO
indicator recomputation inside Nautilus, guaranteeing bitwise consistency.

Action -> target position mapping (paper A={-1,0,+1}):
  0 -> short, 1 -> flat, 2 -> long. Orders move the NET position to target via
  Nautilus order management (no custom execution / order tracking).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from nautilus_trader.model.enums import OrderSide
# pyrefly: ignore [missing-import]
from nautilus_trader.model.identifiers import InstrumentId
# pyrefly: ignore [missing-import]
from nautilus_trader.model.objects import Quantity
# pyrefly: ignore [missing-import]
from nautilus_trader.trading.config import StrategyConfig
# pyrefly: ignore [missing-import]
from nautilus_trader.trading.strategy import Strategy

ACTION_TO_POSITION = {0: -1, 1: 0, 2: 1}


class RLConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str


class RLPolicyStrategy(Strategy):
    def __init__(self, config: RLConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type = config.bar_type
        self._model = None
        self._last_decision_px = 0.0
        self._obs_map: dict = {}
        self._trade_oz = 10.0
        self._commission_per_leg = 0.35
        self._lots_per_trade = 0.1
        self.equity_curve: list = []   # (ts_ns, equity_usd)
        self.action_log: list = []     # (date, action, target_pos, decision_close)
        self.fills_log: list = []      # (fill_ts_ns, fill_px, side, qty, decision_date, decision_close)
        self._instrument = None
        self._pending = None           # (decision_date, decision_close) for the order in flight
        self._cash = None              # self-tracked cash; net-liq sizing (see _net_liq)
        self.dbg = {"bars": 0, "obs_hit": 0, "obs_miss": 0, "orders": 0,
                    "act0": 0, "act1": 0, "act2": 0, "first_keys": []}

    # injected before backtest (model/obs are not config-serializable)
    def attach(
        self,
        model,
        obs_map: dict,
        *,
        trade_oz: float,
        commission_per_leg: float,
        lots_per_trade: float,
    ) -> None:
        self._model = model
        self._obs_map = obs_map
        self._trade_oz = float(trade_oz)
        self._commission_per_leg = float(commission_per_leg)
        self._lots_per_trade = float(lots_per_trade)

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.instrument_id)
        self._cash = self._account_cash()  # added this line
        self.subscribe_bars(self._bar_type_obj())

    def _bar_type_obj(self):
        # pyrefly: ignore [missing-import]
        from nautilus_trader.model.data import BarType 
        return BarType.from_str(self._bar_type)

    def _account_cash(self) -> float:
        acct = self.portfolio.account(self.instrument_id.venue)
        if acct is None:
            return 0.0
        bal = acct.balance_total(self._instrument.quote_currency)
        return float(bal.as_double()) if bal is not None else 0.0

    def _net_liq(self, price: float) -> float:
        """NET-LIQUIDATION equity = self-tracked cash + position marked to market.
        Stable whether long/short/flat — unlike account cash, which is inflated by
        short-sale proceeds and (if used for sizing) compounds short positions to a
        blow-up. Sizing to net-liq bounds notional to ~1x equity (the env's rule)."""
        if self._cash is None:                 # lazy init from the starting balance
            self._cash = self._account_cash()
        return self._cash + self._net_position() * price

    def _net_position(self) -> float:
        return float(self.portfolio.net_position(self.instrument_id))

    @staticmethod
    def _is_buy(side) -> bool:
        try:
            return side == OrderSide.BUY or int(side) == int(OrderSide.BUY)
        except Exception:
            return str(side).upper().endswith("BUY")

    def on_bar(self, bar) -> None:
        # HOURLY parity: key on the exact bar timestamp (integer UTC nanoseconds),
        # NOT .normalize()/.date() (which floor to midnight and collapse all 24
        # hourly bars of a day to one daily key). The ns key is bijective with
        # eval_df.index (build_data uses the same int(pd.Timestamp(ts).value)).
        key = int(bar.ts_event)
        ts = pd.Timestamp(bar.ts_event, tz="UTC")
        equity = self._net_liq(float(bar.close))
        self.equity_curve.append((int(bar.ts_event), equity))
        self.dbg["bars"] += 1
        if len(self.dbg["first_keys"]) < 3:
            self.dbg["first_keys"].append(str(ts))

        obs = self._obs_map.get(key)
        if obs is None or np.any(~np.isfinite(obs)):   # missing/NaN -> hold
            self.dbg["obs_miss"] += 1
            return
        self.dbg["obs_hit"] += 1
        try:
            action, _ = self._model.predict(obs, deterministic=True)
            action = int(action)
        except Exception:                               # inference failure -> hold
            return
        self.dbg[f"act{action}"] += 1
        target_pos = ACTION_TO_POSITION[action]
        price = float(bar.close)
        self.action_log.append((str(ts), action, target_pos, price))

        self._last_decision_px = price
        if price <= 0:
            return
        cur = self._net_position()
        # Fixed lot sizing — matches RL env (lots_per_trade * contract_oz_per_lot).
        target_signed = target_pos * int(self._trade_oz)
        delta = target_signed - cur
        if abs(delta) < 1:
            return
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        qty = Quantity.from_int(int(abs(delta)))
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
        )
        self._pending = (key, price)   # decision context (exact ns) for this order
        self.submit_order(order)
        self.dbg["orders"] += 1

    def on_order_filled(self, event) -> None:
        self.dbg["fills"] = self.dbg.get("fills", 0) + 1
        fpx = float(event.last_px)
        is_buy = self._is_buy(event.order_side)
        sq = float(event.last_qty) * (1 if is_buy else -1)
        # maintain self-tracked cash (cash model): buy -> cash down, sell -> cash up.
        if self._cash is None:
            self._cash = self._account_cash()
        self._cash -= sq * fpx
        self._cash -= self._commission_per_leg * self._lots_per_trade
        dd, dc = self._pending if self._pending else ("", 0.0)
        self.fills_log.append((int(event.ts_event), fpx,
                               "BUY" if is_buy else "SELL", float(event.last_qty), dd, dc))
        fp = self.dbg.setdefault("fill_vs_decision", [])
        if len(fp) < 8:
            fp.append((self._last_decision_px, float(event.last_px)))

    def on_order_denied(self, event) -> None:
        self.dbg["denied"] = self.dbg.get("denied", 0) + 1
        rs = self.dbg.setdefault("deny_reasons", [])
        if len(rs) < 3:
            rs.append(str(getattr(event, "reason", event)))

    def on_order_rejected(self, event) -> None:
        self.dbg["rejected"] = self.dbg.get("rejected", 0) + 1
        rs = self.dbg.setdefault("reject_reasons", [])
        if len(rs) < 3:
            rs.append(str(getattr(event, "reason", event)))

    def on_stop(self) -> None:
        pass
