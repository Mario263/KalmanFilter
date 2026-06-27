"""Shared Nautilus Trader simulation helpers for event-driven backtest."""
from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import List, Tuple

import pandas as pd

from rl_gold_trading.config import nautilus_config

ACTION_TO_POSITION = {0: -1, 1: 0, 2: 1}


@lru_cache(maxsize=1)
def _nt_cfg():
    return nautilus_config()


def starting_usd() -> float:
    return _nt_cfg().starting_usd


def fee_rate() -> float:
    return _nt_cfg().fee


# Backward-compatible aliases for run_backtest imports.
STARTING_USD = starting_usd()
FEE = fee_rate()


def signed_qty(side, qty) -> float:
    text = str(side).upper()
    if text in {"BUY", "ORDERSIDE.BUY", "1"} or text.endswith(".BUY"):
        return float(qty)
    if text in {"SELL", "ORDERSIDE.SELL", "2"} or text.endswith(".SELL"):
        return -float(qty)
    raise ValueError(f"Unknown order side: {side!r}")


def build_instrument():
    from nautilus_trader.model.currencies import USD, XAU  # type: ignore
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue  # type: ignore
    from nautilus_trader.model.instruments import CurrencyPair  # type: ignore
    from nautilus_trader.model.objects import Price, Quantity  # type: ignore

    iid = InstrumentId(Symbol("XAU/USD"), Venue("SIM"))
    return CurrencyPair(
        instrument_id=iid,
        raw_symbol=Symbol("XAU/USD"),
        base_currency=XAU,
        quote_currency=USD,
        price_precision=5,
        size_precision=0,
        price_increment=Price(1e-5, 5),
        size_increment=Quantity.from_int(1),
        lot_size=Quantity.from_int(1),
        max_quantity=Quantity.from_str("100000"),
        min_quantity=Quantity.from_int(1),
        max_price=None,
        min_price=None,
        max_notional=None,
        min_notional=None,
        margin_init=Decimal("0.50"),
        margin_maint=Decimal("0.50"),
        maker_fee=Decimal(str(FEE)),
        taker_fee=Decimal(str(FEE)),
        tick_scheme_name="FOREX_5DECIMAL",
        ts_event=0,
        ts_init=0,
    )


def bar_and_quote(instrument, bar_type, ts, row):
    """Build one bar + quote pair for a single timestep."""
    from nautilus_trader.model.data import Bar, QuoteTick  # type: ignore
    from nautilus_trader.model.objects import Price, Quantity  # type: ignore

    ts_ns = int(pd.Timestamp(ts).value)
    pp = instrument.price_precision
    px = Price(float(row["price"]), pp)
    vol = Quantity(int(max(1, row.get("volume", 1))), 0)
    one = Quantity.from_int(1_000_000)
    # Quote at ts_ns - 1 so market orders on bar t fill at close[t] (env MOC).
    quote = QuoteTick(instrument.id, px, px, one, one, ts_ns - 1, ts_ns - 1)
    bar = Bar(bar_type, px, px, px, px, vol, ts_ns, ts_ns)
    return bar, quote


def build_bar_type(instrument, timeframe: str = "1h"):
    from nautilus_trader.model.data import BarSpecification, BarType  # type: ignore
    from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType  # type: ignore

    tf = timeframe.lower()
    agg = BarAggregation.DAY if tf in {"1d", "d", "day", "daily"} else BarAggregation.HOUR
    spec = BarSpecification(1, agg, PriceType.LAST)
    return BarType(instrument.id, spec, AggregationSource.EXTERNAL)


def build_data(instrument, df, timeframe: str = "1h") -> Tuple[object, List, List]:
    """Build all bars and quotes for a dataframe (inference / batch backtest)."""
    bar_type = build_bar_type(instrument, timeframe=timeframe)
    bars, quotes = [], []
    for ts, row in df.iterrows():
        bar, quote = bar_and_quote(instrument, bar_type, ts, row)
        bars.append(bar)
        quotes.append(quote)
    return bar_type, bars, quotes
