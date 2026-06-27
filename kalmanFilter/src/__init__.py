"""Kalman-enhanced PPO extension layer (OHLC-only filtering; Volume stays raw).

Thin wrapper around the existing Raw PPO pipeline in ``src/rl_gold_trading``.
No existing file is modified; this package only adds a Kalman pre-filter stage.
"""
