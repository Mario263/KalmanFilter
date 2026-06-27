"""Rolling z-score normalization (Paper Section IV.B, Eq.13, PDF p.6).

    z_t = (x_t - mu_{t-W:t}) / sigma_{t-W:t}

The paper specifies a 252-DAY (1 trading year) window. On HOURLY bars
(user-directed deviation) that window is W = ZSCORE_WINDOW = 6048 (252 days x 24h).
CAUSAL: trailing W-observation window only (inclusive of t) -> no look-ahead.
Applied per-feature (not global). Replaces any global/running normalization.
"""
import numpy as np
import pandas as pd

from rl_gold_trading.config import zscore_window


def rolling_zscore(df: pd.DataFrame, feature_cols, window: int | None = None) -> pd.DataFrame:
    if window is None:
        window = zscore_window()
    out = df.copy()
    for col in feature_cols:
        s = out[col].astype(float)
        mu = s.rolling(window, min_periods=window).mean()
        sigma = s.rolling(window, min_periods=window).std(ddof=0)
        out[col] = (s - mu) / sigma.replace(0.0, np.nan)
    out = out.dropna(subset=list(feature_cols)).copy()
    out[list(feature_cols)] = out[list(feature_cols)].replace([np.inf, -np.inf], 0.0)
    return out
