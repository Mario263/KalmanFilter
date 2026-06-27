"""Gate checks: fail loud, never silent. Pure functions returning evidence dicts."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

EXPECTED_ORDER = [
    "open", "high", "low", "close", "volume",
    "sma10", "sma20", "sma50", "ema12", "ema26",
    "macd_line", "macd_signal", "rsi14", "stoch_k", "stoch_d",
    "boll_upper", "boll_lower", "atr14", "obv", "vwap", "cci", "williams_r",
]


def check_volume_unchanged(clean_volume: np.ndarray, model_volume: np.ndarray) -> Dict:
    """Raw Volume must survive untouched into the model-input CSV (KALMAN-A01)."""
    a = np.asarray(clean_volume, dtype=np.float64)
    b = np.asarray(model_volume, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"volume length mismatch: {a.shape} vs {b.shape}")
    max_abs_diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
    ok = max_abs_diff == 0.0
    if not ok:
        raise ValueError(f"VOLUME CHANGED: max|Δ|={max_abs_diff} — must be 0 (raw passthrough)")
    return {"max_abs_diff": max_abs_diff, "unchanged": ok, "negative_rows": int((b < 0).sum())}


def check_ohlc_validity(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Dict:
    """Report (do not clip) filtered-OHLC ordering violations."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    hi_bad = int((h < np.maximum.reduce([o, c, l])).sum())
    lo_bad = int((l > np.minimum.reduce([o, c, h])).sum())
    return {"high_violations": hi_bad, "low_violations": lo_bad, "total": hi_bad + lo_bad, "n": len(o)}


def validate_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Hard checks for Gate D. Raises on any structural failure."""
    cols = list(feature_cols)
    issues: List[str] = []

    if len(cols) != 22:
        issues.append(f"feature count {len(cols)} != 22")
    if cols != EXPECTED_ORDER:
        issues.append("feature order does not match Raw PPO order")
    if len(set(cols)) != len(cols):
        issues.append("duplicate feature names")
    if any(c.startswith("kalman_") for c in cols):
        issues.append("kalman_* column leaked into features")
    if "kalman_volume" in cols:
        issues.append("kalman_volume present")
    for c in cols:
        if c not in df.columns:
            issues.append(f"missing column {c}")

    sub = df[cols].to_numpy(dtype=np.float64)
    n_nan = int(np.isnan(sub).sum())
    n_inf = int(np.isinf(sub).sum())
    if n_nan or n_inf:
        issues.append(f"non-finite after warmup: nan={n_nan} inf={n_inf}")

    if issues:
        raise ValueError("FEATURE MATRIX INVALID: " + "; ".join(issues))
    return {
        "feature_count": len(cols),
        "order_matches_raw": True,
        "nan": n_nan, "inf": n_inf,
        "rows": int(len(df)),
        "filtered_ohlc": ["open", "high", "low", "close"],
        "raw_volume": ["volume"],
        "indicators": cols[5:],
    }
