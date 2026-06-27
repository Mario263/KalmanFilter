"""Gate A: programmatic baseline facts (read-only) so the audit is evidence, not prose."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from .pipeline import ensure_src_on_path
from .validation import EXPECTED_ORDER


def audit_baseline(config_path: str | Path) -> Dict:
    """Confirm the 1d config/pipeline assumptions the Kalman seam relies on."""
    ensure_src_on_path()
    from rl_gold_trading.config import (
        data_config, feature_order, set_config_path, zscore_window,
    )

    set_config_path(config_path)
    dc = data_config()
    order = feature_order()

    facts = {
        "config_path": str(config_path),
        "csv_path": dc.csv_path,
        "skip_resample": bool(dc.skip_resample),
        "timeframe": dc.timeframe,
        "train_end": dc.train_end,
        "test_start": dc.test_start,
        "eval_start": dc.eval_start,
        "eval_end": dc.eval_end,
        "feature_count": len(order),
        "feature_order": order,
        "feature_order_matches_expected": order == EXPECTED_ORDER,
        "zscore_window": zscore_window(),
    }
    problems = []
    if not facts["skip_resample"]:
        problems.append("skip_resample is False — hidden resampling risk")
    if facts["feature_count"] != 22:
        problems.append(f"feature_count {facts['feature_count']} != 22")
    if facts["zscore_window"] != 252:
        problems.append(f"zscore_window {facts['zscore_window']} != 252")
    if not facts["feature_order_matches_expected"]:
        problems.append("feature order differs from expected Raw PPO order")
    facts["problems"] = problems
    facts["gate_a_pass"] = not problems
    return facts
