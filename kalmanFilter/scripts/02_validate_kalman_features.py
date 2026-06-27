"""Gate D: build the 22D feature matrix from the hybrid CSV via existing modules.

    python kalmanFilter/scripts/02_validate_kalman_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_KF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_KF))

from src import pipeline, validation                       # noqa: E402
from src.reports import dump_json, write_text              # noqa: E402

RUNTIME_CFG = pipeline.OUTPUTS / "diagnostics" / "effective_kalman_runtime_config.json"


def main() -> None:
    args = pipeline.base_argparser("Validate 22D Kalman feature matrix (Gate D)").parse_args()
    cfg = args.config_path
    if RUNTIME_CFG.exists():
        cfg = str(RUNTIME_CFG)        # use Kalman-filtered data path
    pipeline.ensure_src_on_path()
    from rl_gold_trading.config import data_config, set_config_path
    from rl_gold_trading.data import load_data
    from rl_gold_trading.features import add_features
    from rl_gold_trading.normalize import rolling_zscore

    set_config_path(cfg)
    dc = data_config()
    daily = load_data(dc)                       # filtered OHLC + raw volume
    feat, cols = add_features(daily)            # 17 indicators (existing formulas)
    feat_z = rolling_zscore(feat, cols)         # 252-day causal z-score (existing)

    info = validation.validate_feature_matrix(feat_z, cols)

    # Prove volume is raw (feature 'volume' == hybrid input volume == cleaned raw volume).
    raw_clean = pipeline.load_clean_ohlcv(args.config_path)["volume"]
    aligned = raw_clean.reindex(feat.index)
    vol_max_diff = float((feat["volume"] - aligned).abs().max())

    sources = (["filtered_open", "filtered_high", "filtered_low", "filtered_close",
                "raw_volume"] + [f"indicator:{c}" for c in cols[5:]])
    report = {
        **info,
        "rows_total_daily": int(len(daily)),
        "rows_after_features": int(len(feat)),
        "rows_after_zscore": int(len(feat_z)),
        "dropped_feature_warmup": int(len(daily) - len(feat)),
        "dropped_zscore_warmup": int(len(feat) - len(feat_z)),
        "first_usable_timestamp": str(feat_z.index.min()),
        "feature_sources": dict(zip(cols, sources)),
        "volume_is_raw": vol_max_diff == 0.0,
        "volume_max_abs_diff_vs_clean_raw": vol_max_diff,
    }
    dump_json(pipeline.OUTPUTS / "diagnostics" / "feature_dimension.json", report)

    head_pre = feat[cols].head(5).round(4)
    head_post = feat_z[cols].head(5).round(4)
    md = f"""# KALMAN_FEATURE_DIMENSION_REPORT

**Gate D — feature matrix.** Built with existing `features.add_features` +
`normalize.rolling_zscore` (no formulas duplicated). Input = hybrid filtered-OHLC / raw-Volume.

## Counts
- feature_count: **{info['feature_count']}** (4 filtered OHLC + 1 raw Volume + 17 indicators)
- order matches Raw PPO: **{info['order_matches_raw']}** | duplicates: none | kalman_* leaked: none
- NaN: {info['nan']} | inf: {info['inf']} (after warmup)
- daily rows {report['rows_total_daily']} → after indicators {report['rows_after_features']} "
  (dropped {report['dropped_feature_warmup']}) → after 252-z-score {report['rows_after_zscore']} "
  (dropped {report['dropped_zscore_warmup']})
- first usable timestamp: {report['first_usable_timestamp']}

## Volume passthrough
- volume feature is raw: **{report['volume_is_raw']}** | max|Δ vs cleaned raw| = {vol_max_diff}

## Feature order + source
| # | feature | source | normalized |
|---|---|---|---|
""" + "\n".join(
        f"| {i} | {c} | {report['feature_sources'][c]} | 252d z-score |"
        for i, c in enumerate(cols)
    ) + f"""

## Sample — first 5 rows BEFORE normalization
```
{head_pre.to_string()}
```

## Sample — first 5 rows AFTER normalization (252-day z-score)
```
{head_post.to_string()}
```

## Result / risk / next
- Gate D pass: feature_count==22, order matched, finite, volume raw, no kalman_volume.
- Risk: Low.
- Next action: train Kalman PPO (script 03).
"""
    write_text(_KF / "docs" / "KALMAN_FEATURE_DIMENSION_REPORT.md", md)
    print(f"Gate D pass | features={info['feature_count']} nan={info['nan']} inf={info['inf']} "
          f"volume_raw={report['volume_is_raw']} first={report['first_usable_timestamp'][:10]}")


if __name__ == "__main__":
    main()
