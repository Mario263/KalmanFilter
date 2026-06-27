"""Orchestration glue: reuse the existing Raw PPO pipeline, swap in filtered OHLC.

No feature/normalization formulas live here — they stay in ``rl_gold_trading``.
This module only: (1) puts the project ``src`` on sys.path, (2) loads the cleaned
OHLCV via the existing loader, (3) writes the hybrid (filtered-OHLC / raw-Volume)
CSVs, and (4) generates an effective runtime config that points the unchanged
pipeline at the hybrid CSV.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "kalmanFilter" / "outputs"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "experiment_dukascopy_1d.json"


def base_argparser(description: str) -> argparse.ArgumentParser:
    """Common CLI surface mandated for every Kalman script."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--config-path", default=str(DEFAULT_CONFIG))
    ap.add_argument("--input-data", default=None, help="Override raw daily CSV (default: from config).")
    ap.add_argument("--output-dir", default=str(OUTPUTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict", action="store_true")
    return ap


def ensure_src_on_path(project_root: Path = PROJECT_ROOT) -> None:
    src = str(project_root / "src")
    if src not in sys.path:
        sys.path.insert(0, str(src))


def load_clean_ohlcv(config_path: str | Path) -> "pd.DataFrame":
    """Cleaned OHLCV exactly as the Raw baseline sees it (existing loader, no edits)."""
    ensure_src_on_path()
    from rl_gold_trading.config import data_config, set_config_path
    from rl_gold_trading.data import load_data

    set_config_path(config_path)
    df = load_data(data_config())
    if list(df.columns) != ["open", "high", "low", "close", "volume"]:
        raise ValueError(f"unexpected loader columns: {list(df.columns)}")
    return df


def write_hybrid_csvs(
    clean: pd.DataFrame,
    filtered_ohlc: np.ndarray,
    *,
    full_path: Path,
    input_path: Path,
) -> None:
    """Write the diagnostic full CSV and the model-input CSV (schema == original)."""
    if filtered_ohlc.shape != (len(clean), 4):
        raise ValueError(f"filtered shape {filtered_ohlc.shape} != ({len(clean)}, 4)")

    full = clean.copy()
    full.columns = ["open", "high", "low", "close", "volume"]  # raw originals
    full["kalman_open"] = filtered_ohlc[:, 0]
    full["kalman_high"] = filtered_ohlc[:, 1]
    full["kalman_low"] = filtered_ohlc[:, 2]
    full["kalman_close"] = filtered_ohlc[:, 3]
    full["raw_volume"] = clean["volume"].to_numpy()
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(full_path, index_label="datetime")

    # Model input: open/high/low/close = filtered, volume = RAW (unchanged).
    model = pd.DataFrame(
        {
            "open": filtered_ohlc[:, 0],
            "high": filtered_ohlc[:, 1],
            "low": filtered_ohlc[:, 2],
            "close": filtered_ohlc[:, 3],
            "volume": clean["volume"].to_numpy(),
        },
        index=clean.index,
    )
    model.to_csv(input_path, index_label="datetime")


def build_runtime_config(
    base_config_path: str | Path,
    *,
    input_csv: Path,
    out_path: Path,
    model_name: str = "ppo_xauusd_kalman_1d",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Deep-copy the read-only 1d config and redirect data + all outputs under kalmanFilter/."""
    base = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)
    models = OUTPUTS / "models"
    diag = OUTPUTS / "diagnostics"

    cfg["_generated"] = {
        "note": "GENERATED RUNTIME CONFIG — not source config. Do not edit by hand.",
        "source_config": str(Path(base_config_path)),
        "purpose": "Kalman-enhanced PPO: filtered-OHLC input, outputs under kalmanFilter/.",
    }
    cfg["experiment"]["model_name"] = model_name
    cfg["experiment"]["baseline"] = "Kalman-enhanced PPO (OHLC filtered, Volume raw)"
    cfg["data"]["csv_path"] = str(input_csv)
    cfg["train"]["save_dir"] = str(models)
    cfg["train"]["device"] = device
    cfg["train"]["validation_log_dir"] = str(diag / "val")
    cfg["train"]["train_log_file"] = str(diag / "kalman_train.log")
    cfg["nautilus"]["model_path"] = str(models / model_name)
    cfg["nautilus"]["metrics_output"] = str(OUTPUTS / "metrics" / "ppo_kalman_nautilus_metrics.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg
