"""Load experiment / env / training settings from config/experiment.json.

All values trace to Kili et al., IJACSA 16(11), 2025 (PPO Raw baseline only).
Edit config/experiment.json to change the experiment; CLI flags override at runtime.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "experiment.json"

_cached: Optional[Dict[str, Any]] = None
_config_path: Optional[Path] = None


def set_config_path(path: str | Path) -> None:
    """Point loaders at a different experiment JSON (clears cache)."""
    global _cached, _config_path
    _config_path = Path(path)
    _cached = None


def _resolve_path() -> Path:
    env = os.environ.get("RL_EXPERIMENT_CONFIG")
    if env:
        return Path(env)
    return _config_path or DEFAULT_CONFIG_PATH


def load_raw_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the full experiment JSON."""
    global _cached
    if path is not None:
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    if _cached is None:
        p = _resolve_path()
        with p.open(encoding="utf-8") as f:
            _cached = json.load(f)
    return _cached


@dataclass
class DataConfig:
    """Paper Section IV.A."""
    csv_path: Optional[str] = None
    hf_dataset: str = "ZombitX64/xauusd-gold-price-historical-data-2004-2025"
    start: str = "2017-01-01"
    end: str = "2025-02-01"
    resample_rule: str = "1h"
    skip_resample: bool = False
    timeframe: str = "1h"
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    eval_start: str = "2023-01-02"
    eval_end: str = "2024-09-12"


@dataclass
class EnvConfig:
    """Paper Section IV.E / IV.F (Eq.22 reward)."""
    initial_capital: float = 100_000.0
    lots_per_trade: float = 0.1
    contract_oz_per_lot: float = 100.0
    commission_per_lot_round_trip: float = 7.0
    commission: float = 0.0
    spread: float = 0.0
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 0.5
    delta: float = 0.1


@dataclass
class TrainConfig:
    """Paper Section IV.G.2 / IV.H."""
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    lr_linear_decay: bool = True
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [512, 512, 256, 128])
    seed: int = 42
    save_dir: str = "models"
    device: str = "cuda"
    validation_enabled: bool = True
    validation_every_epoch: bool = True
    validation_log_dir: str = "logs"
    train_log_file: str = "logs/train.log"


@dataclass
class NautilusConfig:
    """Nautilus Trader backtest settings (inference only)."""
    starting_usd: float = 100_000.0
    fee: float = 0.00015
    venue: str = "SIM"
    symbol: str = "XAU/USD"
    default_leverage: int = 2
    model_path: str = "models/ppo_xauusd_raw"
    metrics_output: str = "nautilus/nautilus_metrics.json"


def _merge(section: str, cls, **overrides):
    raw = load_raw_config()
    base = {k: v for k, v in raw[section].items() if k in cls.__dataclass_fields__}
    base.update(overrides)
    return cls(**base)


def data_config(**overrides) -> DataConfig:
    return _merge("data", DataConfig, **overrides)


def env_config(**overrides) -> EnvConfig:
    return _merge("env", EnvConfig, **overrides)


def train_config(**overrides) -> TrainConfig:
    return _merge("train", TrainConfig, **overrides)


def nautilus_config(**overrides) -> NautilusConfig:
    return _merge("nautilus", NautilusConfig, **overrides)


def feature_order() -> List[str]:
    return list(load_raw_config()["features"]["order"])


def zscore_window() -> int:
    return int(load_raw_config()["features"]["zscore_window"])


def periods_per_year() -> int:
    return int(load_raw_config()["metrics"]["periods_per_year"])


def paper_target() -> Dict[str, float]:
    return dict(load_raw_config()["paper_target"])


def experiment_meta() -> Dict[str, Any]:
    return dict(load_raw_config()["experiment"])


def smoke_timesteps() -> int:
    return int(load_raw_config()["train"]["smoke_timesteps"])


# Backward-compatible module-level names (default config file).
FEATURE_ORDER: List[str] = feature_order()
ZSCORE_WINDOW: int = zscore_window()
PAPER_TARGET: Dict[str, float] = paper_target()

assert len(FEATURE_ORDER) == 22, "State must be exactly 22 dimensions (paper p.7)."
