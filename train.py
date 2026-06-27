"""Entry point: PPO RAW baseline reproduction (no Kalman).

Reproduces the "PPO without Kalman filtering" baseline from Kili et al.
(IJACSA 16(11), 2025). See PPO_RAW_REPRODUCTION_REPORT.md.

Usage:
  python train.py --smoke              # fast 20k-step sanity run
  python train.py --device cpu         # force CPU (avoids SB3 MlpPolicy GPU warning)
  python train.py                      # full 500k-step reproduction (train+eval)
  python train.py --mode eval          # evaluate a saved model
  python train.py --timesteps 200000   # custom budget
  python train.py --config config/experiment.json
"""
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src_path = root / "src"
    sys.path.insert(0, str(src_path))


def main() -> None:
    _ensure_src_on_path()
    # pyrefly: ignore [missing-import]
    from rl_gold_trading.run import main as run_main

    run_main()


if __name__ == "__main__":
    main()
