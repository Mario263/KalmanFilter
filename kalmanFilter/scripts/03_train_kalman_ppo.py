"""Gate E: train PPO on Kalman-filtered data via the UNCHANGED Raw PPO entry point.

Only difference vs Raw PPO: the runtime config's csv_path points at the hybrid
filtered-OHLC / raw-Volume CSV. All PPO hyperparameters/reward/costs are untouched.

    python kalmanFilter/scripts/03_train_kalman_ppo.py            # full (config timesteps)
    python kalmanFilter/scripts/03_train_kalman_ppo.py --smoke    # 20k sanity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_KF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_KF))

from src import pipeline                                  # noqa: E402
from src.reports import dump_json, sha256_file            # noqa: E402

RUNTIME_CFG = pipeline.OUTPUTS / "diagnostics" / "effective_kalman_runtime_config.json"
INPUT_CSV = pipeline.OUTPUTS / "data" / "xauusd_1d_kalman_input.csv"
MODELS = pipeline.OUTPUTS / "models"


def main() -> None:
    ap = pipeline.base_argparser("Train Kalman PPO (Gate E)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    if not RUNTIME_CFG.exists() or not INPUT_CSV.exists():
        raise SystemExit("Run 01_build_filtered_ohlcv.py first (missing runtime config / input CSV).")

    pipeline.ensure_src_on_path()
    from rl_gold_trading.run import main as run_main

    run_args = argparse.Namespace(
        config=str(RUNTIME_CFG), mode="train", timesteps=args.timesteps,
        csv=None, save_dir=str(MODELS), smoke=args.smoke, device=args.device,
    )
    if args.dry_run:
        print(f"[dry-run] would train with {run_args}")
        return

    run_main(run_args)

    model_zip = MODELS / "ppo_xauusd_kalman_1d.zip"
    dump_json(pipeline.OUTPUTS / "metrics" / "ppo_kalman_train_metrics.json", {
        "gate": "E",
        "data_path_proof": str(INPUT_CSV),
        "input_csv_sha256": sha256_file(INPUT_CSV),
        "runtime_config": str(RUNTIME_CFG),
        "model_path": str(model_zip),
        "model_exists": model_zip.exists(),
        "smoke": args.smoke,
        "timesteps_arg": args.timesteps,
        "device": args.device,
        "train_log": str(pipeline.OUTPUTS / "diagnostics" / "kalman_train.log"),
        "note": "Only difference vs Raw PPO is Kalman-filtered OHLC input; hyperparameters unchanged.",
    })
    print(f"Gate E: model_saved={model_zip.exists()} -> {model_zip}")


if __name__ == "__main__":
    main()
