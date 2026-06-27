"""Gate C: fit Q/R on train rows only, filter OHLC, write hybrid CSVs + runtime config.

    python kalmanFilter/scripts/01_build_filtered_ohlcv.py [--max-iter 200] [--tol 1e-6]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_KF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_KF))

from src import pipeline, validation                         # noqa: E402
from src.kalman_ohlc import fit_qr, filter_ohlc, OHLC         # noqa: E402
from src.reports import dump_json, sha256_file, write_text    # noqa: E402

FULL_CSV = pipeline.OUTPUTS / "data" / "xauusd_1d_kalman_ohlc.csv"
INPUT_CSV = pipeline.OUTPUTS / "data" / "xauusd_1d_kalman_input.csv"
RUNTIME_CFG = pipeline.OUTPUTS / "diagnostics" / "effective_kalman_runtime_config.json"


def main() -> None:
    ap = pipeline.base_argparser("Build Kalman-filtered hybrid OHLCV (Gate C)")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--init-diff-window", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    pipeline.ensure_src_on_path()
    from rl_gold_trading.config import data_config, set_config_path
    set_config_path(args.config_path)
    dc = data_config()

    clean = pipeline.load_clean_ohlcv(args.config_path)
    train_end = pd.Timestamp(dc.train_end, tz="UTC")
    train_mask = clean.index <= train_end
    train_ohlc = clean.loc[train_mask, list(OHLC)].to_numpy(float)
    all_ohlc = clean[list(OHLC)].to_numpy(float)

    fit = fit_qr(train_ohlc, tol=args.tol, max_iter=args.max_iter,
                 init_diff_window=args.init_diff_window)
    out = filter_ohlc(all_ohlc, fit)
    filt = out.filtered

    ohlc_valid = validation.check_ohlc_validity(filt[:, 0], filt[:, 1], filt[:, 2], filt[:, 3])

    if args.dry_run:
        print(f"[dry-run] fit iters={fit.n_iter} converged={fit.converged} "
              f"viol={ohlc_valid['total']}")
        return

    pipeline.write_hybrid_csvs(clean, filt, full_path=FULL_CSV, input_path=INPUT_CSV)
    model_vol = pd.read_csv(INPUT_CSV)["volume"].to_numpy(float)
    vol_check = validation.check_volume_unchanged(clean["volume"].to_numpy(float), model_vol)

    pipeline.build_runtime_config(args.config_path, input_csv=INPUT_CSV,
                                  out_path=RUNTIME_CFG, device=args.device)

    diag = {
        "model": "random walk F=H=I_4, OHLC only (Volume excluded, KALMAN-A01)",
        "library": "pykalman.KalmanFilter (.em + .filter, no smoothing)",
        "em": {
            "tol": fit.tol, "converged": fit.converged, "n_iter": fit.n_iter,
            "loglik_first": fit.loglik_history[0], "loglik_last": fit.loglik_history[-1],
            "loglik_history": fit.loglik_history,
        },
        "Q": fit.Q.tolist(), "R": fit.R.tolist(),
        "Q_diag_std": np.sqrt(np.diag(fit.Q)).tolist(),
        "R_diag_std": np.sqrt(np.diag(fit.R)).tolist(),
        "Q_cond": fit.q_cond, "R_cond": fit.r_cond,
        "Q_jitter": fit.q_jitter, "R_jitter": fit.r_jitter,
        "x0": fit.x0.tolist(), "P0": fit.P0.tolist(),
        "train_rows": fit.train_rows, "total_rows": int(len(all_ohlc)),
        "test_rows_untouched": int((clean.index > train_end).sum()),
        "ohlc_validity": ohlc_valid,
        "volume_check": vol_check,
        "filter_diagnostics": {
            "n_rows": out.diagnostics.n_rows, "nonfinite": out.diagnostics.nonfinite,
            "ohlc_violations": out.diagnostics.ohlc_violations,
            "max_abs_state_change": out.diagnostics.max_abs_state_change,
        },
        "sha256": {
            "original_input": sha256_file(dc.csv_path),
            "filtered_full_output": sha256_file(FULL_CSV),
            "filtered_model_input": sha256_file(INPUT_CSV),
        },
    }
    dump_json(pipeline.OUTPUTS / "diagnostics" / "kalman_qr_diagnostics.json", diag)
    _write_report(diag, fit)

    print(f"Gate C: rows={diag['total_rows']} train={fit.train_rows} "
          f"converged={fit.converged}({fit.n_iter}it) Qcond={fit.q_cond:.1f} Rcond={fit.r_cond:.1f} "
          f"ohlc_viol={ohlc_valid['total']} vol_unchanged={vol_check['unchanged']}")


def _write_report(diag: dict, fit) -> None:
    q = np.array(diag["Q"]); r = np.array(diag["R"])
    fmt = lambda m: "\n".join("| " + " | ".join(f"{x:.6g}" for x in row) + " |" for row in m)
    md = f"""# KALMAN_QR_ESTIMATION_REPORT

**Gate C — Q/R estimation.** Source: `kalman_ohlc.py` (pykalman EM), train rows only.

## What was checked
EM fit of full 4×4 Q and R on **training rows only** ({fit.train_rows}), frozen, then
forward **filter** (no smoothing) over all {diag['total_rows']} rows. Test rows
({diag['test_rows_untouched']}) untouched during fitting.

## EM convergence
- threshold (|Δ train logL|): {fit.tol}
- converged: **{fit.converged}**  | iterations: **{fit.n_iter}**
- logL: {diag['em']['loglik_first']:.4f} → {diag['em']['loglik_last']:.4f}

## Process noise Q (4×4, full)
{fmt(q)}

condition number: {fit.q_cond:.4f} | jitter added: {fit.q_jitter:g}
process-noise std (√diag): {[round(x,5) for x in diag['Q_diag_std']]}

## Measurement noise R (4×4, full)
{fmt(r)}

condition number: {fit.r_cond:.4f} | jitter added: {fit.r_jitter:g}
measurement-noise std (√diag): {[round(x,5) for x in diag['R_diag_std']]}

## Initialization (logged assumption)
- x0 = first training OHLC = {[round(x,4) for x in diag['x0']]}
- P0 = cov of first {fit.train_rows and min(30, fit.train_rows)} OHLC first-differences

## OHLC validity (filtered output, reported not clipped)
- high<max(o,c,l): {diag['ohlc_validity']['high_violations']} | low>min(o,c,h): {diag['ohlc_validity']['low_violations']} | of {diag['ohlc_validity']['n']} rows

## Volume passthrough proof
- max|clean_raw_volume − model_input_volume| = **{diag['volume_check']['max_abs_diff']}** (must be 0)
- unchanged: **{diag['volume_check']['unchanged']}** | negative volume rows: {diag['volume_check']['negative_rows']}

## SHA256
- original input: `{diag['sha256']['original_input']}`
- filtered full:  `{diag['sha256']['filtered_full_output']}`
- model input:    `{diag['sha256']['filtered_model_input']}`

## Result / risk / next
- Gate C pass requires: finite filtered output, Q/R 4×4, train-only fit, no smoothing, volume unchanged.
- Risk: Low–Medium (EM convergence + OHLC-ordering violations are reported above).
- Next action: validate 22D feature matrix (script 02).
"""
    write_text(_KF / "docs" / "KALMAN_QR_ESTIMATION_REPORT.md", md)


if __name__ == "__main__":
    main()
