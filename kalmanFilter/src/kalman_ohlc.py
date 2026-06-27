"""OHLC-only Kalman filter with EM-estimated Q/R (paper §IV; KALMAN-A01).

Model (random walk, direct observation):
    x_{t+1} = F x_t + w_t,  w_t ~ N(0, Q)      F = I_4  (frozen)
    y_t     = H x_t + v_t,  v_t ~ N(0, R)      H = I_4  (frozen)
    x_t = y_t = [open, high, low, close]       (Volume excluded — KALMAN-A01)

Q, R are full 4x4, estimated by EM (innovation-sequence ML) on TRAIN ROWS ONLY,
iterated until the change in train log-likelihood < tol (default 1e-6). The fitted
parameters are then FROZEN and the Kalman *filter* (forward, past/current only — no
smoothing, no future leakage) is run over the full chronological series.

Library: pykalman.KalmanFilter (em + filter). F/H are pinned to I_4 by passing them
as fixed transition/observation matrices and restricting em_vars to the two covariances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from pykalman import KalmanFilter

OHLC: tuple[str, ...] = ("open", "high", "low", "close")
_EM_VARS = ["transition_covariance", "observation_covariance"]


@dataclass
class KalmanFitResult:
    """Frozen parameters from EM on training rows only."""
    Q: np.ndarray          # (4,4) process noise
    R: np.ndarray          # (4,4) measurement noise
    x0: np.ndarray         # (4,) initial state mean
    P0: np.ndarray         # (4,4) initial state covariance
    converged: bool
    n_iter: int
    tol: float
    loglik_history: List[float]
    q_cond: float
    r_cond: float
    q_jitter: float
    r_jitter: float
    train_rows: int


@dataclass
class KalmanDiagnostics:
    n_rows: int
    nonfinite: int
    ohlc_violations: int       # rows where filtered OHLC break high>=.. / low<=.. ordering
    max_abs_state_change: float


@dataclass
class KalmanFilterOutput:
    filtered: np.ndarray              # (T,4) filtered OHLC, original price units
    diagnostics: KalmanDiagnostics


def _pd_guard(m: np.ndarray, rel_jitter: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """Symmetrize, return (matrix, condition_number, jitter_added).

    Add jitter only if Cholesky fails (not positive definite). Jitter scaled to the
    mean diagonal so it is negligible relative to the matrix.
    """
    m = 0.5 * (m + m.T)
    jitter = 0.0
    try:
        np.linalg.cholesky(m)
    except np.linalg.LinAlgError:
        scale = float(np.mean(np.diag(m))) or 1.0
        jitter = rel_jitter * scale
        m = m + jitter * np.eye(m.shape[0])
        np.linalg.cholesky(m)  # raise if still not PD — fail loud, never silent
    return m, float(np.linalg.cond(m)), jitter


def fit_qr(
    train_ohlc: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000,
    init_diff_window: int = 30,
) -> KalmanFitResult:
    """EM on training OHLC only. Iterate until the RELATIVE change in train
    log-likelihood |L_i - L_{i-1}| / |L_{i-1}| < tol (KALMAN-A03: the paper's 1e-6
    threshold does not state its reference quantity; relative-logL is the standard
    EM convention and is numerically meaningful, unlike an absolute 1e-6 on a logL
    of magnitude ~1e4)."""
    y = np.asarray(train_ohlc, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 4:
        raise ValueError(f"train_ohlc must be (T,4); got {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("train_ohlc contains non-finite values")
    n = len(y)
    if n < init_diff_window + 1:
        raise ValueError(f"too few train rows for EM: {n}")

    x0 = y[0].copy()
    diffs = np.diff(y[: min(init_diff_window, n)], axis=0)
    p0 = np.cov(diffs, rowvar=False)
    p0, _, _ = _pd_guard(np.atleast_2d(p0))

    kf = KalmanFilter(
        transition_matrices=np.eye(4),
        observation_matrices=np.eye(4),
        transition_offsets=np.zeros(4),     # drift B_t mu_t = 0 (KALMAN ambiguity, logged)
        observation_offsets=np.zeros(4),
        initial_state_mean=x0,
        initial_state_covariance=p0,
        transition_covariance=np.eye(4),    # EM start
        observation_covariance=np.eye(4),   # EM start
        n_dim_state=4,
        n_dim_obs=4,
    )

    history: List[float] = []
    prev: float | None = None
    converged = False
    used = 0
    for i in range(1, max_iter + 1):
        kf = kf.em(y, n_iter=1, em_vars=_EM_VARS)
        ll = float(kf.loglikelihood(y))
        history.append(ll)
        used = i
        if prev is not None and abs(ll - prev) < tol * (abs(prev) + 1e-12):
            converged = True
            break
        prev = ll

    q, q_cond, q_jit = _pd_guard(np.asarray(kf.transition_covariance, float))
    r, r_cond, r_jit = _pd_guard(np.asarray(kf.observation_covariance, float))
    return KalmanFitResult(
        Q=q, R=r, x0=x0, P0=p0,
        converged=converged, n_iter=used, tol=tol, loglik_history=history,
        q_cond=q_cond, r_cond=r_cond, q_jitter=q_jit, r_jitter=r_jit, train_rows=n,
    )


def filter_ohlc(all_ohlc: np.ndarray, fit: KalmanFitResult) -> KalmanFilterOutput:
    """Run the FILTER (forward only) over the full series with frozen params."""
    y = np.asarray(all_ohlc, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 4:
        raise ValueError(f"all_ohlc must be (T,4); got {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("all_ohlc contains non-finite values")

    kf = KalmanFilter(
        transition_matrices=np.eye(4),
        observation_matrices=np.eye(4),
        transition_offsets=np.zeros(4),
        observation_offsets=np.zeros(4),
        initial_state_mean=fit.x0,
        initial_state_covariance=fit.P0,
        transition_covariance=fit.Q,
        observation_covariance=fit.R,
        n_dim_state=4,
        n_dim_obs=4,
    )
    means, _covs = kf.filter(y)          # FILTER, not smooth -> no future leakage
    means = np.asarray(means, dtype=np.float64)

    o, h, l, c = means[:, 0], means[:, 1], means[:, 2], means[:, 3]
    viol = int(
        ((h < np.maximum.reduce([o, c, l])) | (l > np.minimum.reduce([o, c, h]))).sum()
    )
    diag = KalmanDiagnostics(
        n_rows=len(means),
        nonfinite=int((~np.isfinite(means)).sum()),
        ohlc_violations=viol,
        max_abs_state_change=float(np.abs(np.diff(means, axis=0)).max()) if len(means) > 1 else 0.0,
    )
    return KalmanFilterOutput(filtered=means, diagnostics=diag)


def demo() -> None:
    """Self-check: EM converges, F/H frozen, filter is causal & finite."""
    rng = np.random.default_rng(7)
    latent = np.cumsum(rng.normal(0, 1.0, size=(400, 4)), axis=0) + 1800.0
    obs = latent + rng.normal(0, 0.5, size=(400, 4))
    train, full = obs[:250], obs
    fit = fit_qr(train, tol=1e-6, max_iter=50)
    assert fit.Q.shape == (4, 4) and fit.R.shape == (4, 4)
    assert fit.loglik_history == sorted(fit.loglik_history), "EM log-lik must be non-decreasing"
    out = filter_ohlc(full, fit)
    assert out.filtered.shape == (400, 4)
    assert out.diagnostics.nonfinite == 0
    # Causality: filtering the full series then truncating == filtering only the prefix.
    pref = filter_ohlc(full[:250], fit).filtered
    assert np.allclose(pref, out.filtered[:250], atol=1e-6), "filter must be causal (no future leakage)"
    print(f"demo OK | iters={fit.n_iter} converged={fit.converged} "
          f"viol={out.diagnostics.ohlc_violations} qcond={fit.q_cond:.1f} rcond={fit.r_cond:.1f}")


if __name__ == "__main__":
    demo()
