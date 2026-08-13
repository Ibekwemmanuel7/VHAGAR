"""Dozier bispectral sub-pixel retrieval, and an honest account of why it fails.

The system
----------
Two channels, two unknowns: fire fraction ``p`` and fire temperature ``T_f``.

    L_MIR = p * B(3.9, T_f) + (1 - p) * B(3.9, T_b)
    L_TIR = p * B(11,  T_f) + (1 - p) * B(11,  T_b)

Looks well posed. It is not.

Why it is ill-conditioned
-------------------------
``p`` and ``T_f`` trade off along ``p * T_f^b ~ const``. The product is well
constrained; the factors individually are not. That is *precisely* why FRP --
which measures that product -- is robust while ``(p, T_f)`` is not.

The empirical record is blunt about this. GOES Dozier fire-area estimates
correlated with ASTER/ETM+ reference at **r = -0.22** (no skill), with retrieved
fire temperatures biased low against a 688-1153 K field range. Degrading the
background temperature characterisation by only **10 K inflated simulated GOES
FRP by 82%**. The operational consensus for 15 years has been: report FRP, not
``(p, T_f)``. LSA-SAF, MODIS C6, VIIRS and SLSTR all do exactly that.

VHAGAR therefore ships this retrieval **with its condition number attached**.
:func:`retrieve` returns a diagnostic that tells you when the answer is
meaningless, and :func:`is_trustworthy` encodes the gate. Do not strip it.

The real fix is more bands, not better inversion. With N > 2 channels the system
becomes overdetermined and multi-component (flaming + smouldering + background)
fits become identifiable -- which is what VIIRS Nightfire does with 9 bands, and
what hyperspectral VSWIR (EMIT/EnMAP/PRISMA) and 70 m ECOSTRESS make genuinely
tractable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vhagar.physics.planck import planck_radiance, planck_sensitivity_exponent

__all__ = ["DozierResult", "condition_number", "is_trustworthy", "retrieve"]

#: Physical prior on flaming-front temperature. Retrievals outside this are
#: rejected rather than reported.
T_FIRE_PRIOR_K = (600.0, 1400.0)


@dataclass(slots=True)
class DozierResult:
    """Retrieved sub-pixel state with its conditioning diagnostics."""

    fire_fraction: np.ndarray
    t_fire_k: np.ndarray
    #: Jacobian condition number. > ~100 means the (p, T_f) split is not
    #: identifiable from these two channels at this signal level.
    condition: np.ndarray
    #: Number of Newton iterations used; -1 where it did not converge.
    iterations: np.ndarray
    converged: np.ndarray

    def trustworthy(self, max_condition: float = 100.0) -> np.ndarray:
        return is_trustworthy(self, max_condition)

    def summary(self) -> dict[str, float]:
        ok = self.trustworthy()
        n = int(np.size(ok))
        return {
            "n": n,
            "converged_frac": float(np.mean(self.converged)) if n else float("nan"),
            "trustworthy_frac": float(np.mean(ok)) if n else float("nan"),
            "median_condition": float(np.nanmedian(self.condition)) if n else float("nan"),
        }


def _jacobian(p, t_f, t_b, lam_mir, lam_tir):
    """d(L_MIR, L_TIR) / d(p, T_f) at the current estimate."""
    b_mir_f = planck_radiance(lam_mir, t_f)
    b_tir_f = planck_radiance(lam_tir, t_f)
    b_mir_b = planck_radiance(lam_mir, t_b)
    b_tir_b = planck_radiance(lam_tir, t_b)

    # dL/dp
    j11 = b_mir_f - b_mir_b
    j21 = b_tir_f - b_tir_b
    # dL/dT_f = p * b(lambda,T) * B / T
    j12 = p * planck_sensitivity_exponent(lam_mir, t_f) * b_mir_f / t_f
    j22 = p * planck_sensitivity_exponent(lam_tir, t_f) * b_tir_f / t_f
    return j11, j12, j21, j22


def condition_number(fire_fraction, t_fire_k, t_background_k, lam_mir=3.9, lam_tir=11.0):
    """2x2 Jacobian condition number of the Dozier system.

    Large values mean ``p`` and ``T_f`` are not separately identifiable, however
    good your optimiser is. This is a property of the physics and the two
    chosen channels, not of the fitting method.
    """
    p = np.asarray(fire_fraction, dtype=np.float64)
    tf = np.asarray(t_fire_k, dtype=np.float64)
    tb = np.asarray(t_background_k, dtype=np.float64)
    j11, j12, j21, j22 = _jacobian(p, tf, tb, lam_mir, lam_tir)
    j = np.stack(
        [np.stack([j11, j12], axis=-1), np.stack([j21, j22], axis=-1)], axis=-2
    )
    # Scale rows so the condition number is not dominated by the ~1e3 radiance
    # ratio between the two channels -- we want the *shape* of the ill-posedness.
    scale = np.maximum(np.abs(j).max(axis=-1, keepdims=True), 1e-300)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.linalg.cond(j / scale)


def retrieve(
    l_mir,
    l_tir,
    t_background_k,
    lam_mir: float = 3.9,
    lam_tir: float = 11.0,
    t_fire_init_k: float = 800.0,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> DozierResult:
    """Solve the two-channel Dozier system by damped Newton iteration.

    Inputs are **surface-leaving** radiances -- apply the transmittance
    correction from :mod:`vhagar.physics.atmosphere` first.

    The retrieval is constrained to ``p in (0, 1]`` and
    ``T_f in [600, 1400] K``; solutions that leave the box are marked
    non-converged rather than reported, because an unphysical answer that looks
    like a number is worse than an admitted failure.

    >>> import numpy as np
    >>> from vhagar.physics.planck import mixed_pixel_radiance
    >>> p_true, tf_true, tb = 0.004, 900.0, 300.0
    >>> lm = mixed_pixel_radiance(3.9, p_true, tf_true, tb)
    >>> lt = mixed_pixel_radiance(11.0, p_true, tf_true, tb)
    >>> r = retrieve(lm, lt, tb)
    >>> bool(abs(float(r.fire_fraction) - p_true) / p_true < 0.05)
    True
    """
    lm = np.atleast_1d(np.asarray(l_mir, dtype=np.float64))
    lt = np.atleast_1d(np.asarray(l_tir, dtype=np.float64))
    tb = np.broadcast_to(np.asarray(t_background_k, dtype=np.float64), lm.shape).astype(np.float64)

    p = np.full(lm.shape, 1e-3)
    tf = np.full(lm.shape, float(t_fire_init_k))
    iters = np.full(lm.shape, -1, dtype=int)
    done = np.zeros(lm.shape, dtype=bool)

    lo, hi = T_FIRE_PRIOR_K
    for k in range(max_iter):
        b_mir_b = planck_radiance(lam_mir, tb)
        b_tir_b = planck_radiance(lam_tir, tb)
        f1 = p * planck_radiance(lam_mir, tf) + (1 - p) * b_mir_b - lm
        f2 = p * planck_radiance(lam_tir, tf) + (1 - p) * b_tir_b - lt

        newly = (np.abs(f1) < tol * np.maximum(np.abs(lm), 1e-12)) & (
            np.abs(f2) < tol * np.maximum(np.abs(lt), 1e-12)
        )
        iters = np.where(newly & ~done, k, iters)
        done |= newly
        if done.all():
            break

        j11, j12, j21, j22 = _jacobian(p, tf, tb, lam_mir, lam_tir)
        det = j11 * j22 - j12 * j21
        with np.errstate(divide="ignore", invalid="ignore"):
            dp = (-f1 * j22 + f2 * j12) / det
            dt = (-j11 * f2 + j21 * f1) / det
        dp = np.nan_to_num(dp, nan=0.0, posinf=0.0, neginf=0.0)
        dt = np.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0)

        # Damping: fire fraction moves in log space (it spans decades), and the
        # temperature step is capped so a bad Jacobian cannot fling the solution
        # out of the physical box in one iteration.
        step = np.where(done, 0.0, 1.0)
        p = np.clip(p + step * np.clip(dp, -0.5 * p - 1e-6, 2.0 * p + 1e-6), 1e-9, 1.0)
        tf = np.clip(tf + step * np.clip(dt, -200.0, 200.0), lo, hi)

    inside = (tf > lo + 1e-6) & (tf < hi - 1e-6) & (p > 1e-9) & (p <= 1.0)
    converged = done & inside
    cond = condition_number(p, tf, tb, lam_mir, lam_tir)

    return DozierResult(
        fire_fraction=np.where(converged, p, np.nan),
        t_fire_k=np.where(converged, tf, np.nan),
        condition=cond,
        iterations=iters,
        converged=converged,
    )


def is_trustworthy(result: DozierResult, max_condition: float = 100.0) -> np.ndarray:
    """Gate on convergence, the physical prior, and the condition number.

    **Expect this to reject essentially every two-channel retrieval, and
    understand that as the correct answer rather than a bug.** Typical
    condition numbers for realistic fires are 1e4 to 1e8: the MIR and TIR rows
    of the Jacobian are nearly parallel, so the ``(p, T_f)`` split amplifies
    noise enormously even though the *product* that FRP measures is stable.

    :func:`retrieve` will recover a synthetic state exactly from noise-free
    radiances -- and then fall apart under 0.2 K of sensor noise. That is the
    whole story of why the field reports FRP and not ``(p, T_f)``, and why
    GOES Dozier fire-area estimates correlated with reference data at r = -0.22.

    Report ``(p, T_f)`` only where this is True. Report FRP everywhere.
    """
    lo, hi = T_FIRE_PRIOR_K
    with np.errstate(invalid="ignore"):
        return (
            result.converged
            & np.isfinite(result.condition)
            & (result.condition < max_condition)
            & (result.t_fire_k >= lo)
            & (result.t_fire_k <= hi)
        )
