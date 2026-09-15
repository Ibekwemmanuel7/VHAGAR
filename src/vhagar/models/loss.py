"""T5 catastrophe loss: hazard -> exposure -> vulnerability -> loss -> EP curve.

VHAGAR's upstream tiers produce the *hazard* (where fire burns and how severely:
T2/T4) and, in the WUI, which structures are destroyed (:mod:`vhagar.models.wui`).
Catastrophe (CAT) modelling for the (re)insurance market turns that into money and
then into the two curves an insurer actually prices on:

    hazard  ->  exposure (what is at risk, and its value)
            ->  vulnerability (damage ratio given the hazard at a structure)
            ->  ground-up loss per event
            ->  loss catalogue over an event set with annual rates
            ->  EP curve (exceedance probability) + AAL (average annual loss)

This module is the loss half of that chain. It is deliberately generic, it does
not reproduce any vendor platform (e.g. Aon's ELEMENTS); it demonstrates the
standard hazard->exposure->vulnerability->loss->EP pipeline on VHAGAR's own hazard
output, in the vocabulary insurers use: **AAL**, **OEP** (occurrence exceedance
probability), **AEP** (aggregate exceedance probability), and **return period**.

Two EP curves, both standard:

* **OEP** (occurrence): probability that the *largest single event* in a year
  exceeds a loss threshold. Closed form under a Poisson event process:
  ``OEP(x) = 1 - exp(-sum_i rate_i * [loss_i > x])``.
* **AEP** (aggregate): probability that the *annual total* loss exceeds a
  threshold. Estimated by Monte-Carlo simulation of Poisson event years (explicit
  RNG, so it is reproducible).

Average annual loss is rate-weighted: ``AAL = sum_i rate_i * loss_i``.

Pure numpy. The vulnerability curve is a documented logistic surrogate; a curve
fitted to DINS damage classes (see ``docs/17``) drops straight in.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "vulnerability_damage_ratio",
    "ground_up_loss",
    "structure_event_loss",
    "average_annual_loss",
    "oep_curve",
    "aep_curve",
]


def vulnerability_damage_ratio(intensity, midpoint: float = 0.5, steepness: float = 8.0) -> np.ndarray:
    """Damage ratio in [0, 1] as a logistic function of hazard intensity in [0, 1].

    ``intensity`` is a normalised per-structure hazard (e.g. flame/ember exposure,
    or a burn-probability). The logistic ``1 / (1 + exp(-steepness*(I - midpoint)))``
    is a standard, monotone vulnerability shape; ``midpoint`` is the intensity at
    50% damage and ``steepness`` its sharpness. Any curve fitted to DINS damage
    classes can replace it.

    >>> import numpy as np
    >>> dr = vulnerability_damage_ratio(np.array([0.0, 0.5, 1.0]))
    >>> bool(dr[0] < dr[1] < dr[2]) and bool(abs(dr[1] - 0.5) < 1e-9)
    True
    >>> bool(dr.min() >= 0.0 and dr.max() <= 1.0)
    True
    """
    x = np.asarray(intensity, dtype=np.float64)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def ground_up_loss(damage_ratio, exposure_value) -> float:
    """Ground-up loss for one event: sum of ``damage_ratio * exposure_value``.

    >>> ground_up_loss([0.0, 0.5, 1.0], [100.0, 200.0, 300.0])
    400.0
    """
    dr = np.asarray(damage_ratio, dtype=np.float64)
    val = np.asarray(exposure_value, dtype=np.float64)
    if dr.shape != val.shape:
        raise ValueError(f"damage_ratio {dr.shape} and exposure_value {val.shape} must align")
    return float(np.sum(dr * val))


def structure_event_loss(destroyed, exposure_value) -> float:
    """Ground-up loss when the WUI model reports binary destroyed structures.

    Destroyed structures take damage ratio 1, survivors 0, so the event loss is the
    total value of destroyed structures. For graded damage, use
    :func:`ground_up_loss` with a damage ratio per structure instead.

    >>> structure_event_loss([True, False, True], [100.0, 200.0, 300.0])
    400.0
    """
    d = np.asarray(destroyed, dtype=bool)
    val = np.asarray(exposure_value, dtype=np.float64)
    if d.shape != val.shape:
        raise ValueError(f"destroyed {d.shape} and exposure_value {val.shape} must align")
    return float(np.sum(val[d]))


def average_annual_loss(event_losses, rates) -> float:
    """Rate-weighted average annual loss ``AAL = sum_i rate_i * loss_i``.

    ``rates`` are annual occurrence rates (events/year) for each event in the
    catalogue. If ``rates`` is a scalar it applies to every event.

    >>> average_annual_loss([100.0, 400.0], [0.1, 0.01])
    14.0
    """
    losses = np.asarray(event_losses, dtype=np.float64)
    rates = np.broadcast_to(np.asarray(rates, dtype=np.float64), losses.shape)
    return float(np.sum(losses * rates))


def _default_thresholds(losses) -> np.ndarray:
    losses = np.asarray(losses, dtype=np.float64)
    pos = np.unique(losses[losses > 0])
    return np.concatenate([[0.0], pos]) if pos.size else np.array([0.0])


def oep_curve(event_losses, rates, thresholds=None) -> dict:
    """Occurrence exceedance probability curve (closed form, Poisson event set).

    ``OEP(x) = 1 - exp(-Lambda(x))`` where ``Lambda(x) = sum_i rate_i [loss_i > x]``
    is the annual rate of events whose loss exceeds ``x``. Returns a dict with
    ``threshold``, ``oep`` (annual prob the largest event exceeds the threshold),
    and ``return_period`` = 1 / oep (inf where oep = 0).

    >>> import numpy as np
    >>> c = oep_curve([100.0, 400.0], [0.1, 0.02])
    >>> # at x just below 100, both events qualify: OEP = 1 - exp(-(0.1+0.02))
    >>> bool(abs(c["oep"][0] - (1 - np.exp(-0.12))) < 1e-9)
    True
    >>> float(c["oep"][-1])           # above the largest loss, no exceedance
    0.0
    """
    losses = np.asarray(event_losses, dtype=np.float64)
    rates = np.broadcast_to(np.asarray(rates, dtype=np.float64), losses.shape)
    thr = _default_thresholds(losses) if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    lam = np.array([np.sum(rates[losses > x]) for x in thr])
    oep = 1.0 - np.exp(-lam)
    with np.errstate(divide="ignore"):
        rp = np.where(oep > 0, 1.0 / oep, np.inf)
    return {"threshold": thr, "oep": oep, "return_period": rp, "aal": average_annual_loss(losses, rates)}


def aep_curve(event_losses, rates, n_years: int = 100_000, seed: int = 0,
              thresholds=None) -> dict:
    """Aggregate exceedance probability curve by Monte-Carlo Poisson years.

    Simulates ``n_years``: in each year every event ``i`` occurs ``Poisson(rate_i)``
    times, and the year's aggregate loss is the sum of all occurrence losses. Returns
    ``threshold``, ``aep`` (fraction of years whose aggregate exceeds the threshold),
    ``return_period``, and the Monte-Carlo ``aal`` (which should match the analytic
    AAL up to sampling error, a built-in sanity check). Reproducible via ``seed``.
    """
    losses = np.asarray(event_losses, dtype=np.float64)
    rates = np.broadcast_to(np.asarray(rates, dtype=np.float64), losses.shape).astype(np.float64)
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=rates, size=(n_years, losses.size))
    annual = counts @ losses                          # [n_years] aggregate loss
    thr = _default_thresholds(losses) if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    aep = np.array([np.mean(annual > x) for x in thr])
    with np.errstate(divide="ignore"):
        rp = np.where(aep > 0, 1.0 / aep, np.inf)
    return {"threshold": thr, "aep": aep, "return_period": rp,
            "aal": float(annual.mean()), "aal_analytic": average_annual_loss(losses, rates)}
