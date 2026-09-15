"""Calibrate the WUI spread model against observed structure loss (CAL FIRE DINS).

The WUI model (:mod:`vhagar.models.wui`) ships with physically-plausible default
parameters, the ember spotting distance/focus and the structure-to-structure
ignition probability, that have *not* been fit to real loss. This module turns the
prototype into a validated model: it runs :func:`vhagar.models.wui.wui_spread` per
fire, scores predicted-destroyed against DINS-observed-destroyed
(:mod:`vhagar.eval.wui`), grid-searches the parameters, and reports **leave-one-
fire-out** held-out F1 so the chosen parameters are shown to generalise across
fires rather than overfit one.

A :class:`WuiFire` bundles everything a single fire needs (ROS field, ignition
seed, wind, the structure set, and the DINS-derived destroyed labels aligned to
that structure set). Assemble one per calibration fire from DINS + building
footprints (see ``docs/17``), then::

    fires = [build_fire(...), ...]                 # your data assembly
    grid = default_param_grid()
    report = calibrate_lofo(fires, grid)
    print(report["mean_heldout_f1"], report["selected"])

Pure numpy/scipy; deterministic (the model returns probabilities/fixed points, not
random draws), so the calibration is reproducible.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from vhagar.eval.wui import StructureScore, score_structures
from vhagar.models.wui import wui_spread

__all__ = [
    "WuiFire",
    "default_param_grid",
    "predict_fire",
    "score_fire",
    "grid_search",
    "calibrate_lofo",
]


@dataclass(frozen=True, slots=True)
class WuiFire:
    """One fire's inputs and DINS-derived truth for WUI calibration.

    ``ros`` and ``burned_seed`` are ``[H, W]`` fields (rate of spread and the
    initial burning mask). ``struct_rows``/``struct_cols`` index the structures on
    that grid, and ``truth_destroyed`` is the aligned boolean of which were actually
    destroyed (from DINS, associated to this structure set with
    :func:`vhagar.eval.wui.match_points`).
    """

    name: str
    ros: np.ndarray
    burned_seed: np.ndarray
    struct_rows: np.ndarray
    struct_cols: np.ndarray
    truth_destroyed: np.ndarray
    wind_speed: float = 0.0
    wind_dir: float = 0.0
    horizon: float = 12.0
    dx: float = 1.0

    def __post_init__(self):
        n = np.asarray(self.struct_rows).size
        if np.asarray(self.struct_cols).size != n or np.asarray(self.truth_destroyed).size != n:
            raise ValueError(f"fire {self.name!r}: struct rows/cols/truth must be equal length")


#: The parameters that meaningfully move WUI loss. Keep the grid small: these are a
#: handful of global physical knobs, not a high-dimensional model, so coarse search
#: + leave-one-fire-out is the right, honest protocol.
_PARAM_KEYS = ("spotting_max_dist", "spotting_intensity", "struct_radius_cells", "struct_base_p")


def default_param_grid() -> dict[str, list[float]]:
    """A small default search grid over the WUI parameters."""
    return {
        "spotting_max_dist": [4.0, 8.0],
        "spotting_intensity": [1.0, 3.0, 6.0],
        "struct_radius_cells": [2.0, 3.0, 4.0],
        "struct_base_p": [0.5, 0.7, 0.9],
    }


def predict_fire(fire: WuiFire, params: dict) -> np.ndarray:
    """Run the WUI model on one fire with ``params`` and return destroyed mask."""
    out = wui_spread(
        fire.burned_seed, fire.ros, fire.struct_rows, fire.struct_cols,
        horizon=fire.horizon, wind_speed=fire.wind_speed, wind_dir=fire.wind_dir,
        dx=fire.dx,
        spotting_max_dist=params["spotting_max_dist"],
        spotting_intensity=params["spotting_intensity"],
        struct_radius_cells=params["struct_radius_cells"],
        struct_base_p=params["struct_base_p"],
    )
    return out["destroyed"]


def score_fire(fire: WuiFire, params: dict) -> StructureScore:
    """Confusion of predicted vs DINS-observed destroyed structures for one fire."""
    return score_structures(predict_fire(fire, params), fire.truth_destroyed)


def _mean_f1(fires, params: dict) -> float:
    """Mean F1 over fires, ignoring fires whose F1 is undefined (no positives)."""
    f1s = [score_fire(f, params).f1 for f in fires]
    finite = [v for v in f1s if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _iter_param_grid(grid: dict):
    keys = [k for k in _PARAM_KEYS if k in grid]
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo, strict=True))


def grid_search(fires, grid: dict | None = None) -> tuple[dict, float]:
    """Pick the parameter set maximising mean F1 over ``fires``.

    Returns ``(best_params, best_mean_f1)``. Ties break toward the first combo, so
    the grid order encodes a mild preference (put conservative values first).
    """
    grid = grid or default_param_grid()
    best_params, best_f1 = None, -1.0
    for params in _iter_param_grid(grid):
        f1 = _mean_f1(fires, params)
        if np.isfinite(f1) and f1 > best_f1:
            best_f1, best_params = f1, params
    if best_params is None:                      # no combo produced a defined F1
        best_params = next(_iter_param_grid(grid))
        best_f1 = float("nan")
    return best_params, best_f1


def calibrate_lofo(fires, grid: dict | None = None) -> dict:
    """Leave-one-fire-out calibration: fit params on the rest, score the held-out fire.

    For each fire, the parameters are chosen by :func:`grid_search` on *all other*
    fires and then scored on the held-out one, so the reported ``mean_heldout_f1``
    reflects generalisation, not fit-to-self. Also returns a ``selected`` parameter
    set (grid search over *all* fires) to actually deploy, plus per-fold detail.

    Requires >= 2 fires (LOFO is undefined for one). Returns a dict with
    ``folds`` (list of ``{name, f1, pod, far, params}``), ``mean_heldout_f1``,
    ``selected`` (params), and ``selected_f1`` (its mean F1 over all fires).
    """
    fires = list(fires)
    if len(fires) < 2:
        raise ValueError("leave-one-fire-out needs at least 2 fires")
    grid = grid or default_param_grid()
    folds = []
    for i, held in enumerate(fires):
        train = fires[:i] + fires[i + 1:]
        params, _ = grid_search(train, grid)
        s = score_fire(held, params)
        folds.append({"name": held.name, "f1": s.f1, "pod": s.pod, "far": s.far,
                      "params": params})
    heldout = [f["f1"] for f in folds if np.isfinite(f["f1"])]
    selected, selected_f1 = grid_search(fires, grid)
    return {
        "folds": folds,
        "mean_heldout_f1": float(np.mean(heldout)) if heldout else float("nan"),
        "n_fires": len(fires),
        "selected": selected,
        "selected_f1": selected_f1,
    }
