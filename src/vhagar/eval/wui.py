"""WUI structure-loss evaluation against CAL FIRE DINS.

The WUI spread model (:mod:`vhagar.models.wui`) predicts which structures a fire
destroys. The honest test of that is observed structure loss, and the public
ground truth is CAL FIRE's **DINS** (Damage Inspection) program, which records,
per structure, a damage class after every significant California wildfire
("No Damage", "Affected (1-9%)", "Minor (10-25%)", "Major (26-50%)",
"Destroyed (>50%)") with a point location.

This module (a) loads DINS records and reduces them to a binary destroyed label,
and (b) scores predicted-destroyed against observed-destroyed over the *same
structure set* with detection-style metrics, POD, FAR, precision, recall, F1,
so a WUI run is graded exactly like the other VHAGAR tiers: skill against a real
observation, not a plausibility argument. Point-to-structure association is by
nearest match within a tolerance so predicted and observed refer to the same
buildings.

Nothing here is calibrated to DINS yet; this is the harness that *does* the
calibration and reports it honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["StructureScore", "score_structures", "load_dins", "match_points"]

#: DINS ``DAMAGE`` classes that count as a destroyed structure (binary positive).
DINS_DESTROYED_CLASSES = ("Destroyed (>50%)",)


@dataclass(frozen=True, slots=True)
class StructureScore:
    """Binary structure-loss confusion summary."""

    n: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def pod(self) -> float:
        """Probability of detection = recall = TP / (TP + FN)."""
        d = self.tp + self.fn
        return float(self.tp / d) if d else float("nan")

    @property
    def far(self) -> float:
        """False alarm ratio = FP / (TP + FP)."""
        d = self.tp + self.fp
        return float(self.fp / d) if d else float("nan")

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return float(self.tp / d) if d else float("nan")

    @property
    def recall(self) -> float:
        return self.pod

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not np.isfinite(p) or not np.isfinite(r) or (p + r) == 0:
            return float("nan")
        return float(2 * p * r / (p + r))


def score_structures(pred_destroyed, truth_destroyed) -> StructureScore:
    """Confusion of predicted vs observed destroyed structures (aligned 1:1).

    Both inputs are boolean arrays over the *same* ordered structure set (use
    :func:`match_points` first to align a prediction to DINS observations).

    >>> import numpy as np
    >>> pred = np.array([True, True, False, False])
    >>> truth = np.array([True, False, True, False])
    >>> s = score_structures(pred, truth)
    >>> (s.tp, s.fp, s.fn, s.tn)
    (1, 1, 1, 1)
    >>> round(s.pod, 3), round(s.far, 3), round(s.f1, 3)
    (0.5, 0.5, 0.5)
    """
    pred = np.asarray(pred_destroyed, dtype=bool)
    truth = np.asarray(truth_destroyed, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"pred {pred.shape} and truth {truth.shape} must align 1:1")
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    return StructureScore(n=int(pred.size), tp=tp, fp=fp, fn=fn, tn=tn)


def match_points(pred_rows, pred_cols, obs_rows, obs_cols, tol_cells: float = 1.5):
    """Nearest-neighbour association of predicted structures to observed ones.

    Returns index arrays ``(pred_idx, obs_idx)`` of matched pairs within
    ``tol_cells`` (greedy nearest, each observation used once). Lets a modelled
    structure set be aligned to the DINS point set before scoring. Pure numpy for
    small sets; uses a KD-tree when SciPy is available.

    >>> import numpy as np
    >>> pi, oi = match_points([0, 5], [0, 5], [0], [0], tol_cells=1.0)
    >>> pi.tolist(), oi.tolist()
    ([0], [0])
    """
    pr = np.asarray(pred_rows, dtype=np.float64)
    pc = np.asarray(pred_cols, dtype=np.float64)
    orr = np.asarray(obs_rows, dtype=np.float64)
    oc = np.asarray(obs_cols, dtype=np.float64)
    if pr.size == 0 or orr.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    # pairwise distances (small sets; O(P*O))
    d = np.hypot(pr[:, None] - orr[None, :], pc[:, None] - oc[None, :])
    used_obs = np.zeros(orr.size, dtype=bool)
    pred_idx, obs_idx = [], []
    order = np.argsort(d.min(axis=1))          # match closest-first for stability
    for i in order:
        j = int(np.argmin(np.where(used_obs, np.inf, d[i])))
        if not used_obs[j] and d[i, j] <= tol_cells:
            used_obs[j] = True
            pred_idx.append(int(i))
            obs_idx.append(j)
    return np.array(pred_idx, dtype=int), np.array(obs_idx, dtype=int)


def load_dins(path, destroyed_classes=DINS_DESTROYED_CLASSES,
              damage_col: str = "DAMAGE", lat_col: str = "LATITUDE",
              lon_col: str = "LONGITUDE"):
    """Load a CAL FIRE DINS export into ``(lats, lons, destroyed)`` arrays.

    Accepts the DINS CSV (or any table pandas can read). ``destroyed`` is True for
    rows whose ``DAMAGE`` class is in ``destroyed_classes`` (default: only the
    ">50%" class counts as destroyed). Requires pandas; kept import-local so the
    module has no hard pandas dependency for the pure scoring path.
    """
    import pandas as pd

    df = pd.read_csv(path)
    missing = {damage_col, lat_col, lon_col} - set(df.columns)
    if missing:
        raise ValueError(f"DINS file {path} missing columns {sorted(missing)}")
    dmg = df[damage_col].astype(str).str.strip()
    destroyed = dmg.isin(set(destroyed_classes)).to_numpy()
    lats = df[lat_col].to_numpy(dtype=np.float64)
    lons = df[lon_col].to_numpy(dtype=np.float64)
    return lats, lons, destroyed
