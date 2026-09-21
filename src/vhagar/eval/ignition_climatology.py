"""T3 ignition on REAL data: a cause-agnostic occurrence *climatology* from
FPA-FOD, and an honest two-way evaluation of what it can and cannot do.

A per-cell/per-month occurrence frequency is the simplest real ignition layer:
"how often has this cell caught fire in this month, historically." It needs no
external covariates, so it can be fit tonight on real data. The interesting part
is the evaluation, and it is deliberately reported both ways because the two
answers are different and both are true:

* **Temporal holdout** (train early years, test the held-out later year). Fire
  locations persist year to year, so the climatology beats the base rate here:
  this is real, useful skill and the number we serve.
* **Spatial-block holdout** (grouped 5-degree blocks). A per-cell climatology has
  no information about a cell it never saw, so on unseen ground it collapses to
  the base rate. That is the honest limit: to predict ignition in *new* places
  you need covariates (weather, fuel, human factors), which is exactly the
  weather-driven model left as future work.

Metrics are AUPRC (average precision, the right summary under heavy class
imbalance) and the Brier score, each against the base-rate reference. The unit of
analysis is a (cell, month, year) cell-month over the fire-prone domain (cells
that ever burned in training), which is the honest candidate set: scoring against
the whole globe would inflate AUPRC with trivially-negative ocean cells.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.eval.metrics import average_precision, brier_score, skill_score

__all__ = [
    "read_fpa_fod_occurrence",
    "climatology_frequency",
    "evaluate_ignition_climatology",
]


def read_fpa_fod_occurrence(
    path: str | Path, *, table: str = "Fires",
    bbox: tuple[float, float, float, float] | None = None,
    years: Sequence[int] | None = None,
    lon_col: str = "LONGITUDE", lat_col: str = "LATITUDE", year_col: str = "FIRE_YEAR",
    doy_col: str = "DISCOVERY_DOY",
) -> pd.DataFrame:
    """Read FPA-FOD occurrences with month. Returns ``lon, lat, year, month``.

    ``month`` comes from ``DISCOVERY_DOY`` (day-of-year -> calendar month). Rows
    without a usable day-of-year are dropped. Needs a SQLite carrying
    ``DISCOVERY_DOY`` (the full FPA-FOD; the bundled demo lacks it)."""
    import sqlite3

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FPA-FOD SQLite not found at {path}. Download RDS-2013-0009 (SQLITE) "
            "from https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute(f"PRAGMA table_info('{table}')")}
        if doy_col not in cols:
            raise ValueError(
                f"{doy_col!r} not in {table}; monthly climatology needs the full "
                f"FPA-FOD, not the demo (columns present: {sorted(cols)})")
        sel = [f'"{lon_col}"', f'"{lat_col}"', f'"{year_col}"', f'"{doy_col}"']
        where, params = [f'"{doy_col}" IS NOT NULL'], []
        if years is not None:
            where.append(f'"{year_col}" IN ({",".join("?" * len(years))})')
            params += [int(y) for y in years]
        if bbox is not None:
            w, s, e, n = bbox
            where.append(f'"{lon_col}" BETWEEN ? AND ? AND "{lat_col}" BETWEEN ? AND ?')
            params += [w, e, s, n]
        sql = f'SELECT {",".join(sel)} FROM "{table}" WHERE ' + " AND ".join(where)
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    if not rows:
        raise ValueError("no occurrences with day-of-year in the requested bbox/years")
    doy = np.array([r[3] for r in rows], dtype=np.float64)
    month = (pd.to_datetime(
        {"year": [int(r[2]) for r in rows], "month": 1, "day": 1})
        + pd.to_timedelta(doy - 1.0, unit="D")).dt.month.to_numpy()
    return pd.DataFrame({"lon": [float(r[0]) for r in rows],
                         "lat": [float(r[1]) for r in rows],
                         "year": [int(r[2]) for r in rows], "month": month})


def _cell(lon, lat, cell_deg):
    return (np.floor(np.asarray(lon) / cell_deg).astype(np.int64),
            np.floor(np.asarray(lat) / cell_deg).astype(np.int64))


def climatology_frequency(train: pd.DataFrame, cell_deg: float, *, alpha: float = 1.0):
    """Fit the per-(cell, month) occurrence frequency on training years.

    Returns ``(freq, base_rate, cells_seen)`` where ``freq[(cx, cy, m)]`` is the
    fraction of training years in which that cell-month had >= 1 fire, Laplace-
    smoothed by ``alpha`` toward the base rate. ``cells_seen`` is the set of
    ``(cx, cy)`` that ever burned in training (the fire-prone domain)."""
    cx, cy = _cell(train["lon"], train["lat"], cell_deg)
    yrs = np.unique(train["year"])
    n_years = max(len(yrs), 1)
    present = set(zip(cx.tolist(), cy.tolist(), train["month"].astype(int).tolist(),
                      train["year"].astype(int).tolist(), strict=True))
    # count distinct years each (cell, month) burned
    from collections import defaultdict
    cm_years: dict[tuple[int, int, int], set] = defaultdict(set)
    for (a, b, m, y) in present:
        cm_years[(a, b, m)].add(y)
    base = len({(a, b, m, y) for (a, b, m, y) in present}) / max(
        len({(a, b) for (a, b, _m, _y) in present}) * 12 * n_years, 1)
    freq = {k: (len(v) + alpha * base) / (n_years + alpha) for k, v in cm_years.items()}
    cells_seen = {(a, b) for (a, b, _m) in freq}
    return freq, float(base), cells_seen


def _candidate_labels(test: pd.DataFrame, cells_seen, cell_deg):
    """Build (cell, month) candidate rows over the fire-prone domain x 12 months,
    labelled 1 if a fire occurred in the test year. Returns (keys, y_true)."""
    cx, cy = _cell(test["lon"], test["lat"], cell_deg)
    pos = set(zip(cx.tolist(), cy.tolist(), test["month"].astype(int).tolist(), strict=True))
    keys, y = [], []
    for (a, b) in sorted(cells_seen):
        for m in range(1, 13):
            keys.append((a, b, m))
            y.append(1 if (a, b, m) in pos else 0)
    return keys, np.asarray(y, dtype=np.int64)


def evaluate_ignition_climatology(
    path: str | Path, *, bbox: tuple[float, float, float, float] | None = None,
    years: Sequence[int] | None = None, cell_deg: float = 0.5, alpha: float = 1.0,
    block_deg: float = 5.0,
) -> dict:
    """Fit and score the occurrence climatology two honest ways (temporal +
    spatial). Returns AUPRC and Brier for the climatology and the base-rate
    reference under each split, plus provenance."""
    df = read_fpa_fod_occurrence(path, bbox=bbox, years=years)
    all_years = sorted(df["year"].unique())
    if len(all_years) < 2:
        raise ValueError("need >= 2 distinct fire years to hold one out in time")

    # ---- Temporal holdout: train on all but the last year, test the last ----
    test_year = all_years[-1]
    tr = df[df["year"] < test_year]
    te = df[df["year"] == test_year]
    freq, base, cells_seen = climatology_frequency(tr, cell_deg, alpha=alpha)
    keys, y = _candidate_labels(te, cells_seen, cell_deg)
    p_clim = np.array([freq.get(k, base) for k in keys], dtype=np.float64)
    p_base = np.full(y.shape, float(y.mean()) if y.size else 0.0)
    temporal = {
        "test_year": int(test_year), "train_years": [int(x) for x in all_years[:-1]],
        "n_candidates": int(y.size), "n_positive": int(y.sum()),
        "base_rate": float(y.mean()) if y.size else 0.0,
        "auprc_climatology": float(average_precision(y, p_clim)),
        "auprc_baserate": float(average_precision(y, p_base)),
        "brier_climatology": float(brier_score(y, p_clim)),
        "brier_baserate": float(brier_score(y, p_base)),
    }
    temporal["auprc_lift"] = skill_score(
        1 - temporal["auprc_climatology"], 1 - temporal["auprc_baserate"], perfect=0.0)

    # ---- Spatial-block holdout: climatology cannot see held-out cells ----
    cxb, cyb = _cell(df["lon"], df["lat"], block_deg)
    blocks = (cxb * 100_000 + cyb)
    ub = np.unique(blocks)
    auprc_sp, base_sp = [], []
    if len(ub) >= 2:
        for held in ub:
            tr_s = df[blocks != held]
            te_s = df[blocks == held]
            if len(te_s) < 5 or len(tr_s) < 20:
                continue
            fr, bs, seen = climatology_frequency(tr_s, cell_deg, alpha=alpha)
            # score over the held-out block's fire-prone cells (all unseen in train)
            seen_te = set(zip(*_cell(te_s["lon"], te_s["lat"], cell_deg), strict=True))
            k2, y2 = _candidate_labels(te_s, seen_te, cell_deg)
            if y2.sum() == 0:
                continue
            p2 = np.array([fr.get(k, bs) for k in k2], dtype=np.float64)
            auprc_sp.append(float(average_precision(y2, p2)))
            base_sp.append(float(y2.mean()))
    spatial = {
        "n_blocks_scored": len(auprc_sp),
        "auprc_climatology_mean": float(np.mean(auprc_sp)) if auprc_sp else float("nan"),
        "auprc_baserate_mean": float(np.mean(base_sp)) if base_sp else float("nan"),
        "note": ("Per-cell climatology has no signal on cells it never saw, so on "
                 "spatially held-out blocks it collapses toward the base rate. "
                 "Predicting ignition in NEW places needs covariates (future work)."),
    }
    return {"source": "FPA-FOD occurrences (real)", "n_fires": int(len(df)),
            "cell_deg": cell_deg, "years": [int(x) for x in all_years],
            "temporal_holdout": temporal, "spatial_block": spatial}
