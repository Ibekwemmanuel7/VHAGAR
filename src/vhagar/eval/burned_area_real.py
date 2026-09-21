"""T3 burned area on REAL data: the conditional fire-size distribution fit on
FPA-FOD ``FIRE_SIZE``.

The synthetic ``burned_area`` scenario exists to *show* the method (log-space
quantile boosting + a GPD tail, scored with CRPS/pinball). This module runs the
same estimator on real fire sizes from the FPA-FOD SQLite (Short, RDS-2013-0009),
so the numbers are earned rather than asserted.

Honest scope, stated plainly so it can be defended to a fire scientist:

* The **target** is real: ``FIRE_SIZE`` (final mapped/administrative size, acres,
  converted to hectares) for every fire in the requested bbox/years.
* The **features are only what FPA-FOD itself carries and cannot leak location**:
  ignition cause (lightning vs human) and season (day-of-year, encoded as its
  sine and cosine). Weather / fuel / topography covariates that would drive a
  stronger model need external rasters (GridMET, LANDFIRE, ERA5) and are future
  work; leaving them out keeps this an honest lower bound, not an inflated score.
* ``lon``/``lat`` are **excluded from the features** (the T1 leakage lesson) and
  used only to build the 5-degree spatial blocks for grouped cross-validation, so
  a held-out block is genuinely unseen ground.
* The reference is the marginal size **climatology**; a model that does not beat
  it on CRPS has learned nothing, and we report that outcome rather than hide it.

The heavy lifting (``BurnedAreaModel``, ``evaluate_expected_ba``) is reused
unchanged from :mod:`vhagar.eval.burned_area`; this module only supplies the real
data and the leak-free feature builder.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.datasets.ignition_ingest import classify_cause
from vhagar.eval.burned_area import BurnedAreaModel, evaluate_expected_ba

__all__ = [
    "ACRES_TO_HA",
    "read_fpa_fod_sizes",
    "build_size_features",
    "evaluate_real_burned_area",
    "fit_real_burned_area_model",
]

ACRES_TO_HA = 0.404685642


def read_fpa_fod_sizes(
    path: str | Path, *, table: str = "Fires",
    bbox: tuple[float, float, float, float] | None = None,
    years: Sequence[int] | None = None,
    lon_col: str = "LONGITUDE", lat_col: str = "LATITUDE", year_col: str = "FIRE_YEAR",
    size_col: str = "FIRE_SIZE", doy_col: str = "DISCOVERY_DOY",
    id_col: str = "FOD_ID", cause_col: str | None = None,
) -> pd.DataFrame:
    """Read FPA-FOD fires with their sizes into a tidy frame.

    Returns columns ``id, lon, lat, year, doy, cause, area_ha`` with one row per
    fire that has a positive ``FIRE_SIZE``. ``area_ha`` is ``FIRE_SIZE`` (acres)
    converted to hectares. ``doy`` is the discovery day-of-year when the column
    exists, else ``NaN``. Requires the full FPA-FOD SQLite (the bundled demo has
    no ``FIRE_SIZE`` column and will raise a clear error).
    """
    import sqlite3

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FPA-FOD SQLite not found at {path}. Download RDS-2013-0009 (SQLITE) "
            "from https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute(f"PRAGMA table_info('{table}')")}
        if not cols:
            raise ValueError(f"table {table!r} not found or empty in {path}")
        if size_col not in cols:
            raise ValueError(
                f"{size_col!r} not in {table}; this needs the full FPA-FOD, not the "
                f"demo (columns present: {sorted(cols)})")
        if cause_col is None:
            cause_col = next((c for c in ("NWCG_CAUSE_CLASSIFICATION", "STAT_CAUSE_DESCR")
                              if c in cols), None)
        have_doy = doy_col in cols
        sel = [f'"{id_col}"', f'"{lon_col}"', f'"{lat_col}"', f'"{year_col}"', f'"{size_col}"']
        sel.append(f'"{doy_col}"' if have_doy else "NULL")
        sel.append(f'"{cause_col}"' if cause_col else "NULL")
        where, params = [f'"{size_col}" > 0'], []
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
        raise ValueError("no fires with positive FIRE_SIZE in the requested bbox/years")
    recs = [{"id": r[0], "lon": float(r[1]), "lat": float(r[2]), "year": int(r[3]),
             "area_ha": float(r[4]) * ACRES_TO_HA,
             "doy": (float(r[5]) if r[5] is not None else np.nan),
             "cause": classify_cause(r[6])} for r in rows]
    return pd.DataFrame.from_records(recs)


def build_size_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Leak-free real features for the conditional size model.

    ``X`` columns: ``lightning`` (1 for natural/lightning cause, else 0),
    ``sin_doy``/``cos_doy`` (season as a smooth cycle). Missing day-of-year is
    encoded as (0, 0). Returns ``(X, area_ha, year, lon, lat, feature_names)``;
    ``lon``/``lat`` are returned only for spatial blocking, never in ``X``.
    """
    doy = df["doy"].to_numpy(dtype=np.float64)
    ang = 2.0 * np.pi * (doy - 1.0) / 365.0
    sin_doy = np.where(np.isfinite(doy), np.sin(ang), 0.0)
    cos_doy = np.where(np.isfinite(doy), np.cos(ang), 0.0)
    lightning = (df["cause"].to_numpy() == "lightning").astype(np.float64)
    X = np.column_stack([lightning, sin_doy, cos_doy])
    return (X, df["area_ha"].to_numpy(dtype=np.float64), df["year"].to_numpy(dtype=np.int64),
            df["lon"].to_numpy(dtype=np.float64), df["lat"].to_numpy(dtype=np.float64),
            ["lightning", "sin_doy", "cos_doy"])


def evaluate_real_burned_area(
    path: str | Path, *, bbox: tuple[float, float, float, float] | None = None,
    years: Sequence[int] | None = None, n_folds: int = 4, seed: int = 0,
    use_tail: bool = True,
) -> dict:
    """End-to-end real burned-area evaluation: read FPA-FOD sizes, build leak-free
    features, and run the blocked-CV CRPS/pinball evaluation against the size
    climatology. Returns the metrics dict plus data provenance."""
    df = read_fpa_fod_sizes(path, bbox=bbox, years=years)
    X, area, year, lon, lat, fn = build_size_features(df)
    res = evaluate_expected_ba(X, area, lon, lat, n_folds=n_folds, seed=seed, use_tail=use_tail)
    res.update({
        "source": "FPA-FOD FIRE_SIZE (real)",
        "n_fires": int(len(df)),
        "features": fn,
        "median_ha": float(np.median(area)),
        "p95_ha": float(np.quantile(area, 0.95)),
        "max_ha": float(np.max(area)),
        "years": (sorted(set(int(y) for y in year)) if years is None else list(years)),
        "note": ("Real fire sizes; features are cause + season only (no weather/fuel "
                 "covariates yet). CRPS skill vs climatology is the honest verdict."),
    })
    return res


def fit_real_burned_area_model(
    path: str | Path, *, bbox: tuple[float, float, float, float] | None = None,
    years: Sequence[int] | None = None, seed: int = 0, use_tail: bool = True,
) -> tuple[BurnedAreaModel, list[str], dict]:
    """Fit the conditional size model on ALL real fires (for serving), returning
    ``(model, feature_names, summary)``. Predict with ``build_size_features`` rows."""
    df = read_fpa_fod_sizes(path, bbox=bbox, years=years)
    X, area, _year, _lon, _lat, fn = build_size_features(df)
    model = BurnedAreaModel(seed=seed, use_tail=use_tail).fit(X, area)
    summary = {"n_fires": int(len(df)), "features": fn, "median_ha": float(np.median(area)),
               "p95_ha": float(np.quantile(area, 0.95)), "max_ha": float(np.max(area))}
    return model, fn, summary
