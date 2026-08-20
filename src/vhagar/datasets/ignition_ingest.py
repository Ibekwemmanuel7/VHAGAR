"""T3 real ignition ingest: turn a fire-occurrence database into the presence /
candidate design that :mod:`vhagar.datasets.danger` consumes.

Design follows the repo's ingest convention (a pure normaliser plus a thin
reader): :func:`normalize_occurrences` maps already-parsed rows onto the presence
schema and touches no files; :func:`read_fpa_fod_sqlite` is the only part that
does IO, reading the US FPA-FOD SQLite (the standard national fire-occurrence
record, either the 4th-edition ``STAT_CAUSE_DESCR`` or the 5th-edition
``NWCG_CAUSE_CLASSIFICATION`` schema).

The output is exactly what the ``t3-ignition`` CLI already expects: two parquet
tables, ``occurrence.parquet`` (presences) and ``candidates.parquet``
(background), each with ``id, lon, lat, year, stratum, cause`` plus one column
per covariate feature, ready for :func:`vhagar.datasets.danger.frames_to_records`
-> :func:`vhagar.datasets.danger.assemble_ignition_samples`.

Real covariates (dryness, fuel, wind, population, road density) come from rasters
that live outside this repo, so covariate lookup is injected as a callable
``covariate_fn(lon, lat, year) -> {name: array}``. When none is supplied a
labelled synthetic surface is used so the whole pipeline runs and is testable;
that surface is clearly marked and must be replaced with real layers before any
number is quoted.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_FEATURES",
    "classify_cause",
    "normalize_occurrences",
    "read_fpa_fod_sqlite",
    "assign_stratum",
    "build_candidate_grid",
    "synthetic_covariates",
    "attach_features",
    "write_ignition_parquet",
    "ingest_fpa_fod",
]

#: Covariate columns used by the synthetic surface and the t3-ignition demo.
DEFAULT_FEATURES: tuple[str, ...] = ("dryness", "fuel", "wind", "people", "roads")

#: FPA-FOD 4th-edition STAT_CAUSE_DESCR values that are lightning; everything else
#: coded is human. Unknown/undetermined map to "" (unlabelled background-eligible).
_LIGHTNING_DESCR = {"lightning"}
_UNKNOWN_DESCR = {"missing/undefined", "missing data/not specified/undetermined",
                  "undefined", "", "unknown"}


def classify_cause(raw: object) -> str:
    """Map a raw cause label to ``"human"`` / ``"lightning"`` / ``""``.

    Accepts both the 5th-edition classification (``Human`` / ``Natural`` /
    ``Missing data...``) and the 4th-edition free text (``Lightning`` + a dozen
    human causes)."""
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    if s in _UNKNOWN_DESCR:
        return ""
    if s in ("natural",) or s in _LIGHTNING_DESCR:
        return "lightning"
    if s in ("human",):
        return "human"
    # 4th-edition free text: anything named and not lightning is a human cause.
    return "human"


def normalize_occurrences(rows: Iterable[Mapping]) -> pd.DataFrame:
    """Pure row -> presence normaliser. Each row needs ``lon``, ``lat``, ``year``
    and a ``cause`` (raw); an ``id`` is assigned if absent. No IO."""
    recs = []
    for i, r in enumerate(rows):
        lon, lat = r.get("lon"), r.get("lat")
        year = r.get("year")
        if lon is None or lat is None or year is None:
            continue
        recs.append({
            "id": r.get("id", f"fire_{i}"),
            "lon": float(lon), "lat": float(lat), "year": int(year),
            "cause": classify_cause(r.get("cause")),
        })
    df = pd.DataFrame.from_records(recs)
    if df.empty:
        raise ValueError("no valid occurrence rows (need lon, lat, year)")
    return df


def read_fpa_fod_sqlite(
    path: str | Path, *, table: str = "Fires", bbox: tuple[float, float, float, float] | None = None,
    years: Sequence[int] | None = None,
    lon_col: str = "LONGITUDE", lat_col: str = "LATITUDE", year_col: str = "FIRE_YEAR",
    id_col: str = "FOD_ID", cause_col: str | None = None,
) -> pd.DataFrame:
    """Thin reader for the FPA-FOD SQLite. Returns a normalised presence frame.

    ``cause_col`` defaults to whichever of ``NWCG_CAUSE_CLASSIFICATION`` (5th ed.)
    or ``STAT_CAUSE_DESCR`` (4th ed.) exists in the table."""
    import sqlite3

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FPA-FOD SQLite not found at {path}. Download it from "
            "https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute(f"PRAGMA table_info('{table}')")}
        if not cols:
            raise ValueError(f"table {table!r} not found or empty in {path}")
        if cause_col is None:
            cause_col = next((c for c in ("NWCG_CAUSE_CLASSIFICATION", "STAT_CAUSE_DESCR")
                              if c in cols), None)
        sel = [f'"{id_col}"', f'"{lon_col}"', f'"{lat_col}"', f'"{year_col}"']
        if cause_col:
            sel.append(f'"{cause_col}"')
        where, params = [], []
        if years is not None:
            where.append(f'"{year_col}" IN ({",".join("?" * len(years))})')
            params += [int(y) for y in years]
        if bbox is not None:
            w, s, e, n = bbox
            where.append(f'"{lon_col}" BETWEEN ? AND ? AND "{lat_col}" BETWEEN ? AND ?')
            params += [w, e, s, n]
        sql = f'SELECT {",".join(sel)} FROM "{table}"'
        if where:
            sql += " WHERE " + " AND ".join(where)
        cur = con.execute(sql, params)
        raw = cur.fetchall()
    finally:
        con.close()
    rows = [{"id": r[0], "lon": r[1], "lat": r[2], "year": r[3],
             "cause": (r[4] if cause_col else None)} for r in raw]
    return normalize_occurrences(rows)


def assign_stratum(lon, lat, block_deg: float = 5.0) -> np.ndarray:
    """Coarse spatial stratum id from a lon/lat block grid (a stand-in for an
    ecoregion label; pass real ecoregions when you have them)."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    bx = np.floor(lon / block_deg).astype(np.int64)
    by = np.floor(lat / block_deg).astype(np.int64)
    return bx * 100_000 + by


def synthetic_covariates(lon, lat, year, *, seed: int = 0) -> dict[str, np.ndarray]:
    """A LABELLED synthetic covariate surface (not real data). Smoothly varying
    in space so nearby cells correlate, with a mild people/roads confounder that
    the target-group / stratified sampling is meant to defuse."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=5)
    dryness = 0.5 + 0.4 * np.sin(np.radians(lat) * 3 + phase[0])
    fuel = 0.5 + 0.4 * np.cos(np.radians(lon) * 2 + phase[1])
    wind = 0.5 + 0.3 * np.sin(np.radians(lon + lat) + phase[2])
    people = np.clip(0.5 + 0.5 * np.cos(np.radians(lat) * 5 + phase[3]), 0, 1)
    roads = np.clip(people * 0.7 + 0.2 * np.sin(np.radians(lon) * 4 + phase[4]), 0, 1)
    return {"dryness": dryness, "fuel": fuel, "wind": wind, "people": people, "roads": roads}


def attach_features(
    df: pd.DataFrame, covariate_fn: Callable | None, *, seed: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate covariates at each row's lon/lat/year and attach them as columns.
    Returns ``(df_with_features, feature_names)``."""
    fn = covariate_fn or (lambda lo, la, yr: synthetic_covariates(lo, la, yr, seed=seed))
    feats = fn(df["lon"].to_numpy(), df["lat"].to_numpy(), df["year"].to_numpy())
    out = df.copy()
    for name, vals in feats.items():
        out[name] = np.asarray(vals, dtype=np.float64)
    return out, list(feats.keys())


def build_candidate_grid(
    bbox: tuple[float, float, float, float], *, cell_deg: float = 0.25,
    years: Sequence[int], covariate_fn: Callable | None = None,
    block_deg: float = 5.0, seed: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """Regular background grid of candidate cells across ``bbox`` x ``years``.

    Every cell is an ignition-eligible location; the presence set marks the ones
    that actually burned. Returns ``(candidate_df, feature_names)`` with columns
    ``id, lon, lat, year, stratum`` plus covariates."""
    w, s, e, n = bbox
    lons = np.arange(w + cell_deg / 2, e, cell_deg)
    lats = np.arange(s + cell_deg / 2, n, cell_deg)
    if lons.size == 0 or lats.size == 0:
        raise ValueError("empty candidate grid: check bbox and cell_deg")
    gy, gx = np.meshgrid(lats, lons, indexing="ij")
    base = pd.DataFrame({"lon": gx.ravel(), "lat": gy.ravel()})
    frames = []
    for yr in years:
        f = base.copy()
        f["year"] = int(yr)
        frames.append(f)
    grid = pd.concat(frames, ignore_index=True)
    grid.insert(0, "id", [f"cell_{i}" for i in range(len(grid))])
    grid["stratum"] = assign_stratum(grid["lon"], grid["lat"], block_deg=block_deg)
    grid, feats = attach_features(grid, covariate_fn, seed=seed)
    return grid, feats


def write_ignition_parquet(
    presence: pd.DataFrame, candidates: pd.DataFrame, out_dir: str | Path,
) -> dict[str, Path]:
    """Write the two tables the ``t3-ignition`` CLI reads. Returns their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    occ_p, cand_p = out / "occurrence.parquet", out / "candidates.parquet"
    presence.to_parquet(occ_p)
    candidates.to_parquet(cand_p)
    return {"occurrence": occ_p, "candidates": cand_p}


def ingest_fpa_fod(
    sqlite_path: str | Path, *, bbox: tuple[float, float, float, float],
    years: Sequence[int], out_dir: str | Path, cell_deg: float = 0.25,
    block_deg: float = 5.0, covariate_fn: Callable | None = None, seed: int = 0,
) -> dict:
    """End-to-end: read FPA-FOD presences, build the candidate background, attach
    the same covariates to both, and write the parquet pair. Returns a summary."""
    presence = read_fpa_fod_sqlite(sqlite_path, bbox=bbox, years=years)
    presence["stratum"] = assign_stratum(presence["lon"], presence["lat"], block_deg=block_deg)
    presence, pfeat = attach_features(presence, covariate_fn, seed=seed)
    candidates, cfeat = build_candidate_grid(
        bbox, cell_deg=cell_deg, years=years, covariate_fn=covariate_fn,
        block_deg=block_deg, seed=seed)
    if set(pfeat) != set(cfeat):
        raise ValueError("presence and candidate covariates disagree; use one covariate_fn")
    paths = write_ignition_parquet(presence, candidates, out_dir)
    return {"features": pfeat, "n_presence": len(presence), "n_candidates": len(candidates),
            "paths": {k: str(v) for k, v in paths.items()},
            "synthetic_covariates": covariate_fn is None}
