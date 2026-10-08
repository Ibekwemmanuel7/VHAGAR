"""Case-control sampling from the parcel population, and the stage-2 fetch plan.

Cases are all population parcels covered by a facility installed in the label window.
Controls are a simple random sample (without replacement) of non-case parcels, drawn with
a fixed seed. Each control carries the inverse sampling fraction as its weight, so sums
over the sample estimate population totals. Parcels with a small, sub-threshold overlap
with a facility footprint are ambiguous (edge slivers, digitising offsets) and are left
out of both groups; their count is recorded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from vhagar.parcel.study.paths import (
    EQUAL_AREA,
    GEOGRAPHIC,
    N_CONTROLS,
    NFHL_CELL_DEG,
    SEED,
    STUDY_VERSION,
    WORK,
)

__all__ = ["load_population", "draw_sample", "fetch_plan", "dem_tile_name"]

_LIGHT = ["pid", "fips", "county", "area_ha", "lon", "lat", "cx", "cy", "is_case",
          "case_year", "case_facility", "overlap_case_ha"]
AMBIGUOUS_MIN_HA = 0.1


def load_population(work: Path = WORK) -> pd.DataFrame:
    """All county population tables without geometry (a few hundred MB at most)."""
    parts = []
    for p in sorted((work / "population").glob("*.parquet")):
        df = pd.read_parquet(p, columns=_LIGHT)
        df["fips"] = df["fips"].astype(str).str.zfill(5)  # county files mix int and str
        df["county"] = df["county"].astype(str)
        df["src"] = p.stem
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def draw_sample(pop: pd.DataFrame, n_controls: int = N_CONTROLS,
                seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Return the case-control sample (light columns) and the sampling record."""
    rng = np.random.default_rng(seed)
    case = pop["is_case"].to_numpy(bool)
    ambiguous = (~case) & (pop["overlap_case_ha"].to_numpy() >= AMBIGUOUS_MIN_HA)
    eligible = np.flatnonzero(~case & ~ambiguous)
    n_controls = min(n_controls, len(eligible))
    ctrl_idx = rng.choice(eligible, size=n_controls, replace=False)
    cases = pop[case].copy()
    cases["role"] = "case"
    cases["weight"] = 1.0
    ctrls = pop.iloc[np.sort(ctrl_idx)].copy()
    ctrls["role"] = "control"
    ctrls["weight"] = len(eligible) / n_controls
    sample = pd.concat([cases, ctrls], ignore_index=True)
    # Random halves for controls (independent train/test control pools); cases get -1.
    sample["control_half"] = -1
    is_ctrl = sample["role"].eq("control").to_numpy()
    sample.loc[is_ctrl, "control_half"] = rng.integers(0, 2, size=int(is_ctrl.sum()))
    record = {
        "study_version": STUDY_VERSION, "seed": seed,
        "population_parcels": int(len(pop)), "cases": int(case.sum()),
        "ambiguous_excluded": int(ambiguous.sum()),
        "eligible_noncase": int(len(eligible)), "controls": int(n_controls),
        "control_sampling_fraction": n_controls / len(eligible),
        "control_weight": len(eligible) / n_controls,
        "case_facilities": int(pop.loc[case, "case_facility"].nunique()),
        "population_prevalence": float(case.sum() / (case.sum() + len(eligible))),
    }
    return sample, record


def attach_geometry(sample: pd.DataFrame, work: Path = WORK) -> gpd.GeoDataFrame:
    """Load the geometries of the sampled parcels from their county files."""
    out = []
    for src, grp in sample.groupby("src"):
        g = gpd.read_parquet(work / "population" / f"{src}.parquet", columns=["pid", "geometry"],
                             filters=[("pid", "in", grp["pid"].tolist())])
        out.append(grp.merge(g, on="pid", how="left"))
    return gpd.GeoDataFrame(pd.concat(out, ignore_index=True), geometry="geometry",
                            crs=EQUAL_AREA)


def dem_tile_name(lon: float, lat: float) -> str:
    """USGS 1 arc-second tile covering (lon, lat): named by its north-west corner."""
    return f"n{math.floor(lat) + 1:02d}w{-math.floor(lon):03d}"


def fetch_plan(sample_geo: gpd.GeoDataFrame, cell_deg: float = NFHL_CELL_DEG) -> dict:
    """Floodplain query cells and elevation tiles needed for the sampled parcels."""
    b = sample_geo.to_crs(GEOGRAPHIC).bounds
    cells: set[tuple[float, float]] = set()
    tiles: set[str] = set()
    for x0, y0, x1, y1 in b[["minx", "miny", "maxx", "maxy"]].itertuples(index=False):
        for cx in np.arange(math.floor(x0 / cell_deg) * cell_deg, x1, cell_deg):
            for cy in np.arange(math.floor(y0 / cell_deg) * cell_deg, y1, cell_deg):
                cells.add((round(float(cx), 4), round(float(cy), 4)))
        for lon in (x0, x1):
            for lat in (y0, y1):
                tiles.add(dem_tile_name(lon, lat))
    return {
        "study_version": STUDY_VERSION,
        "nfhl_cells": [[x, y, round(x + cell_deg, 4), round(y + cell_deg, 4)]
                       for x, y in sorted(cells)],
        "dem_tiles": sorted(tiles),
    }


def write_plan(plan: dict, work: Path = WORK) -> Path:
    path = work / "fetch_plan.json"
    path.write_text(json.dumps(plan, indent=1))
    return path
