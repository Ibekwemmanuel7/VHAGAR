"""Faithful WUI spread calibration on CAL FIRE DINS with real per-fire drivers.

Unlike the scoped run (structure-network only, centroid seed, no wind), this drives
the WUI model with each fire's **real ignition point and wind**, so the wind-driven
arrival-time front, not building adjacency, decides which structures the fire
reaches, then spotting + structure-to-structure fill in the rest. It runs a
leave-one-fire-out calibration and reports held-out POD/FAR/F1.

Data you supply per fire (a few numbers), in a JSON config (see
``dins_fires_config.example.json``):
  * ignition_lat / ignition_lon  - the fire's origin (CAL FIRE incident record / IR)
  * wind_speed_ms                - representative wind speed during the main run (RAWS)
  * wind_from_deg                - meteorological wind direction (degrees FROM; RAWS)
Structures and destroyed/survived labels come from DINS by ``* Incident Name``.

Performance: the wind-driven arrival field is invariant to the spotting/structure
parameters being searched, so it is solved **once per fire** and reused across the
whole grid search (``wui_spread(..., arrival=cached)``).

Not yet included (documented refinements): LANDFIRE fuels (uniform ROS -> a
homogeneous wind-driven ellipse) and terrain/slope. Ignition + wind are the
first-order drivers this adds over the scoped run.

Usage:
    python scripts/dins_wui_faithful_calibration.py POSTFIRE_MASTER_DATA_SHARE_*.csv \
        dins_fires_config.json --out dins_wui_faithful_summary.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.eval.wui import score_structures
from vhagar.eval.wui_calibrate import fire_from_points
from vhagar.models.spread import anisotropic_arrival
from vhagar.models.wui import wui_spread

DESTROYED = "Destroyed (>50%)"
MAX_STRUCT = 3000
LB_MAX_GRID = [1.6, 2.5]        # front length-to-breadth searched (see docs/17)


def _subsample(lat, lon, dest, cap, rng):
    if lat.size <= cap:
        return lat, lon, dest
    idx_d, idx_s = np.where(dest)[0], np.where(~dest)[0]
    frac = cap / lat.size
    d = rng.choice(idx_d, size=max(1, int(round(idx_d.size * frac))), replace=False)
    s = rng.choice(idx_s, size=max(1, int(round(idx_s.size * frac))), replace=False)
    keep = np.sort(np.concatenate([d, s]))
    return lat[keep], lon[keep], dest[keep]


def _fuel_sampler(path):
    """Build a (lon_grid, lat_grid) -> FBFM40 code-grid sampler over a LANDFIRE raster.

    Reads only the window covering the requested grid (works on the full CONUS FBFM40
    GeoTIFF without loading it all), reprojects grid lon/lat into the raster CRS, and
    indexes the window. Nodata / out-of-range codes become non-burnable (91)."""
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window

    ds = rasterio.open(path)

    def sampler(lon_g, lat_g):
        x0, y0, x1, y1 = transform_bounds("EPSG:4326", ds.crs,
                                          float(lon_g.min()), float(lat_g.min()),
                                          float(lon_g.max()), float(lat_g.max()))
        w = ds.window(x0, y0, x1, y1).round_offsets().round_lengths()
        w = Window(w.col_off - 2, w.row_off - 2, w.width + 4, w.height + 4)
        arr = ds.read(1, window=w, boundless=True, fill_value=91)
        inv = ~ds.window_transform(w)
        xs, ys = warp_transform("EPSG:4326", ds.crs, lon_g.ravel().tolist(), lat_g.ravel().tolist())
        cols, rows = inv * (np.asarray(xs), np.asarray(ys))
        cols = np.clip(np.round(cols).astype(int), 0, arr.shape[1] - 1)
        rows = np.clip(np.round(rows).astype(int), 0, arr.shape[0] - 1)
        codes = arr[rows, cols].reshape(lon_g.shape).astype(np.int64)
        codes[(codes < 91) | (codes > 300)] = 91          # nodata/invalid -> non-burnable
        return codes

    return sampler


def build_fires(df, config):
    dmg = df["* Damage"].astype(str).str.strip()
    df = df.assign(_destroyed=dmg.eq(DESTROYED))
    rng = np.random.default_rng(0)
    fires = []
    for cfg in config:
        d = df[df["* Incident Name"] == cfg["name"]]
        if d.empty:
            print(f"  [skip] no DINS rows for incident {cfg['name']!r}")
            continue
        lat = pd.to_numeric(d["Latitude"], errors="coerce").to_numpy()
        lon = pd.to_numeric(d["Longitude"], errors="coerce").to_numpy()
        dest = d["_destroyed"].to_numpy()
        ok = np.isfinite(lat) & np.isfinite(lon)
        lat, lon, dest = _subsample(lat[ok], lon[ok], dest[ok], MAX_STRUCT, rng)
        fs = _fuel_sampler(cfg["fuel_tif"]) if cfg.get("fuel_tif") else None
        fire = fire_from_points(cfg["name"], lat, lon, dest,
                                ignition_lat=cfg["ignition_lat"], ignition_lon=cfg["ignition_lon"],
                                wind_speed_ms=cfg["wind_speed_ms"], wind_from_deg=cfg["wind_from_deg"],
                                fuel_sampler=fs)
        # Precompute the wind-driven arrival per front length-to-breadth (the only front
        # driver we search); invariant to the spotting/structure params. The solver's
        # default LB=4 over-elongates the least-cost front into a sliver, so we search a
        # realistic range (see docs/17).
        arrivals = {lb: anisotropic_arrival(fire.ros, fire.wind_speed, fire.wind_dir,
                                            fire.burned_seed, dx=fire.dx, lb_max=lb)
                    for lb in LB_MAX_GRID}
        fires.append((fire, arrivals))
        tag = "fuel-aware" if fs is not None else "uniform ROS"
        print(f"  [ok] {cfg['name']}: {int(dest.sum())} destroyed / {dest.size} structures, "
              f"grid {fire.ros.shape}, {tag}")
    return fires


def faithful_grid():
    """A compact grid over spotting, structure, and front controls, kept small so a
    multi-fire leave-one-fire-out sweep is tractable. Spotting distance is included
    because it is what carries fire across gaps in the non-compact fires (Camp, Tubbs)."""
    return {"spotting_max_dist": [6.0, 12.0], "spotting_intensity": [3.0, 6.0],
            "struct_radius_cells": [2.0, 3.0], "struct_base_p": [0.5, 0.9],
            "lb_max": LB_MAX_GRID, "horizon_mult": [3.0, 6.0]}


def _score(fire, arrivals, params):
    arr = arrivals[params["lb_max"]]
    out = wui_spread(fire.burned_seed, fire.ros, fire.struct_rows, fire.struct_cols,
                     horizon=fire.horizon * params.get("horizon_mult", 1.0),
                     wind_speed=fire.wind_speed, wind_dir=fire.wind_dir,
                     dx=fire.dx, anisotropic=True, arrival=arr, struct_edge_cells=3.0,
                     spotting_max_dist=params["spotting_max_dist"],
                     spotting_intensity=params["spotting_intensity"],
                     struct_radius_cells=params["struct_radius_cells"],
                     struct_base_p=params["struct_base_p"])
    return score_structures(out["destroyed"], fire.truth_destroyed)


def _grid(grid):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo, strict=True))


def _best_params(fires, grid):
    best, best_f1 = None, -1.0
    for p in _grid(grid):
        f1s = [_score(f, a, p).f1 for f, a in fires]
        finite = [v for v in f1s if np.isfinite(v)]
        m = float(np.mean(finite)) if finite else float("nan")
        if np.isfinite(m) and m > best_f1:
            best_f1, best = m, p
    return best, best_f1


def calibrate_lofo(fires, grid):
    folds = []
    for i, (held, arr) in enumerate(fires):
        train = fires[:i] + fires[i + 1:]
        params, _ = _best_params(train, grid)
        s = _score(held, arr, params)
        folds.append({"name": held.name, "pod": s.pod, "far": s.far, "f1": s.f1, "params": params})
    sel, sel_f1 = _best_params(fires, grid)
    heldout = [f["f1"] for f in folds if np.isfinite(f["f1"])]
    return {"folds": folds, "mean_heldout_f1": float(np.mean(heldout)) if heldout else float("nan"),
            "selected": sel, "selected_f1": sel_f1, "n_fires": len(fires)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dins_csv", type=Path)
    ap.add_argument("config_json", type=Path)
    ap.add_argument("--out", type=Path, default=Path("dins_wui_faithful_summary.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.dins_csv, low_memory=False)
    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    fires = build_fires(df, config)
    if len(fires) < 2:
        raise SystemExit("need >= 2 configured fires with DINS coverage")

    rep = calibrate_lofo(fires, faithful_grid())
    rep["caveat"] = ("faithful on ignition+wind; uniform ROS (LANDFIRE fuels + slope are "
                     "documented refinements); structures subsampled per fire.")
    args.out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")

    print(f"\nmean leave-one-fire-out held-out F1: {rep['mean_heldout_f1']:.3f}")
    print(f"selected params: {rep['selected']}")
    for f in rep["folds"]:
        print(f"  {f['name']:<16} POD={f['pod']:.2f} FAR={f['far']:.2f} F1={f['f1']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
