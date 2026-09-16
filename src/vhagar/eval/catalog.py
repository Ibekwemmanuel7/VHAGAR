"""T4 batch event-catalog runner: many synthetic ignitions -> per-event spread -> EP curve.

Catastrophe modelling does not price on a single fire; it prices on a *catalogue* of
many events run under varied drivers, then aggregated into an exceedance-probability
(EP) curve. This module is the batch harness for exactly that on VHAGAR's own T4
spread engine:

    ignition catalogue (varied fuel + wind)
        -> per-event anisotropic arrival-time spread (the T4 physics core)
        -> per-event burned area + ground-up structure loss (the T5 loss chain)
        -> catalogue aggregate: AAL, OEP/AEP curves, return-period losses

It runs events in parallel (``concurrent.futures``) and its results are **invariant to
the worker count**: every event is fully seeded, so ``workers=1`` and ``workers=8``
produce identical losses (a property the tests check). The engine and loss chain are
the same code paths used elsewhere (``models.spread``, ``models.fuels``,
``models.loss``), so a catalogue number and a single-fire number are comparable.

Everything here is **SYNTHETIC**: the fuel fields and ignition/wind drivers are
generated, not observed, so the loss figures demonstrate the pipeline, not a real
portfolio. Swap ``event_fuel_codes`` for a LANDFIRE sampler and the structures for a
real exposure file to run it for real; the rest is unchanged.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vhagar.eval.wui_calibrate import wind_from_deg_to_grid
from vhagar.models.fuels import ros_from_fuel_codes
from vhagar.models.loss import (
    average_annual_loss,
    ground_up_loss,
    oep_curve,
    vulnerability_damage_ratio,
)
from vhagar.models.spread import anisotropic_arrival

__all__ = ["EventSpec", "make_catalog", "run_event", "run_catalog", "summarize",
           "landfire_sampler", "load_dins_fire", "run_real_fire", "run_real_catalog"]

#: Base FBFM40 code and base head ROS (m/min) per synthetic fuel scenario.
_FUEL_SCENARIOS = {
    "grass": (102, 8.0),      # GR: fast, flashy
    "shrub": (145, 5.0),      # SH: moderate-high
    "timber": (183, 2.5),     # TL: slow
    "mixed": (0, 5.0),        # patchwork (0 => laid out per cell below)
}


@dataclass(frozen=True)
class EventSpec:
    """One synthetic event: ignition drivers + exposure, all seed-reproducible."""

    event_id: int
    fuel: str
    wind_speed: float          # m/s
    wind_from_deg: float       # meteorological (degrees FROM)
    rate: float                # annual occurrence rate (events/year)
    grid: int = 120            # cells per side
    cell_m: float = 90.0
    horizon_min: float = 1440.0   # 24 h
    lb_max: float = 2.5
    n_structures: int = 200
    value_mean: float = 4.0e5     # per-structure exposure value ($)
    seed: int = 0
    meta: dict = field(default_factory=dict, compare=False)


def make_catalog(n_events: int = 300, seed: int = 0, base_lambda: float = 6.0,
                 grid: int = 120, horizon_min: float = 1440.0) -> list[EventSpec]:
    """Build a synthetic ignition catalogue spanning fuel and wind scenarios.

    ``base_lambda`` is the total annual event rate spread across the catalogue (so
    AAL scales sensibly). Fuel scenario, wind speed, and wind direction are drawn per
    event; each event carries its own seed for reproducible fuel layout and exposure.
    """
    rng = np.random.default_rng(seed)
    fuels = list(_FUEL_SCENARIOS)
    specs: list[EventSpec] = []
    for i in range(n_events):
        fuel = fuels[i % len(fuels)]
        ws = float(rng.uniform(1.5, 11.0))
        wd = float(rng.uniform(0.0, 360.0))
        specs.append(EventSpec(event_id=i, fuel=fuel, wind_speed=ws, wind_from_deg=wd,
                               rate=base_lambda / n_events, grid=grid,
                               horizon_min=horizon_min, seed=int(rng.integers(1, 2**31 - 1)),
                               meta={"scenario": fuel}))
    return specs


def event_fuel_codes(spec: EventSpec, rng: np.random.Generator) -> np.ndarray:
    """Synthetic FBFM40 code grid for an event: a base fuel with fuel breaks + a town.

    Non-burnable road/water breaks (93/98) and an urban cluster (NB1=91) are laid in
    so the wildland front respects fuel continuity, the same behaviour the real
    LANDFIRE-driven runs rely on. Replace this with a LANDFIRE window read for real.
    """
    n = spec.grid
    if spec.fuel == "mixed":
        base = rng.choice([102, 145, 183, 165], size=(n, n),
                          p=[0.35, 0.3, 0.2, 0.15]).astype(np.int64)
    else:
        base = np.full((n, n), _FUEL_SCENARIOS[spec.fuel][0], dtype=np.int64)
    # a few linear fuel breaks (roads) that real fires only cross by spotting
    for _ in range(rng.integers(1, 4)):
        r = int(rng.integers(0, n))
        if rng.random() < 0.5:
            base[r:r + 2, :] = 93
        else:
            base[:, r:r + 2] = 93
    # an urban cluster (the "town") in a random quadrant: NB1, structures live here
    ty, tx = int(rng.integers(n // 6, n - n // 6)), int(rng.integers(n // 6, n - n // 6))
    rad = max(4, n // 12)
    yy, xx = np.ogrid[:n, :n]
    town = (yy - ty) ** 2 + (xx - tx) ** 2 <= rad ** 2
    base[town] = 91
    return base


def _place_structures(spec: EventSpec, codes: np.ndarray, rng: np.random.Generator):
    """Place structures: most inside the urban cluster, some scattered in the WUI."""
    n = spec.grid
    town_r, town_c = np.where(codes == 91)
    rows = np.empty(spec.n_structures, dtype=np.int64)
    cols = np.empty(spec.n_structures, dtype=np.int64)
    n_town = int(spec.n_structures * 0.8)
    if town_r.size:
        pick = rng.integers(0, town_r.size, size=n_town)
        rows[:n_town], cols[:n_town] = town_r[pick], town_c[pick]
    else:
        n_town = 0
    m = spec.n_structures - n_town
    rows[n_town:] = rng.integers(0, n, size=m)
    cols[n_town:] = rng.integers(0, n, size=m)
    values = rng.lognormal(mean=np.log(spec.value_mean), sigma=0.5, size=spec.n_structures)
    return rows, cols, values


def _structure_intensity(arrival: np.ndarray, rows, cols, horizon: float) -> np.ndarray:
    """Per-structure hazard intensity in [0,1] from the arrival-time field.

    A structure's cell may be non-burnable (town), so the front arrives at its edge:
    sample the earliest arrival over the cell and its 8 neighbours. Intensity is the
    fraction of the horizon by which the front arrives (earlier front => more
    exposure); unreached structures get 0. This is the hazard the vulnerability curve
    then turns into a damage ratio.
    """
    H, W = arrival.shape
    best = np.full(rows.shape, np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            r = np.clip(rows + dy, 0, H - 1)
            c = np.clip(cols + dx, 0, W - 1)
            best = np.minimum(best, arrival[r, c])
    reached = best <= horizon
    inten = np.zeros(rows.shape)
    inten[reached] = np.clip((horizon - best[reached]) / horizon, 0.0, 1.0)
    return inten


def run_event(spec: EventSpec) -> dict:
    """Run one event through the T4 spread core and the T5 loss chain. Seed-pure."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(spec.seed)
    codes = event_fuel_codes(spec, rng)
    base_ros = _FUEL_SCENARIOS[spec.fuel][1]
    ros = ros_from_fuel_codes(codes, base=base_ros)
    seed_mask = np.zeros(codes.shape, bool)
    # ignite at a random burnable cell (a real ignition, not in the non-burnable town)
    burnable = np.argwhere(ros > 0)
    iy, ix = burnable[rng.integers(0, len(burnable))]
    seed_mask[iy, ix] = True
    wdir = wind_from_deg_to_grid(spec.wind_from_deg)
    arrival = anisotropic_arrival(ros, spec.wind_speed, wdir, seed_mask,
                                  dx=spec.cell_m, lb_max=spec.lb_max)
    burned = arrival <= spec.horizon_min
    cell_ha = (spec.cell_m ** 2) / 1.0e4
    burned_ha = float(burned.sum()) * cell_ha
    rows, cols, values = _place_structures(spec, codes, rng)
    inten = _structure_intensity(arrival, rows, cols, spec.horizon_min)
    dr = vulnerability_damage_ratio(inten)
    loss = ground_up_loss(dr, values)
    return {"event_id": spec.event_id, "fuel": spec.fuel, "rate": spec.rate,
            "wind_speed": spec.wind_speed, "wind_from_deg": spec.wind_from_deg,
            "burned_ha": burned_ha, "n_reached": int((inten > 0).sum()),
            "loss": loss, "runtime_ms": (time.perf_counter() - t0) * 1e3}


def run_catalog(specs: list[EventSpec], workers: int = 1) -> list[dict]:
    """Run the catalogue, optionally in parallel. Results are worker-count invariant."""
    if workers <= 1:
        results = [run_event(s) for s in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run_event, specs))
    results.sort(key=lambda r: r["event_id"])
    return results


# --------------------------------------------------------------------------- #
# Real-data path: real LANDFIRE fuels + real DINS ignitions, structures, values #
# --------------------------------------------------------------------------- #

_DESTROYED = "Destroyed (>50%)"
#: WUI params (documented faithful selection, docs/17). Real run, not re-fit here.
_REAL_WUI = {"spotting_max_dist": 12.0, "spotting_intensity": 6.0,
             "struct_radius_cells": 3.0, "struct_base_p": 0.9}


def landfire_sampler(path: str):
    """(lon_grid, lat_grid) -> FBFM40 code grid over a LANDFIRE raster (windowed read).

    Reads only the window covering the requested grid, reprojects grid lon/lat into
    the raster CRS, indexes it, and maps nodata / out-of-range codes to non-burnable
    (91). Same reader used by the WUI faithful calibration."""
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
        codes[(codes < 91) | (codes > 300)] = 91
        return codes

    return sampler


def _winsorize(v, q=0.98):
    v = np.asarray(v, dtype=np.float64)
    good = v[np.isfinite(v) & (v > 0)]
    cap = np.quantile(good, q) if good.size else 0.0
    return np.clip(np.nan_to_num(v, nan=0.0), 0.0, cap), float(cap)


def load_dins_fire(dins_csv: str, name: str):
    """Real structures for one incident from DINS: lat, lon, assessed value, destroyed.

    Values are the parcel 'Assessed Improved Value', winsorized at the 98th percentile
    (the documented outlier handling from the T5 loss run), and only finite-located
    structures are kept."""
    import pandas as pd

    df = pd.read_csv(dins_csv, low_memory=False)
    d = df[df["* Incident Name"].astype(str).str.strip() == name]
    if d.empty:
        raise ValueError(f"no DINS rows for incident {name!r}")
    lat = pd.to_numeric(d["Latitude"], errors="coerce").to_numpy()
    lon = pd.to_numeric(d["Longitude"], errors="coerce").to_numpy()
    val = pd.to_numeric(d["Assessed Improved Value (parcel)"], errors="coerce").to_numpy()
    destroyed = d["* Damage"].astype(str).str.strip().eq(_DESTROYED).to_numpy()
    ok = np.isfinite(lat) & np.isfinite(lon)
    val, _ = _winsorize(val[ok])
    return lat[ok], lon[ok], val, destroyed[ok]


def run_real_fire(cfg: dict, dins_csv: str, fuel_tif: str | None, rate: float,
                  horizon_mult: float = 4.0, lb_max: float = 2.5,
                  wui_params: dict | None = None, use_fuel: bool = True) -> dict:
    """One REAL fire: real ignition + real RAWS wind + real DINS structures with real
    assessed values, and (if ``use_fuel``) real LANDFIRE fuels. Returns modeled loss
    and the validation against what actually burned (actual destroyed count and value).

    ``use_fuel=True`` is the fully-real-inputs run; it under-predicts on the compact
    urban fires because a fuel-blocked front cannot enter non-burnable urban cells or
    jump non-fuel gaps that real fires cross by long-range spotting (documented honest
    negative, docs/17). ``use_fuel=False`` uses the validated uniform-ROS field
    (faithful mean held-out F1 ~0.75)."""
    from vhagar.eval.wui_calibrate import fire_from_points, predict_fire

    lat, lon, value, destroyed = load_dins_fire(dins_csv, cfg["name"])
    sampler = landfire_sampler(fuel_tif) if (use_fuel and fuel_tif) else None
    fire = fire_from_points(cfg["name"], lat, lon, destroyed,
                            ignition_lat=cfg["ignition_lat"], ignition_lon=cfg["ignition_lon"],
                            wind_speed_ms=cfg["wind_speed_ms"], wind_from_deg=cfg["wind_from_deg"],
                            fuel_sampler=sampler)
    p = dict(_REAL_WUI if wui_params is None else wui_params)
    # scale the horizon so the front can run the observed extent (documented mult)
    fire = replace_horizon(fire, fire.horizon * horizon_mult)
    modeled = predict_fire(fire, p)
    # recover the real cell size to report burned area in hectares
    cell_m = _real_cell_m(lat, lon, cfg, max_grid=600)
    from vhagar.models.spread import anisotropic_arrival
    arr = anisotropic_arrival(fire.ros, fire.wind_speed, fire.wind_dir, fire.burned_seed,
                              dx=fire.dx, lb_max=lb_max)
    burned_cells = int((arr <= fire.horizon).sum())
    burned_ha = burned_cells * (cell_m ** 2) / 1.0e4
    modeled_loss = float(value[modeled].sum())
    actual_loss = float(value[destroyed].sum())
    return {"event_id": cfg.get("event_id", 0), "fuel": ("real-landfire" if sampler else "real-uniform"),
            "rate": rate, "name": cfg["name"], "burned_ha": burned_ha, "loss": modeled_loss,
            "actual_loss": actual_loss, "n_structures": int(value.size),
            "modeled_destroyed": int(modeled.sum()), "actual_destroyed": int(destroyed.sum()),
            "runtime_ms": 0.0}


def replace_horizon(fire, horizon: float):
    """Return a copy of a WuiFire with a new horizon (WuiFire is frozen)."""
    import dataclasses
    return dataclasses.replace(fire, horizon=horizon)


def _real_cell_m(lat, lon, cfg, cell_m: float = 30.0, max_grid: int = 600) -> float:
    """Mirror fire_from_points' adaptive cell size, so burned area is in real metres."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lat0 = float(np.mean(np.append(lat, cfg["ignition_lat"])))
    lon0 = float(np.mean(np.append(lon, cfg["ignition_lon"])))
    mx = np.cos(np.radians(lat0)) * 111_320.0
    xs = (np.append(lon, cfg["ignition_lon"]) - lon0) * mx
    ys = (np.append(lat, cfg["ignition_lat"]) - lat0) * 110_540.0
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
    return max(cell_m, span / max_grid)


def run_real_catalog(config_path: str, dins_csv: str, fuel_tif: str | None,
                     base_lambda: float = 0.08, use_fuel: bool = True) -> list[dict]:
    """Run the REAL DINS-fire catalogue: real ignitions, structures, values, wind, and
    (if ``use_fuel``) real LANDFIRE fuels.

    Each configured fire is one real event; ``base_lambda`` is the total annual rate
    spread across them (illustrative, not a fitted frequency). Serial, because each
    LANDFIRE windowed read is already vectorised and there are only a handful of fires.
    """
    import json

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    rate = base_lambda / max(len(config), 1)
    results = []
    for i, cfg in enumerate(config):
        cfg = {**cfg, "event_id": i}
        results.append(run_real_fire(cfg, dins_csv, fuel_tif, rate=rate, use_fuel=use_fuel))
    return results


def _loss_at_return_periods(oep: dict, rps) -> dict:
    """Loss at target return periods from an OEP curve (monotone), by interpolation."""
    thr, o = oep["threshold"], oep["oep"]
    order = np.argsort(o)                       # oep increasing for np.interp
    xp, fp = o[order], thr[order]
    return {int(rp): float(np.interp(1.0 / rp, xp, fp)) for rp in rps}


def summarize(results: list[dict], wall_time_s: float = 0.0,
              return_periods=(10, 25, 50, 100, 250)) -> dict:
    """Per-event statistics + the catalogue EP aggregate (AAL, OEP, return periods)."""
    burned = np.array([r["burned_ha"] for r in results], dtype=np.float64)
    losses = np.array([r["loss"] for r in results], dtype=np.float64)
    rates = np.array([r["rate"] for r in results], dtype=np.float64)
    runtime = np.array([r["runtime_ms"] for r in results], dtype=np.float64)
    oep = oep_curve(losses, rates)
    by_fuel = {}
    for r in results:
        by_fuel.setdefault(r["fuel"], []).append((r["burned_ha"], r["loss"]))
    by_fuel = {k: {"n": len(v),
                   "mean_burned_ha": float(np.mean([a for a, _ in v])),
                   "mean_loss": float(np.mean([b for _, b in v]))}
               for k, v in sorted(by_fuel.items())}
    return {
        "n_events": len(results),
        "wall_time_s": float(wall_time_s),
        "burned_ha": {"mean": float(burned.mean()), "median": float(np.median(burned)),
                      "p95": float(np.percentile(burned, 95)), "max": float(burned.max())},
        "runtime_ms": {"mean": float(runtime.mean()), "p95": float(np.percentile(runtime, 95))},
        "aal": average_annual_loss(losses, rates),
        "return_period_loss": _loss_at_return_periods(oep, return_periods),
        "by_fuel": by_fuel,
        "synthetic": True,
    }
