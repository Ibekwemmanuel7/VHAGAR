"""Stage-1 physics features, the cheapest large win, and the guard that keeps it.

The premise
-----------
Wildfire ML models memorise geography. Published result: raw lat/lon supplied
**88.9% of model split-gain** in a FIRMS classification task, and F1 fell from
0.985 (random split) to 0.767 (event-aware) to 0.627 (5-degree spatial block).
Dropping coordinates *raised* spatial-block F1 to 0.818.

The transferable substitute for "detections here are usually industrial" is
physics: geometry, atmospheric state, surface emissivity, spectral shape and
temporal persistence. That is what this module builds.

:func:`build_features` refuses to emit a coordinate feature. The guard is not
advisory -- :data:`FORBIDDEN_FEATURES` is enforced and tested.

Expected impact, honestly ordered
---------------------------------
Large:    glint angle (solar farms, specular reflectors, water glint at once);
          temporal persistence (flares, industrial, volcanic, *and* newly
          commissioned sources a frozen mask cannot cover); emissivity + NDVI
          (desert margins).
Moderate: SWIR ratios where the sensor has them -- the flare colour-temperature
          discriminant (flares ~1750 K vs biomass ~1000 K).
Small for false alarms, large for FRP accuracy: explicit transmittance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vhagar.physics.atmosphere import air_mass_factor, transmittance_mir
from vhagar.physics.geometry import (
    day_of_year_encoding,
    glint_angle_deg,
    local_solar_time_hours,
    pixel_area_growth,
)
from vhagar.physics.planck import planck_sensitivity_exponent

__all__ = [
    "FORBIDDEN_FEATURES",
    "PhysicsInputs",
    "assert_no_forbidden_features",
    "build_features",
    "feature_names",
]

#: Anything that lets a model memorise *where* rather than learn *what*.
#: Enforced by :func:`assert_no_forbidden_features`, which is called inside
#: :func:`build_features` and covered by a test.
FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "lat", "latitude", "lon", "long", "longitude",
        "x", "y", "easting", "northing",
        "row", "col", "i", "j",
        "tile_x", "tile_y", "grid_x", "grid_y", "h", "v",
        "geohash", "h3", "s2_cell", "quadkey",
    }
)


@dataclass(slots=True)
class PhysicsInputs:
    """Everything needed to build the physics feature vector for a detection.

    Coordinates appear here because you need them to *compute* geometry -- and
    then they are discarded. That asymmetry is the entire design.
    """

    bt_mir_k: np.ndarray
    bt_tir_k: np.ndarray
    bt_mir_background_k: np.ndarray
    bt_tir_background_k: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    day_of_year: np.ndarray
    utc_hour: np.ndarray
    view_zenith_deg: np.ndarray
    view_azimuth_deg: np.ndarray
    solar_zenith_deg: np.ndarray
    solar_azimuth_deg: np.ndarray
    tcwv_kg_m2: np.ndarray
    aod_550: np.ndarray | float = 0.0
    #: CAMEL hinge points nearest the fire channels. ASTER GED will NOT do --
    #: it is 8-12 um only and has no MIR coverage at all.
    emissivity_mir: np.ndarray | float = 0.98
    emissivity_tir: np.ndarray | float = 0.98
    elevation_m: np.ndarray | float = 0.0
    ndvi: np.ndarray | float = np.nan
    #: SWIR radiances where the sensor has them (SLSTR S5/S6, ABI C06).
    swir_16_um: np.ndarray | float = np.nan
    swir_22_um: np.ndarray | float = np.nan
    #: Causal temporal context -- computed from PAST observations only.
    persistence_count: np.ndarray | float = 0.0
    persistence_days: np.ndarray | float = 0.0
    #: Normalised temporal anomaly z = (x - mu(pixel, hour)) / sigma(pixel, hour).
    temporal_anomaly_z: np.ndarray | float = np.nan


def feature_names() -> list[str]:
    """Ordered feature names produced by :func:`build_features`."""
    return [
        # --- radiometric contrast -------------------------------------
        "bt_mir_k",
        "bt_tir_k",
        "dt_mir_tir_k",
        "dt_mir_tir_anomaly_k",
        "bt_mir_excess_k",
        "bt_tir_excess_k",
        "mir_tir_excess_ratio",
        "planck_b_mir",
        "planck_b_tir",
        # --- geometry (the cheap, high-yield block) --------------------
        "solar_zenith_deg",
        "view_zenith_deg",
        "glint_angle_deg",
        "relative_azimuth_deg",
        "air_mass_factor",
        "pixel_area_growth",
        "is_night",
        # --- diurnal / seasonal phase ---------------------------------
        "local_solar_time_h",
        "solar_hour_sin",
        "solar_hour_cos",
        "doy_sin",
        "doy_cos",
        # --- atmosphere -----------------------------------------------
        "tcwv_kg_m2",
        "aod_550",
        "transmittance_mir",
        "elevation_km",
        # --- surface --------------------------------------------------
        "emissivity_mir",
        "emissivity_tir",
        "emissivity_contrast",
        "ndvi",
        # --- spectral shape (flare discriminant) -----------------------
        "swir16_mir_ratio",
        "swir22_mir_ratio",
        "has_swir",
        # --- causal temporal context ----------------------------------
        "persistence_count",
        "persistence_days",
        "temporal_anomaly_z",
    ]


def assert_no_forbidden_features(names) -> None:
    """Raise if any feature name would let the model memorise geography."""
    bad = {str(n).lower() for n in names} & FORBIDDEN_FEATURES
    if bad:
        raise ValueError(
            f"forbidden coordinate feature(s) {sorted(bad)}. Raw coordinates "
            "supplied ~89% of split-gain in a published FIRMS classifier while "
            "HARMING out-of-region transfer. Use physics and geometry instead; "
            "see vhagar.features.physics_features.FORBIDDEN_FEATURES."
        )


def _b(x, like) -> np.ndarray:
    return np.broadcast_to(np.asarray(x, dtype=np.float64), np.shape(like)).astype(np.float64)


def build_features(inputs: PhysicsInputs) -> tuple[np.ndarray, list[str]]:
    """Build the stage-1 physics feature matrix.

    Returns ``(X, names)`` with ``X`` of shape ``(n_samples, n_features)``.

    >>> import numpy as np
    >>> pi = PhysicsInputs(
    ...     bt_mir_k=np.array([340.0]), bt_tir_k=np.array([300.0]),
    ...     bt_mir_background_k=np.array([305.0]), bt_tir_background_k=np.array([298.0]),
    ...     latitude_deg=np.array([38.0]), longitude_deg=np.array([-120.0]),
    ...     day_of_year=np.array([200]), utc_hour=np.array([21.0]),
    ...     view_zenith_deg=np.array([25.0]), view_azimuth_deg=np.array([100.0]),
    ...     solar_zenith_deg=np.array([30.0]), solar_azimuth_deg=np.array([200.0]),
    ...     tcwv_kg_m2=np.array([18.0]))
    >>> X, names = build_features(pi)
    >>> X.shape[1] == len(names)
    True
    """
    names = feature_names()
    assert_no_forbidden_features(names)

    ref = np.asarray(inputs.bt_mir_k, dtype=np.float64)
    bt_mir = ref
    bt_tir = _b(inputs.bt_tir_k, ref)
    bg_mir = _b(inputs.bt_mir_background_k, ref)
    bg_tir = _b(inputs.bt_tir_background_k, ref)

    dt = bt_mir - bt_tir
    dt_bg = bg_mir - bg_tir
    excess_mir = bt_mir - bg_mir
    excess_tir = bt_tir - bg_tir

    vz = _b(inputs.view_zenith_deg, ref)
    va = _b(inputs.view_azimuth_deg, ref)
    sz = _b(inputs.solar_zenith_deg, ref)
    sa = _b(inputs.solar_azimuth_deg, ref)
    tcwv = _b(inputs.tcwv_kg_m2, ref)
    aod = _b(inputs.aod_550, ref)

    lst = local_solar_time_hours(inputs.longitude_deg, inputs.utc_hour)
    lst = _b(lst, ref)
    doy_s, doy_c = day_of_year_encoding(inputs.day_of_year)

    e_mir = _b(inputs.emissivity_mir, ref)
    e_tir = _b(inputs.emissivity_tir, ref)

    swir16 = _b(inputs.swir_16_um, ref)
    swir22 = _b(inputs.swir_22_um, ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        # A uniform warm surface raises MIR and TIR together; a sub-pixel fire
        # raises them by wildly different amounts. This ratio is the cleanest
        # scalar expression of that distinction.
        excess_ratio = np.where(np.abs(excess_tir) > 0.05, excess_mir / excess_tir, np.nan)
        r16 = swir16 / np.maximum(bt_mir, 1e-6)
        r22 = swir22 / np.maximum(bt_mir, 1e-6)

    columns = [
        bt_mir,
        bt_tir,
        dt,
        dt - dt_bg,
        excess_mir,
        excess_tir,
        excess_ratio,
        planck_sensitivity_exponent(3.9, bt_mir),
        planck_sensitivity_exponent(11.0, bt_tir),
        sz,
        vz,
        glint_angle_deg(sz, vz, sa, va),
        np.mod(sa - va, 360.0),
        air_mass_factor(vz),
        pixel_area_growth(vz),
        (sz > 90.0).astype(np.float64),
        lst,
        np.sin(2 * np.pi * lst / 24.0),
        np.cos(2 * np.pi * lst / 24.0),
        _b(doy_s, ref),
        _b(doy_c, ref),
        tcwv,
        aod,
        transmittance_mir(tcwv, vz, aod),
        _b(inputs.elevation_m, ref) / 1000.0,
        e_mir,
        e_tir,
        e_mir - e_tir,
        _b(inputs.ndvi, ref),
        r16,
        r22,
        np.isfinite(swir16).astype(np.float64),
        _b(inputs.persistence_count, ref),
        _b(inputs.persistence_days, ref),
        _b(inputs.temporal_anomaly_z, ref),
    ]

    x = np.stack([np.asarray(c, dtype=np.float64).ravel() for c in columns], axis=-1)
    if x.shape[1] != len(names):
        raise AssertionError(f"feature count mismatch: {x.shape[1]} columns vs {len(names)} names")
    return x, names


def missingness_indicators(x: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Append explicit NaN indicators rather than imputing.

    SWIR features in particular are *missing-not-at-random*: SWIR is degraded by
    smoke more severely than MIR, so a missing SWIR ratio is itself evidence
    about the scene. Imputing a mean destroys that signal and quietly biases the
    model toward treating smoky wildfires like clear-sky flares.
    """
    ind = (~np.isfinite(x)).astype(np.float64)
    keep = ind.any(axis=0)
    return (
        np.concatenate([x, ind[:, keep]], axis=1),
        names + [f"missing__{n}" for n, k in zip(names, keep, strict=True) if k],
    )
