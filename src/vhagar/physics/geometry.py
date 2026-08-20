"""Observation geometry, the cheapest large win in false-alarm reduction.

The single most decision-relevant published result for VHAGAR is this: in a
FIRMS wildfire/non-wildfire classification, **raw lat/lon supplied 88.9% of
model split-gain**, and F1 fell 0.985 (random split) -> 0.767 (event-aware) ->
0.627 (5-degree spatial block). Dropping the coordinates *raised* spatial-block
F1 to 0.818.

Coordinates are a memorisation shortcut. **Geometry and physics are the
transferable substitute.** The features in this module -- solar zenith, view
zenith, glint angle, relative azimuth, local solar time, pixel area growth --
are what let a model learn "this is a specular reflector" rather than
"detections here are usually industrial".

Glint angle in particular is close to free and addresses three false-alarm
classes at once: solar farms, specular reflectors, and sun glint over water.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "day_of_year_encoding",
    "glint_angle_deg",
    "local_solar_time_hours",
    "pixel_area_growth",
    "solar_position",
    "viirs_pixel_area_m2",
]


def _rad(x) -> np.ndarray:
    return np.radians(np.asarray(x, dtype=np.float64))


def solar_position(
    latitude_deg, longitude_deg, day_of_year, utc_hour
) -> tuple[np.ndarray, np.ndarray]:
    """Solar zenith and azimuth in degrees (NOAA low-precision algorithm).

    Accurate to roughly +-0.2 degrees, which is far better than needed for a
    feature. Azimuth is measured clockwise from north.

    >>> z, a = solar_position(40.0, -100.0, 172, 19.0)   # summer solstice, local noon-ish
    >>> bool(0 < float(z) < 30)
    True
    """
    lat = _rad(latitude_deg)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    doy = np.asarray(day_of_year, dtype=np.float64)
    hour = np.asarray(utc_hour, dtype=np.float64)

    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    time_offset = eqtime + 4.0 * lon
    tst = hour * 60.0 + time_offset
    ha = _rad(tst / 4.0 - 180.0)

    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    cos_z = np.clip(cos_z, -1.0, 1.0)
    zenith = np.degrees(np.arccos(cos_z))

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_az = (np.sin(decl) - np.sin(lat) * cos_z) / (np.cos(lat) * np.sin(np.arccos(cos_z)))
    cos_az = np.clip(np.nan_to_num(cos_az, nan=0.0), -1.0, 1.0)
    azimuth = np.where(ha > 0, 360.0 - np.degrees(np.arccos(cos_az)), np.degrees(np.arccos(cos_az)))
    return zenith, azimuth


def glint_angle_deg(
    solar_zenith_deg, view_zenith_deg, solar_azimuth_deg, view_azimuth_deg
) -> np.ndarray:
    """Angle between the specular reflection direction and the sensor.

    ``cos(glint) = cos(sz)cos(vz) - sin(sz)sin(vz)cos(rel_az)``

    Small glint angles mean the sensor is looking at the sun's mirror image.
    MODIS Collection 6 rejects fire detections at ``glint < 2 deg`` outright,
    and at ``glint < 10 deg`` when the visible/SWIR reflectances are also high.

    This one feature covers solar farms, other specular reflectors, and water
    glint -- all three of which have the same geometric signature and none of
    which have a real thermal one.
    """
    sz = _rad(solar_zenith_deg)
    vz = _rad(view_zenith_deg)
    rel = _rad(np.asarray(solar_azimuth_deg, dtype=np.float64)
               - np.asarray(view_azimuth_deg, dtype=np.float64))
    cos_g = np.cos(sz) * np.cos(vz) - np.sin(sz) * np.sin(vz) * np.cos(rel)
    return np.degrees(np.arccos(np.clip(cos_g, -1.0, 1.0)))


def local_solar_time_hours(longitude_deg, utc_hour) -> np.ndarray:
    """Local solar time in hours [0, 24).

    The diurnal cycle is the dominant confound in thermal fire detection: hot
    bare soil peaks at solar noon plus a thermal-inertia lag, wildfire peaks
    mid-to-late afternoon, and gas flares have no diurnal phase at all. Give
    the model the phase directly rather than making it infer it.
    """
    lon = np.asarray(longitude_deg, dtype=np.float64)
    return np.mod(np.asarray(utc_hour, dtype=np.float64) + lon / 15.0, 24.0)


def day_of_year_encoding(day_of_year) -> tuple[np.ndarray, np.ndarray]:
    """Cyclic (sin, cos) encoding so 31 Dec is adjacent to 1 Jan."""
    theta = 2.0 * np.pi * np.asarray(day_of_year, dtype=np.float64) / 365.25
    return np.sin(theta), np.cos(theta)


def pixel_area_growth(view_zenith_deg, orbit_altitude_km: float = 833.0) -> np.ndarray:
    """Along-scan x along-track pixel area growth factor relative to nadir.

    Derived from slant-range geometry, not a small-angle secant approximation.
    Let ``rho`` be the slant range to the ground pixel and ``h`` the nadir
    altitude. The along-track footprint stretches with the range ratio,
    ``rho / h``; the along-scan footprint stretches by a further ``1 / cos(z)``
    as the line of sight grazes the surface. So the area growth is

        (rho / h)**2 / cos(z),

    where ``rho`` follows from the Earth triangle (scan angle at the sensor,
    Earth-central angle, sine rule). This reproduces the published anchors:
    MODIS (h=705 km) reaches ~9.7x area (~2.0x along-track, ~4.8x along-scan) at
    its ~65 deg scan-edge view zenith, and a geostationary sensor reaches ~2.33x
    at 60 deg view zenith. The previous ``cos(scan)/cos(z)**2`` form overstated
    both (~14x MODIS edge, ~7.9x GEO at 60 deg) because it applied secant growth
    to the wide ground incidence angle rather than to the range ratio.

    This matters twice over: FRP is proportional to pixel area, and detection
    sensitivity degrades as the fire's fractional coverage shrinks.
    """
    r_e = 6371.0
    z = np.radians(np.clip(np.asarray(view_zenith_deg, dtype=np.float64), 0.0, 80.0))
    h = float(orbit_altitude_km)
    # Earth triangle: scan angle at the sensor from the view zenith (sine rule),
    # then the Earth-central angle, then slant range over nadir altitude.
    alpha = np.arcsin(np.clip(r_e * np.sin(z) / (r_e + h), -1.0, 1.0))
    gamma = z - alpha
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_over_h = np.where(z > 0.0, (r_e + h) * np.sin(gamma) / (h * np.sin(z)), 1.0)
    # area = along_track * along_scan = (rho/h) * (rho/h / cos z)
    return rho_over_h**2 / np.cos(z)


def viirs_pixel_area_m2(view_zenith_deg, nominal_m: float = 375.0) -> np.ndarray:
    """VIIRS I-band pixel area with aggregation-limited growth.

    VIIRS aggregates 3x1, 2x1 and 1x1 detector samples across the scan, which
    caps pixel growth at roughly 800 m at swath edge instead of MODIS' ~4.8 km.
    Modelled here as growth capped at 4x nadir area.
    """
    growth = np.minimum(pixel_area_growth(view_zenith_deg), 4.0)
    return nominal_m**2 * growth
