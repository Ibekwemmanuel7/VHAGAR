"""ABI fixed-grid navigation, scan angles to geodetic coordinates.

Every GOES-R product is on the **ABI fixed grid**: a pair of scan angles
``(x, y)`` in radians as seen from the satellite, not a map projection you can
hand to PROJ without setup. Converting correctly matters more than it looks:
a navigation bug puts your fires in the wrong place, and unlike most bugs it
produces perfectly plausible output.

The algorithm is from the GOES-R Product User Guide, Volume 5, section 4.2.8.
It intersects the line of sight with the WGS-84 ellipsoid, which is a quadratic
in the satellite-to-ground range ``r_s``:

    a = sin^2(x) + cos^2(x)*[cos^2(y) + (r_eq^2/r_pol^2)*sin^2(y)]
    b = -2*H*cos(x)*cos(y)
    c = H^2 - r_eq^2
    r_s = (-b - sqrt(b^2 - 4ac)) / (2a)

The discriminant goes negative for lines of sight that miss the Earth, those
are off-disk pixels and must become NaN, not a complex number or a clamp.

Two things this module gets right that naive implementations often do not:

* **Off-disk pixels return NaN.** Roughly 15 % of a full-disk grid is space.
* **View zenith angle is returned alongside**, because it drives the
  atmospheric air-mass correction (a 2.1x FRP factor at 60 degrees, see
  :mod:`vhagar.physics.atmosphere`) and the pixel-area growth that FRP is
  directly proportional to. Getting lat/lon without the geometry means
  re-deriving it later, badly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ABIProjection", "GOES_EAST_LON", "GOES_WEST_LON"]

#: Nominal sub-satellite longitudes, degrees east.
GOES_EAST_LON = -75.0    # GOES-19 operates at 75.2 W; read it from the file
GOES_WEST_LON = -137.0   # GOES-18


@dataclass(frozen=True, slots=True)
class ABIProjection:
    """ABI fixed-grid projection parameters, normally read from the file.

    Defaults are the GOES-R standard values; always prefer the
    ``goes_imager_projection`` variable attributes in the actual granule, since
    the sub-satellite longitude differs between satellites and can be adjusted.
    """

    lon_origin_deg: float = -75.0
    #: Height of the satellite above the ellipsoid, metres.
    perspective_point_height: float = 35786023.0
    semi_major_axis: float = 6378137.0
    semi_minor_axis: float = 6356752.31414

    @property
    def h(self) -> float:
        """Distance from Earth centre to satellite, metres."""
        return self.perspective_point_height + self.semi_major_axis

    @classmethod
    def from_dataset(cls, ds) -> ABIProjection:
        """Read projection parameters from an open ABI xarray Dataset."""
        p = ds["goes_imager_projection"]
        return cls(
            lon_origin_deg=float(p.attrs["longitude_of_projection_origin"]),
            perspective_point_height=float(p.attrs["perspective_point_height"]),
            semi_major_axis=float(p.attrs["semi_major_axis"]),
            semi_minor_axis=float(p.attrs["semi_minor_axis"]),
        )

    # -- forward: scan angles -> geodetic --------------------------------

    def to_latlon(self, x_rad, y_rad) -> tuple[np.ndarray, np.ndarray]:
        """Scan angles (radians) to geodetic latitude/longitude (degrees).

        Off-disk lines of sight return NaN in both outputs.

        >>> proj = ABIProjection(lon_origin_deg=-75.0)
        >>> lat, lon = proj.to_latlon(0.0, 0.0)
        >>> float(round(lat, 6)), float(round(lon, 4))
        (0.0, -75.0)
        """
        x = np.asarray(x_rad, dtype=np.float64)
        y = np.asarray(y_rad, dtype=np.float64)
        req, rpol, h = self.semi_major_axis, self.semi_minor_axis, self.h
        ratio2 = (req / rpol) ** 2

        sin_x, cos_x = np.sin(x), np.cos(x)
        sin_y, cos_y = np.sin(y), np.cos(y)

        a = sin_x**2 + cos_x**2 * (cos_y**2 + ratio2 * sin_y**2)
        b = -2.0 * h * cos_x * cos_y
        c = h**2 - req**2

        disc = b**2 - 4.0 * a * c
        with np.errstate(invalid="ignore"):
            r_s = np.where(disc >= 0.0, (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a), np.nan)

        s_x = r_s * cos_x * cos_y
        s_y = -r_s * sin_x
        s_z = r_s * cos_x * sin_y

        with np.errstate(invalid="ignore", divide="ignore"):
            lat = np.degrees(np.arctan(ratio2 * s_z / np.sqrt((h - s_x) ** 2 + s_y**2)))
            lon = self.lon_origin_deg - np.degrees(np.arctan(s_y / (h - s_x)))
        return lat, lon

    # -- inverse: geodetic -> scan angles --------------------------------

    def to_scan_angles(self, lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray]:
        """Geodetic latitude/longitude to scan angles (radians).

        Needed to crop a granule to an area of interest **before** decoding. 
        which is the difference between reading a few hundred kilobytes and
        pulling a whole 50 MB full-disk file for one fire.

        Points on the far side of the Earth (not visible from the satellite)
        return NaN.
        """
        lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
        lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
        req, rpol, h = self.semi_major_axis, self.semi_minor_axis, self.h
        lon0 = np.radians(self.lon_origin_deg)

        # Geocentric latitude on the ellipsoid.
        lat_c = np.arctan((rpol**2 / req**2) * np.tan(lat))
        r_c = rpol / np.sqrt(1.0 - (1.0 - rpol**2 / req**2) * np.cos(lat_c) ** 2)

        s_x = h - r_c * np.cos(lat_c) * np.cos(lon - lon0)
        s_y = -r_c * np.cos(lat_c) * np.sin(lon - lon0)
        s_z = r_c * np.sin(lat_c)

        # Visibility: the dot product test from the PUG.
        visible = h * (h - s_x) >= (s_y**2 + (req**2 / rpol**2) * s_z**2)

        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.arcsin(-s_y / np.sqrt(s_x**2 + s_y**2 + s_z**2))
            y = np.arctan(s_z / s_x)
        return np.where(visible, x, np.nan), np.where(visible, y, np.nan)

    # -- geometry --------------------------------------------------------

    def view_zenith_deg(self, lat_deg, lon_deg) -> np.ndarray:
        """Satellite view zenith angle at a ground point, degrees.

        Drives the atmospheric air-mass factor and pixel-area growth. At the
        geostationary disk edge this approaches 90 degrees; fire products cut
        off processing beyond 80.

        >>> proj = ABIProjection(lon_origin_deg=-75.0)
        >>> float(round(proj.view_zenith_deg(0.0, -75.0), 3))
        0.0
        """
        lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
        lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
        lon0 = np.radians(self.lon_origin_deg)
        re, h = self.semi_major_axis, self.h

        cos_psi = np.cos(lat) * np.cos(lon - lon0)
        cos_psi = np.clip(cos_psi, -1.0, 1.0)
        psi = np.arccos(cos_psi)
        # Plane triangle: Earth centre, ground point, satellite.
        denom = np.sqrt(1.0 + (re / h) ** 2 - 2.0 * (re / h) * cos_psi)
        with np.errstate(invalid="ignore", divide="ignore"):
            sin_z = np.sin(psi) / denom
        return np.degrees(np.arcsin(np.clip(sin_z, -1.0, 1.0)))

    def pixel_area_m2(self, lat_deg, lon_deg, nominal_m: float = 2000.0) -> np.ndarray:
        """Ground area of a nominally ``nominal_m`` pixel, accounting for obliquity.

        FRP is directly proportional to pixel area, so using the nominal 2 km
        everywhere under-reports FRP off nadir by the same factor the footprint
        grows, a factor of several near the disk edge.
        """
        z = np.radians(np.clip(self.view_zenith_deg(lat_deg, lon_deg), 0.0, 85.0))
        re, h = self.semi_major_axis, self.h
        sin_scan = np.clip(re * np.sin(z) / h, -1.0, 1.0)
        scan = np.arcsin(sin_scan)
        # Along-scan stretches as 1/cos of the incidence angle; along-track as sec.
        growth = (np.cos(scan) / np.cos(z) ** 2) * (np.cos(scan) / np.cos(z))
        return nominal_m**2 * growth
