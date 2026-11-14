"""Terrain derivatives for a projected or locally metric T4-M analysis grid."""

from __future__ import annotations

import numpy as np

__all__ = ["slope_aspect_from_dem"]


def slope_aspect_from_dem(
    elevation_m: np.ndarray, *, cell_size_m: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return terrain slope (rise/run) and *downslope* aspect in radians.

    ``cell_size_m`` is ``(north_south, east_west)``. The input grid must be
    north-up with rows increasing southward and use locally meaningful metric
    cell sizes. For a geographic analysis grid, callers must convert its degree
    spacing to local metres and record that approximation in provenance.
    """
    elevation = np.asarray(elevation_m, dtype=float)
    cell_y_m, cell_x_m = cell_size_m
    if (
        elevation.ndim != 2
        or min(elevation.shape) < 2
        or not np.isfinite(elevation).all()
        or cell_y_m <= 0
        or cell_x_m <= 0
    ):
        raise ValueError("DEM must be finite 2-D and cell sizes positive")

    # np.gradient's row derivative is southward. Negate it to express the
    # elevation gradient in a conventional north/east coordinate system.
    south_gradient, east_gradient = np.gradient(elevation, cell_y_m, cell_x_m)
    north_gradient = -south_gradient
    slope_tan = np.hypot(east_gradient, north_gradient)
    # Aspect is the downhill compass direction: 0=north, pi/2=east.
    aspect_rad = np.mod(np.arctan2(-east_gradient, -north_gradient), 2 * np.pi)
    return slope_tan, aspect_rad
