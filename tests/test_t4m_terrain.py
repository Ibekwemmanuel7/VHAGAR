import numpy as np
import pytest

from vhagar.datasets.t4m_terrain import slope_aspect_from_dem


def test_plane_has_expected_slope_and_downslope_aspect() -> None:
    rows, columns = np.mgrid[:5, :5]
    elevation = 2 * columns - 3 * rows
    slope, aspect = slope_aspect_from_dem(elevation, cell_size_m=(1, 1))
    assert np.allclose(slope, np.hypot(2, 3))
    assert np.allclose(aspect, np.arctan2(-2, -3) % (2 * np.pi))


def test_invalid_dem_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite 2-D"):
        slope_aspect_from_dem(np.array([[1.0, np.nan], [3.0, 4.0]]), cell_size_m=(1, 1))
