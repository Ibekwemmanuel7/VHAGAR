"""FIRMS adapter tests: real-format timing and first-seen cell semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vhagar.datasets.spread_ingest import GridSpec
from vhagar.datasets.t4m_firms import firms_to_t4m_observations, footprint_fuel_support


def test_footprint_support_recovers_burnable_area_around_nonburnable_centroid():
    ros = np.zeros((5, 5))
    ros[2, 3] = 2.0
    arrival = np.full((5, 5), np.inf)
    arrival[2, 3] = 4.0
    frac, effective_ros, effective_arrival = footprint_fuel_support(
        ros, arrival, 2, 2, cell_size_m=(100.0, 100.0), radius_m=150.0
    )
    assert 0 < frac < 1 and effective_ros == 2.0 and effective_arrival == 4.0


def test_firms_adapter_preserves_first_seen_timing_and_sensor_metadata():
    firms = pd.DataFrame({
        "latitude": [0.05, 0.05, 0.55], "longitude": [0.05, 0.05, 0.55],
        "acq_date": ["2026-08-01"] * 3, "acq_time": [822, 945, 1030],
        "confidence": ["n", "h", "l"], "satellite": ["N20"] * 3,
        "instrument": ["VIIRS"] * 3, "frp": [3.0, 4.0, 5.0],
    })
    spec = GridSpec.from_bbox_res((0.0, 0.0, 1.0, 1.0), cell_deg=0.5)
    grid = np.ones(spec.shape)
    out = firms_to_t4m_observations(
        firms, fire_id="f", spec=spec, analysis_grid_id="f-50km", prior_arrival_hours=grid * 2,
        prior_ros_m_min=grid, covariate_grids={"fuel_moisture": grid * 0.08,
                                                "wind_midflame_ms": grid * 3,
                                                "slope_tan": grid * 0.1},
    )
    assert len(out) == 2                       # later detection in same cell is deduplicated
    assert out.loc[0, "first_seen_hours"] == 0
    assert out.loc[1, "first_seen_hours"] == 2.0 + 8 / 60
    assert out.loc[0, "source_product"] == "N20 VIIRS"
    assert out.loc[0, "quality"] == 0.75
    assert out.loc[0, "time_sigma_min"] == 360.0
