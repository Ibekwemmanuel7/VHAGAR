"""FWI System regression tests.

The reference values below are the canonical Van Wagner & Pickett (1985)
worked example, which every implementation (the R ``cffdrs`` package included)
reproduces. If these drift, the implementation has a bug -- do not "update the
expected values".
"""

from __future__ import annotations

import numpy as np
import pytest

from vhagar.features import fwi as F


def test_van_wagner_worked_example():
    """Van Wagner & Pickett 1985, Table: T=17C, H=42%, W=25 km/h, rain=0, April."""
    state = F.FWIState.season_start()  # FFMC 85, DMC 6, DC 15
    out, _ = F.fwi_system(17.0, 42.0, 25.0, 0.0, state, month=4)
    assert float(out["ffmc"]) == pytest.approx(87.7, abs=0.15)
    assert float(out["dmc"]) == pytest.approx(8.5, abs=0.2)
    assert float(out["dc"]) == pytest.approx(19.0, abs=0.5)
    assert float(out["isi"]) == pytest.approx(10.9, abs=0.3)
    assert float(out["bui"]) == pytest.approx(8.5, abs=0.3)
    assert float(out["fwi"]) == pytest.approx(10.1, abs=0.5)


def test_dsr_relationship():
    assert float(F.dsr(10.1)) == pytest.approx(0.0272 * 10.1**1.77, rel=1e-9)


def test_rain_reduces_ffmc():
    state = F.FWIState.season_start()
    dry, _ = F.fwi_system(20.0, 40.0, 15.0, 0.0, state, month=6)
    wet, _ = F.fwi_system(20.0, 40.0, 15.0, 20.0, state, month=6)
    assert float(wet["ffmc"]) < float(dry["ffmc"])
    assert float(wet["dmc"]) < float(dry["dmc"])
    assert float(wet["dc"]) < float(dry["dc"])


def test_codes_stay_in_range_under_extremes():
    state = F.FWIState.season_start()
    for temp in (-30.0, 0.0, 45.0):
        for rh in (0.0, 50.0, 100.0):
            for wind in (0.0, 60.0, 120.0):
                for rain in (0.0, 1.0, 100.0):
                    out, _ = F.fwi_system(temp, rh, wind, rain, state, month=7)
                    assert 0.0 <= float(out["ffmc"]) <= 101.0
                    assert float(out["dmc"]) >= 0.0
                    assert float(out["dc"]) >= 0.0
                    assert float(out["isi"]) >= 0.0
                    assert float(out["bui"]) >= 0.0
                    assert np.isfinite(float(out["fwi"]))


def test_vectorised_matches_scalar():
    shape = (4, 5)
    state = F.FWIState.season_start(shape)
    temp = np.full(shape, 22.0)
    out_grid, _ = F.fwi_system(temp, 35.0, 18.0, 0.0, state, month=8)
    out_scalar, _ = F.fwi_system(22.0, 35.0, 18.0, 0.0, F.FWIState.season_start(), month=8)
    assert out_grid["fwi"].shape == shape
    assert np.allclose(out_grid["fwi"], float(out_scalar["fwi"]))


def test_state_advances_and_drought_accumulates():
    state = F.FWIState.season_start()
    dc_series = []
    for _ in range(30):
        out, state = F.fwi_system(28.0, 25.0, 20.0, 0.0, state, month=7)
        dc_series.append(float(out["dc"]))
    assert dc_series == sorted(dc_series), "DC must increase monotonically under drought"
    assert dc_series[-1] > dc_series[0] + 50


def test_southern_hemisphere_day_length_is_phase_shifted():
    assert F.day_length_dmc(1, lat=-35.0) == F.day_length_dmc(7, lat=35.0)
    assert F.day_length_dc(1, lat=-35.0) == F.day_length_dc(7, lat=35.0)


def test_effis_classes():
    assert int(F.effis_class(3.0)) == 0     # very low
    assert int(F.effis_class(15.0)) == 2    # moderate
    assert int(F.effis_class(80.0)) == 6    # very extreme
