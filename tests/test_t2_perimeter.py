"""T2 perimeter-vs-severity commission analysis (census from a class histogram)."""

from __future__ import annotations

import numpy as np
import pytest

from vhagar.eval.t2_perimeter import perimeter_vs_severity

PX = 0.09


def _hist(counts: dict[int, int], size: int = 7) -> np.ndarray:
    h = np.zeros(size, dtype=np.int64)
    for k, v in counts.items():
        h[k] = v
    return h


def test_commission_with_strict_burned_classes():
    # perimeter interior = classes 1-5; burned = 2,3,4; islands = classes 1 and 5
    hist = _hist({0: 1000, 1: 100, 2: 300, 3: 200, 4: 100, 5: 50, 6: 10})
    r = perimeter_vs_severity(hist, pixel_area_ha=PX, burned_classes=(2, 3, 4))
    perimeter_px = 100 + 300 + 200 + 100 + 50   # 750
    burned_px = 300 + 200 + 100                 # 600
    assert r.perimeter_ha == pytest.approx(perimeter_px * PX)
    assert r.severity_burned_ha == pytest.approx(burned_px * PX)
    assert r.unburned_within_ha == pytest.approx((perimeter_px - burned_px) * PX)
    assert r.commission_fraction == pytest.approx((100 + 50) / 750)


def test_burned_class_definition_moves_the_commission():
    hist = _hist({0: 1000, 1: 100, 2: 300, 3: 200, 4: 100, 5: 50, 6: 10})
    strict = perimeter_vs_severity(hist, burned_classes=(2, 3, 4))
    lenient = perimeter_vs_severity(hist, burned_classes=(1, 2, 3, 4))
    # including "unburned to low" as burned shrinks the commission to just class 5
    assert lenient.commission_fraction < strict.commission_fraction
    assert lenient.commission_fraction == pytest.approx(50 / 750)


def test_background_and_nodata_are_excluded_from_the_perimeter():
    # class 0 (background) and 6 (nodata within) are not part of the perimeter
    hist = _hist({0: 10_000, 1: 0, 2: 100, 3: 0, 4: 0, 5: 0, 6: 500})
    r = perimeter_vs_severity(hist, burned_classes=(2, 3, 4))
    assert r.perimeter_ha == pytest.approx(100 * PX)   # only class 2 is mapped here
    assert r.commission_fraction == pytest.approx(0.0)


def test_empty_perimeter_is_nan_not_a_crash():
    hist = _hist({0: 5000})
    r = perimeter_vs_severity(hist, burned_classes=(2, 3, 4))
    assert np.isnan(r.commission_fraction)
