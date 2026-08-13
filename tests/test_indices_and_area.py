from __future__ import annotations

import numpy as np
import pytest

from vhagar.eval.area_estimation import (
    allocate_samples,
    estimate_areas,
    sample_size_stratified,
)
from vhagar.features import indices as I

# ----------------------------------------------------------- indices ------


def test_nbr_drops_after_a_burn():
    pre_nir, pre_swir = 0.35, 0.10       # healthy vegetation
    post_nir, post_swir = 0.12, 0.30     # charred
    assert I.nbr(pre_nir, pre_swir) > I.nbr(post_nir, post_swir)
    assert I.dnbr(I.nbr(pre_nir, pre_swir), I.nbr(post_nir, post_swir)) > 0


def test_rbr_avoids_rdnbr_singularity():
    """As pre-fire NBR approaches zero, RdNBR explodes and RBR does not."""
    nbr_post = -0.4
    small = 0.001
    rd = abs(float(I.rdnbr(small, nbr_post)))
    rb = abs(float(I.rbr(small, nbr_post)))
    assert rd > 5 * rb


def test_rbr_preserves_sign_of_prefire_nbr():
    d_pos = float(I.rbr(0.5, -0.3))
    d_neg = float(I.rbr(-0.5, -0.9))
    assert np.isfinite(d_pos) and np.isfinite(d_neg)
    # Denominator stays positive for NBR_pre > -1, so dNBR sign carries through.
    assert d_pos > 0 and d_neg > 0


def test_severity_classification_and_nan_handling():
    idx = np.array([50.0, 200.0, 350.0, 550.0, 900.0, np.nan])
    cls = I.classify_severity(idx)
    assert cls.tolist() == [0, 1, 2, 3, 4, -1]


def test_index_bounds():
    rng = np.random.default_rng(0)
    a, b = rng.random(1000), rng.random(1000)
    nd = I.normalized_difference(a, b)
    assert np.nanmin(nd) >= -1.0 and np.nanmax(nd) <= 1.0


# ------------------------------------------------- area estimation --------


def test_olofsson_adjusts_area_and_gives_a_ci():
    # 20,000 ha mapped burned out of 220,000. The burned class has 10%
    # commission; the unburned class has ~3% omission, which at scale adds a
    # lot of real burned area the map missed.
    conf = np.array([[97.0, 3.0], [10.0, 90.0]])
    areas = np.array([200_000.0, 20_000.0])
    est = estimate_areas(conf, areas, ["unburned", "burned"])
    burned = est[1]
    assert burned.adjusted_area != burned.mapped_area
    assert burned.ci95_low < burned.adjusted_area < burned.ci95_high
    assert 0.0 <= burned.users_accuracy <= 1.0
    assert 0.0 <= burned.producers_accuracy <= 1.0


def test_olofsson_adjusted_areas_sum_to_total():
    conf = np.array([[80.0, 20.0], [15.0, 85.0]])
    areas = np.array([500_000.0, 50_000.0])
    est = estimate_areas(conf, areas)
    assert sum(e.adjusted_area for e in est) == pytest.approx(areas.sum(), rel=1e-9)


def test_perfect_map_reproduces_mapped_areas_with_zero_error():
    conf = np.array([[100.0, 0.0], [0.0, 100.0]])
    areas = np.array([300_000.0, 30_000.0])
    for e in estimate_areas(conf, areas):
        assert e.adjusted_area == pytest.approx(e.mapped_area)
        assert e.standard_error == pytest.approx(0.0)


def test_tiny_strata_are_rejected_not_silently_estimated():
    conf = np.array([[50.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="reference samples"):
        estimate_areas(conf, np.array([100.0, 10.0]))


def test_sample_size_and_allocation():
    w = np.array([0.97, 0.03])
    n = sample_size_stratified(w, np.array([0.90, 0.75]), target_se_overall=0.01)
    assert n > 500
    alloc = allocate_samples(n, w, rare_classes=[1], min_per_rare=75)
    assert alloc[1] == 75
    assert alloc.sum() == n


def test_allocation_rejects_impossible_budget():
    with pytest.raises(ValueError, match="rare-class floor"):
        allocate_samples(50, np.array([0.9, 0.1]), rare_classes=[1], min_per_rare=75)


def test_classify_severity_handles_scalars_and_zero_d_arrays():
    """0-d inputs must behave exactly like arrays (regression: masked assignment)."""
    assert int(I.classify_severity(500.0)) == 3
    assert int(I.classify_severity(np.float64(50.0))) == 0
    assert int(I.classify_severity(np.array(np.nan))) == -1
    assert int(I.classify_severity(I.rbr(0.58, -0.48))) >= 3
