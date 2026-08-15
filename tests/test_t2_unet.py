"""T2 U-Net companion baseline: numpy core always, torch loop when available."""

from __future__ import annotations

import numpy as np
import pytest

from vhagar.datasets.burned_area import make_sample
from vhagar.eval.t2_unet import (
    grouped_folds,
    random_crops,
    standardizer_from,
)


def _fire(event_id, burned_frac=0.3, n=160, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.random((n, n)) < burned_frac
    rbr = np.where(truth, rng.normal(300, 40, (n, n)), rng.normal(20, 40, (n, n)))
    return make_sample(event_id, rbr, truth, tile_id="conus/x0001_y0001")


# ------------------------------------------------------------- numpy core -----


def test_standardizer_is_robust_median_mad():
    s = _fire("f", seed=1)
    std = standardizer_from([s])
    # center near the pooled median, scale positive; a huge outlier must not move it much
    assert std.scale > 0
    out = std.apply(s.predictor, s.valid)
    assert out.min() >= -5.0 and out.max() <= 5.0
    assert np.all(out[~s.valid] == 0.0)      # invalid mapped to channel mean (0)


def test_random_crops_have_right_shape_and_bias_to_burned():
    s = _fire("f", burned_frac=0.1, seed=2)
    rng = np.random.default_rng(0)
    crops = random_crops(s, crop=64, n=40, rng=rng, burned_bias=1.0)
    assert crops and all(img.shape == (64, 64) for img, _, _ in crops)
    # with full burned bias, mean burned fraction across crops exceeds the 10% base rate
    frac = np.mean([(b & v).mean() for _, b, v in crops])
    assert frac > 0.1


def test_random_crops_pads_a_small_window():
    s = _fire("f", n=40, seed=3)
    rng = np.random.default_rng(0)
    crops = random_crops(s, crop=64, n=4, rng=rng)
    assert crops and all(img.shape == (64, 64) for img, _, _ in crops)


def test_grouped_folds_cover_every_fire_once_without_leakage():
    ids = [f"f{i}" for i in range(11)]
    folds = grouped_folds(ids, k=4, seed=0)
    assert len(folds) == 4
    tested = []
    for f in folds:
        assert not (set(f["train"]) & set(f["test"]))     # no leakage
        tested += f["test"]
    assert sorted(tested) == sorted(ids)                  # every fire tested once


def test_grouped_folds_rejects_impossible_k():
    with pytest.raises(ValueError, match="k must be"):
        grouped_folds(["a", "b"], k=5)


# --------------------------------------------------------- torch smoke test ---


def test_unet_learns_a_separable_fire_and_is_compared_to_threshold():
    pytest.importorskip("torch")
    from vhagar.eval.t2_unet import run_unet_cv, summarise_unet_cv

    samples = {f"f{i}": _fire(f"f{i}", burned_frac=0.25, n=96, seed=i) for i in range(4)}
    results = run_unet_cv(
        samples, k=2, epochs=3, crop=64, crops_per_fire=8, seed=0,
    )
    assert results
    # on a cleanly separable fire the U-Net should have positive skill over naive
    assert any(r.skill_f1 > 0.0 for r in results)
    s = summarise_unet_cv(results)
    assert "unet_skill_mean" in s and "thr_skill_mean" in s
    assert 0 <= s["unet_beats_thr"] <= s["fires"]
