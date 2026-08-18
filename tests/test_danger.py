"""T3 ignition sampling design + evaluation tests."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.datasets import danger as dg


def test_rare_event_correction_maps_to_base_rate():
    # A balanced-trained 0.5 maps to ~tau; ordering preserved; monotone in p.
    assert abs(float(dg.rare_event_correction(0.5, tau=0.01, ybar=0.5)) - 0.01) < 1e-3
    lo = float(dg.rare_event_correction(0.4, 0.02, 0.5))
    hi = float(dg.rare_event_correction(0.8, 0.02, 0.5))
    assert hi > lo
    # vectorised + clipped
    out = dg.rare_event_correction(np.array([0.0, 0.5, 1.0]), 0.05, 0.5)
    assert np.all((out >= 0) & (out <= 1))


def test_rare_event_correction_rejects_bad_args():
    with pytest.raises(ValueError):
        dg.rare_event_correction(0.5, tau=0.0, ybar=0.5)
    with pytest.raises(ValueError):
        dg.rare_event_correction(0.5, tau=0.5, ybar=1.0)


def test_target_group_background_excludes_presences():
    rng = np.random.default_rng(0)
    pool = np.arange(200)
    pres = np.array([1, 2, 3, 4, 5])
    bg = dg.target_group_background(pres, pool, 30, rng)
    assert len(bg) == 30
    assert not (set(bg.tolist()) & set(pres.tolist()))


def test_target_group_background_weight_biases_selection():
    rng = np.random.default_rng(0)
    pool = np.arange(1000)
    w = np.where(pool < 100, 10.0, 0.01)  # first 100 heavily weighted
    bg = dg.target_group_background(np.array([]), pool, 100, rng, pool_weight=w)
    assert (bg < 100).mean() > 0.7  # most draws land in the high-weight region


def test_stratify_negatives_matches_positive_distribution():
    rng = np.random.default_rng(0)
    pos_strata = np.array([0, 0, 0, 1, 1, 2])          # 3:2:1
    cand_ids = np.arange(600)
    cand_strata = np.repeat([0, 1, 2], 200)
    neg = dg.stratify_negatives(pos_strata, cand_ids, cand_strata, rng, ratio=2.0)
    counts = {s: int((cand_strata[neg] == s).sum()) for s in (0, 1, 2)}
    assert counts == {0: 6, 1: 4, 2: 2}


def test_synthetic_scenario_has_reporting_bias():
    rng = np.random.default_rng(1)
    pres, cand, fn, tau = dg.synthetic_reporting_scenario(rng, n_cells=2000)
    assert 0.0 < tau < 1.0
    assert len(pres["id"]) > 50
    # reported ignitions over-represent populated cells: the confounder
    assert pres["people"].mean() - cand["people"].mean() > 0.02
    assert fn == ["dryness", "fuel", "wind", "people", "roads"]


def test_assemble_shapes_and_ybar():
    rng = np.random.default_rng(2)
    pres, cand, fn, tau = dg.synthetic_reporting_scenario(rng, n_cells=1500)
    s = dg.assemble_ignition_samples(pres, cand, fn, rng, tau=tau, neg_per_pos=3.0)
    assert s.X.shape[1] == len(fn)
    assert s.X.shape[0] == s.y.shape[0] == s.lon.shape[0]
    assert abs(s.ybar - 0.25) < 0.03
    assert set(np.unique(s.cause)) <= {"human", "lightning", ""}
    # lon/lat are NOT in the feature matrix
    assert "lon" not in s.feature_names and "lat" not in s.feature_names


def test_block_group_and_cause_mask():
    rng = np.random.default_rng(2)
    pres, cand, fn, tau = dg.synthetic_reporting_scenario(rng, n_cells=1200)
    s = dg.assemble_ignition_samples(pres, cand, fn, rng, tau=tau)
    g = s.block_group()
    assert g.shape[0] == s.y.shape[0] and len(np.unique(g)) > 1
    hm = s.cause_mask("human")
    # a cause head sees that cause's positives plus all backgrounds, never the other cause
    assert not np.any((s.cause == "lightning") & hm)
    assert np.all(s.y[~hm] == 1)  # excluded rows are all positives of the other cause


def test_scorecard_prior_correction_moves_mean_to_base_rate():
    pytest.importorskip("sklearn")
    from vhagar.eval.danger import ignition_scorecard
    rng = np.random.default_rng(4)
    y = (rng.random(2000) < 0.2).astype(int)
    prob = np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(0, 0.1, y.size), 1e-3, 1 - 1e-3)
    raw = ignition_scorecard(y, prob, tau=0.02, ybar=0.2, prior_correct=False)
    cor = ignition_scorecard(y, prob, tau=0.02, ybar=0.2, prior_correct=True)
    # correction pulls the mean probability toward the true (much lower) base rate
    assert cor["mean_prob"] < raw["mean_prob"]
    assert cor["mean_prob"] < 0.1
    for k in ("auprc", "brier", "reliability", "resolution", "ece", "bss_vs_climatology"):
        assert k in cor


def test_reporting_weight_tracks_occurrence_density():
    # Presences clustered at the origin; candidates near vs far. Nearby candidate
    # gets more weight, and weights normalise.
    pres_lon = np.zeros(50)
    pres_lat = np.zeros(50)
    cand_lon = np.array([0.0, 0.1, 5.0])
    cand_lat = np.zeros(3)
    w = dg.reporting_weight(cand_lon, cand_lat, pres_lon, pres_lat, bandwidth_deg=0.25)
    assert w[0] > w[2]
    assert abs(float(w.sum()) - 1.0) < 1e-9


def test_frames_to_records_roundtrip_and_autoweight():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    pres, cand, fn, tau = dg.synthetic_reporting_scenario(rng, n_cells=800)
    pdf = pd.DataFrame({k: pres[k] for k in ("id", "lon", "lat", "year", "stratum", "cause", *fn)})
    cdf = pd.DataFrame({k: cand[k] for k in ("id", "lon", "lat", "year", "stratum", *fn)})  # no weight
    P, C, FN = dg.frames_to_records(pdf, cdf, fn)
    assert fn == FN
    assert "weight" in C and abs(float(C["weight"].sum()) - 1.0) < 1e-6   # auto target-group weight
    assert {"cause", *fn} <= set(P) and P["lon"].shape[0] == len(pdf)
    assert "lon" not in FN  # features stay clean


@pytest.mark.slow
def test_target_group_background_defuses_the_trap():
    """The regression that matters: target-group sampling collapses the model's
    reliance on the human-footprint covariate that naive sampling inflates."""
    pytest.importorskip("sklearn")
    from vhagar.eval.danger import top_features
    rng = np.random.default_rng(3)
    pres, cand, fn, tau = dg.synthetic_reporting_scenario(rng, n_cells=2500)

    def footprint_importance(**kw):
        s = dg.assemble_ignition_samples(pres, cand, fn, np.random.default_rng(7),
                                         tau=tau, neg_per_pos=3.0, **kw)
        tf = dict(top_features(s, n_folds=3, seed=0, k=5))
        return tf.get("people", 0.0) + tf.get("roads", 0.0)

    naive = footprint_importance(use_target_group=False, stratify=False)
    tgb = footprint_importance(use_target_group=True, stratify=False)
    assert tgb < naive  # the artefact reliance is reduced by target-group sampling
