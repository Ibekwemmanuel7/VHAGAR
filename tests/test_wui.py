"""WUI spread extension (T4) + structure-loss scoring. Deterministic, no RNG, no torch."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.eval import wui as ewui
from vhagar.models import wui


def test_rasterize_structures_conserves_weight_and_drops_oob():
    g = wui.rasterize_structures([0, 0, 2, 5], [0, 0, 1, 5], (3, 3))
    assert g[0, 0] == 2.0                 # two structures stack in one cell
    assert g[2, 1] == 1.0
    assert g.sum() == 3.0                 # (5, 5) is out of bounds, dropped
    assert wui.rasterize_structures([], [], (3, 3)).sum() == 0.0


def test_spotting_kernel_normalized_center_zero():
    k = wui.spotting_kernel(4, wind_speed=1.0, wind_dir=0.0)
    R = k.shape[0] // 2
    assert k[R, R] == 0.0                 # a brand is not a self-spot
    assert abs(k.sum() - 1.0) < 1e-9      # conserves brands


def test_spotting_kernel_downwind_bias_and_isotropy():
    R = 4
    k = wui.spotting_kernel(R, wind_speed=1.0, wind_dir=0.0)   # downwind = +x (cols)
    c = k.shape[1] // 2
    assert k[:, c + 1:].sum() > k[:, :c].sum()                 # mass shifted downwind
    k0 = wui.spotting_kernel(R, wind_speed=0.0, wind_dir=0.0)
    assert abs(k0[:, c + 1:].sum() - k0[:, :c].sum()) < 1e-9   # no wind -> symmetric


def test_spot_ignition_probability_bounds_and_downwind():
    em = np.zeros((21, 21))
    em[10, 10] = 1.0
    rec = np.ones((21, 21))
    p = wui.spot_ignition_probability(em, rec, wind_speed=1.0, wind_dir=0.0,
                                      max_dist_cells=6, intensity=5.0)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert p[10, 13] > p[10, 7]                                # downwind more likely
    p0 = wui.spot_ignition_probability(em, np.zeros((21, 21)), 1.0, 0.0)
    assert np.allclose(p0, 0.0)                                # no fuel -> no spots


def test_structures_reached_by_front():
    arr = np.full((3, 3), np.inf)
    arr[0, 0] = 2.0
    arr[2, 2] = 20.0
    reached = wui.structures_reached_by_front([0, 2, 9], [0, 2, 9], arr, horizon=10.0)
    assert reached.tolist() == [True, False, False]           # third is out of bounds


def test_s2s_line_propagates_downwind():
    r, c = [0, 0, 0, 0], [0, 1, 2, 3]
    seed = np.array([True, False, False, False])
    out = wui.structure_to_structure_spread(r, c, seed, wind_speed=1.0, wind_dir=0.0,
                                            radius_cells=3, base_p=0.9)
    assert out.tolist() == [True, True, True, True]


def test_s2s_out_of_radius_isolated():
    out = wui.structure_to_structure_spread([0, 0], [0, 20], np.array([True, False]),
                                            radius_cells=3, base_p=0.9)
    assert out.tolist() == [True, False]


def test_s2s_no_seed_no_ignition():
    out = wui.structure_to_structure_spread([0, 1], [0, 1], np.array([False, False]),
                                            radius_cells=3, base_p=0.9)
    assert not out.any()


def test_wui_spread_end_to_end():
    H = W = 41
    ros = np.ones((H, W))
    burned = np.zeros((H, W), bool)
    burned[20, 20] = True
    # a downwind (+x) cluster near the ignition, plus one far corner structure
    srows = [20, 20, 20, 20, 0]
    scols = [24, 25, 26, 27, 40]
    out = wui.wui_spread(burned, ros, srows, scols, horizon=8.0,
                         wind_speed=1.0, wind_dir=0.0,
                         struct_radius_cells=3, struct_base_p=0.9)
    assert out["n_structures"] == 5
    assert bool(out["destroyed"][:4].all())      # near cluster destroyed
    assert not bool(out["destroyed"][4])         # far corner survives
    assert out["arrival"].shape == (H, W)


def test_wui_spread_anisotropic_front_elongates_downwind():
    H = W = 61
    ros = np.ones((H, W))
    burned = np.zeros((H, W), bool)
    burned[30, 30] = True
    # structures equidistant downwind (col 45) and upwind (col 15) of the ignition
    out = wui.wui_spread(burned, ros, [30, 30], [45, 15], horizon=8.0,
                         wind_speed=1.0, wind_dir=0.0, anisotropic=True,
                         struct_radius_cells=1, struct_base_p=0.9)
    arr = out["arrival"]
    assert arr[30, 45] < arr[30, 15]      # wind-driven front reaches downwind far sooner


def test_wui_spread_edge_seeding_reaches_adjacent_structures():
    H = W = 41
    ros = np.ones((H, W))
    burned = np.zeros((H, W), bool)
    burned[20, 20] = True
    # structure a few cells from the ignition; with horizon 1 the front does not reach its
    # own cell, but edge-seeding (3 cells) ignites it as adjacent to the burned area.
    base = dict(horizon=1.0, struct_radius_cells=1.0, struct_base_p=0.9)
    off = wui.wui_spread(burned, ros, [20], [23], struct_edge_cells=0.0, **base)
    on = wui.wui_spread(burned, ros, [20], [23], struct_edge_cells=3.0, **base)
    assert bool(on["reached"][0]) and not bool(off["reached"][0])


def test_score_structures_confusion():
    pred = np.array([True, True, False, False])
    truth = np.array([True, False, True, False])
    s = ewui.score_structures(pred, truth)
    assert (s.tp, s.fp, s.fn, s.tn) == (1, 1, 1, 1)
    assert abs(s.pod - 0.5) < 1e-9
    assert abs(s.far - 0.5) < 1e-9
    assert abs(s.f1 - 0.5) < 1e-9
    with pytest.raises(ValueError):
        ewui.score_structures(np.array([True]), np.array([True, False]))


def test_match_points_nearest_within_tol():
    pi, oi = ewui.match_points([0, 5, 10], [0, 5, 10], [0, 10], [0, 10], tol_cells=1.0)
    # pred 0 -> obs 0, pred 10 -> obs 10, pred 5 (mid) unmatched (out of tol)
    assert set(zip(pi.tolist(), oi.tolist(), strict=True)) == {(0, 0), (2, 1)}
