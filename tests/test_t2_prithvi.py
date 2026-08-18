"""T2 Prithvi glue: leakage-proof split, chipping, and fair skill scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vhagar.eval.t2_prithvi import (
    chip_sample,
    grouped_split,
    score_masks,
    stitch_chip_predictions,
    summarise_scores,
)


@dataclass
class _Sample:
    features: np.ndarray
    reference: np.ndarray
    valid: np.ndarray
    is_usable: bool = True


def _sample(H=300, W=260, C=6, burn_top=True):
    feats = np.random.default_rng(0).normal(0.2, 0.05, (C, H, W)).astype("float32")
    feats[:, 0, 0] = np.nan
    burned = np.zeros((H, W), bool)
    if burn_top:
        burned[: H // 2] = True
    valid = np.ones((H, W), bool)
    valid[0, 0] = False
    return _Sample(feats, burned, valid)


def test_grouped_split_puts_each_fire_in_one_split():
    ids = [f"fire{i}" for i in range(10)]
    sp = grouped_split(ids, val_frac=0.2, test_frac=0.2, seed=1)
    allids = sp["train"] + sp["val"] + sp["test"]
    assert sorted(allids) == sorted(ids)                 # partition, nothing lost
    assert len(set(allids)) == len(ids)                  # and nothing duplicated
    assert len(sp["test"]) == 2 and len(sp["val"]) == 2


def test_chip_sample_makes_six_band_chips_with_signed_labels():
    s = _sample()
    chips = chip_sample(s, chip=224, min_valid_frac=0.1)
    assert chips, "expected at least one chip"
    ch = chips[0]
    assert ch["image"].shape == (6, 224, 224)
    assert not np.isnan(ch["image"]).any()               # NaN filled with 0
    labs = np.unique(ch["label"])
    assert set(labs.tolist()) <= {-1, 0, 1}              # HLS Burn Scars convention
    assert ch["label"].dtype == np.int8


def test_burn_balance_rebalances_chips_toward_fire():
    # a big window with a small burn: uniform tiling is mostly all-unburned chips;
    # burn_balance keeps the burn chips and caps background to 1x, so >= ~half contain fire
    H = W = 900
    feats = np.random.default_rng(0).normal(0.2, 0.05, (6, H, W)).astype("float32")
    burned = np.zeros((H, W), bool)
    burned[400:520, 400:520] = True                    # one ~120px burn blob
    valid = np.ones((H, W), bool)
    s = _Sample(feats, burned, valid)

    plain = chip_sample(s, chip=224, burn_balance=False)
    bal = chip_sample(s, chip=224, burn_balance=True, max_bg_ratio=1.0, seed=0)
    frac_burn = np.mean([bool((c["label"] == 1).any()) for c in bal])
    frac_burn_plain = np.mean([bool((c["label"] == 1).any()) for c in plain])
    assert frac_burn > frac_burn_plain                 # rebalanced set is burn-richer
    assert frac_burn >= 0.45                            # roughly balanced (>= 1:1 with bg cap)


def test_chip_sample_pads_a_small_sample_up_to_one_chip():
    s = _sample(H=100, W=120)                             # smaller than a 224 chip
    chips = chip_sample(s, chip=224, min_valid_frac=0.0)
    assert len(chips) == 1
    assert chips[0]["image"].shape == (6, 224, 224)


def test_stitch_reassembles_chips_into_a_fire_mask():
    # two chips of a 300x160 fire: left half and right half, each predicts its quadrant burned
    manifest = {
        "fireA_0": {"event_id": "fireA", "y0": 0, "x0": 0, "H": 300, "W": 160},
        "fireA_1": {"event_id": "fireA", "y0": 0, "x0": 80, "H": 300, "W": 160},
    }
    left = np.zeros((300, 80), np.uint8)
    left[:150] = 1                                              # top-left burned
    right = np.zeros((300, 80), np.uint8)
    right[150:] = 1                                             # bottom-right burned
    out = stitch_chip_predictions({"fireA_0": left, "fireA_1": right}, manifest)
    assert set(out) == {"fireA"}
    m = out["fireA"]
    assert m.shape == (300, 160)
    assert m[:150, :80].all() and m[150:, 80:].all()           # placed at the right offsets
    assert not m[150:, :80].any() and not m[:150, 80:].any()   # the other quadrants unburned


def test_stitch_clips_edge_chip_overhang():
    # a chip larger than the remaining fire extent is clipped, not out-of-bounds
    manifest = {"fB_0": {"event_id": "fB", "y0": 0, "x0": 0, "H": 100, "W": 90}}
    pred = np.ones((224, 224), np.uint8)                       # overhangs 100x90
    out = stitch_chip_predictions({"fB_0": pred}, manifest)
    assert out["fB"].shape == (100, 90) and out["fB"].all()


def test_nbr_threshold_baseline_separates_burn_and_scores():
    from vhagar.eval.t2_prithvi import nbr_threshold_baseline

    # four fires; burned pixels have low NBR (low NIR / high SWIR2), unburned high NBR
    def fire(seed):
        rng = np.random.default_rng(seed)
        H = W = 260
        f = rng.normal(0.2, 0.02, (6, H, W)).astype("float32")
        burned = np.zeros((H, W), bool)
        burned[: H // 2] = True
        f[3][burned] = 0.15   # NIR low  where burned
        f[5][burned] = 0.35   # SWIR2 high where burned  -> NBR very negative
        f[3][~burned] = 0.40
        f[5][~burned] = 0.10  # -> NBR positive where unburned
        return _Sample(f, burned, np.ones((H, W), bool))

    samples = {f"fire{i}": fire(i) for i in range(4)}
    scores, thr = nbr_threshold_baseline(samples, seed=0)
    assert scores, "expected at least one test fire"
    assert scores[0].skill_f1 > 0                      # a clean NBR split beats predict-all-burned
    assert -1.0 < thr < 1.0


def test_nbr_transfer_tunes_on_train_scores_on_test():
    from vhagar.eval.t2_prithvi import nbr_threshold_transfer

    def fire(seed):
        rng = np.random.default_rng(seed)
        H = W = 240
        f = rng.normal(0.2, 0.02, (6, H, W)).astype("float32")
        burned = np.zeros((H, W), bool)
        burned[: H // 2] = True
        f[3][burned], f[5][burned] = 0.15, 0.35        # low NBR where burned
        f[3][~burned], f[5][~burned] = 0.40, 0.10      # high NBR where unburned
        s = _Sample(f, burned, np.ones((H, W), bool))
        s.event_id = f"fire{seed}"                     # score_masks/_score_nbr read event_id
        return s

    train = [fire(i) for i in range(3)]
    test = [fire(10)]
    scores, thr = nbr_threshold_transfer(train, test)
    assert scores[0].skill_f1 > 0                       # a CONUS-tuned cut transfers to a clean test fire
    assert -1.0 < thr < 1.0


def test_score_masks_matches_confusion_and_naive():
    s = _sample()                                        # top half burned
    # a perfect prediction leads the predict-all-burned naive on this half-burned window
    pred_perfect = s.reference.copy()
    scores = score_masks({"fireA": pred_perfect}, {"fireA": s})
    assert len(scores) == 1
    assert scores[0].f1 > scores[0].naive_f1             # positive skill
    assert scores[0].skill_f1 > 0
    summ = summarise_scores(scores)
    assert summ["fires"] == 1 and summ["fires_positive_skill"] == 1
