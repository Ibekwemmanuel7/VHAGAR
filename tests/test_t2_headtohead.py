"""Prithvi-vs-U-Net head-to-head harness tests (no torch, no GPU needed)."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.datasets.burned_area import T2Sample
from vhagar.eval import t2_headtohead as hh
from vhagar.eval import t2_prithvi as t2p


def test_restrict_to_heldout_filters_and_guards():
    split = {"train": ["a", "b"], "val": ["c"], "test": ["d", "e"]}
    # strict: a prediction for a train/val fire is a hard error
    with pytest.raises(ValueError, match="trained on"):
        t2p.restrict_to_heldout({"a": 1, "d": 1}, split, strict=True)
    # non-strict: in-sample fires are dropped, only test fires survive
    kept = t2p.restrict_to_heldout({"a": 1, "d": 1, "e": 1, "z": 1}, split, strict=False)
    assert set(kept) == {"d", "e"}                      # "z" is outside the split entirely


def test_load_split_validates_keys(tmp_path):
    import json
    good = tmp_path / "s.json"
    good.write_text(json.dumps({"train": ["a"], "val": [], "test": ["b"]}), encoding="utf-8")
    assert t2p.load_split(good)["test"] == ["b"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"train": ["a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        t2p.load_split(bad)


def test_bootstrap_paired_diff_separates_a_clear_winner():
    a = np.array([0.6, 0.7, 0.65, 0.72, 0.68])
    b = a - 0.15                                   # a beats b by 0.15 on every fire
    pd_ = hh.bootstrap_paired_diff(a, b, n_boot=2000, seed=0, label_a="prithvi", label_b="rbr")
    assert pd_.mean_diff == pytest.approx(0.15, abs=1e-6)
    assert pd_.ci_lo > 0.0 and pd_.separable
    assert pd_.prob_a_better > 0.99
    assert pd_.n_fires == 5


def test_bootstrap_paired_diff_not_separable_when_noisy():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.1, size=8)
    b = a + rng.normal(0.0, 0.1, size=8)           # no systematic difference
    pd_ = hh.bootstrap_paired_diff(a, b, n_boot=2000, seed=2)
    assert not pd_.separable                        # CI straddles zero
    with pytest.raises(ValueError):
        hh.bootstrap_paired_diff([1.0], [1.0, 2.0])


def _burn_sample(eid: str, *, burned_frac: float, seed: int) -> T2Sample:
    """A tiny 6-band fire: burned pixels have low NBR (low NIR idx3, high SWIR2 idx5)."""
    rng = np.random.default_rng(seed)
    H = W = 16
    truth = np.zeros((H, W), dtype=bool)
    cut = int(W * burned_frac)
    truth[:, :cut] = True
    stack = np.zeros((6, H, W), dtype=np.float64)
    for c in range(6):
        stack[c] = rng.uniform(0.05, 0.15, size=(H, W))
    # NIR (3) high on unburned, low on burned; SWIR2 (5) the reverse -> NBR separates.
    stack[3] = np.where(truth, 0.10, 0.45) + rng.normal(0, 0.01, (H, W))
    stack[5] = np.where(truth, 0.40, 0.10) + rng.normal(0, 0.01, (H, W))
    predictor = (stack[3] - stack[5]) / (stack[3] + stack[5] + 1e-6)   # NBR-like
    valid = np.ones((H, W), dtype=bool)
    return T2Sample(event_id=eid, tile_id=None, predictor=predictor,
                    reference=truth, valid=valid, stack=stack)


def test_per_fire_skill_prithvi_scores_supplied_masks():
    samples = {f"f{i}": _burn_sample(f"f{i}", burned_frac=0.4, seed=i) for i in range(3)}
    # a near-perfect predicted mask should beat predict-all-burned (positive skill)
    preds = {eid: s.reference.copy() for eid, s in samples.items()}
    skill = hh.per_fire_skill_prithvi(preds, samples)
    assert set(skill) == set(samples)
    assert all(v > 0.0 for v in skill.values())


def test_head_to_head_runs_rbr_and_prithvi_without_torch():
    # enough fires that grouped_split yields a non-empty test set
    samples = {f"f{i}": _burn_sample(f"f{i}", burned_frac=0.4, seed=i) for i in range(8)}
    split = t2p.grouped_split(list(samples), seed=0)
    # Prithvi only ever predicts the fires IT held out (the test split).
    preds = {eid: samples[eid].reference.copy() for eid in split["test"]}
    rep = hh.head_to_head(samples, prithvi_pred_by_event=preds, prithvi_split=split,
                          run_unet=False, n_boot=500, seed=0)
    assert rep["n_test_fires"] >= 1
    assert "rbr" in rep["per_fire_skill"] and "prithvi" in rep["per_fire_skill"]
    assert "rbr" in rep["mean_skill"] and "prithvi" in rep["mean_skill"]
    # every scored Prithvi fire is a held-out test fire, none from train/val
    assert set(rep["per_fire_skill"]["prithvi"]) <= set(split["test"])
    # a Prithvi-vs-RBR paired diff is produced over the shared test fires
    names = {(d.a, d.b) for d in rep["paired_diffs"]}
    assert ("prithvi", "rbr") in names
    assert any("u-net skipped" in n for n in rep["notes"])


def test_head_to_head_requires_split_when_prithvi_supplied():
    # Supplying Prithvi masks without its split cannot be verified out-of-sample -> hard error.
    samples = {f"f{i}": _burn_sample(f"f{i}", burned_frac=0.4, seed=i) for i in range(8)}
    preds = {eid: s.reference.copy() for eid, s in samples.items()}
    with pytest.raises(ValueError, match="without prithvi_split"):
        hh.head_to_head(samples, prithvi_pred_by_event=preds, run_unet=False, seed=0)


def test_head_to_head_rejects_in_sample_prithvi_prediction():
    # A Prithvi mask for a TRAIN fire is in-sample; the harness must refuse to score it.
    samples = {f"f{i}": _burn_sample(f"f{i}", burned_frac=0.4, seed=i) for i in range(8)}
    split = t2p.grouped_split(list(samples), seed=0)
    leaked_fire = split["train"][0]
    preds = {eid: samples[eid].reference.copy() for eid in split["test"]}
    preds[leaked_fire] = samples[leaked_fire].reference.copy()   # the leak
    with pytest.raises(ValueError, match="trained on"):
        hh.head_to_head(samples, prithvi_pred_by_event=preds, prithvi_split=split,
                        run_unet=False, seed=0)
