"""T2 deep models on the stack: T2Sample.stack, numpy core, torch smoke."""

from __future__ import annotations

import numpy as np
import pytest

from vhagar.datasets.burned_area import T2Sample, make_sample
from vhagar.eval.t2_deep import (
    channel_standardizer_from,
    random_feature_crops,
)


def _stack_fire(event_id, burned_frac=0.3, n=120, seed=0):
    """Synthetic bi-temporal fire: burned pixels drop post-NBR and spike dNBR."""
    rng = np.random.default_rng(seed)
    truth = rng.random((n, n)) < burned_frac
    pre = rng.normal(0.4, 0.08, (n, n))                       # healthy veg NBR
    post = np.where(truth, rng.normal(-0.2, 0.08, (n, n)), pre + rng.normal(0, 0.05, (n, n)))
    dnbr = (pre - post) * 1000.0
    stack = np.stack([pre, post, dnbr]).astype(np.float32)
    rbr = dnbr / (pre + 1.001)
    return make_sample(event_id, rbr, truth, stack=stack, tile_id="conus/x1_y1")


# ------------------------------------------------------- T2Sample.stack --------


def test_sample_carries_stack_and_features_prefers_it():
    s = _stack_fire("f", seed=1)
    assert s.stack is not None and s.stack.shape == (3, 120, 120)
    assert s.features.shape == (3, 120, 120)          # uses the stack
    # a sample with no stack falls back to the single predictor channel
    plain = make_sample("p", s.predictor, s.reference)
    assert plain.stack is None
    assert plain.features.shape == (1, 120, 120)


def test_stack_round_trips_through_npz(tmp_path):
    s = _stack_fire("mtbs:CA1", seed=2)
    back = T2Sample.load(s.save(tmp_path / "s.npz"))
    assert back.stack is not None
    assert np.allclose(back.stack, s.stack)
    assert back.features.shape == s.features.shape


def test_make_sample_rejects_a_misshaped_stack():
    with pytest.raises(ValueError, match="stack shape"):
        make_sample("f", np.zeros((10, 10)), np.zeros((10, 10), bool),
                    stack=np.zeros((3, 10, 9)))


# --------------------------------------------------------- numpy core ----------


def test_channel_standardizer_is_per_channel():
    s = _stack_fire("f", seed=3)
    std = channel_standardizer_from([s])
    assert std.center.shape == (3,) and std.scale.shape == (3,)
    out = std.apply(s.features, s.valid)
    assert out.shape == s.features.shape
    assert out.min() >= -5.0 and out.max() <= 5.0
    assert np.all(out[:, ~s.valid] == 0.0)


def test_feature_crops_keep_all_channels():
    s = _stack_fire("f", burned_frac=0.1, seed=4)
    rng = np.random.default_rng(0)
    crops = random_feature_crops(s, crop=64, n=20, rng=rng, burned_bias=1.0)
    assert crops and all(f.shape == (3, 64, 64) for f, _, _ in crops)
    frac = np.mean([(b & v).mean() for _, b, v in crops])
    assert frac > 0.1                                  # burned-biased


# ------------------------------------------------------ torch smoke ------------


@pytest.mark.parametrize("model_kind", ["siamese", "unet"])
def test_deep_model_learns_a_separable_fire(model_kind):
    pytest.importorskip("torch")
    from vhagar.eval.t2_deep import run_deep_cv, summarise_deep_cv

    samples = {f"f{i}": _stack_fire(f"f{i}", burned_frac=0.25, n=80, seed=i) for i in range(4)}
    results = run_deep_cv(
        samples, model_kind=model_kind, k=2, epochs=3, crop=48, crops_per_fire=8, seed=0,
    )
    assert results
    assert any(r.skill_f1 > 0.0 for r in results)
    s = summarise_deep_cv(results)
    assert "deep_skill_mean" in s and 0 <= s["deep_beats_thr"] <= s["fires"]
