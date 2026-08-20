"""T2 decisive experiment: Prithvi vs U-Net vs RBR-threshold, same fires, with CIs.

The project's declared bar for the burned-area claim is not "beat the RBR
threshold" but "beat the in-repo U-Net on the identical transfer fires, and show
the margin is real, not a fold-variance artefact". This module runs that
head-to-head:

* one leakage-proof grouped split by whole fire (``t2_prithvi.grouped_split``);
* per-fire skill-over-naive (``f1 - predict_all_burned_f1``) for each model,
  scored through the same ``eval.metrics.confusion_counts`` path so the three
  numbers are directly comparable;
* paired bootstrap confidence intervals on the per-fire differences
  (Prithvi - U-Net, Prithvi - RBR, U-Net - RBR).

The RBR threshold is always available (pure numpy). The U-Net path needs torch;
if it is absent the harness reports RBR alone and whatever Prithvi masks were
supplied. Prithvi itself is fine-tuned out of repo (TerraTorch, GPU); its
per-fire predicted masks are handed in via ``prithvi_pred_by_event`` (build them
with ``t2_prithvi.stitch_chip_predictions`` from the TerraTorch inference run),
so this module never needs a GPU.

Nothing here fabricates a comparison: a model only appears in the report if it
actually produced a prediction for the held-out fire.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PairedDiff",
    "bootstrap_paired_diff",
    "per_fire_skill_rbr",
    "per_fire_skill_unet",
    "per_fire_skill_prithvi",
    "head_to_head",
]


@dataclass(frozen=True)
class PairedDiff:
    """A paired per-fire skill difference ``a - b`` with a bootstrap CI.

    ``prob_a_better`` is the bootstrap fraction of resamples with a positive mean
    difference: a one-sided credibility that model ``a`` beats model ``b`` on
    these fires. It is descriptive, not a p-value.
    """

    a: str
    b: str
    n_fires: int
    mean_diff: float
    ci_lo: float
    ci_hi: float
    prob_a_better: float

    @property
    def separable(self) -> bool:
        """True when the CI excludes zero (the margin is not a coin-flip)."""
        return self.ci_lo > 0.0 or self.ci_hi < 0.0


def bootstrap_paired_diff(
    a: Sequence[float], b: Sequence[float], *, n_boot: int = 10_000, ci: float = 0.95,
    seed: int = 0, label_a: str = "a", label_b: str = "b",
) -> PairedDiff:
    """Paired bootstrap of the mean of ``a - b`` over fires (resamples fires, not
    pixels: the fire is the independent unit, per the leakage doctrine)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be equal-length, non-empty per-fire vectors")
    d = a - b
    n = d.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo = float(np.quantile(boot, (1.0 - ci) / 2.0))
    hi = float(np.quantile(boot, 1.0 - (1.0 - ci) / 2.0))
    return PairedDiff(
        a=label_a, b=label_b, n_fires=n, mean_diff=float(d.mean()),
        ci_lo=lo, ci_hi=hi, prob_a_better=float((boot > 0.0).mean()),
    )


def per_fire_skill_rbr(train_samples: Sequence, test_samples: Sequence) -> dict[str, float]:
    """RBR-threshold transfer baseline: tune the cut on the training fires, apply
    it unchanged to each held-out fire, return ``{event_id: skill_f1}``."""
    from vhagar.eval.t2_prithvi import nbr_threshold_transfer

    scores, _thr = nbr_threshold_transfer(train_samples, test_samples)
    return {s.event_id: float(s.skill_f1) for s in scores}


def per_fire_skill_unet(
    train_samples: Sequence, test_samples: Sequence, *, seed: int = 0, **train_kw,
) -> dict[str, float]:
    """Train the in-repo U-Net on the training fires and score each held-out fire.
    Returns ``{event_id: skill_f1}``. Requires torch (raises ImportError if absent)."""
    from vhagar.eval.t2_unet import evaluate_unet, train_unet

    model, std = train_unet(train_samples, seed=seed, **train_kw)
    out: dict[str, float] = {}
    for s in test_samples:
        f1, _iou, naive = evaluate_unet(model, std, s)
        out[s.event_id] = float(f1) - float(naive)
    return out


def per_fire_skill_prithvi(
    prithvi_pred_by_event: Mapping, samples_by_id: Mapping,
) -> dict[str, float]:
    """Skill of externally-produced Prithvi masks, scored on the same pixels/metric
    as the other two. ``prithvi_pred_by_event`` maps event id to a burned mask."""
    from vhagar.eval.t2_prithvi import score_masks

    scores = score_masks(dict(prithvi_pred_by_event), dict(samples_by_id))
    return {s.event_id: float(s.skill_f1) for s in scores}


def _aligned(x: Mapping[str, float], y: Mapping[str, float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    fires = sorted(set(x) & set(y))
    return fires, np.array([x[f] for f in fires]), np.array([y[f] for f in fires])


def head_to_head(
    samples_by_id: Mapping,
    *,
    prithvi_pred_by_event: Mapping | None = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    n_boot: int = 10_000,
    ci: float = 0.95,
    run_unet: bool = True,
    unet_kw: Mapping | None = None,
) -> dict:
    """Run the head-to-head on one grouped split and return a report dict.

    ``samples_by_id`` maps event id -> :class:`vhagar.datasets.burned_area.T2Sample`.
    The report holds each model's mean per-fire skill, the held-out fire list, and
    the paired bootstrap differences among whichever models produced predictions.
    """
    from vhagar.eval.t2_prithvi import grouped_split

    split = grouped_split(list(samples_by_id), val_frac=val_frac, test_frac=test_frac, seed=seed)
    train_ids = list(split["train"]) + list(split["val"])
    test_ids = list(split["test"])
    train_samples = [samples_by_id[i] for i in train_ids]
    test_samples = [samples_by_id[i] for i in test_ids]
    test_by_id = {i: samples_by_id[i] for i in test_ids}

    skills: dict[str, dict[str, float]] = {}
    notes: list[str] = []

    # RBR threshold: always available.
    skills["rbr"] = per_fire_skill_rbr(train_samples, test_samples)

    # U-Net: needs torch.
    if run_unet:
        try:
            skills["unet"] = per_fire_skill_unet(
                train_samples, test_samples, seed=seed, **(dict(unet_kw) if unet_kw else {}))
        except ImportError:
            notes.append("u-net skipped: torch not installed")
    else:
        notes.append("u-net skipped: run_unet=False")

    # Prithvi: only if external predictions were supplied.
    if prithvi_pred_by_event:
        skills["prithvi"] = per_fire_skill_prithvi(prithvi_pred_by_event, test_by_id)
    else:
        notes.append("prithvi skipped: no predicted masks supplied")

    means = {m: (float(np.mean(list(s.values()))) if s else float("nan"))
             for m, s in skills.items()}

    # Paired differences among the models that ran, most-capable first.
    order = [m for m in ("prithvi", "unet", "rbr") if m in skills]
    diffs: list[PairedDiff] = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            fires, va, vb = _aligned(skills[a], skills[b])
            if not fires:
                continue
            diffs.append(bootstrap_paired_diff(
                va, vb, n_boot=n_boot, ci=ci, seed=seed, label_a=a, label_b=b))

    return {
        "split": split,
        "n_test_fires": len(test_ids),
        "test_fires": test_ids,
        "per_fire_skill": skills,
        "mean_skill": means,
        "paired_diffs": diffs,
        "notes": notes,
    }
