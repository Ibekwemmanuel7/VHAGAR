"""T2 Stage-0: the calibrated-threshold burned-area baseline, per fold.

This is the first honest number. For each fold of a leakage-proof split it
calibrates a dNBR/RBR threshold on the training fires, applies it to the test
fires, and reports the map accuracy plus an **Olofsson error-adjusted burned area
with a 95% confidence interval**, never a raw pixel count. The threshold is the
Stage-0 "model"; the architecture is explicit that a tuned spectral baseline may
well be the product, and that it must be reported next to everything fancier.

Two disciplines are enforced here:

* **No leakage.** The threshold is tuned only on ``fold["train"]`` and applied to
  ``fold["test"]``. The split machinery guarantees the ids are disjoint.
* **Area, not pixels.** The burned class is rare, so the reference sample floors
  it (``allocate_samples``) and the area is the Olofsson estimate with its CI.
  The reference draw is seeded, so the CI is reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from vhagar.eval.area_estimation import allocate_samples, estimate_areas
from vhagar.eval.baselines import threshold_baseline, tune_threshold
from vhagar.eval.metrics import confusion_counts

__all__ = ["FoldResult", "run_stage0", "summarise_stage0"]

#: MTBS rasters are 30 m; a pixel is 0.09 ha. Change for a different predictor.
MTBS_PIXEL_AREA_HA = 0.09


@dataclass(slots=True)
class FoldResult:
    """One fold's Stage-0 outcome."""

    held_out: str
    threshold: float
    n_test_valid: int
    f1: float
    iou: float
    mapped_burned_ha: float
    adjusted_burned_ha: float | None
    ci95_ha: float | None
    note: str = ""


def _pool(samples: Sequence, cap: int | None, rng) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate valid predictor and reference pixels, optionally subsampled.

    When a cap is given the subsampling happens **per sample**, before
    concatenation, so the pool never materialises every training fire's pixels at
    once. That keeps memory flat across a leave-one-fire-out fold with a dozen
    large fires, which otherwise concatenates hundreds of millions of pixels.
    """
    if not samples:
        return np.empty(0), np.empty(0, dtype=bool)
    per = None if cap is None else max(1, cap // len(samples))
    preds, refs = [], []
    for s in samples:
        v = s.valid
        p = s.predictor[v]
        r = s.reference[v]
        if per is not None and p.size > per:
            idx = rng.choice(p.size, size=per, replace=False)
            p, r = p[idx], r[idx]
        preds.append(p)
        refs.append(r)
    return np.concatenate(preds), np.concatenate(refs)


def _stratified_confusion(pred_mask, truth, n_alloc, rng) -> np.ndarray:
    """2x2 map-vs-reference counts from a stratified reference draw.

    Rows are map classes (0 unburned, 1 burned), columns reference classes, as
    :func:`estimate_areas` requires.
    """
    conf = np.zeros((2, 2), dtype=np.int64)
    for k in (0, 1):
        idx = np.flatnonzero(pred_mask == k)
        if idx.size == 0:
            continue
        take = min(int(n_alloc[k]), idx.size)
        chosen = rng.choice(idx, size=take, replace=False)
        t = truth[chosen].astype(int)
        conf[k, 0] = int(np.count_nonzero(t == 0))
        conf[k, 1] = int(np.count_nonzero(t == 1))
    return conf


def evaluate_fold(
    train_samples: Sequence,
    test_samples: Sequence,
    held_out: str = "",
    pixel_area_ha: float = MTBS_PIXEL_AREA_HA,
    n_reference: int = 500,
    min_per_rare: int = 75,
    max_tune_pixels: int = 2_000_000,
    objective: str = "f1",
    seed: int = 0,
) -> FoldResult:
    """Calibrate on train, evaluate on test, return the fold's Stage-0 result."""
    rng = np.random.default_rng(seed)
    tr_pred, tr_truth = _pool(train_samples, max_tune_pixels, rng)
    if tr_pred.size == 0:
        raise ValueError("no training pixels")
    threshold, _ = tune_threshold(tr_pred, tr_truth.astype(np.uint8), objective=objective)

    te_pred, te_truth = _pool(test_samples, None, rng)
    if te_pred.size == 0:
        raise ValueError("no test pixels")
    pred_mask = threshold_baseline(te_pred, threshold)
    truth_u8 = te_truth.astype(np.uint8)

    cc = confusion_counts(truth_u8, pred_mask)
    n_map = np.array([int(np.count_nonzero(pred_mask == 0)), int(np.count_nonzero(pred_mask == 1))])
    mapped_burned_ha = float(n_map[1] * pixel_area_ha)

    adjusted, ci, note = None, None, ""
    if n_map[1] == 0:
        note = "no pixels mapped as burned; nothing to adjust"
    else:
        weights = n_map / n_map.sum()
        try:
            alloc = allocate_samples(n_reference, weights, rare_classes=[1], min_per_rare=min_per_rare)
            conf = _stratified_confusion(pred_mask, truth_u8, alloc, rng)
            est = estimate_areas(conf, n_map * pixel_area_ha, class_names=["unburned", "burned"])
            burned = est[1]
            adjusted = float(burned.adjusted_area)
            ci = float(burned.margin_of_error)
        except (ValueError, ZeroDivisionError) as exc:
            note = f"Olofsson skipped: {exc}"

    return FoldResult(
        held_out=held_out or "",
        threshold=float(threshold),
        n_test_valid=int(te_pred.size),
        f1=float(cc.f1),
        iou=float(cc.iou),
        mapped_burned_ha=mapped_burned_ha,
        adjusted_burned_ha=adjusted,
        ci95_ha=ci,
        note=note,
    )


def run_stage0(
    samples_by_id: Mapping[str, object],
    manifest,
    pixel_area_ha: float = MTBS_PIXEL_AREA_HA,
    n_reference: int = 500,
    seed: int = 0,
    **kw,
) -> list[FoldResult]:
    """Run Stage 0 across every fold of a split manifest.

    ``samples_by_id`` maps event id to a :class:`~vhagar.datasets.burned_area.T2Sample`.
    Fires without a sample (no raster available) are skipped. Per-fold reporting
    is mandatory; that is what this returns.
    """
    results: list[FoldResult] = []
    for i, fold in enumerate(manifest.folds):
        train = [samples_by_id[u] for u in fold.get("train", []) if u in samples_by_id]
        test = [samples_by_id[u] for u in fold.get("test", []) if u in samples_by_id]
        if not train or not test:
            continue
        results.append(
            evaluate_fold(
                train, test,
                held_out=str(fold.get("held_out", i)),
                pixel_area_ha=pixel_area_ha, n_reference=n_reference,
                seed=seed + i, **kw,
            )
        )
    return results


def summarise_stage0(results: Sequence[FoldResult]) -> dict:
    """Mean and standard deviation of the fold metrics.

    Fold std is reported because on these tasks it rivals the spread between
    models; a single mean without it is not a defensible claim.
    """
    if not results:
        return {"folds": 0}
    f1 = np.array([r.f1 for r in results])
    iou = np.array([r.iou for r in results])
    return {
        "folds": len(results),
        "f1_mean": float(f1.mean()),
        "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        "iou_mean": float(iou.mean()),
        "iou_std": float(iou.std(ddof=1)) if len(iou) > 1 else 0.0,
    }
