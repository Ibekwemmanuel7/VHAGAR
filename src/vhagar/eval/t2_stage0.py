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
from vhagar.eval.baselines import otsu_threshold, threshold_baseline, tune_threshold
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
    #: F1 of the trivial "predict everything burned" classifier on this fold's
    #: test pixels. On a window that is ~90% burned this is ~0.95, so a model F1
    #: below it has *negative skill*: it is worse than ignoring the imagery. This
    #: is the permanent no-skill baseline the results must always be read against.
    naive_f1: float = 0.0
    naive_iou: float = 0.0
    note: str = ""

    @property
    def skill_f1(self) -> float:
        """F1 improvement over the predict-all-burned baseline. Can be negative."""
        return self.f1 - self.naive_f1


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
    method: str = "global",
    strata: Mapping[str, object] | None = None,
    seed: int = 0,
) -> FoldResult:
    """Evaluate a fold. ``method`` selects the thresholding strategy.

    ``"global"``: one threshold tuned on the training fires, applied to every test
    pixel. ``"otsu"``: an adaptive threshold per test fire from its own
    distribution (calibration-free). ``"perstratum"``: for each test fire, tune a
    threshold on the training fires that share its stratum (e.g. its Köppen
    climate zone), falling back to all training fires when the stratum is unseen.
    Per-stratum is the like-for-like transfer method: a threshold learned on US
    Mediterranean fires applied to Greek Mediterranean fires. ``strata`` maps
    event id to a stratum label.
    """
    rng = np.random.default_rng(seed)
    strata = strata or {}

    if method == "perstratum":
        preds, truths, thrs = [], [], []
        for s in test_samples:
            v = s.valid
            x = s.predictor[v]
            if x.size == 0:
                continue
            strat = strata.get(s.event_id)
            same = [t for t in train_samples if strata.get(t.event_id) == strat]
            xtr, ttr = _pool(same or train_samples, max_tune_pixels, rng)
            if xtr.size == 0:
                continue
            thr, _ = tune_threshold(xtr, ttr.astype(np.uint8), objective=objective)
            preds.append((x > thr).astype(np.uint8))
            truths.append(s.reference[v].astype(np.uint8))
            thrs.append(thr)
        if not preds:
            raise ValueError("no test pixels")
        pred_mask = np.concatenate(preds)
        truth_u8 = np.concatenate(truths)
        threshold = float(np.mean(thrs))
    elif method == "otsu":
        preds, truths, thrs = [], [], []
        for s in test_samples:
            v = s.valid
            x = s.predictor[v]
            if x.size == 0:
                continue
            thr = otsu_threshold(x)
            preds.append((x > thr).astype(np.uint8))
            truths.append(s.reference[v].astype(np.uint8))
            thrs.append(thr)
        if not preds:
            raise ValueError("no test pixels")
        pred_mask = np.concatenate(preds)
        truth_u8 = np.concatenate(truths)
        threshold = float(np.mean(thrs))
    elif method == "global":
        tr_pred, tr_truth = _pool(train_samples, max_tune_pixels, rng)
        if tr_pred.size == 0:
            raise ValueError("no training pixels")
        threshold, _ = tune_threshold(tr_pred, tr_truth.astype(np.uint8), objective=objective)
        te_pred, te_truth = _pool(test_samples, None, rng)
        if te_pred.size == 0:
            raise ValueError("no test pixels")
        pred_mask = threshold_baseline(te_pred, threshold)
        truth_u8 = te_truth.astype(np.uint8)
    else:
        raise ValueError(f"method must be 'global' or 'otsu', got {method!r}")

    cc = confusion_counts(truth_u8, pred_mask)
    # The no-skill baseline: predict every valid pixel burned. On burn-heavy
    # windows this scores a high F1 that owes nothing to the predictor, so it is
    # reported alongside every fold to keep the model honest (see docs/11).
    naive = confusion_counts(truth_u8, np.ones_like(pred_mask))
    n_map = np.array([int(np.count_nonzero(pred_mask == 0)), int(np.count_nonzero(pred_mask == 1))])
    mapped_burned_ha = float(n_map[1] * pixel_area_ha)

    adjusted, ci, note = None, None, ""
    if int(n_map.min()) == 0:
        # A single-class map (all burned or all unburned in-window) cannot be
        # stratified into two strata, so the error-adjusted area is undefined.
        # The per-pixel F1/IoU above still stand; only the area estimate is
        # skipped. Common for a small fire whose window is almost entirely burned.
        note = "single-class map in window; area not adjusted"
    else:
        weights = n_map / n_map.sum()
        try:
            alloc = allocate_samples(n_reference, weights, rare_classes=[1], min_per_rare=min_per_rare)
            conf = _stratified_confusion(pred_mask, truth_u8, alloc, rng)
            est = estimate_areas(conf, n_map * pixel_area_ha, class_names=["unburned", "burned"])
            burned = est[1]
            adjusted = float(burned.adjusted_area)
            ci = float(burned.margin_of_error)
        except Exception as exc:  # noqa: BLE001  (never let one fold crash the run)
            note = f"Olofsson skipped: {exc}"

    return FoldResult(
        held_out=held_out or "",
        threshold=float(threshold),
        n_test_valid=int(pred_mask.size),
        f1=float(cc.f1),
        iou=float(cc.iou),
        mapped_burned_ha=mapped_burned_ha,
        adjusted_burned_ha=adjusted,
        ci95_ha=ci,
        naive_f1=float(naive.f1),
        naive_iou=float(naive.iou),
        note=note,
    )


def run_stage0(
    samples_by_id: Mapping[str, object],
    manifest,
    pixel_area_ha: float = MTBS_PIXEL_AREA_HA,
    n_reference: int = 500,
    seed: int = 0,
    method: str = "global",
    **kw,
) -> list[FoldResult]:
    """Run Stage 0 across every fold of a split manifest.

    ``samples_by_id`` maps event id to a :class:`~vhagar.datasets.burned_area.T2Sample`.
    Fires without a sample (no raster available) are skipped. Per-fold reporting
    is mandatory; that is what this returns.
    """
    results: list[FoldResult] = []
    for i, fold in enumerate(manifest.folds):
        train = [
            samples_by_id[u] for u in fold.get("train", [])
            if u in samples_by_id and samples_by_id[u].n_valid > 0
        ]
        test = [
            samples_by_id[u] for u in fold.get("test", [])
            if u in samples_by_id and samples_by_id[u].n_valid > 0
        ]
        # Calibration (global, perstratum) needs training pixels of both classes;
        # Otsu is calibration-free and only needs a usable test fire. A degenerate
        # test fire (all cloud or single-class) is skipped rather than crashing.
        if not test:
            continue
        if method in ("global", "perstratum") and (
            not train or not any(0.0 < s.burned_fraction < 1.0 for s in train)
        ):
            continue
        if not any(0.0 < s.burned_fraction < 1.0 for s in test):
            continue
        results.append(
            evaluate_fold(
                train, test,
                held_out=str(fold.get("held_out", i)),
                pixel_area_ha=pixel_area_ha, n_reference=n_reference,
                method=method, seed=seed + i, **kw,
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
    naive = np.array([r.naive_f1 for r in results])
    skill = f1 - naive
    return {
        "folds": len(results),
        "f1_mean": float(f1.mean()),
        "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        "iou_mean": float(iou.mean()),
        "iou_std": float(iou.std(ddof=1)) if len(iou) > 1 else 0.0,
        # Mean F1 improvement over predict-all-burned, and how many folds clear it.
        # A negative or near-zero skill means the headline F1 is a window artefact.
        "naive_f1_mean": float(naive.mean()),
        "skill_f1_mean": float(skill.mean()),
        "folds_beating_naive": int((skill > 0).sum()),
    }
