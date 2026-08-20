"""T3 ignition-probability evaluation: cause-stratified, blocked, properly scored.

The architecture's evidence (``docs/00`` section 5.3) is that the state of the
art here is gradient boosting, not a deep net: ECMWF's operational
Probability-of-Fire uses XGBoost on ~19 predictors and found input quality (fuel
state above all) mattered more than architecture. So the model is a
gradient-boosted classifier, fit per cause, and the work is in honest
evaluation:

* **Blocked splits.** Spatial-block GroupKFold on the same 5-degree blocks the
  rest of VHAGAR uses, so a fold cannot memorise a location.
* **Proper scores only.** AUPRC, Brier with its Murphy decomposition
  (reliability / resolution), ECE, log loss, and Brier skill score against a
  base-rate climatology. F1/CSI are improper and are not used for selection.
* **Rare-event correction.** Predicted probabilities are mapped from the
  down-sampled design's base rate back to the true one before calibration is
  scored, so reliability means what it says operationally.
* **lon/lat are excluded** from the model (the T1 leakage lesson); they only
  build the spatial blocks.

:func:`evaluate_ignition` scores one design. :func:`top_features` returns the
permutation-importance ranking on a held-out block, which is how the sampling
trap is exposed: with naive background the top predictor is the human-footprint
covariate (an observation artefact); with target-group + stratified background
it is weather / fuel state.
"""

from __future__ import annotations

import numpy as np

from vhagar.datasets.danger import IgnitionSamples, rare_event_correction
from vhagar.eval.metrics import (
    average_precision,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    log_loss,
    skill_score,
)

__all__ = ["evaluate_ignition", "top_features", "ignition_scorecard"]


def _gbdt(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.06,
        l2_regularization=1.0, random_state=seed,
    )


def _blocked_oof(X, y, groups, n_folds: int, seed: int):
    """Out-of-fold probabilities under spatial-block GroupKFold."""
    from sklearn.model_selection import GroupKFold

    y = np.asarray(y)
    oof = np.full(y.shape[0], np.nan, dtype=np.float64)
    n_groups = len(np.unique(groups))
    folds = int(min(n_folds, n_groups))
    if folds < 2:
        raise ValueError(f"need >= 2 spatial blocks, found {n_groups}")
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = float(y[tr].mean())
            continue
        m = _gbdt(seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def ignition_scorecard(y, prob, tau: float, ybar: float, prior_correct: bool) -> dict:
    """Proper-scoring card for one set of out-of-fold probabilities."""
    y = np.asarray(y)
    p = np.asarray(prob, dtype=np.float64)
    if prior_correct:
        p = rare_event_correction(p, tau, ybar)
    base = float(y.mean()) if not prior_correct else tau
    # The climatology reference must sit on the SAME probability scale as p: the
    # population base rate tau when p is prior-corrected, else the design mean.
    # Using y.mean() unconditionally scored a tau-scale p (~0.01) against a
    # ~0.25 constant, making the reference absurdly bad and inflating the skill.
    brier_ref = brier_score(y, np.full_like(p, base))
    dec = brier_decomposition(y, p, n_bins=10)
    return {
        "auprc": average_precision(y, prob),   # ranking is correction-invariant
        "brier": dec["brier"],
        "reliability": dec["reliability"],     # lower = better calibrated
        "resolution": dec["resolution"],       # higher = more discriminating
        "ece": expected_calibration_error(y, p, n_bins=10, equal_mass=True),
        "log_loss": log_loss(y, p),
        "bss_vs_climatology": skill_score(dec["brier"], brier_ref, perfect=0.0),
        "mean_prob": float(p.mean()),
        "base_rate": base,
    }


def top_features(samples: IgnitionSamples, n_folds: int, seed: int, k: int = 3):
    """Permutation-importance ranking on one held-out spatial block.

    The tell for the sampling trap: which covariate the model leans on. Uses a
    single block split (train on the rest, permute on the held-out block) so the
    importance is measured out of sample.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupKFold

    X, y, g = samples.X, samples.y, samples.block_group()
    folds = int(min(n_folds, len(np.unique(g))))
    tr, te = next(GroupKFold(n_splits=folds).split(X, y, g))
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return []
    m = _gbdt(seed).fit(X[tr], y[tr])
    imp = permutation_importance(m, X[te], y[te], n_repeats=8,
                                 random_state=seed, scoring="average_precision")
    order = np.argsort(imp.importances_mean)[::-1]
    return [(samples.feature_names[i], float(imp.importances_mean[i])) for i in order[:k]]


def evaluate_ignition(samples: IgnitionSamples, n_folds: int = 5, seed: int = 0,
                      prior_correct: bool = True) -> dict:
    """Cause-stratified, blocked, properly scored ignition evaluation.

    Returns ``{"pooled": card, "human": card, "lightning": card,
    "top_features": [...], "n": {...}}``. Each cause head is fit on that cause's
    positives plus all backgrounds, so the two orthogonal ignition processes are
    modelled separately.
    """
    out: dict = {"n": dict(samples.meta), "tau": samples.tau, "ybar": samples.ybar}

    oof = _blocked_oof(samples.X, samples.y, samples.block_group(), n_folds, seed)
    ok = ~np.isnan(oof)
    out["pooled"] = ignition_scorecard(samples.y[ok], oof[ok], samples.tau, samples.ybar, prior_correct)

    for cause in ("human", "lightning"):
        mask = samples.cause_mask(cause)
        if int(samples.y[mask].sum()) < 10 or len(np.unique(samples.block_group()[mask])) < 2:
            continue
        yc = samples.y[mask]
        ybar_c = float(yc.mean())
        oofc = _blocked_oof(samples.X[mask], yc, samples.block_group()[mask], n_folds, seed)
        okc = ~np.isnan(oofc)
        out[cause] = ignition_scorecard(yc[okc], oofc[okc], samples.tau, ybar_c, prior_correct)

    out["top_features"] = top_features(samples, n_folds, seed)
    return out
