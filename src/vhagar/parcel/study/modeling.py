"""Models, splits, and population-weighted metrics for the case-control study.

Everything here works on a plain ``pandas.DataFrame`` with one row per sampled parcel:
the feature columns, ``y`` (1 = case), ``case_year``, ``case_facility``, ``role``,
``control_half``, ``weight`` (inverse sampling fraction), ``area_ha``, and a location
(``lon``, ``lat``, plus ``block_lon``/``block_lat`` for spatial blocking). It has no I/O and
is unit-tested on synthetic data.

Metrics are re-weighted to the parcel population with the control weights, so precision,
average precision, and "recall in the top k% of land area" describe Texas, not the
sample. Probabilities from models fit on the case-control sample are mapped to the
population with the King and Zeng (2001) prior correction before calibration is assessed.
Uncertainty comes from a cluster bootstrap that resamples whole facilities (parcels of
one facility are not independent) and controls independently.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from vhagar.eval.splits import SplitUnit, spatial_block_split
from vhagar.parcel.engine import score_parcel
from vhagar.parcel.schemas import EvidenceStatus, FeatureValue, Parcel, ProposedUse

__all__ = [
    "FEATURES", "MONOTONE", "Split", "temporal_split", "spatial_splits", "weighted_metrics",
    "recall_at_area", "recall_curve", "prior_correct", "reliability", "MODELS", "fit_predict",
    "rules_engine_scores", "cluster_bootstrap", "Model", "make_hgb",
]

FEATURES = [
    "slope_mean_pct", "slope_p90_pct", "ghi_kwh_m2_day", "dist_tx_robust_km", "dist_tx_hv_km",
    "dist_road_km", "whp_mean", "floodplain_frac", "protected_overlap_frac", "log_area_ha",
    "nlcd_developed_frac", "nlcd_cropland_frac", "nlcd_pasture_frac", "nlcd_shrub_grass_frac",
    "nlcd_forest_frac", "nlcd_wetland_water_frac", "nlcd_barren_frac",
]

#: Monotone constraints encode prior physical and economic knowledge so the learned
#: shapes stay explainable: +1 increasing, -1 decreasing, 0 unconstrained.
MONOTONE = {
    "slope_mean_pct": -1, "slope_p90_pct": -1, "ghi_kwh_m2_day": 1, "dist_tx_robust_km": -1,
    "dist_tx_hv_km": -1, "dist_road_km": 0, "whp_mean": 0, "floodplain_frac": -1,
    "protected_overlap_frac": -1, "log_area_ha": 1, "nlcd_developed_frac": -1,
    "nlcd_cropland_frac": 0, "nlcd_pasture_frac": 0, "nlcd_shrub_grass_frac": 0,
    "nlcd_forest_frac": -1, "nlcd_wetland_water_frac": -1, "nlcd_barren_frac": 0,
}


# ---------------------------------------------------------------------------- splits


@dataclass
class Split:
    """One train/test partition with population weights for the test rows."""

    name: str
    train: np.ndarray  # row indices
    test: np.ndarray
    test_weight: np.ndarray  # population weight per test row (cases 1, controls N/n)
    note: str = ""


def temporal_split(df: pd.DataFrame, last_train_year: int, noncase_population: int) -> Split:
    """Prospective holdout: train on cases installed up to ``last_train_year`` plus control
    half 0; test on later cases plus control half 1 (independent control pools)."""
    case = df["y"].to_numpy() == 1
    yr = df["case_year"].to_numpy()
    half = df["control_half"].to_numpy()
    train = np.flatnonzero((case & (yr <= last_train_year)) | (~case & (half == 0)))
    test = np.flatnonzero((case & (yr > last_train_year)) | (~case & (half == 1)))
    n_ctrl_test = int((~case[test]).sum())
    w = np.where(case[test], 1.0, noncase_population / max(n_ctrl_test, 1))
    return Split(f"temporal (train cases <= {last_train_year}, test later)", train, test, w,
                 "control pools are disjoint random halves")


def spatial_splits(df: pd.DataFrame, n_folds: int, block_degrees: float, seed: int,
                   noncase_population: int) -> list[Split]:
    """Spatial-block folds via ``vhagar.eval.splits``. Cases are blocked by their
    facility's location (``block_lon``/``block_lat``) so one facility never straddles
    train and test."""
    units = [SplitUnit(uid=str(i), lon=float(lo), lat=float(la),
                       when=pd.Timestamp("2016-01-01").date())
             for i, (lo, la) in enumerate(zip(df["block_lon"], df["block_lat"], strict=True))]
    manifest = spatial_block_split(units, n_folds=n_folds, block_degrees=block_degrees,
                                   seed=seed)
    case = df["y"].to_numpy() == 1
    out = []
    for k, fold in enumerate(manifest.folds):
        train = np.array(sorted(int(u) for u in fold["train"]))
        test = np.array(sorted(int(u) for u in fold["test"]))
        n_ctrl_total = int((~case).sum())
        w = np.where(case[test], 1.0, noncase_population / max(n_ctrl_total, 1))
        out.append(Split(f"spatial fold {k}", train, test, w,
                         f"{block_degrees} deg blocks, {n_folds} folds"))
    return out


# ---------------------------------------------------------------------------- metrics


def prior_correct(p: np.ndarray, sample_case_frac: float, population_prev: float) -> np.ndarray:
    """King and Zeng (2001) prior correction from case-control to population scale."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    shift = math.log(((1 - population_prev) / population_prev)
                     * (sample_case_frac / (1 - sample_case_frac)))
    logit = np.log(p / (1 - p)) - shift
    return 1.0 / (1.0 + np.exp(-logit))


def recall_curve(y: np.ndarray, score: np.ndarray, w: np.ndarray, area: np.ndarray,
                 fracs: np.ndarray) -> np.ndarray:
    """Expected share of cases captured when screening the highest-scored parcels that
    together make up each fraction of the (weighted) population land area.

    Tied scores are handled exactly: within a tie group, cases are credited in proportion
    to the area screened, which is the expectation under a random order. A constant score
    therefore traces the diagonal (recall equals the share of land screened), and missing
    scores rank last as one tied group."""
    s = np.where(np.isfinite(score), score, -np.inf)
    df = pd.DataFrame({"s": s, "a": w * area, "c": (y == 1).astype(float)})
    g = df.groupby("s", sort=True)[["a", "c"]].sum().iloc[::-1]
    a = g["a"].to_numpy() / g["a"].sum()
    c = g["c"].to_numpy() / max(g["c"].sum(), 1e-12)
    ca = np.concatenate([[0.0], np.cumsum(a)])
    cc = np.concatenate([[0.0], np.cumsum(c)])
    return np.interp(np.asarray(fracs, dtype=float), ca, cc)


def recall_at_area(y: np.ndarray, score: np.ndarray, w: np.ndarray, area: np.ndarray,
                   fracs=(0.01, 0.05, 0.10, 0.20)) -> dict[str, float]:
    """Expected recall at fixed shares of screened land area (see ``recall_curve``)."""
    if not (y == 1).any():
        return {f"recall_top{int(round(f * 100))}pct_area": math.nan for f in fracs}
    r = recall_curve(y, score, w, area, np.array(fracs))
    return {f"recall_top{int(round(f * 100))}pct_area": float(v) for f, v in zip(fracs, r, strict=True)}


def reliability(y: np.ndarray, p: np.ndarray, w: np.ndarray, n_bins: int = 10) -> dict:
    """Weighted equal-mass reliability table, Brier score, and expected calibration error."""
    order = np.argsort(p)
    cw = np.cumsum(w[order]) / w.sum()
    bins = np.minimum((cw * n_bins).astype(int), n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = order[bins == b]
        if m.size == 0:
            continue
        ww = w[m]
        mp = float(np.average(p[m], weights=ww))
        ob = float(np.average(y[m], weights=ww))
        rows.append({"bin": b, "n": int(m.size), "weight": float(ww.sum()),
                     "mean_pred": mp, "observed": ob})
        ece += ww.sum() / w.sum() * abs(mp - ob)
    brier = float(np.average((p - y) ** 2, weights=w))
    base = float(np.average(y, weights=w))
    brier_ref = float(np.average((base - y) ** 2, weights=w))
    return {"brier": brier, "brier_climatology": brier_ref,
            "brier_skill": 1 - brier / brier_ref if brier_ref else math.nan,
            "ece": float(ece), "bins": rows}


def weighted_metrics(y: np.ndarray, score: np.ndarray, w: np.ndarray, area: np.ndarray) -> dict:
    """Population-weighted ranking metrics for one score vector."""
    finite = np.isfinite(score)
    floor = (np.min(score[finite]) - 1.0) if finite.any() else 0.0
    s = np.where(finite, score, floor)  # missing scores rank last, as one tied group
    prev = float(np.average(y, weights=w))
    out = {
        "n": int(len(y)), "cases": int(y.sum()),
        "population_prevalence": prev,
        "average_precision": float(average_precision_score(y, s, sample_weight=w)),
        "ap_lift": float(average_precision_score(y, s, sample_weight=w) / prev) if prev else math.nan,
        "roc_auc": float(roc_auc_score(y, s, sample_weight=w)),
    }
    out.update(recall_at_area(y, score, w, area))
    return out


# ---------------------------------------------------------------------------- models


@dataclass
class Model:
    name: str
    kind: str  # baseline | rules | learned | diagnostic
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
    probabilistic: bool = False
    note: str = ""
    features: list[str] = field(default_factory=list)


def _const(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(test))


def _neg(col: str):
    def f(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return -test[col].to_numpy(dtype=float)
    return f


def _pos(col: str):
    def f(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return test[col].to_numpy(dtype=float)
    return f


def make_hgb(features: list[str], monotone: bool, seed: int = 0) -> HistGradientBoostingClassifier:
    """The gradient-boosting configuration used throughout (fixed before evaluation)."""
    cst = [MONOTONE.get(c, 0) for c in features] if monotone else None
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, monotonic_cst=cst, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30, random_state=seed)


def _hgb(features: list[str], monotone: bool, seed: int = 0):
    def f(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        m = make_hgb(features, monotone, seed)
        m.fit(train[features], train["y"])
        return m.predict_proba(test[features])[:, 1]
    return f


def _gam_lr(features: list[str]):
    def f(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             StandardScaler(), SplineTransformer(n_knots=5, degree=3),
                             LogisticRegression(C=0.3, max_iter=2000))
        pipe.fit(train[features], train["y"])
        return pipe.predict_proba(test[features])[:, 1]
    return f


RULES_MAP = {
    "slope_pct": "slope_mean_pct", "irradiance_kwh_m2_day": "ghi_kwh_m2_day",
    "transmission_distance_km": "dist_tx_robust_km", "road_distance_km": "dist_road_km",
    "wildfire_exposure_0_100": "whp_pct", "floodplain_frac": "floodplain_frac",
    "protected_overlap_frac": "protected_overlap_frac",
}
_RULES_PARCEL = Parcel("p", [[0, 0], [0.05, 0], [0.05, 0.05], [0, 0.05]])  # area gate passes


def rules_engine_scores(test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Score parcels with the transparent v0.2 solar rules (not fitted to any label).
    Returns (score, status) with ineligible -> 0 and insufficient evidence -> NaN. The
    parcel-size gate is applied with the real area."""
    scores = np.full(len(test), np.nan)
    status = np.empty(len(test), dtype=object)
    cols = {k: test[v].to_numpy(dtype=float) for k, v in RULES_MAP.items()}
    area = test["area_ha"].to_numpy(dtype=float)
    for i in range(len(test)):
        feats = {}
        for k, arr in cols.items():
            v = arr[i]
            feats[k] = FeatureValue(k, None if not np.isfinite(v) else float(v), "",
                                    EvidenceStatus.MISSING if not np.isfinite(v)
                                    else EvidenceStatus.DERIVED)
        _RULES_PARCEL.area_ha = float(area[i])
        r = score_parcel(_RULES_PARCEL, ProposedUse.SOLAR, feats)
        status[i] = r.overall_status
        if r.overall_status == "ineligible":
            scores[i] = 0.0
        elif r.overall_score is not None:
            scores[i] = r.overall_score
    return scores, status


def _rules(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return rules_engine_scores(test)[0]


MODELS: list[Model] = [
    Model("null (prevalence)", "baseline", _const, note="every parcel tied"),
    Model("nearest transmission", "baseline", _neg("dist_tx_robust_km"),
          note="single feature: closer is better"),
    Model("irradiance", "baseline", _pos("ghi_kwh_m2_day"), note="single feature"),
    Model("rules engine v0.2", "rules", _rules,
          note="transparent prior-knowledge screen, not fitted; abstentions rank last"),
    Model("spline logistic (GAM)", "learned", _gam_lr(FEATURES), probabilistic=True,
          features=FEATURES),
    Model("gradient boosting, monotone", "learned", _hgb(FEATURES, True), probabilistic=True,
          features=FEATURES),
    Model("gradient boosting, unconstrained", "learned", _hgb(FEATURES, False),
          probabilistic=True, features=FEATURES),
    Model("coordinates only (diagnostic)", "diagnostic", _hgb(["lon", "lat"], False),
          probabilistic=True, note="memorises where solar already is; should not transfer",
          features=["lon", "lat"]),
]


def fit_predict(model: Model, df: pd.DataFrame, split: Split) -> np.ndarray:
    return model.fit_predict(df.iloc[split.train], df.iloc[split.test])


# ---------------------------------------------------------------------------- bootstrap


def cluster_bootstrap(y: np.ndarray, groups: np.ndarray, w: np.ndarray, area: np.ndarray,
                      scores: dict[str, np.ndarray], metric: str, n_boot: int = 300,
                      seed: int = 0, reference: str | None = None) -> dict[str, dict]:
    """Percentile intervals for ``metric`` by resampling case facilities (``groups``) and
    controls independently. With ``reference``, also the interval of the difference."""
    rng = np.random.default_rng(seed)
    case_idx = np.flatnonzero(y == 1)
    ctrl_idx = np.flatnonzero(y == 0)
    fac = groups[case_idx]
    uniq = np.unique(fac)
    members = {g: case_idx[fac == g] for g in uniq}
    draws: dict[str, list[float]] = {k: [] for k in scores}
    diffs: dict[str, list[float]] = {k: [] for k in scores}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ci = np.concatenate([members[g] for g in pick])
        co = rng.choice(ctrl_idx, size=len(ctrl_idx), replace=True)
        idx = np.concatenate([ci, co])
        vals = {k: weighted_metrics(y[idx], s[idx], w[idx], area[idx])[metric]
                for k, s in scores.items()}
        for k, v in vals.items():
            draws[k].append(v)
            if reference is not None:
                diffs[k].append(v - vals[reference])
    out = {}
    for k in scores:
        d = np.array(draws[k])
        out[k] = {"lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}
        if reference is not None and k != reference:
            dd = np.array(diffs[k])
            out[k]["diff_vs_" + reference] = {
                "lo": float(np.nanpercentile(dd, 2.5)), "hi": float(np.nanpercentile(dd, 97.5)),
                "p_le_0": float(np.mean(dd <= 0))}
    return out
