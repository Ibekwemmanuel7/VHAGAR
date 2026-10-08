"""Run the study evaluation and write results, figures, and per-parcel predictions.

Outputs (``outputs/parcel_study/``): ``results.json`` (every number quoted in docs/28),
``fig_*.png``. Per-parcel out-of-sample predictions go to
``data/parcel/work/predictions.parquet`` (not committed).
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import average_precision_score

from vhagar.parcel.study import modeling as M
from vhagar.parcel.study.features import FEATURE_SPECS
from vhagar.parcel.study.paths import (
    LABEL_FIRST_YEAR,
    LABEL_LAST_YEAR,
    OUT,
    SEED,
    SNAPSHOT_YEAR,
    STUDY_VERSION,
    TRAIN_LAST_YEAR,
    WORK,
)

__all__ = ["run_evaluation"]

# Reference palette (dataviz skill, light mode). Fixed slot per entity, never by rank.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SURFACE = "#fcfcfb"
COLOR = {"gradient boosting, monotone": "#2a78d6", "rules engine v0.2": "#eb6834",
         "nearest transmission": "#1baf7a", "null (prevalence)": MUTED,
         "coordinates only (diagnostic)": "#4a3aa7"}
FOCUS = ["gradient boosting, monotone", "rules engine v0.2", "nearest transmission",
         "null (prevalence)"]


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2.0, "savefig.dpi": 160,
        "savefig.bbox": "tight"})
    return plt


def run_evaluation(df: pd.DataFrame, record: dict, *, n_boot: int = 300,
                   log: Callable[[str], None] = print,
                   facilities: gpd.GeoDataFrame | None = None,
                   label_source: str = "") -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    noncase_pop = int(record["eligible_noncase"])
    res: dict = {
        "study_version": STUDY_VERSION, "label_source": label_source,
        "design": {"snapshot_year": SNAPSHOT_YEAR,
                   "label_window": [LABEL_FIRST_YEAR, LABEL_LAST_YEAR],
                   "temporal_train_last_year": TRAIN_LAST_YEAR, "seed": SEED},
        "sampling": record,
        "features": {k: vars(v) for k, v in FEATURE_SPECS.items()},
        "missing_share": {c: round(float(df[c].isna().mean()), 4) for c in M.FEATURES},
        "cases_by_year": df[df.y == 1].groupby("case_year").size().astype(int).to_dict(),
    }
    res["cases_by_year"] = {int(k): int(v) for k, v in res["cases_by_year"].items()}
    preds = pd.DataFrame({"pid": df["pid"], "y": df["y"], "case_year": df["case_year"]})

    # ------------------------------------------------------------- temporal holdout
    sp = M.temporal_split(df, TRAIN_LAST_YEAR, noncase_pop)
    te = df.iloc[sp.test]
    y, w, area = te["y"].to_numpy(), sp.test_weight, te["area_ha"].to_numpy()
    scores: dict[str, np.ndarray] = {}
    tbl = {}
    for m in M.MODELS:
        s = M.fit_predict(m, df, sp)
        scores[m.name] = s
        tbl[m.name] = {"kind": m.kind, "note": m.note, **M.weighted_metrics(y, s, w, area)}
        log(f"temporal  {m.name:<34} AP={tbl[m.name]['average_precision']:.4f} "
            f"lift={tbl[m.name]['ap_lift']:.1f} AUC={tbl[m.name]['roc_auc']:.3f} "
            f"R@10%area={tbl[m.name]['recall_top10pct_area']:.3f}")
    _, status = M.rules_engine_scores(te)
    rules_cov = pd.Series(status).value_counts().to_dict()
    groups = np.where(y == 1, te["case_facility"].to_numpy(), -1)
    boot_ap = M.cluster_bootstrap(y, groups, w, area, scores, "average_precision",
                                  n_boot=n_boot, seed=SEED, reference="nearest transmission")
    boot_r10 = M.cluster_bootstrap(y, groups, w, area, scores, "recall_top10pct_area",
                                   n_boot=n_boot, seed=SEED + 1, reference="rules engine v0.2")
    train_case_frac = float(df.iloc[sp.train]["y"].mean())
    prev = float(np.average(y, weights=w))
    calib = {}
    for m in M.MODELS:
        if m.probabilistic:
            p = M.prior_correct(scores[m.name], train_case_frac, prev)
            calib[m.name] = M.reliability(y, p, w)
    res["temporal"] = {
        "split": sp.name, "note": sp.note, "n_train": int(len(sp.train)),
        "n_test": int(len(sp.test)), "test_cases": int(y.sum()),
        "test_case_facilities": int(te.loc[te.y == 1, "case_facility"].nunique()),
        "metrics": tbl, "bootstrap_ap": boot_ap, "bootstrap_recall10": boot_r10,
        "rules_engine_status": {str(k): int(v) for k, v in rules_cov.items()},
        "calibration": calib, "n_boot": n_boot}
    preds.loc[sp.test, "temporal_hgb"] = scores["gradient boosting, monotone"]
    preds.loc[sp.test, "temporal_rules"] = scores["rules engine v0.2"]

    # ------------------------------------------------------------- leakage ablations
    abl = {}
    variants = {
        "robust transmission (main)": M.FEATURES,
        "raw HIFLD transmission (leaky)": [c if c != "dist_tx_robust_km" else "dist_tx_km"
                                           for c in M.FEATURES],
        "no transmission features": [c for c in M.FEATURES if not c.startswith("dist_tx")],
        "no land cover": [c for c in M.FEATURES if not c.startswith("nlcd_")],
    }
    for name, feats in variants.items():
        m = M.Model(name, "learned", M._hgb(feats, True), probabilistic=True)
        s = M.fit_predict(m, df, sp)
        abl[name] = M.weighted_metrics(y, s, w, area)
        log(f"ablation  {name:<34} AP={abl[name]['average_precision']:.4f} "
            f"R@10%area={abl[name]['recall_top10pct_area']:.3f}")
    res["ablations_temporal"] = abl

    # ------------------------------------------------------------- spatial holdout
    folds = M.spatial_splits(df, n_folds=5, block_degrees=2.0, seed=SEED,
                             noncase_population=noncase_pop)
    oof = {m.name: np.full(len(df), np.nan) for m in M.MODELS}
    for f in folds:
        for m in M.MODELS:
            oof[m.name][f.test] = M.fit_predict(m, df, f)
    ya = df["y"].to_numpy()
    wa = np.where(ya == 1, 1.0, noncase_pop / max(int((ya == 0).sum()), 1))
    aa = df["area_ha"].to_numpy()
    sp_tbl = {}
    for m in M.MODELS:
        sp_tbl[m.name] = {"kind": m.kind, **M.weighted_metrics(ya, oof[m.name], wa, aa)}
        log(f"spatial   {m.name:<34} AP={sp_tbl[m.name]['average_precision']:.4f} "
            f"R@10%area={sp_tbl[m.name]['recall_top10pct_area']:.3f}")
    groups_all = np.where(ya == 1, df["case_facility"].to_numpy(), -1)
    boot_sp = M.cluster_bootstrap(ya, groups_all, wa, aa, oof, "average_precision",
                                  n_boot=max(100, n_boot // 2), seed=SEED + 2,
                                  reference="nearest transmission")
    res["spatial"] = {"folds": len(folds), "block_degrees": 2.0, "metrics": sp_tbl,
                      "bootstrap_ap": boot_sp,
                      "fold_sizes": [[int(len(f.train)), int(len(f.test))] for f in folds]}
    preds["spatial_hgb"] = oof["gradient boosting, monotone"]

    # ------------------------------------------------------------- explanation
    tr = df.iloc[sp.train]
    model = M.make_hgb(M.FEATURES, True)
    model.fit(tr[M.FEATURES], tr["y"])

    def ap_scorer(est, X, yy):
        return average_precision_score(yy, est.predict_proba(X)[:, 1], sample_weight=w)

    pi = permutation_importance(model, te[M.FEATURES], y, scoring=ap_scorer, n_repeats=5,
                                random_state=SEED)
    imp = (pd.DataFrame({"feature": M.FEATURES, "ap_drop": pi.importances_mean,
                         "sd": pi.importances_std}).sort_values("ap_drop", ascending=False))
    res["permutation_importance_temporal"] = imp.to_dict(orient="records")
    top = imp["feature"].head(6).tolist()
    pdp = {}
    for feat in top:
        pdr = partial_dependence(model, tr[M.FEATURES], [feat], kind="average",
                                 grid_resolution=30, percentiles=(0.02, 0.98))
        pdp[feat] = {"grid": pdr["grid_values"][0].tolist(),
                     "avg": pdr["average"][0].tolist()}
    res["partial_dependence"] = pdp

    # ------------------------------------------------------------- figures + outputs
    _figures(res, y, w, area, scores, imp, pdp, calib, df, preds, facilities)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=_json))
    preds.to_parquet(WORK / "predictions.parquet", index=False)
    return str(OUT / "results.json")


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _figures(res, y, w, area, scores, imp, pdp, calib, df, preds, facilities):
    plt = _style()
    # 1. Recall of later-built solar parcels vs share of land screened (temporal test)
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in FOCUS:
        c = M.recall_curve(y, scores[name], w, area, grid)
        label = "random screening (null)" if name.startswith("null") else name
        ax.plot(grid * 100, c * 100, color=COLOR[name], lw=1.2 if label.startswith("random") else 2,
                ls="--" if label.startswith("random") else "-")
        ax.annotate(label, (30, c[60] * 100), xytext=(6, -3), textcoords="offset points",
                    color=INK2, fontsize=8.5, annotation_clip=False)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of Texas parcel land area screened, highest score first (%)")
    ax.set_ylabel("Later solar parcels captured (%)")
    ax.set_title("Temporal holdout: cases installed 2021 to 2025", loc="left", fontsize=11)
    fig.savefig(OUT / "fig_recall_vs_area.png")
    plt.close(fig)

    # 2. Permutation importance
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    d = imp.head(10).iloc[::-1]
    ax.barh(d["feature"], d["ap_drop"], xerr=d["sd"], color="#2a78d6", height=0.6,
            error_kw={"ecolor": MUTED, "lw": 1})
    ax.set_xlabel("Drop in population-weighted average precision when shuffled")
    ax.set_title("What the monotone gradient-boosting model relies on", loc="left",
                 fontsize=11)
    fig.savefig(OUT / "fig_importance.png")
    plt.close(fig)

    # 3. Partial dependence (small multiples, one hue)
    feats = list(pdp)
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.0))
    for ax, f in zip(axes.ravel(), feats, strict=False):
        ax.plot(pdp[f]["grid"], pdp[f]["avg"], color="#2a78d6", lw=2)
        ax.set_title(f, fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(feats):]:
        ax.set_visible(False)
    fig.suptitle("Partial dependence (sample-scale probability), top six features",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_partial_dependence.png")
    plt.close(fig)

    # 4. Reliability of the main model after prior correction
    rel = calib.get("gradient boosting, monotone")
    if rel:
        b = pd.DataFrame(rel["bins"])
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        hi = max(b["mean_pred"].max(), b["observed"].max()) * 1.1
        ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--")
        ax.plot(b["mean_pred"], b["observed"], color="#2a78d6", marker="o", ms=5, lw=2)
        ax.set_xlabel("Predicted probability (population scale)")
        ax.set_ylabel("Observed share of cases")
        ax.set_title(f"Calibration, temporal test (ECE {rel['ece']:.4f})", loc="left",
                     fontsize=11)
        fig.savefig(OUT / "fig_calibration.png")
        plt.close(fig)

    # 5. Map: test parcels, top-decile scores vs later cases
    te = preds.dropna(subset=["temporal_hgb"]).merge(df[["pid", "lon", "lat"]], on="pid")
    thr = np.nanpercentile(te["temporal_hgb"], 90)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ctrl = te[te.y == 0]
    ax.scatter(ctrl["lon"], ctrl["lat"], s=1.5, color="#d7d6d1", lw=0, label="sampled parcels")
    topc = ctrl[ctrl["temporal_hgb"] >= thr]
    ax.scatter(topc["lon"], topc["lat"], s=3, color="#86b6ef", lw=0,
               label="top-decile score, no solar by 2025")
    cs = te[te.y == 1]
    ax.scatter(cs["lon"], cs["lat"], s=10, color="#eb6834", edgecolor=SURFACE, lw=0.4,
               label="solar parcel installed 2021 to 2025")
    ax.set_aspect(1 / math.cos(math.radians(31)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=3)
    ax.set_title("Where the model points vs where solar was built", loc="left", fontsize=11)
    fig.savefig(OUT / "fig_map.png")
    plt.close(fig)
