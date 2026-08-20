"""T1 Stage-2 preview: does raw lat/lon leak in a GOES fire-event classifier?

The architecture's central warning for T1 (docs/00, section 3.2): in a published FIRMS
wildfire/non-wildfire classification, raw latitude and longitude supplied ~89% of the
model's gain while *harming* out-of-region transfer, F1 collapsing 0.985 (random split)
-> 0.767 (event-aware) -> 0.627 (5-degree spatial block). VHAGAR reports all three on
every release and excludes raw coordinates from the production feature set by
construction (``fusion.event_features``).

This module reproduces that collapse on our own GOES-18 FDC + VIIRS data, so the rule is
evidence, not received wisdom. Each GOES detection is a sample; the label is whether a
VIIRS detection coincides with it in space and time (a proxy for "real, VIIRS-confirmable
fire" vs an unconfirmed geostationary flag). A gradient-boosted classifier is trained
twice, on physical features alone and on physical features plus raw lon/lat, under three
increasingly honest splits:

* **random**: detections shuffled. Same location appears in train and test, so lon/lat
  memorises "fires happen here".
* **cell-grouped** (the event-aware analog): whole 4 km cells held out together.
* **spatial-block**: whole 5-degree cells held out, the out-of-region test.

If adding lon/lat lifts random-split F1 but that lift evaporates on the spatial block,
the coordinates were memorising geography, not learning fire physics. The numpy feature
build is pure and testable; the classifier needs scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LabeledSamples", "build_samples", "evaluate_leakage"]

#: Physical, transfer-safe features. No raw coordinates, by construction; the only
#: thing adding raw lon/lat can do is memorise *where* fires tend to be confirmed.
PHYSICAL_FEATURES = ("frp_mw", "temp_k", "confidence", "area_m2", "view_zenith_deg", "hour")


@dataclass(slots=True)
class LabeledSamples:
    """Detection-level samples for the leakage experiment."""

    X: np.ndarray                 # [n, n_physical] physical features
    lonlat: np.ndarray            # [n, 2] raw coordinates (the leaky features)
    y: np.ndarray                 # [n] 1 = VIIRS-confirmed, 0 = not
    cell_group: np.ndarray        # [n] int id of the ~4 km cell (event-aware grouping)
    block_group: np.ndarray       # [n] int id of the 5-degree block (spatial split)
    feature_names: tuple[str, ...]


def _cells(x, y, size):
    return (np.floor(x / size).astype(np.int64), np.floor(y / size).astype(np.int64))


def build_samples(
    fdc_df, viirs_lonlat, viirs_times, region_crs: str = "EPSG:5070",
    cell_m: float = 4_000.0, window_min: float = 30.0, block_deg: float = 5.0,
) -> LabeledSamples:
    """Label each GOES detection by VIIRS coincidence and assemble features.

    ``fdc_df`` is the FDC parquet (lon, lat, t, frp_mw, temp_k, confidence, area_m2,
    view_zenith_deg). ``viirs_lonlat`` is ``[m, 2]`` VIIRS lon/lat and ``viirs_times``
    the matching POSIX seconds. A GOES detection is labelled 1 (VIIRS-confirmed) when a
    VIIRS detection lies in the same ``cell_m`` cell (8-neighbour) within ``window_min``,
    else 0. The label carries spatial structure (confirmation rate varies by region), so
    raw lon/lat can memorise it, which is the leakage this experiment exposes. Pure numpy
    + pandas + pyproj.
    """
    import pandas as pd
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", region_crs, always_xy=True)
    lon = fdc_df["lon"].to_numpy(float)
    lat = fdc_df["lat"].to_numpy(float)
    gx, gy = tf.transform(lon, lat)
    t = pd.to_datetime(fdc_df["t"])
    gt = (t - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()

    vx, vy = tf.transform(viirs_lonlat[:, 0], viirs_lonlat[:, 1])
    vt = np.asarray(viirs_times, float)
    vcx, vcy = _cells(vx, vy, cell_m)
    from collections import defaultdict
    tmp: dict[tuple[int, int], list[float]] = defaultdict(list)
    for i in range(len(vx)):
        tmp[(int(vcx[i]), int(vcy[i]))].append(vt[i])
    vindex = {k: np.sort(np.asarray(v)) for k, v in tmp.items()}

    gcx, gcy = _cells(gx, gy, cell_m)
    w = window_min * 60.0
    y = np.zeros(len(gx), dtype=np.int64)
    for i in range(len(gx)):
        t0 = gt[i]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                arr = vindex.get((int(gcx[i]) + dx, int(gcy[i]) + dy))
                if arr is None:
                    continue
                j = int(np.searchsorted(arr, t0))
                if any(0 <= jj < len(arr) and abs(float(arr[jj]) - t0) <= w for jj in (j - 1, j)):
                    y[i] = 1
                    break
            if y[i]:
                break

    hour = (t.dt.hour + t.dt.minute / 60.0).to_numpy()
    cols = [hour if n == "hour" else fdc_df[n].to_numpy(float) for n in PHYSICAL_FEATURES]
    X = np.column_stack(cols).astype(np.float64)
    cell_id = gcx.astype(np.int64) * 100_000 + gcy.astype(np.int64)
    block_id = (np.floor(lon / block_deg).astype(np.int64) * 1_000
                + np.floor(lat / block_deg).astype(np.int64))
    return LabeledSamples(
        X=X, lonlat=np.column_stack([lon, lat]), y=y,
        cell_group=cell_id, block_group=block_id, feature_names=PHYSICAL_FEATURES,
    )


def _cv_f1(model_factory, X, y, groups, n_folds, seed):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    scores = []
    if groups is None:
        # deliberate non-spatial baseline, contrasted with GroupKFold below
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True,  # leakage-ok
                                   random_state=seed).split(X, y)
    else:
        # cap folds at the number of groups
        k = min(n_folds, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=max(2, k)).split(X, y, groups)
    for tr, te in splitter:
        if len(np.unique(y[tr])) < 2:
            continue
        m = model_factory()
        m.fit(X[tr], y[tr])
        scores.append(f1_score(y[te], m.predict(X[te]), zero_division=0))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_leakage(s: LabeledSamples, n_folds: int = 5, seed: int = 0) -> dict:
    """F1 of a gradient-boosted classifier with vs without raw lon/lat, across the
    three splits. Needs scikit-learn. Returns ``{split: {"physical": f1, "with_latlon": f1}}``."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("t1-classify needs scikit-learn: install the gbdt extra") from exc

    def factory():
        return HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=seed)

    Xphys = s.X
    Xgeo = np.column_stack([s.X, s.lonlat])
    splits = {
        "random": None,
        "cell_grouped": s.cell_group,
        "spatial_block_5deg": s.block_group,
    }
    out: dict[str, dict] = {}
    for name, groups in splits.items():
        out[name] = {
            "physical": _cv_f1(factory, Xphys, s.y, groups, n_folds, seed),
            "with_latlon": _cv_f1(factory, Xgeo, s.y, groups, n_folds, seed),
        }
        out[name]["latlon_gain"] = out[name]["with_latlon"] - out[name]["physical"]
    return out
