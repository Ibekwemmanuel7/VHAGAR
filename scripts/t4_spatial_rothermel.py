"""T4 spatial-Rothermel replay on a REAL historical fire, validated against its
real MTBS perimeter.

This is the reviewer's key T4 upgrade: replace the uniform rate-of-spread grid with
a SPATIAL ROS raster computed cell by cell from real LANDFIRE FBFM40 fuel codes via
the Rothermel model, run the anisotropic arrival-time solver from the ignition point,
and score the reconstructed front against the observed MTBS burned perimeter, next to
a uniform-ROS front and an equal-area circle baseline.

Pipeline (all inputs co-registered in EPSG:5070, 30 m):
    MTBS severity  -> observed burned mask (largest connected component)
    LANDFIRE FBFM40 -> Rothermel head ROS field (m/min) per cell
    ignition point + wind -> anisotropic arrival-time front
    threshold to observed area -> predicted mask -> IoU / Sorensen vs observed

Honest scope: LANDFIRE is LF2025 (current fuels; a pre-fire vintage would be more
correct), slope is a documented proxy (no DEM wired here, so slope_tan=0), and the
wind is a single representative value (real time-varying wind is the next step). The
point of the run is the SPATIAL fuel-driven ROS and its validation against a real
perimeter, not an operational forecast.

    python scripts/t4_spatial_rothermel.py --fig outputs/t4_spatial_rothermel.png \
        --out outputs/t4_spatial_rothermel.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vhagar.eval.perimeter_shape import equal_area_disk, iou
from vhagar.eval.wui_calibrate import wind_from_deg_to_grid
from vhagar.models.rothermel import rothermel_head_ros_field
from vhagar.models.spread import anisotropic_arrival

_ROOT = Path(__file__).resolve().parents[1]
MTBS = _ROOT / "mtbs_extract" / "mtbs_CONUS_2021.tif"
LF_ZIP = _ROOT / "LF2025_FBFM40_CONUS.zip"
LF_VSI = f"/vsizip/{LF_ZIP}/LF2025_FBFM40_CONUS/Tif/LF2025_FBFM40_CONUS.tif"
BURNED = (2, 3, 4)                      # MTBS low/moderate/high severity

# name, ignition lon, lat, half-window km, midflame wind m/s, wind-from deg, dead-fuel moisture
FIRES = [
    ("Dixie", -121.387, 39.877, 75, 3.0, 225, 0.06),
]


def _read_window(ds, cx, cy, half_m, down):
    from rasterio.windows import from_bounds
    win = from_bounds(cx - half_m, cy - half_m, cx + half_m, cy + half_m, ds.transform)
    a = ds.read(1, window=win)
    H, W = a.shape
    Hc, Wc = H // down, W // down
    return a[:Hc * down, :Wc * down], (Hc, Wc)


def replay(name, lon, lat, half_km, wind_ms, wind_deg, m_f, down):
    import rasterio
    from pyproj import Transformer
    from scipy import ndimage

    tf = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    ix, iy = tf.transform(lon, lat)
    half = half_km * 1000.0
    dsm = rasterio.open(str(MTBS))
    dsl = rasterio.open(LF_VSI)
    sev, (Hc, Wc) = _read_window(dsm, ix, iy, half, down)
    fuel, _ = _read_window(dsl, ix, iy, half, down)
    # align shapes
    Hc = min(Hc, fuel.shape[0] // down if fuel.shape[0] >= down else Hc)
    # observed burned mask, downsample by max-pool (burned if any fine cell burned)
    Hs, Ws = sev.shape[0] // down, sev.shape[1] // down
    obs = np.isin(sev[:Hs * down, :Ws * down], BURNED).reshape(Hs, down, Ws, down).max(axis=(1, 3))
    lbl, n = ndimage.label(obs)
    if n:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        obs = lbl == int(np.argmax(sizes)) + 1
    # fuel codes: subsample at block centre (fast, adequate for a demo grid)
    codes = fuel[down // 2::down, down // 2::down][:Hs, :Ws]
    H, W = obs.shape
    codes = codes[:H, :W]
    cell_m = 30.0 * down

    # spatial Rothermel ROS (m/min) per cell from real fuel codes
    ros = rothermel_head_ros_field(codes, m_f=m_f, wind_ms=wind_ms, slope_tan=0.0, wind_adj=1.0)
    ros = np.clip(np.asarray(ros, float), 0.0, None)

    # ignition seed cell (from the real ignition point), small disk
    ir = int(round((iy - (iy + half)) / -cell_m))      # rows from north edge
    ic = int(round((ix - (ix - half)) / cell_m))
    ir = min(max(ir, 0), H - 1)
    ic = min(max(ic, 0), W - 1)
    yy, xx = np.mgrid[0:H, 0:W]
    seed = (yy - ir) ** 2 + (xx - ic) ** 2 <= 4

    # rasters are north-up (row 0 = north), so flip the y-up wind convention to row-down
    wdir = -wind_from_deg_to_grid(float(wind_deg))
    area = int(obs.sum())

    def front(ros_field):
        T = anisotropic_arrival(np.where(ros_field > 0, ros_field, 1e-6), wind_ms, wdir,
                                seed, dx=cell_m, lb_max=3.0)
        T = np.where(ros_field > 0, T, np.inf)         # non-burnable stay unreached
        fin = np.isfinite(T)
        k = min(area, int(fin.sum()))
        if k < 1:
            return np.zeros_like(obs)
        thr = np.partition(T[fin], k - 1)[k - 1]
        return fin & np.less_equal(T, thr)

    pred_spatial = front(ros)
    ros_uniform = np.full_like(ros, float(np.mean(ros[ros > 0])) if (ros > 0).any() else 6.0)
    ros_uniform[ros <= 0] = 0.0                        # keep the same fuel breaks
    pred_uniform = front(ros_uniform)
    circle = equal_area_disk((H, W), (ir, ic), area)

    res = {"fire": name, "cells": f"{H}x{W}", "cell_m": cell_m, "obs_ha": round(area * (cell_m ** 2) / 1e4),
           "iou_spatial": round(iou(pred_spatial, obs), 4), "iou_uniform": round(iou(pred_uniform, obs), 4),
           "iou_circle": round(iou(circle, obs), 4),
           "burnable_frac": round(float((ros > 0).mean()), 3),
           "ros_mean_mpermin": round(float(np.mean(ros[ros > 0])) if (ros > 0).any() else 0.0, 2)}
    res["spatial_beats_uniform"] = bool(res["iou_spatial"] > res["iou_uniform"])
    return res, dict(obs=obs, codes=codes, ros=ros, pred_spatial=pred_spatial,
                     pred_uniform=pred_uniform, ig=(ir, ic))


def _figure(name, P, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    fig.patch.set_facecolor("#0E1117")
    ax[0].imshow(np.where(P["ros"] > 0, P["ros"], np.nan), cmap="YlOrRd")
    ax[0].set_title("Spatial Rothermel ROS (m/min)\nfrom real LANDFIRE FBFM40", color="w", fontsize=9)
    for a, key, ttl in ((ax[1], "pred_spatial", f"Spatial front vs observed\nIoU {res['iou_spatial']:.2f}"),
                        (ax[2], "pred_uniform", f"Uniform-ROS front vs observed\nIoU {res['iou_uniform']:.2f}")):
        img = np.zeros((*P["obs"].shape, 3))
        img[P["obs"]] = [0.28, 0.31, 0.36]
        img[P[key] & P["obs"]] = [1.0, 0.55, 0.12]
        img[P[key] & ~P["obs"]] = [0.85, 0.1, 0.1]
        a.imshow(img)
        a.plot(P["ig"][1], P["ig"][0], "o", color="#37a3ff", ms=5)
        a.set_title(ttl, color="w", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"VHAGAR T4 spatial-Rothermel replay: {name} (real MTBS perimeter grey, hit orange, false alarm red)",
                 color="w", fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=130, facecolor="#0E1117")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--down", type=int, default=8)     # 30 m -> 240 m
    ap.add_argument("--fig", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results, panels = [], None
    hdr = f"{'fire':<8}{'cells':>10}{'obs_ha':>9}{'IoU_spatial':>12}{'IoU_uniform':>12}{'IoU_circle':>11}{'beats?':>8}"
    print(hdr)
    for name, lon, lat, hk, w, wd, mf in FIRES:
        res, P = replay(name, lon, lat, hk, w, wd, mf, args.down)
        results.append(res)
        if panels is None:
            panels = (name, P, res)
        print(f"{name:<8}{res['cells']:>10}{res['obs_ha']:>9,}{res['iou_spatial']:>12.3f}"
              f"{res['iou_uniform']:>12.3f}{res['iou_circle']:>11.3f}{'yes' if res['spatial_beats_uniform'] else 'no':>8}")
    if args.out:
        Path(args.out).write_text(json.dumps({"scope": ("Spatial Rothermel ROS from real LANDFIRE FBFM40 vs "
            "uniform ROS, validated on the real MTBS perimeter. LF2025 fuels (post-fire), slope proxy (no DEM), "
            "single representative wind. Ignition from public record."), "fires": results}, indent=2), encoding="utf-8")
        print("wrote", args.out)
    if args.fig and panels:
        _figure(*panels, args.fig)
        print("wrote", args.fig)


if __name__ == "__main__":
    main()
