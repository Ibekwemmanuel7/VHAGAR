"""T4 real-data forward validation: assimilate real VIIRS timed detections.

Runs VHAGAR's T4 arrival-time assimilation on REAL active-fire detections (NASA
FIRMS VIIRS, with acquisition timestamps), for the largest multi-day fire complexes
in the input. For each fire it calibrates the per-fire rate-of-spread scale on the
early detections, forecasts, and scores the forecast against the held-out later
detections on NEW burn only (calibration cells excluded), reporting Sorensen (Dice),
POD, and FAR. It writes a per-fire figure and a JSON summary.

This is a genuine held-out forward validation on real fire progression, the number
the earlier synthetic-truth evaluation could not give. Honest scope: unless a fuel
raster is supplied with ``--fuel-tif``, the prior rate-of-spread field is nominal, so
only the scalar per-fire ROS scale is calibrated from real data, not the spatial fuel
pattern; the "fires" are half-degree FIRMS detection clusters (a complex may hold more
than one ignition); VIIRS detections are the truth proxy, not mapped perimeters.

Usage:
    python scripts/t4_real_viirs.py viirs_truth.csv --fig outputs/t4_real_fire.png \
        --out outputs/t4_real_viirs_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.datasets.spread_ingest import (
    GridSpec,
    ignition_from_detections,
    prior_ros_from_covariates,
    rasterize_detections,
)
from vhagar.eval.metrics import dice, pod_far
from vhagar.models.state_estimation import estimate_arrival_field


def _load(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    dt = (pd.to_datetime(df["acq_date"])
          + pd.to_timedelta(df["acq_time"].astype(int) // 100, unit="h")
          + pd.to_timedelta(df["acq_time"].astype(int) % 100, unit="m"))
    return df.assign(_dt=dt)


def _clusters(df: pd.DataFrame, min_days: int, min_dets: int, top: int) -> pd.DataFrame:
    df = df.assign(cy=(df.latitude / 0.5).round().astype(int),
                   cx=(df.longitude / 0.5).round().astype(int))
    g = (df.groupby(["cy", "cx"]).agg(n=("latitude", "size"), days=("acq_date", "nunique"))
         .reset_index())
    g = g[(g.days >= min_days) & (g.n >= min_dets)].sort_values("n", ascending=False)
    return g.head(top), df


def _case_and_masks(sub: pd.DataFrame, cell_deg: float, split_frac: float, fuel_tif: str | None):
    """Build a case, run the calibrate/forecast split, return metrics and plot masks."""
    lo, la = sub.longitude.to_numpy(), sub.latitude.to_numpy()
    pad = 0.02
    spec = GridSpec.from_bbox_res((lo.min() - pad, la.min() - pad, lo.max() + pad, la.max() + pad),
                                  cell_deg=cell_deg)
    th = ((sub["_dt"] - sub["_dt"].min()).dt.total_seconds() / 3600.0).to_numpy()
    det_rc, det_times = rasterize_detections(lo, la, th, spec)
    if det_rc.shape[0] < 20:
        return None
    ignition = ignition_from_detections(det_rc, det_times, spec, seed_quantile=0.05)
    fuel = _fuel_from_tif(spec, fuel_tif) if fuel_tif else None
    prior_ros = prior_ros_from_covariates(spec, fuel=fuel)

    t_split = float(np.quantile(det_times, split_frac))
    early = det_times <= t_split
    if not early.any():
        early = det_times == det_times.min()
    late = ~early
    state = estimate_arrival_field(prior_ros, ignition, det_rc[early], det_times[early])
    t_eval = float(det_times.max())
    H, W = spec.shape
    seen = np.zeros((H, W), bool)
    seen[det_rc[early, 0], det_rc[early, 1]] = True
    truth = np.zeros((H, W), bool)
    truth[det_rc[late, 0], det_rc[late, 1]] = True
    eval_mask = ~seen
    truth &= eval_mask
    pred = state.burned_by(t_eval) & eval_mask
    if truth.any():
        d = float(dice(truth, pred))
        pod, far = pod_far(truth, pred)
    else:
        d = pod = far = float("nan")
    return {"spec": spec, "k": float(state.k), "dice": d, "pod": float(pod), "far": float(far),
            "n_dets": int(len(sub)), "n_eval_cells": int(truth.sum()),
            "seen": seen, "truth": truth, "pred": pred,
            "center": (float(la.mean()), float(lo.mean()))}


def _fuel_from_tif(spec: GridSpec, path: str):
    """Optional: sample a LANDFIRE FBFM40 raster into a fuel-code grid over the spec."""
    import rasterio
    from rasterio.warp import transform as wt
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window
    ds = rasterio.open(path)
    H, W = spec.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    lon, lat = spec.rc_to_lonlat(rr.ravel(), cc.ravel()) if hasattr(spec, "rc_to_lonlat") else (None, None)
    if lon is None:  # derive lon/lat from bbox linearly
        w, s, e, n = spec.bbox
        lon = (w + (cc.ravel() + 0.5) / W * (e - w))
        lat = (n - (rr.ravel() + 0.5) / H * (n - s))
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", ds.crs, float(min(lon)), float(min(lat)),
                                      float(max(lon)), float(max(lat)))
    win = ds.window(x0, y0, x1, y1).round_offsets().round_lengths()
    win = Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
    arr = ds.read(1, window=win, boundless=True, fill_value=91)
    inv = ~ds.window_transform(win)
    xs, ys = wt("EPSG:4326", ds.crs, list(lon), list(lat))
    col, row = inv * (np.asarray(xs), np.asarray(ys))
    col = np.clip(np.round(col).astype(int), 0, arr.shape[1] - 1)
    row = np.clip(np.round(row).astype(int), 0, arr.shape[0] - 1)
    codes = arr[row, col].reshape(H, W).astype(np.int64)
    from vhagar.models.fuels import ros_from_fuel_codes
    return ros_from_fuel_codes(codes, base=1.0)          # relative fuel factor as the prior fuel field


def _figure(fire: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    seen, truth, pred = fire["seen"], fire["truth"], fire["pred"]
    H, W = seen.shape
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    base = np.full((H, W, 3), 0.10)                        # slate background
    base[..., 0] = 0.08
    base[..., 1] = 0.11
    base[..., 2] = 0.16
    ax.imshow(base, origin="upper")
    overlay = np.zeros((H, W, 4))                          # forecast new burn = orange fill
    overlay[pred] = [1.0, 0.5, 0.12, 0.38]
    ax.imshow(overlay, origin="upper")
    yr, xr = np.where(seen)
    ax.scatter(xr, yr, s=9, c="#37a3ff", label="calibration detections (early)")
    yt, xt = np.where(truth)
    ax.scatter(xt, yt, s=14, c="#ffd166", marker="s", edgecolors="k", linewidths=0.3,
               label="held-out later detections (truth)")
    la, lo = fire["center"]
    ax.set_title(f"T4 real forward forecast, VIIRS fire ~{la:.2f}, {lo:.2f}\n"
                 f"calibrate early, forecast held-out new burn  ·  Sorensen {fire['dice']:.2f}, "
                 f"POD {fire['pod']:.2f}, FAR {fire['far']:.2f}", fontsize=10, color="white")
    ax.legend(handles=[Patch(color="#ff7f0e", alpha=0.5, label="forecast burned (front)"),
                       Patch(color="#37a3ff", label="calibration detections (early)"),
                       Patch(color="#ffd166", label="held-out detections (truth)")],
              loc="lower right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor("#0a0f16")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#0a0f16")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--min-days", type=int, default=4)
    ap.add_argument("--min-dets", type=int, default=1500)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--cell-deg", type=float, default=0.01)
    ap.add_argument("--split-frac", type=float, default=0.5)
    ap.add_argument("--fuel-tif", default=None)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = _load(args.csv)
    g, df = _clusters(df, args.min_days, args.min_dets, args.top)
    fires, results = [], []
    print(f"{'fire (center)':<20} {'dets':>6} {'grid':>9} {'k':>5} {'Dice':>6} {'POD':>5} {'FAR':>5} {'eval':>6}")
    for _, r in g.iterrows():
        sub = df[(df.cy == r.cy) & (df.cx == r.cx)]
        f = _case_and_masks(sub, args.cell_deg, args.split_frac, args.fuel_tif)
        if f is None:
            continue
        la, lo = f["center"]
        H, W = f["spec"].shape
        print(f"{la:.2f},{lo:.2f}{'':<7} {f['n_dets']:>6} {H}x{W:<6} {f['k']:>5.2f} "
              f"{f['dice']:>6.3f} {f['pod']:>5.2f} {f['far']:>5.2f} {f['n_eval_cells']:>6}")
        fires.append(f)
        results.append({k: f[k] for k in ("k", "dice", "pod", "far", "n_dets", "n_eval_cells", "center")})
    dv = [x["dice"] for x in results if np.isfinite(x["dice"])]
    summary = {"n_fires": len(results), "mean_dice": float(np.mean(dv)) if dv else float("nan"),
               "mean_pod": float(np.mean([x["pod"] for x in results])),
               "mean_far": float(np.mean([x["far"] for x in results])),
               "prior": "landfire" if args.fuel_tif else "nominal (scalar ROS scale calibrated from real detections)",
               "scoring": "held-out post-cutoff new burn; calibration cells excluded", "fires": results}
    print(f"\nREAL held-out forward validation: {summary['n_fires']} fires, "
          f"mean Sorensen(Dice)={summary['mean_dice']:.3f}, mean POD={summary['mean_pod']:.2f}, "
          f"mean FAR={summary['mean_far']:.2f}")
    if args.fig and fires:
        best = max(fires, key=lambda f: (f["dice"] if np.isfinite(f["dice"]) else -1))
        _figure(best, args.fig)
        print(f"wrote {args.fig}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
