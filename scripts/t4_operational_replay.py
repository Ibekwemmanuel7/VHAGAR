"""T4 operational replay: a spatial-Rothermel, short-horizon spread forecast for one
REAL incident, initialized at an observed cutoff time and scored against the incident's
own LATER observed progression, next to persistence / equal-area circle / uniform-ROS
baselines.

Claim under test, stated narrowly (per the operational-baseline plan):
    "VHAGAR provides short-horizon, surface-fire directional spread decision support
     for a selected incident."
It does NOT claim structure-to-structure spread, spotting, or authoritative agency
perimeters. Those are separate modules with their own validation.

Pipeline (using only information available at the cutoff T):
    timed VIIRS detections   -> rasterize to a local grid, earliest time per cell
    detections with t <= T   -> observed initial front + ignition seed
    LANDFIRE FBFM40 (real)   -> cell-by-cell Rothermel head ROS (m/min); non-burnable = 0
    representative wind + dead-fuel moisture -> ROS magnitude
    FMM arrival field, per-fire scale k calibrated on t <= T detections ONLY
    forecast arrival <= t_eval -> predicted new burn beyond the initial front

Scoring (on NEW burn only; initial-front cells excluded) against the held-out t > T
detections: arrival-time MAE (hours), IoU, Dice, POD (1 - omission) and FAR
(commission), each compared with three mandatory baselines:
    persistence     the initial front frozen (predicts no growth);
    equal-area disk a circle grown to the observed new-burn area at the front centroid;
    uniform ROS     the same solver on a spatially-flat ROS (mean of the Rothermel field).

Honest scope / documented gaps (staged deliberately):
  * truth is VIIRS ~375 m active-fire detections, a coarse recall proxy, NOT an agency
    perimeter; POD/FAR are against that proxy;
  * wind is a single representative value; time-varying wind is the next input;
  * slope/aspect from a DEM is absent here (slope_tan = 0), a documented degradation;
  * fuels are LF2025 (current), not a pre-fire vintage;
  * ONE incident is a milestone, not a validated domain. Only after this runs across a
    representative set of fires (fuels, regions, wind regimes, seasons) is "operational
    decision support" an honest label.

    python scripts/t4_operational_replay.py viirs_truth.csv \
        --lf-zip LF2025_FBFM40_CONUS.zip \
        --fig outputs/t4_operational_replay.png --out outputs/t4_operational_replay.json
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
    rasterize_detections,
)
from vhagar.eval.metrics import dice, iou, pod_far
from vhagar.eval.perimeter_shape import arrival_time_mae, equal_area_disk
from vhagar.models.rothermel import rothermel_head_ros_field
from vhagar.models.state_estimation import estimate_arrival_field

_ROOT = Path(__file__).resolve().parents[1]

# refusal thresholds: below these the operational forecast is withheld with a reason.
MIN_EARLY_CELLS = 15       # too few pre-cutoff detections to define a front / calibrate
MIN_LATE_CELLS = 10        # too little later progression to score against
MIN_BURNABLE_FRAC = 0.05   # fuels essentially all non-burnable over the window


def _load(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    dt = (pd.to_datetime(df["acq_date"])
          + pd.to_timedelta(df["acq_time"].astype(int) // 100, unit="h")
          + pd.to_timedelta(df["acq_time"].astype(int) % 100, unit="m"))
    return df.assign(_dt=dt)


def _clusters(df: pd.DataFrame, min_days: int, min_dets: int, top: int):
    df = df.assign(cy=(df.latitude / 0.5).round().astype(int),
                   cx=(df.longitude / 0.5).round().astype(int))
    g = (df.groupby(["cy", "cx"]).agg(n=("latitude", "size"), days=("acq_date", "nunique"))
         .reset_index())
    g = g[(g.days >= min_days) & (g.n >= min_dets)].sort_values("n", ascending=False)
    return g.head(top), df


def _fuel_codes(spec: GridSpec, lf_vsi: str) -> np.ndarray:
    """Sample the LANDFIRE FBFM40 raster into an (H, W) fuel-code grid over ``spec``
    (nearest cell), reprojecting grid lon/lat into the raster CRS."""
    import rasterio
    from rasterio.warp import transform as wt
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window
    ds = rasterio.open(lf_vsi)
    H, W = spec.shape
    w, s, e, n = spec.bbox
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    lon = w + (cc.ravel() + 0.5) / W * (e - w)
    lat = n - (rr.ravel() + 0.5) / H * (n - s)
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", ds.crs, w, s, e, n)
    win = ds.window(x0, y0, x1, y1).round_offsets().round_lengths()
    win = Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
    arr = ds.read(1, window=win, boundless=True, fill_value=-9999)
    inv = ~ds.window_transform(win)
    xs, ys = wt("EPSG:4326", ds.crs, list(lon), list(lat))
    col, row = inv * (np.asarray(xs), np.asarray(ys))
    col = np.clip(np.round(col).astype(int), 0, arr.shape[1] - 1)
    row = np.clip(np.round(row).astype(int), 0, arr.shape[0] - 1)
    return arr[row, col].reshape(H, W).astype(np.int64)


def _time_grid(det_rc: np.ndarray, det_times: np.ndarray, shape) -> np.ndarray:
    """Observed first-arrival time per cell (hours); non-detected cells are +inf."""
    tg = np.full(shape, np.inf)
    tg[det_rc[:, 0], det_rc[:, 1]] = det_times
    return tg


def _forecast(prior_ros, ignition, det_rc_early, det_times_early, t_eval, eval_mask):
    """Calibrate the per-fire ROS scale on the early detections only, then return the
    predicted new-burn mask and the calibrated arrival field."""
    state = estimate_arrival_field(prior_ros, ignition, det_rc_early, det_times_early)
    pred = state.burned_by(t_eval) & eval_mask
    return pred, state


def _score(pred, truth, arrival=None, truth_time=None):
    """IoU/Dice/POD/FAR on the new-burn mask. Arrival-time MAE is reported only for
    solver-based forecasts (``arrival`` given); the circle and persistence baselines
    have no arrival field, so their MAE is NaN, not a spurious number."""
    mae = (arrival_time_mae(pred, truth, arrival, truth_time)
           if arrival is not None else float("nan"))
    return {"iou": round(float(iou(truth, pred)), 4),
            "dice": round(float(dice(truth, pred)), 4),
            "pod": round(float(pod_far(truth, pred)[0]), 4),
            "far": round(float(pod_far(truth, pred)[1]), 4),
            "arrival_mae_h": round(mae, 3)}


def replay(sub: pd.DataFrame, *, cell_deg: float, split_frac: float, lf_vsi: str,
           wind_ms: float, m_f: float, horizon_h: float | None) -> dict:
    lo, la = sub.longitude.to_numpy(), sub.latitude.to_numpy()
    pad = 0.02
    spec = GridSpec.from_bbox_res((lo.min() - pad, la.min() - pad, lo.max() + pad, la.max() + pad),
                                  cell_deg=cell_deg)
    H, W = spec.shape
    th = ((sub["_dt"] - sub["_dt"].min()).dt.total_seconds() / 3600.0).to_numpy()
    det_rc, det_times = rasterize_detections(lo, la, th, spec)

    t_split = float(np.quantile(det_times, split_frac)) if det_times.size else 0.0
    early = det_times <= t_split
    if not early.any():
        early = det_times == det_times.min()
    late = ~early

    seen = np.zeros((H, W), bool)
    seen[det_rc[early, 0], det_rc[early, 1]] = True
    eval_mask = ~seen
    t_last = float(det_times.max()) if det_times.size else 0.0
    t_eval = float(min(t_split + horizon_h, t_last)) if horizon_h else t_last
    # short-horizon target: NEW burn observed within the forecast window (t_split, t_eval],
    # not the whole multi-day record. Scoring a 24 h forecast against detections days later
    # would unfairly penalise both extent and timing.
    truth_time_full = _time_grid(det_rc[late], det_times[late], (H, W))
    truth = (truth_time_full <= t_eval) & eval_mask

    codes = _fuel_codes(spec, lf_vsi)
    ros = rothermel_head_ros_field(codes, m_f=m_f, wind_ms=wind_ms, slope_tan=0.0)
    ros = np.clip(np.asarray(ros, float), 0.0, None)
    burnable_frac = float((ros > 0).mean())
    floor = float(np.mean(ros[ros > 0])) if (ros > 0).any() else 1.0
    ros_prior = ros.copy()
    ros_prior[seen & (ros_prior <= 0)] = floor          # real fire burned here; don't seed on a barrier

    acq_max = str(sub["_dt"].max())
    prov = {"incident_center": [round(float(la.mean()), 4), round(float(lo.mean()), 4)],
            "grid_cells": f"{H}x{W}", "cell_deg": cell_deg,
            "cutoff_hours_from_first": round(t_split, 2),
            "forecast_horizon_hours": round(t_eval - t_split, 2),
            "n_early_cells": int(seen.sum()), "n_late_new_cells": int(truth.sum()),
            "burnable_fraction": round(burnable_frac, 3),
            "inputs": {"detections": "VIIRS active-fire (NASA FIRMS), ~375 m recall proxy",
                       "detections_latest_acq_utc": acq_max,
                       "fuels": "LANDFIRE FBFM40 LF2025 (current vintage)",
                       "wind": f"single representative {wind_ms} m/s (time-varying wind: not wired)",
                       "terrain_dem": "absent (slope_tan=0): documented degradation",
                       "dead_fuel_moisture": m_f}}

    refuse = None
    if seen.sum() < MIN_EARLY_CELLS:
        refuse = f"too few pre-cutoff detections ({int(seen.sum())} < {MIN_EARLY_CELLS})"
    elif truth.sum() < MIN_LATE_CELLS:
        refuse = f"too little later progression to score ({int(truth.sum())} < {MIN_LATE_CELLS})"
    elif burnable_frac < MIN_BURNABLE_FRAC:
        refuse = f"fuels essentially non-burnable over window ({burnable_frac:.2f} < {MIN_BURNABLE_FRAC})"
    if refuse:
        return {"refused": refuse, "provenance": prov}

    ignition = ignition_from_detections(det_rc[early], det_times[early], spec, seed_quantile=0.1)
    if not ignition.any():
        ignition = seen.copy()
    truth_time = _time_grid(det_rc[late], det_times[late], (H, W))

    # model: spatial Rothermel ROS
    pred_m, state_m = _forecast(ros_prior, ignition, det_rc[early], det_times[early], t_eval, eval_mask)
    # baseline: uniform ROS (same solver, spatially-flat rate, same fuel breaks kept as barriers)
    ros_uniform = np.where(ros_prior > 0, floor, 0.0)
    pred_u, state_u = _forecast(ros_uniform, ignition, det_rc[early], det_times[early], t_eval, eval_mask)
    # baseline: persistence (front frozen, predicts no new burn)
    pred_p = np.zeros((H, W), bool)
    # baseline: equal-area circle grown to the observed new-burn area at the front centroid
    cy, cx = (np.argwhere(seen).mean(0) if seen.any() else (H / 2, W / 2))
    pred_c = equal_area_disk((H, W), (cy, cx), int(truth.sum())) & eval_mask

    res = {"provenance": prov,
           "ros_scale_k": round(float(state_m.k), 3),
           "ros_mean_mpermin": round(floor, 2),
           "model_spatial_rothermel": _score(pred_m, truth, state_m.arrival, truth_time),
           "baseline_uniform_ros": _score(pred_u, truth, state_u.arrival, truth_time),
           "baseline_persistence": _score(pred_p, truth),
           "baseline_equal_area_circle": _score(pred_c, truth)}
    res["model_beats_uniform_iou"] = bool(res["model_spatial_rothermel"]["iou"] > res["baseline_uniform_ros"]["iou"])
    res["model_beats_circle_iou"] = bool(res["model_spatial_rothermel"]["iou"] > res["baseline_equal_area_circle"]["iou"])
    res["_masks"] = dict(seen=seen, truth=truth, pred=pred_m, pred_circle=pred_c)
    return res


def _figure(res: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    m = res["_masks"]
    seen, truth, pred = m["seen"], m["truth"], m["pred"]
    H, W = seen.shape
    img = np.full((H, W, 3), [0.06, 0.08, 0.12])
    img[pred & truth] = [1.0, 0.55, 0.12]        # hit (forecast new burn that was observed)
    img[pred & ~truth] = [0.80, 0.12, 0.12]      # commission (forecast, not observed)
    img[truth & ~pred] = [0.95, 0.82, 0.30]      # omission (observed, missed)
    img[seen] = [0.22, 0.45, 0.85]               # pre-cutoff front (information at T)
    fig, ax = plt.subplots(figsize=(8, 6.6))
    fig.patch.set_facecolor("#0a0f16")
    ax.imshow(img, origin="upper")
    s = res["model_spatial_rothermel"]
    p = res["provenance"]
    ax.set_title("VHAGAR T4 operational replay (one real incident)\n"
                 f"spatial Rothermel over real LANDFIRE fuels  ·  cutoff {p['cutoff_hours_from_first']:.0f} h, "
                 f"horizon {p['forecast_horizon_hours']:.0f} h\n"
                 f"new-burn IoU {s['iou']:.2f} (uniform {res['baseline_uniform_ros']['iou']:.2f}, "
                 f"circle {res['baseline_equal_area_circle']['iou']:.2f})  ·  arrival MAE {s['arrival_mae_h']:.1f} h",
                 color="white", fontsize=10)
    ax.legend(handles=[Patch(color=[0.22, 0.45, 0.85], label="pre-cutoff front (info at T)"),
                       Patch(color=[1.0, 0.55, 0.12], label="forecast hit"),
                       Patch(color=[0.80, 0.12, 0.12], label="commission (false alarm)"),
                       Patch(color=[0.95, 0.82, 0.30], label="omission (missed)")],
              loc="lower right", fontsize=8, facecolor="#11161f", labelcolor="white")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#0a0f16")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--lf-zip", default=str(_ROOT / "LF2025_FBFM40_CONUS.zip"))
    ap.add_argument("--min-days", type=int, default=4)
    ap.add_argument("--min-dets", type=int, default=1500)
    ap.add_argument("--cell-deg", type=float, default=0.005)
    ap.add_argument("--split-frac", type=float, default=0.5, help="fraction of the detection timeline used as the cutoff T")
    ap.add_argument("--horizon-h", type=float, default=None, help="forecast horizon in hours past T (default: to last observation)")
    ap.add_argument("--wind-ms", type=float, default=3.0)
    ap.add_argument("--moisture", type=float, default=0.06)
    ap.add_argument("--incidents", type=int, default=6,
                    help="how many of the largest multi-day complexes to replay (a validation sweep, not one showcase fire)")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lf_vsi = f"/vsizip/{args.lf_zip}/LF2025_FBFM40_CONUS/Tif/LF2025_FBFM40_CONUS.tif"
    df = _load(args.csv)
    g, df = _clusters(df, args.min_days, args.min_dets, args.incidents)
    if g.empty:
        raise SystemExit("no multi-day complex met the size threshold")

    scored, refused, fig_res = [], [], None
    hdr = f"{'incident':<22}{'grid':>9}{'sp_IoU':>8}{'un_IoU':>8}{'ci_IoU':>8}{'sp_POD':>8}{'sp_FAR':>8}{'MAE_h':>7}"
    print(hdr)
    for _, r in g.iterrows():
        sub = df[(df.cy == r.cy) & (df.cx == r.cx)]
        res = replay(sub, cell_deg=args.cell_deg, split_frac=args.split_frac, lf_vsi=lf_vsi,
                     wind_ms=args.wind_ms, m_f=args.moisture, horizon_h=args.horizon_h)
        if "refused" in res:
            refused.append(res)
            c = res["provenance"]["incident_center"]
            print(f"{str(c):<22}{'':>9}  REFUSED: {res['refused']}")
            continue
        p, s = res["provenance"], res["model_spatial_rothermel"]
        u, ci = res["baseline_uniform_ros"], res["baseline_equal_area_circle"]
        print(f"{str(p['incident_center']):<22}{p['grid_cells']:>9}{s['iou']:>8.3f}{u['iou']:>8.3f}"
              f"{ci['iou']:>8.3f}{s['pod']:>8.3f}{s['far']:>8.3f}{s['arrival_mae_h']:>7.2f}")
        scored.append(res)
        if fig_res is None or res["provenance"]["n_late_new_cells"] > fig_res["provenance"]["n_late_new_cells"]:
            fig_res = res

    def _mean(key, sub):
        vals = [x[key][sub] for x in scored if np.isfinite(x[key][sub])]
        return float(np.mean(vals)) if vals else float("nan")

    agg = {}
    if scored:
        agg = {"n_incidents_scored": len(scored), "n_refused": len(refused),
               "mean_iou_spatial": round(_mean("model_spatial_rothermel", "iou"), 4),
               "mean_iou_uniform": round(_mean("baseline_uniform_ros", "iou"), 4),
               "mean_iou_circle": round(_mean("baseline_equal_area_circle", "iou"), 4),
               "mean_pod_spatial": round(_mean("model_spatial_rothermel", "pod"), 4),
               "mean_far_spatial": round(_mean("model_spatial_rothermel", "far"), 4),
               "mean_arrival_mae_h": round(_mean("model_spatial_rothermel", "arrival_mae_h"), 3),
               "far_caveat": "FAR is against VIIRS ~375 m detections, a recall-limited proxy that misses burned cells, so commission is overstated; POD is the more trustworthy VIIRS metric",
               "spatial_ge_uniform_count": int(sum(x["model_beats_uniform_iou"] for x in scored)),
               "spatial_ge_circle_count": int(sum(x["model_beats_circle_iou"] for x in scored))}
        print(f"\nSWEEP ({agg['n_incidents_scored']} incidents, {agg['n_refused']} refused): "
              f"mean new-burn IoU spatial {agg['mean_iou_spatial']:.3f}, uniform {agg['mean_iou_uniform']:.3f}, "
              f"circle {agg['mean_iou_circle']:.3f}; mean POD {agg['mean_pod_spatial']:.3f}, "
              f"FAR {agg['mean_far_spatial']:.3f} (vs sparse VIIRS), mean arrival MAE {agg['mean_arrival_mae_h']:.1f} h")
        print(f"spatial >= circle on {agg['spatial_ge_circle_count']}/{agg['n_incidents_scored']} incidents, "
              f">= uniform on {agg['spatial_ge_uniform_count']}/{agg['n_incidents_scored']}")

    if args.fig and fig_res is not None:
        _figure(fig_res, args.fig)
        print("wrote", args.fig)
    if args.out:
        out = {"claim": "short-horizon, surface-fire directional spread decision support for a selected incident",
               "aggregate": agg,
               "incidents": [{k: v for k, v in x.items() if k != "_masks"} for x in scored],
               "refused": [x for x in refused]}
        Path(args.out).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
