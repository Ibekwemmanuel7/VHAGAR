"""T4 mapped-perimeter validation against REAL MTBS 2021 burned perimeters.

For each fire, crop the MTBS CONUS severity mosaic, take the burned mask (low/
moderate/high severity), isolate the fire as the largest connected component,
downsample to a workable grid, and score the elliptical arrival-time front's
reconstructed perimeter against the real one (IoU, Sorensen) versus an equal-area
circle baseline (see vhagar.eval.perimeter_shape for the honest scope).

    python scripts/t4_mtbs_perimeter.py \
        --mtbs mtbs_extract/mtbs_CONUS_2021.tif \
        --fig outputs/t4_mtbs_perimeter.png --out outputs/t4_mtbs_perimeter.json

Needs rasterio + pyproj + scipy (the geospatial extra), not the core test env.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vhagar.eval.perimeter_shape import evaluate_perimeter_shape, front_prediction

# Well-documented large 2021 CONUS fires: (name, ignition lon, lat, half-window km).
FIRES = [
    ("Dixie", -121.20, 40.05, 130),
    ("Caldor", -120.45, 38.65, 65),
    ("Bootleg", -121.30, 42.65, 60),
    ("Monument", -123.30, 40.60, 50),
    ("River Complex", -123.10, 41.20, 45),
]
BURNED_CLASSES = (2, 3, 4)          # MTBS low / moderate / high severity


def _fire_mask(ds, tf, lon, lat, half_km, down):
    from rasterio.windows import from_bounds
    from scipy import ndimage

    x, y = tf.transform(lon, lat)
    h = half_km * 1000.0
    win = from_bounds(x - h, y - h, x + h, y + h, ds.transform)
    sev = ds.read(1, window=win)
    burned = np.isin(sev, BURNED_CLASSES)
    lbl, n = ndimage.label(burned)
    if n == 0:
        return None, False
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    fire = lbl == (int(np.argmax(sizes)) + 1)
    clipped = bool(fire[0, :].any() or fire[-1, :].any() or fire[:, 0].any() or fire[:, -1].any())
    # downsample: a coarse cell is burned if any fine cell in it is (max-pool)
    H, W = fire.shape
    Hc, Wc = H // down, W // down
    fire = fire[:Hc * down, :Wc * down].reshape(Hc, down, Wc, down).max(axis=(1, 3))
    return fire.astype(bool), clipped


def _figure(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.4))
    if ncol == 1:
        axes = [axes]
    for ax, (name, mask, pred, res) in zip(axes, panels, strict=True):
        H, W = mask.shape
        img = np.zeros((H, W, 3))
        img[mask] = [0.25, 0.28, 0.33]                 # real perimeter, grey
        img[pred & mask] = [1.0, 0.55, 0.12]           # hit, orange
        img[pred & ~mask] = [0.85, 0.1, 0.1]           # false alarm, red
        ax.imshow(img, origin="upper")
        ax.set_title(f"{name}\nIoU {res['iou_front']:.2f} vs circle {res['iou_circle_centroid']:.2f}",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("T4 elliptical front vs real MTBS perimeter (grey=real, orange=hit, red=false alarm)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mtbs", default="mtbs_extract/mtbs_CONUS_2021.tif")
    ap.add_argument("--down", type=int, default=8)          # 30 m -> 240 m cells
    ap.add_argument("--fig", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import rasterio
    from pyproj import Transformer
    ds = rasterio.open(args.mtbs)
    tf = Transformer.from_crs("EPSG:4326", ds.crs.to_string(), always_xy=True)

    results, panels = [], []
    hdr = f"{'fire':<14}{'ha':>10}{'LB':>6}{'IoU_front':>11}{'Dice':>7}{'IoU_circle':>12}{'beats?':>8}"
    print(hdr)
    for name, lon, lat, half_km in FIRES:
        mask, clipped = _fire_mask(ds, tf, lon, lat, half_km, args.down)
        if mask is None or mask.sum() < 50:
            print(f"{name:<14} (too small / not found)")
            continue
        res = evaluate_perimeter_shape(mask)
        ha = mask.sum() * (30 * args.down) ** 2 / 1e4
        res.update({"fire": name, "ha": round(ha), "clipped_by_window": clipped})
        results.append(res)
        pred, _ = front_prediction(mask)
        panels.append((name, mask, pred, res))
        print(f"{name:<14}{ha:>10,.0f}{res['lb']:>6.2f}{res['iou_front']:>11.3f}"
              f"{res['dice_front']:>7.3f}{res['iou_circle_centroid']:>12.3f}"
              f"{'yes' if res['front_beats_circle'] else 'no':>8}"
              + ("  [clipped]" if clipped else ""))

    if results:
        mf = float(np.mean([r["iou_front"] for r in results]))
        mc = float(np.mean([r["iou_circle_centroid"] for r in results]))
        nbeat = sum(r["front_beats_circle"] for r in results)
        print(f"\nMEAN IoU: elliptical front {mf:.3f} vs equal-area circle {mc:.3f}; "
              f"front wins on {nbeat}/{len(results)} fires.")
        summary = {"n_fires": len(results), "mean_iou_front": mf,
                   "mean_iou_circle_centroid": mc, "front_wins": nbeat,
                   "scope": ("Shape-family adequacy vs real MTBS perimeters: final area, dominant "
                             "axis and length-to-breadth taken from truth; only the elliptical front "
                             "geometry is tested, against an equal-area circle baseline. Not a blind "
                             "forecast."),
                   "fires": results}
        if args.out:
            Path(args.out).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
            print(f"wrote {args.out}")
        if args.fig and panels:
            _figure(panels, args.fig)
            print(f"wrote {args.fig}")


if __name__ == "__main__":
    main()
