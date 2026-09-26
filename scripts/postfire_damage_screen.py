"""Post-fire structure-damage screen, validated against real CAL FIRE DINS.

The rapid-damage-assessment tier from the product review, done on real data. For a
named fire it takes the DINS post-fire structure inspections (the accepted US ground
truth), samples the MTBS burn-severity raster at each structure, predicts damage two
ways, and scores against DINS with the mandatory baselines:

  binary destroyed:  inside-burn screen (severity low/moderate/high) and a stricter
                     high-severity-only screen, each vs DINS "Destroyed (>50%)",
                     with POD/FAR/precision/F1/accuracy and the predict-all-destroyed
                     baseline plus the F1 skill over it;
  four-class grade:  severity mapped to the xView2/xBD scale (No Damage, Minor, Major,
                     Destroyed) vs the DINS class, scored with accuracy, per-class
                     recall/precision, and the quadratic-weighted kappa.

Honest scope: MTBS severity is a 30 m dNBR-derived product, not a per-structure
inspection, so this is a SCREEN, not a damage adjudication. DINS is destroyed-heavy
(inspectors concentrate near destruction), so the predict-all-destroyed baseline is
strong and the skill-over-baseline column, not the raw F1, is the honest read. MTBS
2021 is an end-of-year mosaic, appropriate for these 2021 fires.

    python scripts/postfire_damage_screen.py --incident Caldor \
        --fig outputs/postfire_Caldor.png --out outputs/postfire_Caldor.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vhagar.eval.damage_screen import (
    SEVERITY_CLASS_NAMES,
    binary_screen,
    confusion,
    ordinal_metrics,
    severity_to_class_index,
)

_ROOT = Path(__file__).resolve().parents[1]
DINS_CSV = _ROOT / "POSTFIRE_MASTER_DATA_SHARE_-519030234411900050.csv"
MTBS_TIF = _ROOT / "mtbs_extract" / "mtbs_CONUS_2021.tif"

# MTBS thematic burn severity: 1 unburned-to-low, 2 low, 3 moderate, 4 high,
# 5 increased greenness, 6 non-processing mask (0 = background/no data).
BURNED = (2, 3, 4)
CLASS_NAMES = SEVERITY_CLASS_NAMES

# DINS "* Damage" -> ordinal xView2/xBD class index (Inaccessible dropped by caller).
DINS_TO_IDX = {
    "No Damage": 0,
    "Affected (>0-10%)": 1, "Affected (1-9%)": 1, "Minor (10-25%)": 1,
    "Major (25-50%)": 2, "Major (26-50%)": 2,
    "Destroyed (>50%)": 3,
}


_severity_to_idx = severity_to_class_index


def _load_dins(incident: str):
    import pandas as pd
    df = pd.read_csv(DINS_CSV, low_memory=False)
    sub = df[df["* Incident Name"].astype(str).str.strip().str.casefold() == incident.casefold()]
    sub = sub.dropna(subset=["Latitude", "Longitude", "* Damage"])
    dmg = sub["* Damage"].astype(str).str.strip()
    keep = dmg.isin(DINS_TO_IDX)                  # drop Inaccessible / unknowns
    sub, dmg = sub[keep], dmg[keep]
    lat = sub["Latitude"].to_numpy(float)
    lon = sub["Longitude"].to_numpy(float)
    truth_idx = dmg.map(DINS_TO_IDX).to_numpy(int)
    return lat, lon, truth_idx


def _sample_mtbs(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Nearest MTBS severity code at each structure (windowed read around the fire)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds
    ds = rasterio.open(str(MTBS_TIF))
    tf = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    xs, ys = tf.transform(lon, lat)
    pad = 2000.0
    win = from_bounds(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad, ds.transform)
    win = win.round_offsets().round_lengths()
    arr = ds.read(1, window=win)
    wt = ds.window_transform(win)
    inv = ~wt
    cols, rows = inv * (np.asarray(xs), np.asarray(ys))
    rows = np.clip(np.round(rows).astype(int), 0, arr.shape[0] - 1)
    cols = np.clip(np.round(cols).astype(int), 0, arr.shape[1] - 1)
    return arr[rows, cols].astype(int)


def _figure(incident, lat, lon, truth_idx, sev, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pred_idx = _severity_to_idx(sev)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.patch.set_facecolor("#0E1117")
    colors = np.array([[0.5, 0.6, 0.7], [1.0, 0.85, 0.3], [1.0, 0.5, 0.1], [0.85, 0.12, 0.12]])
    ax[0].scatter(lon, lat, s=6, c=colors[truth_idx], linewidths=0)
    ax[0].set_title(f"DINS observed damage: {incident}", color="w", fontsize=10)
    ax[1].scatter(lon, lat, s=6, c=colors[pred_idx], linewidths=0)
    ax[1].set_title("MTBS severity screen (predicted)", color="w", fontsize=10)
    for a in ax:
        a.set_facecolor("#0E1117")
        a.tick_params(colors="w", labelsize=7)
        for sp in a.spines.values():
            sp.set_color("#444")
    b = res["binary_inside_burn"]
    fig.suptitle(f"Post-fire damage screen vs DINS  ·  {incident}  ·  inside-burn destroyed "
                 f"F1 {b['f1']:.2f} (baseline {b['predict_all_destroyed_f1']:.2f}, "
                 f"skill {b['skill_f1_over_baseline']:+.2f})  ·  4-class QWK "
                 f"{res['four_class']['quadratic_weighted_kappa']:.2f}",
                 color="w", fontsize=10, y=0.99)
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color=colors[i], label=CLASS_NAMES[i]) for i in range(4)],
                 loc="lower right", fontsize=7, facecolor="#11161f", labelcolor="white")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130, facecolor="#0E1117")


def screen(incident: str) -> dict:
    lat, lon, truth_idx = _load_dins(incident)
    if lat.size < 20:
        raise SystemExit(f"too few DINS structures for '{incident}' ({lat.size})")
    sev = _sample_mtbs(lat, lon)
    truth_destroyed = truth_idx == 3
    pred_inside = np.isin(sev, BURNED)
    pred_high = sev == 4
    pred_idx = _severity_to_idx(sev)
    cm = confusion(truth_idx, pred_idx, 4)
    res = {
        "incident": incident, "n_structures": int(lat.size),
        "dins_class_support": {CLASS_NAMES[i]: int(np.sum(truth_idx == i)) for i in range(4)},
        "mtbs_burned_fraction": round(float(pred_inside.mean()), 4),
        "binary_inside_burn": binary_screen(pred_inside, truth_destroyed),
        "binary_high_severity_only": binary_screen(pred_high, truth_destroyed),
        "four_class": ordinal_metrics(cm),
        "four_class_confusion_rows_true_cols_pred": cm.tolist(),
        "scope": ("MTBS 30 m severity screen vs CAL FIRE DINS inspections. A screen, "
                  "not a per-structure adjudication. DINS is destroyed-heavy, so read "
                  "skill_f1_over_baseline, not raw F1."),
    }
    res["_plot"] = (lat, lon, truth_idx, sev)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incident", default="Caldor", help="DINS '* Incident Name' (e.g. Caldor, Dixie)")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = screen(args.incident)
    plot = res.pop("_plot")
    b, h, fc = res["binary_inside_burn"], res["binary_high_severity_only"], res["four_class"]
    print(f"{res['incident']}: {res['n_structures']} DINS structures, "
          f"{res['dins_class_support']}")
    print(f"  inside-burn destroyed : POD {b['pod']:.2f} FAR {b['far']:.2f} F1 {b['f1']:.2f} "
          f"(baseline {b['predict_all_destroyed_f1']:.2f}, skill {b['skill_f1_over_baseline']:+.2f})")
    print(f"  high-severity only    : POD {h['pod']:.2f} FAR {h['far']:.2f} F1 {h['f1']:.2f} "
          f"(skill {h['skill_f1_over_baseline']:+.2f})")
    print(f"  4-class               : accuracy {fc['accuracy']:.2f}, QWK {fc['quadratic_weighted_kappa']:.2f}")
    if args.fig:
        _figure(args.incident, *plot, res, args.fig)
        print("wrote", args.fig)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
