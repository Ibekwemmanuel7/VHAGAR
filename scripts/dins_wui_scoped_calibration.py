"""Scoped WUI structure-spread calibration on CAL FIRE DINS (assumption-laden).

This calibrates the *structure-to-structure conflagration* component of the WUI
model (:func:`vhagar.models.wui.structure_to_structure_spread`) against real
destroyed/survived structures from DINS, WITHOUT the wildland-front driver.

Honest scope and caveats (read before quoting any number):
  * DINS gives structure locations + a destroyed/survived label per fire, but NOT
    the ignition point, perimeter, wind, or fuels that drive where the fire went.
  * So this does not test the physical wildland spread. It tests only: given a
    small SEED of first-ignited structures (proxied by the destroyed structures
    nearest the destroyed-cluster centroid), does proximity-based
    structure-to-structure propagation recover the *rest* of the destroyed set
    without over-igniting survivors?
  * No wind (isotropic propagation), because DINS carries none. Scoring excludes
    the seed structures (we grade recovered propagation, not the given seed).
  * Large fires are stratified-subsampled for tractability (preserving the
    destroyed/survivor ratio), with a fixed seed.

Interpretation: expect decent recall (proximity carries fire through dense
neighbourhoods) but a non-trivial false-alarm rate, because without the wildland
front the structure graph cannot know which side of a community actually burned.
That gap is exactly the value the faithful calibration (ignition + wind + fuels)
adds; this scoped run quantifies the structure-network component alone.

Usage:
    python scripts/dins_wui_scoped_calibration.py POSTFIRE_MASTER_DATA_SHARE_*.csv \
        --out dins_wui_scoped_summary.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.eval.wui import score_structures
from vhagar.models.wui import structure_to_structure_spread

DESTROYED = "Destroyed (>50%)"
RNG_SEED = 0
CELL_M = 1.0            # coords already in metres; radius params are in metres
SEED_FRAC = 0.05        # fraction of destroyed used as the (proxy) ignition core
MAX_STRUCT = 3500       # per-fire cap for tractability (stratified subsample)


def _to_metres(lat, lon):
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    x = (lon - lon0) * 111_320.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110_540.0
    return x, y


def _subsample(idx_destroyed, idx_survivor, cap, rng):
    n = idx_destroyed.size + idx_survivor.size
    if n <= cap:
        return np.concatenate([idx_destroyed, idx_survivor])
    frac = cap / n
    d = rng.choice(idx_destroyed, size=max(1, int(round(idx_destroyed.size * frac))), replace=False)
    s = rng.choice(idx_survivor, size=max(1, int(round(idx_survivor.size * frac))), replace=False)
    return np.concatenate([d, s])


def build_fires(df: pd.DataFrame, min_destroyed=150, min_survivors=100, max_fires=6):
    """Assemble per-fire (x, y, truth_destroyed, seed) from DINS, largest fires first."""
    dmg = df["* Damage"].astype(str).str.strip()
    df = df.assign(_destroyed=dmg.eq(DESTROYED))
    rng = np.random.default_rng(RNG_SEED)
    fires = []
    counts = df.groupby("* Incident Name")["_destroyed"].agg(["sum", "size"])
    counts = counts.sort_values("sum", ascending=False)
    for name in counts.index:
        d = df[df["* Incident Name"] == name]
        lat = pd.to_numeric(d["Latitude"], errors="coerce").to_numpy()
        lon = pd.to_numeric(d["Longitude"], errors="coerce").to_numpy()
        dest = d["_destroyed"].to_numpy()
        ok = np.isfinite(lat) & np.isfinite(lon)
        lat, lon, dest = lat[ok], lon[ok], dest[ok]
        idx_d = np.where(dest)[0]
        idx_s = np.where(~dest)[0]
        if idx_d.size < min_destroyed or idx_s.size < min_survivors:
            continue
        keep = np.sort(_subsample(idx_d, idx_s, MAX_STRUCT, rng))
        lat, lon, dest = lat[keep], lon[keep], dest[keep]
        x, y = _to_metres(lat, lon)
        # seed = destroyed structures nearest the destroyed centroid (proxy ignition core)
        dmask = dest.astype(bool)
        cx, cy = x[dmask].mean(), y[dmask].mean()
        dd = np.hypot(x - cx, y - cy)
        dd_masked = np.where(dmask, dd, np.inf)
        n_seed = max(3, int(round(SEED_FRAC * dmask.sum())))
        seed_idx = np.argsort(dd_masked)[:n_seed]
        seed = np.zeros(dest.shape, dtype=bool)
        seed[seed_idx] = True
        fires.append({"name": name, "x": x, "y": y, "truth": dmask, "seed": seed})
        if len(fires) >= max_fires:
            break
    return fires


def _predict(fire, radius_m, base_p):
    return structure_to_structure_spread(
        fire["y"], fire["x"], fire["seed"], wind_speed=0.0, wind_dir=0.0,
        radius_cells=radius_m, base_p=base_p)


def _score_fold(fire, radius_m, base_p):
    """Score on NON-seed structures (grade recovered propagation, not the given seed)."""
    pred = _predict(fire, radius_m, base_p)
    ev = ~fire["seed"]
    return score_structures(pred[ev], fire["truth"][ev])


def _mean_f1(fires, radius_m, base_p):
    vals = [_score_fold(f, radius_m, base_p).f1 for f in fires]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def calibrate(fires, radii, base_ps):
    grid = list(itertools.product(radii, base_ps))
    folds = []
    for i, held in enumerate(fires):
        train = fires[:i] + fires[i + 1:]
        best, best_f1 = grid[0], -1.0
        for r, p in grid:
            f1 = _mean_f1(train, r, p)
            if np.isfinite(f1) and f1 > best_f1:
                best_f1, best = f1, (r, p)
        s = _score_fold(held, *best)
        folds.append({"name": held["name"], "radius_m": best[0], "base_p": best[1],
                      "pod": s.pod, "far": s.far, "f1": s.f1,
                      "n": s.n, "tp": s.tp, "fp": s.fp, "fn": s.fn})
    # selected params over all fires
    best, best_f1 = grid[0], -1.0
    for r, p in grid:
        f1 = _mean_f1(fires, r, p)
        if np.isfinite(f1) and f1 > best_f1:
            best_f1, best = f1, (r, p)
    heldout = [f["f1"] for f in folds if np.isfinite(f["f1"])]
    return {"folds": folds, "mean_heldout_f1": float(np.mean(heldout)) if heldout else float("nan"),
            "selected_radius_m": best[0], "selected_base_p": best[1], "selected_f1": best_f1,
            "n_fires": len(fires)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dins_csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("dins_wui_scoped_summary.json"))
    ap.add_argument("--max-fires", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.dins_csv, low_memory=False)
    fires = build_fires(df, max_fires=args.max_fires)
    if len(fires) < 2:
        raise SystemExit("need >= 2 qualifying fires")
    rep = calibrate(fires, radii=[30.0, 60.0, 90.0, 120.0], base_ps=[0.4, 0.6, 0.8])
    rep["caveat"] = ("scoped: structure-to-structure propagation only, no wildland front / "
                     "wind; seed = destroyed nearest cluster centroid; scored on non-seed "
                     "structures; large fires stratified-subsampled.")
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"fires: {[f['name'] for f in fires]}")
    print(f"selected: radius={rep['selected_radius_m']} m, base_p={rep['selected_base_p']}, "
          f"train F1={rep['selected_f1']:.3f}")
    print(f"mean leave-one-fire-out held-out F1: {rep['mean_heldout_f1']:.3f}")
    print("per-fold (held-out):")
    for f in rep["folds"]:
        print(f"  {f['name']:<16} POD={f['pod']:.2f} FAR={f['far']:.2f} F1={f['f1']:.3f}  "
              f"(r={f['radius_m']}m p={f['base_p']})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
