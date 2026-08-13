"""Phase 0, Step 1, first real bytes end to end.

Pulls real GOES-19/18 ABI fire detections from NOAA's public S3 bucket and real
VIIRS detections from NASA FIRMS, fuses them into events with VHAGAR's
parallax-aware clusterer, and reports what it found.

    python scripts/step1_first_light.py --bbox -124 36 -118 42 --hours 6

Requires network. GOES needs no credentials (the NOAA buckets are public and
not requester-pays). FIRMS needs a free map key:

    https://firms.modaps.eosdis.nasa.gov/api/map_key/
    export FIRMS_MAP_KEY=...          # Windows: set FIRMS_MAP_KEY=...

What this is for
----------------
Everything in VHAGAR up to this point is apparatus. This script is the first
thing that turns assumptions into facts. Expect it to surface problems, that
is the point. Specifically, watch for:

* how many GOES detections are low-probability (code 15) versus good (10/30),
* how far GOES and VIIRS detections of the same fire actually are apart,
* how many events are single-sensor, which is your false-alarm suspect pool,
* whether any detections land on the FIRMS static thermal anomaly mask.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np


def log(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str) -> None:
    log(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def preflight() -> list[str]:
    """Check the optional stack up front and say exactly what is missing.

    Reporting a missing dependency once, by name, beats reporting it N times as
    a bare exception type, which is what this script did in v0.3 and is the
    reason you are reading this docstring.
    """
    problems: list[str] = []
    for mod, hint in (
        ("s3fs", "pip install s3fs"),
        ("xarray", "pip install -U 'xarray numpy pandas'"),
        ("h5netcdf", "pip install h5netcdf h5py"),
        ("h5py", "pip install h5py"),
        ("pyproj", "pip install pyproj"),
    ):
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001 - we want the real message
            problems.append(f"    {mod:<12} {type(exc).__name__}: {exc}\n                 fix: {hint}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="VHAGAR Phase 0 Step 1, first light")
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        default=[-124.0, 36.0, -118.0, 42.0],
        help="west south east north, degrees (default: northern California)",
    )
    ap.add_argument("--satellite", type=int, default=19, choices=[16, 18, 19])
    ap.add_argument("--domain", default="C", choices=["C", "F"], help="C=CONUS 5min, F=full disk")
    ap.add_argument("--hours", type=float, default=6.0, help="look-back window")
    ap.add_argument("--max-granules", type=int, default=24, help="cap on granules to read")
    ap.add_argument("--region", default="conus", choices=["conus", "canada", "europe"])
    ap.add_argument("--out", default="", help="optional GeoJSON output path")
    args = ap.parse_args()

    west, south, east, north = args.bbox
    bbox = (west, south, east, north)

    from vhagar.grid import REGION_CRS
    from vhagar.harmonize.fusion import cluster_detections, event_features
    from vhagar.io.goes_reader import (
        list_fdc_granules,
        mask_summary,
        open_fdc,
        read_fdc_detections,
    )

    crs = REGION_CRS[args.region]
    end = datetime.now(UTC)
    start = end - timedelta(hours=args.hours)

    rule("Step 1, first real bytes")
    log(f"  bbox        {bbox}")
    log(f"  window      {start:%Y-%m-%d %H:%M} to {end:%H:%M} UTC")
    log(f"  satellite   GOES-{args.satellite}  domain {args.domain}")
    log(f"  analysis CRS {crs}")

    # -- GOES ------------------------------------------------------------
    rule("0. Preflight")
    problems = preflight()
    if problems:
        log("  [FAIL] the optional geospatial stack is not importable:\n")
        for p in problems:
            log(p)
        log("\n  Nothing below will work until these import. Most often this is")
        log("  xarray refusing an older numpy/pandas -- upgrade all three together.")
        return 3
    log("  s3fs, xarray, h5netcdf, h5py, pyproj all import OK")

    rule(f"1. GOES ABI L2 FDC from s3://noaa-goes{args.satellite} (anonymous)")
    try:
        keys = list_fdc_granules(args.satellite, start, end, domain=args.domain)
    except Exception as exc:  # noqa: BLE001
        log(f"  [FAIL] could not list S3: {type(exc).__name__}: {exc}")
        log("  Check network access. The bucket is public; no credentials needed.")
        return 2

    log(f"  found {len(keys)} granules")
    if not keys:
        log("  Nothing to read. Widen --hours, or check the satellite/domain.")
        return 1
    keys = keys[-args.max_granules :]
    log(f"  reading the most recent {len(keys)}")

    goes_dets = []
    totals: dict[str, int] = {}
    read_ok = 0
    n_failed = 0
    for i, key in enumerate(keys, 1):
        try:
            g = open_fdc(key, args.satellite, bbox=bbox)
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            log(
                f"  [{i:>3}/{len(keys)}] {key.rsplit('/', 1)[-1][:44]}  "
                f"SKIP {type(exc).__name__}: {exc}"
            )
            if n_failed == 1:
                import traceback

                log("\n  --- full traceback for the first failure ---")
                traceback.print_exc()
                log("  --- end traceback ---\n")
            if n_failed >= 3 and read_ok == 0:
                log("\n  [ABORT] first 3 granules all failed the same way. Fix the")
                log("  error above rather than waiting for 21 more identical lines.")
                return 4
            continue
        read_ok += 1
        dets = read_fdc_detections(g, crs=crs)
        goes_dets.extend(dets)
        for k, v in mask_summary(g).items():
            totals[k] = totals.get(k, 0) + v
        if i <= 3 or dets:
            log(
                f"  [{i:>3}/{len(keys)}] {g.scan_start:%H:%M:%S}  grid {g.mask.shape}  "
                f"new {g.n_fire_pixels(filtered=False):>3}  "
                f"confirmed {g.n_fire_pixels(filtered=True):>3}  "
                f"detections {len(dets):>3}"
            )

    log(f"\n  granules read : {read_ok}/{len(keys)}")
    log(f"  GOES detections: {len(goes_dets)}")
    if totals:
        log("  mask breakdown (all granules):")
        for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
            log(f"    {k:<45} {v:>6}")

    # -- FIRMS -----------------------------------------------------------
    rule("2. VIIRS from NASA FIRMS")
    viirs_dets = []
    if not os.environ.get("FIRMS_MAP_KEY"):
        log("  [SKIP] FIRMS_MAP_KEY not set. GOES-only run.")
        log("  Get a free key: https://firms.modaps.eosdis.nasa.gov/api/map_key/")
    else:
        from pyproj import Transformer

        from vhagar.harmonize.fusion import Detection
        from vhagar.io.firms import FirmsClient

        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        client = FirmsClient()
        # NOAA-21 is primary; NOAA-20 secondary. S-NPP delivery ends 2026-11-01.
        for source in ("viirs_noaa21_nrt", "viirs_noaa20_nrt"):
            try:
                recs = client.area(source=source, bbox=bbox, day_range=1)
            except Exception as exc:  # noqa: BLE001
                log(f"  {source:<22} FAILED ({type(exc).__name__}: {str(exc)[:60]})")
                continue
            recs = [r for r in recs if r.acq_datetime >= start]
            log(f"  {source:<22} {len(recs):>4} detections in window")
            for r in recs:
                x, y = tf.transform(r.longitude, r.latitude)
                viirs_dets.append(
                    Detection(
                        sensor="viirs",
                        x=float(x),
                        y=float(y),
                        when=r.acq_datetime,
                        frp_mw=r.frp if np.isfinite(r.frp) else None,
                        bt_mir_k=r.brightness if np.isfinite(r.brightness) else None,
                        bt_tir_k=r.bright_t31 if np.isfinite(r.bright_t31) else None,
                        confidence={"l": 0.25, "n": 0.60, "h": 0.90}.get(
                            str(r.confidence).lower()[:1], 0.5
                        ),
                    )
                )
        log(f"  VIIRS detections: {len(viirs_dets)}")

    # -- Fusion ----------------------------------------------------------
    rule("3. Multi-sensor fusion (parallax-aware)")
    all_dets = goes_dets + viirs_dets
    if not all_dets:
        log("  No detections in this window. Try a bigger bbox, a longer")
        log("  --hours, or an area with active fire. This is a normal outcome.")
        return 0

    events = cluster_detections(all_dets, max_gap_hours=12.0, extra_tolerance_m=2000.0)
    multi = [e for e in events if len(e.sensors) > 1]
    log(f"  {len(all_dets)} detections -> {len(events)} events")
    log(f"  multi-sensor confirmed: {len(multi)}  ({100 * len(multi) / max(len(events), 1):.0f}%)")
    log("  Single-sensor events are your false-alarm suspect pool.")

    ranked = sorted(events, key=lambda e: len(e.detections), reverse=True)[:10]
    log("\n  largest events:")
    log(f"    {'event':<14}{'det':>5}{'sens':>5}{'hours':>7}{'peakFRP':>10}{'growth/h':>10}")
    for e in ranked:
        f = event_features(e)
        peak = f["peak_frp_mw"]
        growth = f["frp_growth_mw_per_h"]
        log(
            f"    {e.event_id:<14}{len(e.detections):>5}{len(e.sensors):>5}"
            f"{e.duration_h:>7.1f}"
            f"{('  n/a' if not np.isfinite(peak) else f'{peak:>10.1f}')}"
            f"{('       n/a' if not np.isfinite(growth) else f'{growth:>10.1f}')}"
        )

    # -- GEO/LEO separation ----------------------------------------------
    if goes_dets and viirs_dets:
        rule("4. GEO vs LEO separation, why the tolerance is not a hyperparameter")
        from vhagar.harmonize.fusion import geo_leo_tolerance_m

        gx = np.array([d.x for d in goes_dets])
        gy = np.array([d.y for d in goes_dets])
        vza = np.array([d.view_zenith_deg for d in goes_dets if d.view_zenith_deg is not None])
        seps = np.array([float(np.hypot(gx - d.x, gy - d.y).min()) for d in viirs_dets])

        median_vza = float(np.median(vza)) if vza.size else float("nan")
        tol = float(geo_leo_tolerance_m(median_vza)) if vza.size else 2000.0

        log(f"  nearest GOES pixel to each VIIRS detection, n={len(seps)}")
        for q in (50, 75, 90, 95):
            log(f"    p{q:<3} {np.percentile(seps, q):>9,.0f} m")

        log(f"\n  median GOES view zenith here: {median_vza:.1f} deg")
        log(f"  geometry-derived tolerance:  {tol:,.0f} m")
        log("    = half the GOES pixel diagonal at this angle, plus terrain")
        log("      parallax (elevation x tan(vza)). Not a tuned constant.")
        log(f"  VIIRS detections inside it:  {100 * np.mean(seps <= tol):.0f}%")

        # Beyond a few pixel widths these are not the same fire at all. VIIRS
        # at 375 m sees fires that a 2 km ABI pixel cannot resolve, so the
        # nearest GOES detection is simply some other fire.
        far = seps > 5 * tol
        log(
            f"\n  {100 * np.mean(far):.0f}% of VIIRS detections are more than "
            f"{5 * tol / 1000:.0f} km from any GOES detection."
        )
        log("  Those are not mismatches. They are fires VIIRS resolves at 375 m")
        log("  and ABI cannot see at all, so p90/p95 above measure the distance")
        log("  to an unrelated fire and say nothing about parallax.")
        near = seps[~far]
        if near.size:
            log(
                f"  Among the {near.size} plausibly matched pairs: median "
                f"{np.median(near):,.0f} m, p75 {np.percentile(near, 75):,.0f} m."
            )

    # -- Output ----------------------------------------------------------
    if args.out:
        import json

        from pyproj import Transformer

        back = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        feats = []
        for e in events:
            cx, cy = e.centroid()
            lon, lat = back.transform(cx, cy)
            props = {k: (None if not np.isfinite(v) else v) for k, v in event_features(e).items()}
            props |= {
                "event_id": e.event_id,
                "first_seen": e.start.isoformat(),
                "last_seen": e.end.isoformat(),
                "sensors": sorted(e.sensors),
            }
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": props,
                }
            )
        with open(args.out, "w") as fh:
            json.dump({"type": "FeatureCollection", "features": feats}, fh, indent=2)
        log(f"\n  wrote {len(feats)} events to {args.out}")

    rule("Done")
    log("  Next: docs/07_PHASE0.md step 2, the tile archive backfill.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
