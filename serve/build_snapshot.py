"""Build a near-real-time, multi-sensor VHAGAR snapshot for the live console.

Pulls the newest fire detections from every configured source, GOES-18 and
GOES-19 ABI FDC off the public NOAA S3 buckets (anonymous), plus VIIRS
(S-NPP / NOAA-20 / NOAA-21) and MODIS off NASA FIRMS when FIRMS_MAP_KEY is set,
then fuses them in one parallax-aware clustering pass and packages the result as
``snapshot.tgz``
(``detections.parquet`` + ``events.pkl``) that ``serve/vhagar_api.py`` serves
when ``VHAGAR_SNAPSHOT_URL`` points at it.

Designed for a scheduled job (see .github/workflows/live-snapshot.yml): each run
is self-contained (it pulls a fixed lookback window), so no persistent store is
needed between runs. Weather is NOT baked in here; the API enriches events with
live current conditions per request when ``VHAGAR_WEATHER`` is set, keeping fire
data and weather as separate, honestly-labelled measurements.

Needs internet to noaa-goes18/19 and the decode stack (s3fs, xarray, h5netcdf,
h5py, pyproj). It does not run inside the offline sandbox.

    python -m serve.build_snapshot --sats 18,19 --lookback-hours 12 --out snapshot.tgz
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import tarfile
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def _log(msg: str) -> None:
    print(f"[vhagar-snapshot] {msg}", flush=True)


def _ingest_goes(sat: int, domain: str, region: str, lookback_hours: float, workers: int,
                 bbox=None):
    """Pull one GOES satellite into its own store (backfill refuses to mix
    satellites in one directory) and return its detections as a post-processed
    DataFrame with t + sensor columns, or None if nothing landed. ``bbox`` clips
    the pull to the region so neighbouring regions do not double-cluster the same
    fire (their grids overlap)."""
    import pandas as pd

    from serve.ingest import ingest_once
    from serve.vhagar_api import _sensor_from_granule

    sub = Path(tempfile.mkdtemp(prefix=f"vhagar_g{sat}_"))
    _log(f"pulling GOES-{sat} {domain} last {lookback_hours:g} h ...")
    try:
        ingest_once(
            out_dir=sub, satellite=sat, domain=domain, region=region, bbox=bbox,
            lookback_min=lookback_hours * 60.0, workers=workers, min_confidence=0.0,
            drop_filtered=False, retention_days=0.0,  # each run is self-contained
        )
    except Exception as exc:  # one unavailable feed must not sink the run
        _log(f"GOES-{sat} pull failed ({type(exc).__name__}: {exc}); continuing")
        return None
    det = sub / "detections"
    parts = list(det.glob("year=*/tile=*/part-*.parquet")) if det.exists() else []
    if not parts:
        _log(f"GOES-{sat}: no granules in the window")
        return None
    g = pd.read_parquet(det)
    g["t"] = pd.to_datetime(g["t"], utc=False)
    g["sensor"] = g["granule_key"].map(_sensor_from_granule)
    _log(f"GOES-{sat}: {len(g)} detections")
    return g


def _bake_weather(events) -> int:
    """Attach current wind/RH/temp + spread-risk to each event record in place.
    Best-effort: returns how many events got weather, 0 on any failure."""
    if not events:
        return 0
    try:
        from vhagar.features.spread_risk import risk_class, spread_risk_score
        from vhagar.weather import fetch_weather
        pts = [(e["centroid_lat"], e["centroid_lon"]) for e in events]
        wx = fetch_weather(pts)
        got = 0
        for e, w in zip(events, wx, strict=False):
            if not w:
                continue
            got += 1
            e.update({k: w.get(k) for k in
                      ("temp_c", "rh_pct", "wind_speed_ms", "wind_dir_deg", "wind_gust_ms")})
            s = spread_risk_score(w.get("temp_c"), w.get("rh_pct"), w.get("wind_speed_ms"))
            e["risk_score"] = s
            e["risk_class"] = risk_class(s)
        return got
    except Exception as exc:  # noqa: BLE001
        _log(f"weather bake failed ({type(exc).__name__}: {exc})")
        return 0


def build(sats: list[int], domain: str, regions: list[str], lookback_hours: float,
          workers: int, out: Path, min_detections: int, firms_key: str | None = None) -> int:
    import pandas as pd

    # Do not let the build recurse into a published snapshot or the committed demo.
    os.environ.pop("VHAGAR_SNAPSHOT_URL", None)
    os.environ.pop("VHAGAR_FROZEN", None)
    from serve import vhagar_api as api

    if not firms_key:
        _log("FIRMS_MAP_KEY not set; VIIRS/MODIS skipped (GOES only)")
    frames = []
    for region in regions:
        _log(f"region: {region}")
        bbox = api.REGIONS.get(region, api.REGIONS["conus"])["bbox"]
        # GEO: each GOES satellite, clipped + tiled into this region's grid.
        for sat in sats:
            g = _ingest_goes(sat, domain, region, lookback_hours, workers, bbox=bbox)
            if g is not None and len(g):
                frames.append(g)
        # LEO: VIIRS (S-NPP / NOAA-20 / NOAA-21) + MODIS via NASA FIRMS.
        if firms_key:
            from serve.firms_ingest import fetch_firms
            leo = fetch_firms(firms_key, bbox, region=region, hours=lookback_hours)
            if len(leo):
                leo["t"] = pd.to_datetime(leo["t"], utc=True).dt.tz_localize(None)
                _log(f"FIRMS LEO [{region}]: {len(leo)} detections "
                     f"across {leo['sensor'].nunique()} sensor(s)")
                frames.append(leo)

    if not frames:
        _log("no detections from any source (quiet period or fetch failed); not publishing")
        return 3
    df = pd.concat(frames, ignore_index=True)
    if len(df) < min_detections:
        _log(f"only {len(df)} detections (< {min_detections}); not publishing")
        return 3
    if "confidence" in df.columns:                 # mixed str (VIIRS) / numeric (GOES)
        df["confidence"] = df["confidence"].astype("string")

    # One fused clustering pass across every sensor (parallax-aware tolerances).
    events = api._cluster_all(df)
    sensors = ", ".join(sorted(df["sensor"].astype(str).unique()))
    _log(f"clustered {len(df)} detections [{sensors}] into {len(events)} events")

    # Bake current weather + spread risk into each event here, on the runner, so
    # the deployed API never calls the weather service (whose free tier rate-
    # limits a shared cloud egress IP). One batched call per build; best-effort.
    got = _bake_weather(events)
    _log(f"weather baked into {got}/{len(events)} events")

    work = Path(tempfile.mkdtemp(prefix="vhagar_pack_"))
    df.to_parquet(work / "detections.parquet")
    (work / "events.pkl").write_bytes(pickle.dumps(events))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tf:
        tf.add(work / "detections.parquet", arcname="detections.parquet")
        tf.add(work / "events.pkl", arcname="events.pkl")
    _log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): "
         f"{len(df)} detections, {len(events)} events, window {lookback_hours:g} h")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a near-real-time VHAGAR snapshot")
    ap.add_argument("--sats", default="18,19",
                    help="comma list of GOES satellites (18=West, 19=East)")
    ap.add_argument("--domain", default="C", help="ABI domain: C, F, M1, M2")
    ap.add_argument("--regions", default="conus,canada",
                    help="comma list of analysis regions to build and fuse")
    ap.add_argument("--lookback-hours", type=float, default=12.0,
                    help="how far back to pull each run (self-contained window)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-detections", type=int, default=1,
                    help="publish only if at least this many detections were found")
    ap.add_argument("--out", type=Path, default=_ROOT / "snapshot.tgz")
    a = ap.parse_args()
    sats = [int(s) for s in a.sats.split(",") if s.strip()]
    regions = [r.strip() for r in a.regions.split(",") if r.strip()]
    firms_key = os.environ.get("FIRMS_MAP_KEY", "").strip() or None
    code = build(sats, a.domain, regions, a.lookback_hours, a.workers, a.out,
                 a.min_detections, firms_key)
    sys.exit(code)


if __name__ == "__main__":
    main()
