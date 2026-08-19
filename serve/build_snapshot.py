"""Build a near-real-time VHAGAR snapshot for the live console.

Pulls the newest GOES ABI L2 FDC granules from the public NOAA S3 buckets
(anonymous, no credentials), clusters them into fire events with VHAGAR's own
parallax-aware fusion, and packages the result as ``snapshot.tgz``
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


def _ingest(store: Path, sats: list[int], domain: str, region: str,
            lookback_hours: float, workers: int) -> None:
    """Pull the lookback window for each satellite into one rolling store."""
    from serve.ingest import ingest_once

    for sat in sats:
        _log(f"pulling GOES-{sat} {domain} last {lookback_hours:g} h ...")
        try:
            ingest_once(
                out_dir=store, satellite=sat, domain=domain, region=region, bbox=None,
                lookback_min=lookback_hours * 60.0, workers=workers, min_confidence=0.0,
                drop_filtered=False, retention_days=0.0,  # each run is self-contained
            )
        except Exception as exc:  # one unavailable feed must not sink the run
            _log(f"GOES-{sat} pull failed ({type(exc).__name__}: {exc}); continuing")


def build(sats: list[int], domain: str, region: str, lookback_hours: float,
          workers: int, out: Path, min_detections: int) -> int:
    store = Path(tempfile.mkdtemp(prefix="vhagar_nrt_"))
    _ingest(store, sats, domain, region, lookback_hours, workers)

    det_dir = store / "detections"
    parts = list(det_dir.glob("year=*/tile=*/part-*.parquet")) if det_dir.exists() else []
    if not parts:
        _log("no FDC granules in the window (quiet period or fetch failed); not publishing")
        return 3

    # Cluster through the exact serving path so the snapshot matches what the API
    # expects. DET_DIR / NO_CACHE must be set before importing the API module,
    # since it reads them at import time. SNAPSHOT_URL / FROZEN are cleared so the
    # build does not recurse into a fetch or the committed demo.
    os.environ["VHAGAR_DET_DIR"] = str(det_dir)
    os.environ["VHAGAR_NO_CACHE"] = "1"
    os.environ.pop("VHAGAR_SNAPSHOT_URL", None)
    os.environ.pop("VHAGAR_FROZEN", None)
    from serve import vhagar_api as api

    df, events = api._build_state()
    if len(df) < min_detections:
        _log(f"only {len(df)} detections (< {min_detections}); not publishing")
        return 3
    _log(f"clustered {len(df)} detections into {len(events)} events")

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
    ap.add_argument("--region", default="conus")
    ap.add_argument("--lookback-hours", type=float, default=12.0,
                    help="how far back to pull each run (self-contained window)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-detections", type=int, default=1,
                    help="publish only if at least this many detections were found")
    ap.add_argument("--out", type=Path, default=_ROOT / "snapshot.tgz")
    a = ap.parse_args()
    sats = [int(s) for s in a.sats.split(",") if s.strip()]
    code = build(sats, a.domain, a.region, a.lookback_hours, a.workers, a.out,
                 a.min_detections)
    sys.exit(code)


if __name__ == "__main__":
    main()
