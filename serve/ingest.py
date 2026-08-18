"""VHAGAR near-real-time FDC ingester.

Turns the console from a fixed August window into a live feed. It reuses
VHAGAR's own resumable archive builder (vhagar.archive.backfill) to pull the
newest GOES ABI L2 FDC granules from the public NOAA S3 bucket, decode them,
and append to a rolling detection store that the API serves. Because backfill
skips granules it already has, calling this every few minutes is exactly an
NRT poller; pruning keeps the store bounded so clustering stays fast.

This needs network access to the public noaa-goes18/19 buckets (anonymous, no
credentials) and the decode stack (s3fs, xarray, h5netcdf). Run it on a machine
with internet; it does not run inside the sandbox.

Once:
    python -m serve.ingest --out data/detections_nrt --sat 18 --lookback-min 30

Loop (every 5 minutes, keep 3 days):
    python -m serve.ingest --out data/detections_nrt --sat 18 --interval 300 --retention-days 3

Then point the API at that store and turn on live refresh:
    set VHAGAR_DET_DIR=%CD%\data\detections_nrt\detections   (PowerShell: $env:...)
    set VHAGAR_REFRESH_SEC=300
    uvicorn serve.vhagar_api:app
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def _log(msg: str) -> None:
    print(f"[vhagar-ingest {datetime.now(UTC):%H:%M:%S}Z] {msg}", flush=True)


def prune(out_dir: Path, retention_days: float) -> int:
    """Delete detection partition files older than the retention window. Keeps
    the rolling store bounded so a background re-cluster stays quick. The
    manifest is left as-is (it only feeds coverage accounting, not serving)."""
    if retention_days <= 0:
        return 0
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).date()
    removed = 0
    for part in (out_dir / "detections").glob("year=*/tile=*/part-*.parquet"):
        stamp = part.stem.split("-")[-1]  # part-YYYYMMDD
        try:
            day = datetime.strptime(stamp, "%Y%m%d").date()
        except ValueError:
            continue
        if day < cutoff:
            try:
                part.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def ingest_once(out_dir: Path, satellite: int, domain: str, region: str,
                bbox, lookback_min: float, workers: int, min_confidence: float,
                drop_filtered: bool, retention_days: float) -> dict:
    """Pull + decode + append the last `lookback_min` minutes of FDC, then prune."""
    from vhagar.archive.backfill import BackfillConfig, backfill

    now = datetime.now(UTC)
    start = now - timedelta(minutes=lookback_min)
    cfg = BackfillConfig(
        out_dir=out_dir, start=start, end=now, satellite=satellite, domain=domain,
        region=region, bbox=bbox, workers=workers,
        include_filtered=not drop_filtered, min_confidence=min_confidence,
    )
    result = backfill(cfg)
    removed = prune(out_dir, retention_days)
    summary = {"window": f"{start:%H:%M}-{now:%H:%M}Z", "result": str(result), "pruned": removed}
    _log(f"ingested {summary['window']} -> {summary['result']}; pruned {removed} old part(s)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="VHAGAR near-real-time FDC ingester")
    ap.add_argument("--out", type=Path, default=_ROOT / "data" / "detections_nrt",
                    help="rolling store directory (the API reads <out>/detections)")
    ap.add_argument("--sat", type=int, default=18, choices=[16, 18, 19],
                    help="GOES satellite (18=West, 19=East)")
    ap.add_argument("--domain", default="C", help="ABI domain: C, F, M1, M2")
    ap.add_argument("--region", default="conus")
    ap.add_argument("--bbox", default="", help="west,south,east,north degrees; empty reads all")
    ap.add_argument("--lookback-min", type=float, default=30.0,
                    help="how far back to pull each cycle (GOES cadence is 5 min)")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="seconds between cycles; 0 runs once and exits")
    ap.add_argument("--retention-days", type=float, default=3.0,
                    help="drop partitions older than this so the store stays bounded")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--drop-filtered", action="store_true",
                    help="keep only the 10-15 mask series (early, noisier)")
    a = ap.parse_args()

    box = None
    if a.bbox:
        parts = [float(p) for p in a.bbox.split(",")]
        if len(parts) != 4:
            ap.error("bbox needs four numbers: west,south,east,north")
        box = tuple(parts)

    a.out.mkdir(parents=True, exist_ok=True)
    _log(f"store={a.out}  GOES-{a.sat} {a.domain}  lookback={a.lookback_min:.0f}m  "
         f"retention={a.retention_days:g}d  {'loop @'+str(int(a.interval))+'s' if a.interval else 'once'}")

    def cycle():
        try:
            ingest_once(a.out, a.sat, a.domain, a.region, box, a.lookback_min,
                        a.workers, a.min_confidence, a.drop_filtered, a.retention_days)
        except Exception as exc:
            _log(f"cycle error: {type(exc).__name__}: {exc}")

    cycle()
    if a.interval > 0:
        _log("looping; Ctrl+C to stop")
        try:
            while True:
                time.sleep(a.interval)
                cycle()
        except KeyboardInterrupt:
            _log("stopped")


if __name__ == "__main__":
    main()
