"""Fetch selected NOAA HRRR surface forecasts from one prior model run.

The output manifest records run time, lead, valid time, URL, and a caller-
declared availability time. Availability is intentionally not inferred from an
archive download: validate it against production ingestion logs before scoring
an operational replay.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

HRRR_S3 = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


def _utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include timezone, e.g. 2026-08-03T23:00:00Z")
    return timestamp.tz_convert("UTC")


def _download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "VHAGAR/0.1 weather-ingest"})
    with urlopen(request, timeout=180) as response, destination.open("wb") as output:  # nosec B310
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-utc", type=_utc_timestamp, required=True)
    parser.add_argument("--lead-hours", type=int, nargs="+", required=True)
    parser.add_argument(
        "--available-at-utc",
        type=_utc_timestamp,
        required=True,
        help="declared latest availability for all selected frames; audit before replay scoring",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if any(lead < 0 or lead > 48 for lead in args.lead_hours):
        parser.error("lead hours must be in [0, 48]")
    if args.available_at_utc < args.run_utc:
        parser.error("availability cannot precede model run")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for lead in sorted(set(args.lead_hours)):
        valid = args.run_utc + pd.Timedelta(hours=lead)
        filename = f"hrrr.t{args.run_utc.hour:02d}z.wrfsfcf{lead:02d}.grib2"
        url = f"{HRRR_S3}/hrrr.{args.run_utc:%Y%m%d}/conus/{filename}"
        destination = out_dir / filename
        if destination.exists() and destination.stat().st_size > 1_000_000:
            print(f"reusing {destination}")
        else:
            print(f"downloading {url}")
            _download(url, destination)
        frames.append(
            {
                "run_time_utc": args.run_utc.isoformat(),
                "lead_hours": lead,
                "valid_time_utc": valid.isoformat(),
                "available_time_utc": args.available_at_utc.isoformat(),
                "url": url,
                "path": str(destination),
                "bytes": destination.stat().st_size,
            }
        )
    manifest_path = out_dir / "hrrr_manifest.json"
    existing_frames = []
    if manifest_path.exists():
        existing_frames = json.loads(manifest_path.read_text(encoding="utf-8")).get("frames", [])
    by_path = {frame.get("path"): frame for frame in existing_frames}
    by_path.update({frame["path"]: frame for frame in frames})
    frames = sorted(by_path.values(), key=lambda frame: (frame["run_time_utc"], frame["lead_hours"]))
    manifest_path.write_text(
        json.dumps(
            {
                "source": "NOAA HRRR surface forecast (wrfsfc)",
                "retrieved_utc": datetime.now(UTC).isoformat(),
                "availability_note": "Caller-declared availability; validate against operational logs.",
                "frames": frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}: {len(frames)} HRRR frame(s)")


if __name__ == "__main__":
    main()
