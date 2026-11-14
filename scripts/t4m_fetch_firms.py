"""Fetch a bounded historical FIRMS active-fire export for one T4-M case.

NASA FIRMS requires a MAP_KEY for its area API.  Put that key in an environment
variable (default ``FIRMS_MAP_KEY``); it is never printed or stored in the
output manifest.  FIRMS limits one area query to five days, so this command
partitions longer incident windows and writes one reproducible CSV plus source
and request provenance.

Example (PowerShell)::

    $env:FIRMS_MAP_KEY = "your-key"
    python scripts/t4m_fetch_firms.py --bbox=-121.0,39.6,-120.2,40.2 \
      --start-date 2020-09-07 --end-date 2020-09-17 \
      --source VIIRS_SNPP_SP --out data/raw/firms/creek-2020.csv

For scientific historical replay prefer ``*_SP`` standard-processing products.
``*_NRT`` is appropriate only when recreating the data product actually
available operationally at the time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
_MAX_DAYS_PER_REQUEST = 5


def _bbox(value: str) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = (float(v) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise argparse.ArgumentTypeError("bbox must be a valid west,south,east,north extent")
    return west, south, east, north


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dates must be YYYY-MM-DD") from exc


def _date_batches(start: date, end: date, *, maximum_days: int = _MAX_DAYS_PER_REQUEST):
    """Yield inclusive FIRMS request windows no longer than ``maximum_days``."""
    cursor = start
    while cursor <= end:
        batch_end = min(cursor + timedelta(days=maximum_days - 1), end)
        yield cursor, batch_end
        cursor = batch_end + timedelta(days=1)


def _read_csv(url: str) -> pd.DataFrame:
    try:
        with urllib.request.urlopen(url, timeout=90) as response:  # noqa: S310 - fixed HTTPS endpoint
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"FIRMS returned HTTP {exc.code}; check MAP_KEY, source, date, and bbox") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("could not reach FIRMS") from exc
    if not payload.strip():
        return pd.DataFrame()
    from io import BytesIO

    return pd.read_csv(BytesIO(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", type=_bbox, required=True)
    parser.add_argument("--start-date", type=_parse_date, required=True)
    parser.add_argument("--end-date", type=_parse_date, required=True)
    parser.add_argument("--source", default="VIIRS_SNPP_SP", help="FIRMS source, usually VIIRS_SNPP_SP or VIIRS_NOAA20_SP")
    parser.add_argument("--map-key-env", default="FIRMS_MAP_KEY")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")
    if not re.fullmatch(r"[A-Z0-9_]+", args.source):
        parser.error("--source must be a FIRMS product identifier such as VIIRS_SNPP_SP")
    key = os.environ.get(args.map_key_env)
    if not key:
        parser.error(f"environment variable {args.map_key_env} is not set; request a free FIRMS MAP_KEY")

    bbox_text = ",".join(f"{value:g}" for value in args.bbox)
    frames: list[pd.DataFrame] = []
    requests: list[dict[str, object]] = []
    for window_start, window_end in _date_batches(args.start_date, args.end_date):
        days = (window_end - window_start).days + 1
        # Do not print the URL: it includes a credential.
        url = "/".join((
            _API, urllib.parse.quote(key, safe=""), args.source, bbox_text,
            str(days), window_start.isoformat(),
        ))
        frame = _read_csv(url)
        frames.append(frame)
        requests.append({"start_date": window_start.isoformat(), "end_date": window_end.isoformat(),
                         "day_range": days, "records": int(len(frame))})
        print(f"received {len(frame)} records for {window_start.isoformat()} through {window_end.isoformat()}")

    # FIRMS legitimately returns header-only/empty windows after containment.
    # Exclude them from concatenation while retaining their zero-record request
    # entries in the manifest; this avoids dtype-dependent pandas warnings.
    nonempty_frames = [frame for frame in frames if not frame.empty]
    combined = pd.concat(nonempty_frames, ignore_index=True) if nonempty_frames else pd.DataFrame()
    if not combined.empty:
        dedupe_columns = [column for column in ("latitude", "longitude", "acq_date", "acq_time", "satellite")
                          if column in combined]
        if dedupe_columns:
            combined = combined.drop_duplicates(dedupe_columns, keep="first").reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "provider": "NASA FIRMS area API",
        "api_endpoint": _API,
        "source": args.source,
        "bbox": args.bbox,
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "requests": requests,
        "records_after_deduplication": int(len(combined)),
        "sha256": digest,
        "credential": f"read from environment variable {args.map_key_env}; not persisted",
        "scientific_note": (
            "FIRMS first-detection points are sparse thermal observations, not a complete perimeter. "
            "Use standard-processing products for retrospective science; retain availability evidence "
            "separately for operational replay."
        ),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}: {len(combined)} records; sha256={digest}")
    print(f"wrote {out.with_suffix('.json')}: request and product provenance (no MAP_KEY)")


if __name__ == "__main__":
    main()
