"""Wildfire Event Evidence Pack: intersect a customer portfolio of locations against
VHAGAR fire events and emit an affected-location report.

This is the market-entry product: a customer supplies a portfolio (CSV of locations),
and VHAGAR returns, per location, the nearest event, whether it falls inside the
detection footprint or within a screening buffer, the first and last detection times,
the sensors that saw it, a corroboration confidence, and the data age, with an explicit
disclosure that a footprint is a detection hull, not an agency perimeter.

Portfolio CSV: a header row with latitude and longitude columns (any of lat/latitude,
lon/lng/longitude, case-insensitive) and an optional id/name column.

Events come from either a local GeoJSON file (--events) or a live VHAGAR API
(--base-url with --region/--days), so it runs against the deployed console feed.

    python scripts/evidence_pack.py portfolio.csv --events events.geojson \
        --buffer-m 1000 --out pack.json --csv affected.csv
    python scripts/evidence_pack.py portfolio.csv \
        --base-url https://vhagar-console.onrender.com --region california --days 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vhagar.intersect import intersect_portfolio  # noqa: E402

_LAT = {"lat", "latitude", "y"}
_LON = {"lon", "lng", "long", "longitude", "x"}
_ID = {"id", "name", "location_id", "policy_id", "ref"}


def _read_portfolio(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        latc = next((cols[c] for c in cols if c in _LAT), None)
        lonc = next((cols[c] for c in cols if c in _LON), None)
        idc = next((cols[c] for c in cols if c in _ID), None)
        if not latc or not lonc:
            raise SystemExit(f"portfolio needs latitude and longitude columns; saw {reader.fieldnames}")
        for i, r in enumerate(reader):
            try:
                rows.append({"id": (r.get(idc) or f"loc-{i}") if idc else f"loc-{i}",
                             "lat": float(r[latc]), "lon": float(r[lonc])})
            except (TypeError, ValueError):
                continue
    if not rows:
        raise SystemExit("no valid rows parsed from the portfolio CSV")
    return rows


def _load_events(args) -> dict:
    if args.events:
        return json.loads(Path(args.events).read_text(encoding="utf-8"))
    url = f"{args.base_url.rstrip('/')}/api/events?region={args.region}&days={args.days}"
    with urllib.request.urlopen(url, timeout=60) as resp:   # noqa: S310 (user-supplied base URL)
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("portfolio", help="CSV of locations (lat/lon columns, optional id)")
    ap.add_argument("--events", help="local events GeoJSON file")
    ap.add_argument("--base-url", default="https://vhagar-console.onrender.com",
                    help="VHAGAR API base URL (used when --events is not given)")
    ap.add_argument("--region", default="california")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--buffer-m", type=float, default=1000.0,
                    help="flag a location if within this many metres of a footprint")
    ap.add_argument("--out", help="write the full JSON report here")
    ap.add_argument("--csv", help="write the affected-location table here")
    args = ap.parse_args()

    portfolio = _read_portfolio(args.portfolio)
    fc = _load_events(args)
    res = intersect_portfolio(portfolio, fc, buffer_m=args.buffer_m)
    meta = fc.get("metadata", {})
    res["metadata"] = {"region": args.region, "days": args.days,
                       "mode": meta.get("mode"), "event_count": meta.get("event_count"),
                       "window_end_utc": meta.get("window_end_utc")}

    s = res["summary"]
    print(f"portfolio {s['portfolio_size']}  affected {s['affected_count']} "
          f"(inside {s['inside_footprint_count']}, buffer {s['within_buffer_count']})  "
          f"events {res['metadata'].get('event_count')}")
    print(f"{'id':<16}{'status':<18}{'dist_m':>8}  {'event':<10}{'age_h':>6}  confidence")
    for r in res["affected"]:
        print(f"{str(r['id']):<16}{r['status']:<18}{r['distance_m']:>8}  "
              f"{str(r['event_id'] or '-'):<10}{str(r['data_age_hours'] if r['data_age_hours'] is not None else '-'):>6}  {r['confidence']}")
    print("NOTE:", res["disclosure"])

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        print("wrote", args.out)
    if args.csv:
        cols = ["id", "lat", "lon", "status", "distance_m", "event_id", "event_label",
                "first_seen_utc", "last_seen_utc", "data_age_hours", "n_detections",
                "max_frp_mw", "confidence"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in res["affected"]:
                row = dict(r)
                row["sensors"] = ";".join(r.get("sensors", []))
                w.writerow(row)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
