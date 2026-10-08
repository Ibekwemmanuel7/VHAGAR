"""VHAGAR Southeast historical replay: how big is the nuisance-alert problem?

Downloads the NASA FIRMS archive for Georgia and Florida, clusters detections
into fire events, and reports how events distribute over the year. With an
optional reference file of reported wildfires (for example FPA FOD, or a state
wildfire export), it also reports the share of burn-season events that match no
reported wildfire.

Read the result carefully. An event that matches no reported wildfire is NOT
proven to be a prescribed burn. That group also holds agricultural and debris
burning, unreported wildfires, and wildfires that the reference misses. It is a
measure of the alert volume a permit-unaware system would produce, and the
upper bound of what a permit-aware filter could remove. It is not a label set.

Raw FIRMS downloads are cached under data/southeast/firms/ and reused.

    set FIRMS_MAP_KEY=...            (free: https://firms.modaps.eosdis.nasa.gov/api/map_key/)
    python scripts/southeast_replay.py --years 2019 2020 --source VIIRS_SNPP_SP \
        --wildfire-ref data/southeast/fpa_fod_ga_fl.csv

Reference CSV columns: lon, lat, date (YYYY-MM-DD) and optional acres. FPA FOD
column names (LONGITUDE, LATITUDE, DISCOVERY_DATE, FIRE_SIZE) are accepted.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vhagar.intersect import haversine_m, point_in_ring  # noqa: E402
from vhagar.io.firms import FirmsClient, parse_firms_csv  # noqa: E402
from vhagar.southeast.events import cluster_obs_grid, obs_from_firms  # noqa: E402

#: Approximate state boxes, used only when no boundary file is given. They
#: overlap neighbouring states (parts of AL, SC, TN, NC), so a bbox-only run
#: says so in its output.
STATE_BBOX = {
    "GA": (-85.61, 30.36, -80.84, 35.00),
    "FL": (-87.64, 24.40, -79.97, 31.00),
}
#: FIRMS ``type`` codes that are not vegetation fires (SP products): 1 volcano,
#: 2 other static land source, 3 offshore.
STATIC_TYPES = {"1", "2", "3"}


def union_bbox(states):
    boxes = [STATE_BBOX[s] for s in states]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def load_boundaries(path: Path | None, states):
    """{state: [rings]} from a GeoJSON whose features carry a STUSPS, STATE or
    NAME property. None means bbox mode."""
    if path is None:
        return None
    fc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list] = {}
    for f in fc["features"]:
        p = f.get("properties") or {}
        code = p.get("STUSPS") or p.get("STATE") or p.get("NAME")
        if code not in states:
            continue
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        out.setdefault(code, []).extend(poly[0] for poly in polys)
    missing = set(states) - set(out)
    if missing:
        raise SystemExit(f"boundary file has no feature for {sorted(missing)}")
    return out


def state_of(lon, lat, states, boundaries):
    if boundaries is not None:
        for s in states:
            if any(point_in_ring(lon, lat, ring) for ring in boundaries[s]):
                return s
        return None
    for s in states:
        w, so, e, n = STATE_BBOX[s]
        if w <= lon <= e and so <= lat <= n:
            return s
    return None


def fetch_year(client, source, bbox, year, cache: Path, sleep_s: float):
    """FIRMS area API in 10-day chunks for one calendar year, cached per chunk."""
    cache.mkdir(parents=True, exist_ok=True)
    texts = []
    d = date(year, 1, 1)
    end = date(year, 12, 31)
    while d <= end:
        span = min(10, (end - d).days + 1)
        f = cache / f"{source}_{d.isoformat()}_{span}d.csv"
        if f.exists():
            texts.append(f.read_text(encoding="utf-8"))
        else:
            if client is None:
                raise SystemExit(f"missing cache {f.name} and no FIRMS_MAP_KEY to download it")
            text = client.area_csv(source, bbox, day_range=span, start=d)
            if text.lstrip().lower().startswith(("invalid", "error")):
                raise SystemExit(f"FIRMS refused {d}: {text[:200]}")
            f.write_text(text, encoding="utf-8")
            texts.append(text)
            time.sleep(sleep_s)
        d += timedelta(days=span)
    return texts


def _key(lat, lon, day: str, hhmm: int):
    return round(float(lat), 4), round(float(lon), 4), day, int(hhmm)


def parse_with_type(texts):
    """FirmsRecords plus the set of static-source keys from the ``type`` column."""
    records, static = [], set()
    for text in texts:
        for row in csv.DictReader(io.StringIO(text)):
            if str(row.get("type", "")).strip() in STATIC_TYPES:
                static.add(_key(row["latitude"], row["longitude"], row["acq_date"],
                                row["acq_time"]))
        records.extend(parse_firms_csv(text))
    return records, static


def load_reference(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            lon = r.get("lon") or r.get("LONGITUDE") or r.get("longitude")
            lat = r.get("lat") or r.get("LATITUDE") or r.get("latitude")
            day = r.get("date") or r.get("DISCOVERY_DATE") or r.get("discovery_date")
            if not (lon and lat and day):
                continue
            rows.append({"lon": float(lon), "lat": float(lat),
                         "date": date.fromisoformat(day.strip()[:10]),
                         "acres": float(r.get("acres") or r.get("FIRE_SIZE") or "nan")})
    return rows


def match_reference(events, ref, radius_m, days_before, days_after):
    """event_id -> True when a reported wildfire lies within radius and window."""
    by_day: dict[date, list] = {}
    for r in ref:
        by_day.setdefault(r["date"], []).append(r)
    out = {}
    for e in events:
        lon = sum(o.lon for o in e.detections) / len(e.detections)
        lat = sum(o.lat for o in e.detections) / len(e.detections)
        d0 = e.start.date()
        hit = False
        for k in range(-days_after, days_before + 1):
            for r in by_day.get(d0 + timedelta(days=k), []):
                if haversine_m(lat, lon, r["lat"], r["lon"]) <= radius_m:
                    hit = True
                    break
            if hit:
                break
        out[e.event_id] = hit
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--states", nargs="+", default=["GA", "FL"], choices=sorted(STATE_BBOX))
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--source", default="VIIRS_SNPP_SP",
                    help="FIRMS source id, e.g. VIIRS_SNPP_SP, VIIRS_NOAA20_SP, VIIRS_NOAA20_NRT")
    ap.add_argument("--season-months", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--boundary", type=Path, help="state boundary GeoJSON (else bbox mode)")
    ap.add_argument("--wildfire-ref", type=Path, help="reported wildfires CSV")
    ap.add_argument("--match-radius-m", type=float, default=5_000.0)
    ap.add_argument("--match-days-before", type=int, default=1,
                    help="days the event may start before the reported date")
    ap.add_argument("--match-days-after", type=int, default=3,
                    help="days the event may start after the reported date")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "southeast" / "firms")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "southeast_replay")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between API calls")
    args = ap.parse_args(argv)

    try:
        client = FirmsClient()
    except ValueError:
        client = None
    bbox = union_bbox(args.states)
    boundaries = load_boundaries(args.boundary, args.states)
    ref = load_reference(args.wildfire_ref) if args.wildfire_ref else None
    season = set(args.season_months)

    per_year = {}
    all_rows = []
    for year in args.years:
        texts = fetch_year(client, args.source, bbox, year, args.cache / args.source, args.sleep)
        records, static = parse_with_type(texts)
        obs = obs_from_firms(records)
        kept = []
        for r, o in zip(records, obs, strict=True):
            st = state_of(o.lon, o.lat, args.states, boundaries)
            if st is None:
                continue
            key = _key(r.latitude, r.longitude, r.acq_datetime.date().isoformat(),
                       r.acq_datetime.strftime("%H%M"))
            kept.append((st, replace(o, static_anomaly=key in static)))
        events = cluster_obs_grid([o for _, o in kept], id_prefix=f"se{year}")
        state_by_obs = {id(o): s for s, o in kept}
        matched = match_reference(events, ref, args.match_radius_m, args.match_days_before,
                                  args.match_days_after) if ref else {}

        months = Counter()
        season_events = season_static = season_matched = 0
        sizes = Counter()
        for e in events:
            m = e.start.month
            months[m] += 1
            static_e = e.static_fraction >= 0.5
            n = len(e.detections)
            sizes["1" if n == 1 else "2-5" if n <= 5 else "6-20" if n <= 20 else ">20"] += 1
            if m in season:
                season_events += 1
                season_static += static_e
                season_matched += bool(matched.get(e.event_id))
            lon = sum(o.lon for o in e.detections) / n
            lat = sum(o.lat for o in e.detections) / n
            frps = [o.frp_mw for o in e.detections if o.frp_mw is not None]
            all_rows.append({
                "event_id": e.event_id, "state": state_by_obs[id(e.detections[0])],
                "start_utc": e.start.isoformat(), "end_utc": e.end.isoformat(),
                "month": m, "n_detections": n, "lon": round(lon, 5), "lat": round(lat, 5),
                "peak_frp_mw": round(max(frps), 1) if frps else "",
                "static": int(static_e),
                "matched_reported_wildfire": "" if ref is None else int(bool(matched.get(e.event_id))),
            })
        non_static_season = season_events - season_static
        per_year[year] = {
            "detections": len(kept),
            "events": len(events),
            "events_by_month": {str(k): months[k] for k in sorted(months)},
            "events_by_detection_count": dict(sizes),
            "season_events": season_events,
            "season_share_of_year": round(season_events / len(events), 3) if events else None,
            "season_static_events": season_static,
            "season_events_matched_reported_wildfire": season_matched if ref else None,
            "season_events_unmatched_share": round(
                (non_static_season - season_matched) / non_static_season, 3)
            if ref and non_static_season else None,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
    summary = {
        "generated_utc": stamp,
        "states": args.states,
        "years": args.years,
        "source": args.source,
        "season_months": sorted(season),
        "clip": "boundary" if boundaries else "bbox (overlaps neighbouring states)",
        "wildfire_reference": str(args.wildfire_ref) if ref else None,
        "match_rule": {"radius_m": args.match_radius_m, "days_before": args.match_days_before,
                       "days_after": args.match_days_after} if ref else None,
        "clustering": {"tolerance_m": 1125.0, "max_gap_hours": 24.0},
        "per_year": per_year,
        "caveat": ("Unmatched events are not proven prescribed burns. They include "
                   "agricultural and debris burning and wildfires missing from the reference."),
    }
    (args.out / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / f"events_{stamp}.csv").open("w", newline="", encoding="utf-8") as fh:
        if all_rows:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
    print(json.dumps(summary["per_year"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
