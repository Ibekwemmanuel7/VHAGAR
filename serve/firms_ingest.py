"""NASA FIRMS active-fire ingest: VIIRS (S-NPP / NOAA-20 / NOAA-21) and MODIS.

Adds polar-orbiting (LEO) fire detections to VHAGAR's live feed so the fusion
clusters GEO (GOES) and LEO sensors together, which is the multi-sensor design.
Free: needs a MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/ (instant
registration). Set it as FIRMS_MAP_KEY.

Each FIRMS record is mapped into VHAGAR's detection schema (lon, lat, frp_mw,
temp_k, t, confidence, sensor) and placed on the shared CONUS grid (x/y meters +
tile_id) so the serving clustering treats it exactly like a GOES detection.
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# FIRMS near-real-time source -> VHAGAR sensor label.
SOURCES: dict[str, str] = {
    "VIIRS_SNPP_NRT": "VIIRS-SNPP",
    "VIIRS_NOAA20_NRT": "VIIRS-NOAA20",
    "VIIRS_NOAA21_NRT": "VIIRS-NOAA21",
    "MODIS_NRT": "MODIS",
}


def _log(msg: str) -> None:
    print(f"[vhagar-firms] {msg}", flush=True)


def _acq_dt(date_s: str, time_s: str) -> datetime:
    """FIRMS acq_date (YYYY-MM-DD) + acq_time (HHMM, UTC) -> aware datetime."""
    return datetime.strptime(f"{date_s} {str(time_s).zfill(4)}", "%Y-%m-%d %H%M").replace(tzinfo=UTC)


def _fetch_source(map_key: str, source: str, bbox, day_range: int, timeout: float) -> list[dict]:
    west, south, east, north = bbox
    url = f"{FIRMS_URL}/{map_key}/{source}/{west},{south},{east},{north}/{day_range}"
    req = urllib.request.Request(url, headers={"User-Agent": "vhagar-fire/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        text = r.read().decode("utf-8", "replace")
    first = text.splitlines()[0].lower() if text.strip() else ""
    if not first or "latitude" not in first:   # error string or empty payload
        _log(f"{source}: no data ({text.strip()[:80]!r})")
        return []
    return list(csv.DictReader(io.StringIO(text)))


def fetch_firms(map_key: str, bbox, region: str = "conus", hours: float = 12.0,
                sources=None, timeout: float = 30.0) -> pd.DataFrame:
    """Detections from FIRMS within ``bbox`` over the last ``hours``, placed on
    ``region``'s analysis grid, as a DataFrame in VHAGAR's detection schema.
    Empty DataFrame on any failure."""
    if not map_key:
        return pd.DataFrame()
    from vhagar.grid import AnalysisGrid
    from vhagar.labels.tiles import _transformer
    grid = AnalysisGrid(region)
    tf = _transformer(region)
    # FIRMS day_range already bounds the window; do not add a now-based cutoff
    # (this environment's clock can differ from the live service, which would
    # silently drop every row). Keep the freshest `hours` relative to the data's
    # own newest acquisition instead.
    day_range = max(1, min(10, round(hours / 24) or 1))
    rows: list[dict] = []
    for src in (sources or SOURCES):
        label = SOURCES[src]
        try:
            recs = _fetch_source(map_key, src, bbox, day_range=day_range, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            _log(f"{src} fetch failed ({type(exc).__name__}: {exc})")
            continue
        parsed = []
        for d in recs:
            try:
                lat, lon = float(d["latitude"]), float(d["longitude"])
                t = _acq_dt(d["acq_date"], d["acq_time"])
            except (KeyError, ValueError):
                continue
            frp = d.get("frp")
            bright = d.get("bright_ti4") or d.get("brightness")
            scan, track = d.get("scan"), d.get("track")
            parsed.append({
                "lon": lon, "lat": lat, "t": t,
                "frp_mw": float(frp) if frp not in (None, "", "nan") else np.nan,
                "temp_k": float(bright) if bright not in (None, "", "nan") else np.nan,
                "confidence": d.get("confidence"),
                "view_zenith_deg": np.nan,   # LEO: fusion uses the sensor footprint
                "sensor": label, "granule_key": src,
                "area_m2": (float(scan) * float(track) * 1e6) if scan and track else np.nan,
            })
        # Keep only the freshest `hours` relative to this source's own newest row.
        if parsed:
            newest = max(r["t"] for r in parsed)
            cutoff = newest - timedelta(hours=hours)
            kept = [r for r in parsed if r["t"] >= cutoff]
        else:
            newest, kept = None, []
        rows.extend(kept)
        stamp = newest.strftime("%Y-%m-%d %H:%MZ") if newest else "n/a"
        _log(f"{label}: {len(recs)} rows returned, kept {len(kept)} within {hours:g} h "
             f"(day_range={day_range}, newest={stamp})")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    xs, ys = tf.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"] = np.asarray(xs, dtype=float)
    df["y"] = np.asarray(ys, dtype=float)
    tiles = []
    for x, y in zip(df["x"], df["y"], strict=False):
        try:
            tiles.append(grid.tile_for_point(x, y).tile_id)
        except Exception:  # noqa: BLE001 - outside the CONUS grid
            tiles.append(None)
    df["tile_id"] = tiles
    return df[df["tile_id"].notna()].reset_index(drop=True)
