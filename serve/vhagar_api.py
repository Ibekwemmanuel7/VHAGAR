"""VHAGAR fire API: serves the real GOES FDC detections and clustered events.

Self-hosted, no external services. It reads the cached FDC detection parquet
(data/detections), clusters detections into fire events with VHAGAR's own
parallax-aware fusion (harmonize.fusion.cluster_detections), and serves the
result as GeoJSON in the shape vhagar_console.html expects:

    GET /api/detections?region=&days=   point features (FDC pixels)
    GET /api/events?region=&days=       polygon features (clustered events)
    GET /api/export/geojson?region=&days=   events as a download
    GET /console                        the operations console
    GET /api/health

Honesty note. GOES FDC gives us position, FRP, brightness temperature,
confidence, view zenith and time. It does NOT give spread risk, fire weather,
or a validated burned area, so those fields are absent here rather than
invented. The event polygon is the convex hull of a cluster's detection
pixels: a DETECTION FOOTPRINT, not a burned-area measurement. The console is
told schema="fdc" so it labels every number for what it is.

Run:
    pip install fastapi uvicorn pandas pyarrow numpy
    uvicorn serve.vhagar_api:app --reload
    open http://127.0.0.1:8000/console
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# We reuse VHAGAR's own parallax-aware GEO/LEO tolerance (the measured
# footprint-quantisation + terrain-parallax radius) and the same single-link
# rule as harmonize.fusion.cluster_detections; only the neighbour search is
# swapped for a KD-tree so it scales to the full CONUS week.
from vhagar.harmonize.fusion import SENSOR_TOLERANCE_M, geo_leo_tolerance_m  # noqa: E402

MAX_GAP_S = 12 * 3600  # same 12 h temporal link as cluster_detections

DET_DIR = Path(os.environ.get("VHAGAR_DET_DIR", _ROOT / "data" / "detections" / "detections"))
CONSOLE = _ROOT / "vhagar_console.html"
CACHE_DIR = _ROOT / "serve" / ".cache"
# A prebuilt, self-contained snapshot committed to the repo so a hosted deploy
# (Render, a container) starts instantly with no raw parquet and no clustering.
# Set VHAGAR_FROZEN=1 to force it; it is also used automatically when the raw
# detection dataset is absent (the usual case on a fresh clone / a slim image).
FROZEN_DIR = Path(os.environ.get("VHAGAR_FROZEN_DIR", _ROOT / "serve" / "demo"))
# Near-real-time deploy: a scheduled job publishes a fresh snapshot tarball
# (detections.parquet + events.pkl) at this URL, e.g. a rolling GitHub Release
# asset. When set, the API pulls it on load and on every background refresh, so
# a free (sleeping) host still serves the latest feed on each wake without any
# always-on ingester. Falls back to the committed demo if the fetch fails.
SNAPSHOT_URL = os.environ.get("VHAGAR_SNAPSHOT_URL", "").strip()

# Region windows in lon/lat, matching the console's region picker.
REGIONS: dict[str, dict] = {
    "california": {"label": "California / US West", "bbox": (-124.6, 32.0, -114.0, 42.3)},
    "us_west":    {"label": "US West",             "bbox": (-125.0, 31.0, -102.0, 49.2)},
    "conus":      {"label": "Continental US",      "bbox": (-125.0, 24.0, -66.5, 50.0)},
}

# Minimum detections before a cluster is drawn as an event polygon; smaller
# clusters stay as individual detection points only.
MIN_EVENT_DETECTIONS = 4
# Cap on points returned per request so the browser stays responsive; the
# strongest FRP pixels are kept.
MAX_POINTS = 5000
# Bound the O(n^2) clustering per tile.
MAX_DET_PER_TILE = 3000


def _sensor_from_granule(key: str) -> str:
    if not isinstance(key, str):
        return "GOES"
    if "_G19_" in key:
        return "GOES-19"
    if "_G18_" in key:
        return "GOES-18"
    if "_G16_" in key:
        return "GOES-16"
    return "GOES"


def _dataset_sig() -> list:
    """Cheap fingerprint of the detection dataset: mtime + size of its manifest
    and config, not a walk of all 1600+ partition files."""
    marks = []
    for name in ("_manifest.jsonl", "_config.json"):
        p = DET_DIR.parent / name
        if p.exists():
            st = p.stat()
            marks.append([name, round(st.st_mtime, 3), st.st_size])
    if not marks:
        st = DET_DIR.stat()
        marks.append(["dir", round(st.st_mtime, 3)])
    return marks


def _load_snapshot_url(url: str) -> tuple[pd.DataFrame, list[dict]]:
    """Download a snapshot tarball and load it. The tarball holds a
    ``detections.parquet`` and an ``events.pkl`` at any depth; we extract by
    basename only (no path traversal) into a temp dir. The source is our own
    published asset, but the extraction is sanitised regardless."""
    import io
    import tarfile
    import tempfile
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "vhagar-api"})
    with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310 (our asset)
        raw = resp.read()
    dest = Path(tempfile.mkdtemp(prefix="vhagar_snap_"))
    wanted = {"detections.parquet", "events.pkl"}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for member in tf.getmembers():
            base = os.path.basename(member.name)
            if member.isfile() and base in wanted:
                member.name = base  # flatten, drop any directory component
                tf.extract(member, dest)
    df_p, ev_p = dest / "detections.parquet", dest / "events.pkl"
    if not (df_p.exists() and ev_p.exists()):
        raise FileNotFoundError("snapshot tarball missing detections.parquet or events.pkl")
    return pd.read_parquet(df_p), pickle.loads(ev_p.read_bytes())


def _build_state() -> tuple[pd.DataFrame, list[dict]]:
    """Load all FDC detections once and precompute clustered events per tile.

    Returns (detections dataframe, event records). Cached for process lifetime.
    """
    # Near-real-time deploy: pull the freshest published snapshot. Tried first so
    # a background refresh keeps the feed current; any failure falls through to
    # the committed demo (below) rather than taking the service down.
    if SNAPSHOT_URL:
        try:
            return _load_snapshot_url(SNAPSHOT_URL)
        except Exception as exc:  # network / format problem: keep serving
            print(f"[vhagar-api] snapshot fetch failed ({exc}); using fallback",
                  file=sys.stderr)

    # Frozen deploy: load the committed snapshot directly and skip the raw read
    # + clustering entirely. Explicit via VHAGAR_FROZEN, or automatic when the
    # raw dataset is missing but a bundled snapshot is present.
    fz_df, fz_ev = FROZEN_DIR / "detections.parquet", FROZEN_DIR / "events.pkl"
    if (os.environ.get("VHAGAR_FROZEN") or not DET_DIR.exists()) and fz_df.exists() and fz_ev.exists():
        return pd.read_parquet(fz_df), pickle.loads(fz_ev.read_bytes())

    if not DET_DIR.exists():
        raise FileNotFoundError(f"no FDC parquet under {DET_DIR}")

    # Disk cache: skip the read + cluster (about 45 s) when the dataset is
    # unchanged. Fingerprint from the dataset manifest (cheap) rather than
    # walking every partition file (~7 s over a network mount). Set
    # VHAGAR_NO_CACHE=1 to force a rebuild; any cache problem falls through to a
    # clean recompute.
    sig = _dataset_sig()
    df_cache, ev_cache, sig_cache = (CACHE_DIR / "detections.parquet",
                                     CACHE_DIR / "events.pkl", CACHE_DIR / "sig.json")
    if not os.environ.get("VHAGAR_NO_CACHE") and df_cache.exists() and ev_cache.exists() and sig_cache.exists():
        try:
            if json.loads(sig_cache.read_text()) == sig:
                return pd.read_parquet(df_cache), pickle.loads(ev_cache.read_bytes())
        except (OSError, ValueError, pickle.UnpicklingError):
            pass

    df = pd.read_parquet(DET_DIR)
    df["t"] = pd.to_datetime(df["t"], utc=False)
    df["sensor"] = df["granule_key"].map(_sensor_from_granule)
    events = _cluster_all(df)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(df_cache)
        ev_cache.write_bytes(pickle.dumps(events))
        sig_cache.write_text(json.dumps(sig))
    except OSError:
        pass
    return df, events


def _tol_vector(vza: np.ndarray) -> np.ndarray:
    """Per-detection matching radius (m): VHAGAR's parallax-aware GEO tolerance
    where a view zenith is known, else the flat 3x2 km GOES buffer."""
    out = np.where(np.isnan(vza), SENSOR_TOLERANCE_M["goes"],
                   geo_leo_tolerance_m(np.nan_to_num(vza, nan=45.0), 1000.0))
    return np.maximum(out.astype(float), 1000.0)


def _fast_groups(x, y, tsec, tol):
    """Single-link clusters (same rule as cluster_detections: join when planar
    separation <= max of the two tolerances AND time gap <= 12 h), via a
    KD-tree so it is O(n log n) not O(n^2)."""
    n = len(x)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if n > 1:
        pairs = cKDTree(np.column_stack([x, y])).query_pairs(float(tol.max()), output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            tmax = np.maximum(tol[i], tol[j])
            ok = (d2 <= tmax * tmax) & (np.abs(tsec[i] - tsec[j]) <= MAX_GAP_S)
            for a, b in pairs[ok]:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def _cluster_all(df: pd.DataFrame) -> list[dict]:
    """Cluster per tile across the whole window using VHAGAR fusion, then
    reduce each cluster to a compact, honest event record."""
    records: list[dict] = []
    for _tile, g in df.groupby("tile_id", sort=False):
        if len(g) > MAX_DET_PER_TILE:
            g = g.sort_values("frp_mw", ascending=False).head(MAX_DET_PER_TILE)
        x = g["x"].to_numpy(float)
        y = g["y"].to_numpy(float)
        tsec = (g["t"].astype("int64").to_numpy()) / 1e9
        tol = _tol_vector(g["view_zenith_deg"].to_numpy(float))
        for members in _fast_groups(x, y, tsec, tol):
            if len(members) < MIN_EVENT_DETECTIONS:
                continue
            rec = _event_record(g.iloc[members])
            if rec is not None:
                records.append(rec)
    # stable, strongest first
    records.sort(key=lambda r: (r["_frp_sort"]), reverse=True)
    for i, r in enumerate(records, 1):
        r["event_id"] = i
        r["label"] = f"Cluster {i}"
    return records


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Monotonic-chain convex hull of an (n,2) lon/lat array. Returns closed ring."""
    p = np.unique(pts, axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper = []
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    ring = np.array(lower[:-1] + upper[:-1])
    return np.vstack([ring, ring[0]])


def _hull_area_perimeter(ring: np.ndarray, lat0: float) -> tuple[float, float]:
    """Footprint area (ha) and perimeter (km) of a lon/lat ring via a local
    equirectangular projection about lat0."""
    mx = 111_320.0 * math.cos(math.radians(lat0))
    my = 110_540.0
    xs = (ring[:, 0] - ring[0, 0]) * mx
    ys = (ring[:, 1] - ring[0, 1]) * my
    area = 0.5 * abs(np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1]))
    seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    return area / 10_000.0, float(seg.sum()) / 1000.0


def _event_record(rows: pd.DataFrame) -> dict | None:
    lon = rows["lon"].to_numpy(dtype=float)
    lat = rows["lat"].to_numpy(dtype=float)
    ring = _convex_hull(np.column_stack([lon, lat]))
    if len(ring) < 4:
        return None
    clat, clon = float(lat.mean()), float(lon.mean())
    area_ha, perim_km = _hull_area_perimeter(ring, clat)
    frp = rows["frp_mw"].to_numpy(dtype=float)
    frp_valid = frp[~np.isnan(frp)]
    total = float(np.nansum(frp)) if frp_valid.size else None
    mx = float(np.nanmax(frp)) if frp_valid.size else None
    mean = float(np.nanmean(frp)) if frp_valid.size else None
    t0, t1 = rows["t"].min(), rows["t"].max()
    return {
        "geometry": [[[round(float(x), 4), round(float(y), 4)] for x, y in ring]],
        "centroid_lat": round(clat, 4), "centroid_lon": round(clon, 4),
        "n_detections": int(len(rows)),
        "total_frp_mw": None if total is None else round(total, 1),
        "max_frp_mw": None if mx is None else round(mx, 1),
        "mean_frp_mw": None if mean is None else round(mean, 1),
        "footprint_ha": round(area_ha, 1), "perimeter_km": round(perim_km, 1),
        "first_seen": t0.isoformat(), "last_seen": t1.isoformat(),
        "sensors": ", ".join(sorted(rows["sensor"].unique())),
        "bbox": (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())),
        "_t0": t0, "_t1": t1,
        "_frp_sort": (mx if mx is not None else 0.0),
    }


def _window(df: pd.DataFrame, days: int) -> pd.Timestamp:
    return df["t"].max() - timedelta(days=days)


def _in_bbox(lon, lat, bbox) -> bool:
    w, s, e, n = bbox
    return (w <= lon <= e) and (s <= lat <= n)


# ------------------------------------------------------------------ live state
# One in-memory (df, events) snapshot, swapped atomically. A background thread
# rebuilds it on a schedule so a running server picks up newly ingested granules
# without a restart, and requests never block on the ~30 s clustering: they read
# the last-good snapshot until the new one is ready.
_STATE: tuple[pd.DataFrame, list[dict]] | None = None
_BUILD_LOCK = threading.Lock()
_REFRESH_COUNT = 0
_LAST_REFRESH: float | None = None


def get_state() -> tuple[pd.DataFrame, list[dict]]:
    global _STATE
    if _STATE is None:
        with _BUILD_LOCK:
            if _STATE is None:
                _STATE = _build_state()
    return _STATE


def refresh_state() -> None:
    """Rebuild the snapshot (cheap when the dataset fingerprint is unchanged)
    and swap it in. Built outside the lock so reads are never blocked."""
    new = _build_state()
    global _STATE, _REFRESH_COUNT, _LAST_REFRESH
    _STATE = new
    _REFRESH_COUNT += 1
    _LAST_REFRESH = time.time()


def _refresh_loop(interval_s: float) -> None:
    while True:
        time.sleep(interval_s)
        try:
            refresh_state()
        except Exception as exc:  # keep the server up even if a refresh fails
            print(f"[vhagar-api] refresh failed: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------- API
from fastapi import FastAPI, Query, Response  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

app = FastAPI(title="VHAGAR fire API", version="0.1",
              description="Real GOES FDC detections and clustered fire events.")


@app.on_event("startup")
def _start_refresher() -> None:
    """When VHAGAR_REFRESH_SEC>0 (live mode), rebuild the snapshot on that cadence
    so newly ingested granules appear without a restart. Off by default so the
    static demo does not re-cluster pointlessly."""
    sec = float(os.environ.get("VHAGAR_REFRESH_SEC", "0") or 0)
    if sec > 0:
        threading.Thread(target=_refresh_loop, args=(sec,), daemon=True).start()
        print(f"[vhagar-api] live: background refresh every {sec:.0f}s", file=sys.stderr)


@app.get("/api/health")
def health():
    try:
        df, ev = get_state()
        return {"status": "ok", "detections": int(len(df)), "events": len(ev),
                "window": [str(df["t"].min()), str(df["t"].max())],
                "regions": list(REGIONS),
                "refreshes": _REFRESH_COUNT,
                "last_refresh": _LAST_REFRESH}
    except FileNotFoundError as e:
        return JSONResponse({"status": "no_data", "detail": str(e)}, status_code=503)


@app.get("/api/detections")
def detections(region: str = Query("california"), days: int = Query(3, ge=1, le=14),
               filter_fa: bool = Query(False)):
    df, _ = get_state()
    bbox = REGIONS.get(region, REGIONS["california"])["bbox"]
    w, s, e, n = bbox
    cut = _window(df, days)
    m = ((df["t"] >= cut) & (df["lon"] >= w) & (df["lon"] <= e)
         & (df["lat"] >= s) & (df["lat"] <= n))
    sub = df.loc[m]
    if len(sub) > MAX_POINTS:
        sub = sub.assign(_f=sub["frp_mw"].fillna(0.0)).sort_values("_f", ascending=False).head(MAX_POINTS)
    feats = []
    for r in sub.itertuples():
        feats.append({"type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(float(r.lon), 4), round(float(r.lat), 4)]},
            "properties": {
                "frp_mw": None if pd.isna(r.frp_mw) else round(float(r.frp_mw), 1),
                "brightness_k": None if pd.isna(r.temp_k) else round(float(r.temp_k), 1),
                "confidence": None if pd.isna(r.confidence) else round(float(r.confidence) * 100),
                "sensor": r.sensor, "acq_datetime": r.t.isoformat(),
                "view_zenith_deg": None if pd.isna(r.view_zenith_deg) else round(float(r.view_zenith_deg), 1),
                "is_false_alarm": False}})
    return JSONResponse({"type": "FeatureCollection", "features": feats,
        "metadata": {"mode": "live", "schema": "fdc", "region": region,
                     "source": "GOES-18/19 ABI FDC (VHAGAR)", "count": len(feats)}})


def _enrich_weather(feats):
    """Attach current wind/RH/temp + an operational spread-risk score/class to
    each event, when VHAGAR_WEATHER is set. Weather is CURRENT conditions at the
    location (coincident with a live NRT feed; present-day for archived data).
    Degrades gracefully and returns a status string for the response metadata so
    a failure is visible rather than silently blank."""
    if not feats:
        return "no-events"
    if not os.environ.get("VHAGAR_WEATHER"):
        return "off"
    try:
        from vhagar.features.spread_risk import risk_class, spread_risk_score
        from vhagar.weather import fetch_weather
        pts = [(f["properties"]["centroid_lat"], f["properties"]["centroid_lon"]) for f in feats]
        wx = fetch_weather(pts)
        got = 0
        for f, w in zip(feats, wx, strict=False):
            if not w:
                continue
            got += 1
            p = f["properties"]
            p.update({k: w.get(k) for k in
                      ("temp_c", "rh_pct", "wind_speed_ms", "wind_dir_deg", "wind_gust_ms")})
            score = spread_risk_score(w.get("temp_c"), w.get("rh_pct"), w.get("wind_speed_ms"))
            p["risk_score"] = score
            p["risk_class"] = risk_class(score)
        return f"current:{got}/{len(feats)}" if got else "unavailable"
    except Exception as exc:            # noqa: BLE001
        return f"error:{type(exc).__name__}"


def _events_fc(region: str, days: int) -> dict:
    df, evs = get_state()
    bbox = REGIONS.get(region, REGIONS["california"])["bbox"]
    cut = _window(df, days)
    feats = []
    for r in evs:
        if not _in_bbox(r["centroid_lon"], r["centroid_lat"], bbox):
            continue
        if r["_t1"] < cut:
            continue
        p = {k: r[k] for k in ("event_id", "label", "centroid_lat", "centroid_lon",
             "n_detections", "total_frp_mw", "max_frp_mw", "mean_frp_mw",
             "perimeter_km", "footprint_ha", "first_seen", "last_seen", "sensors")}
        p["area_ha"] = r["footprint_ha"]          # console reads area_ha; labelled "footprint"
        p["risk_class"] = "Unknown"               # FDC carries no spread risk (unless enriched)
        p["perimeter_method"] = "detection convex hull"
        feats.append({"type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": r["geometry"]},
            "properties": p})
    feats.sort(key=lambda f: (f["properties"]["max_frp_mw"] or 0), reverse=True)
    wx_status = _enrich_weather(feats)
    return {"type": "FeatureCollection", "features": feats,
            "metadata": {"mode": "live", "schema": "fdc", "region": region,
                         "source": "GOES-18/19 ABI FDC (VHAGAR)", "event_count": len(feats),
                         "detection_count": int((df["t"] >= cut).sum()),
                         "weather": wx_status}}


@app.get("/api/events")
def events(region: str = Query("california"), days: int = Query(3, ge=1, le=14),
           filter_fa: bool = Query(False)):
    return JSONResponse(_events_fc(region, days))


@app.get("/api/export/geojson")
def export_geojson(region: str = Query("california"), days: int = Query(3, ge=1, le=14)):
    return Response(content=json.dumps(_events_fc(region, days)),
        media_type="application/geo+json",
        headers={"Content-Disposition": f"attachment; filename=vhagar_events_{region}.geojson"})


@app.get("/api/export/kmz")
def export_kmz(region: str = Query("california"), days: int = Query(3, ge=1, le=14)):
    """Agency-ready KMZ (Google Earth / GIS): risk-styled event perimeters."""
    from vhagar.export import events_to_kmz
    kmz = events_to_kmz(_events_fc(region, days))
    return Response(content=kmz, media_type="application/vnd.google-earth.kmz",
        headers={"Content-Disposition": f"attachment; filename=vhagar_events_{region}.kmz"})


@app.get("/console")
def console():
    """Serve the console, injecting the Mapbox token from VHAGAR_MAPBOX_TOKEN."""
    from fastapi.responses import HTMLResponse
    html = CONSOLE.read_text(encoding="utf-8")
    html = html.replace("__MAPBOX_TOKEN__", os.environ.get("VHAGAR_MAPBOX_TOKEN", ""))
    return HTMLResponse(html)


@app.get("/favicon.ico")
def favicon():
    ico = _ROOT / "brand" / "favicon.png"
    if ico.exists():
        return FileResponse(ico, media_type="image/png")
    return Response(status_code=204)


# ---------------------------------------------------------- T3 danger (/v1)
# The three fire-danger quantities the architecture refuses to collapse into one
# number: FWI (conditional danger), P(ignition), and E[BA]. Models are trained
# lazily on VHAGAR's synthetic danger scenarios and cached; wire real fuels /
# weather / occurrence for operational values. This is a demo of the contract,
# labelled as such, never a claim of calibrated real danger.
_DANGER = None
_DANGER_LOCK = threading.Lock()


def _danger_state():
    global _DANGER
    if _DANGER is None:
        with _DANGER_LOCK:
            if _DANGER is None:
                import numpy as _np
                from sklearn.ensemble import HistGradientBoostingClassifier

                from vhagar.datasets.danger import (
                    assemble_ignition_samples,
                    rare_event_correction,
                    synthetic_reporting_scenario,
                )
                from vhagar.eval.burned_area import (
                    BurnedAreaModel,
                    synthetic_burned_area_scenario,
                )
                rng = _np.random.default_rng(0)
                pres, cand, ign_fn, tau = synthetic_reporting_scenario(rng, n_cells=2500)
                s = assemble_ignition_samples(pres, cand, ign_fn, rng, tau=tau,
                                              use_target_group=True, stratify=False)
                ign = HistGradientBoostingClassifier(max_depth=4, max_iter=200,
                                                     random_state=0).fit(s.X, s.y)
                Xb, area, *_ = synthetic_burned_area_scenario(_np.random.default_rng(1), n=3000)
                eba = BurnedAreaModel(seed=0).fit(Xb, area)
                _DANGER = {"ign": ign, "ign_fn": ign_fn, "tau": s.tau, "ybar": s.ybar,
                           "eba": eba, "rec": rare_event_correction}
    return _DANGER


def _fwi_point(temp, rh, wind, rain, month):
    from vhagar.features import fwi as F
    ff = float(F.ffmc(temp, rh, wind, rain, 85.0))
    dm = float(F.dmc(temp, rh, rain, 6.0, month))
    dc = float(F.dc(temp, rain, 15.0, month))
    return float(F.fwi(F.isi(ff, wind), F.bui(dm, dc)))


def _fwi_class(v):
    for thr, name in ((38, "Extreme"), (21, "Very high"), (12, "High"), (5, "Moderate")):
        if v >= thr:
            return name
    return "Low"


@app.get("/v1/danger")
def danger(dryness: float = Query(0.6, ge=0, le=1), fuel: float = Query(0.6, ge=0, le=1),
           wind: float = Query(0.5, ge=0, le=1), slope: float = Query(0.3, ge=0, le=1),
           temp: float = Query(28.0), rh: float = Query(25.0), rainfall: float = Query(0.0),
           month: int = Query(8, ge=1, le=12)):
    """The three T3 danger quantities for one cell-day, kept separate."""
    import numpy as _np
    try:
        st = _danger_state()
    except ImportError:
        return JSONResponse({"error": "danger models need scikit-learn"}, status_code=503)
    # ignition features: [dryness, fuel, wind, people, roads]; people/roads neutral
    xig = _np.array([[dryness, fuel, wind, 0.3, 0.3]])
    p_raw = float(st["ign"].predict_proba(xig)[:, 1][0])
    p_ig = float(st["rec"](p_raw, st["tau"], st["ybar"]))
    # E[BA | ignition] from the quantile model, then E[BA]
    q = st["eba"].predict_quantiles(_np.array([[dryness, fuel, wind, slope]]))
    eba_cond = float(_np.mean(q))
    fwi_v = _fwi_point(temp, rh, wind * 40.0, rainfall, month)
    return JSONResponse({
        "schema": "t3-danger-demo",
        "fire_danger": {"fwi": round(fwi_v, 1), "class": _fwi_class(fwi_v)},
        "ignition_probability": round(p_ig, 5),
        "expected_burned_area_ha": round(p_ig * eba_cond, 2),
        "e_ba_given_ignition_ha": round(eba_cond, 1),
        "note": ("Three separate quantities (never one 'risk' number). Models trained on VHAGAR's "
                 "synthetic danger scenarios; wire real fuels / weather / occurrence for operational "
                 "values. FWI is the Canadian Fire Weather Index from the supplied weather."),
    })


@app.get("/")
def root():
    return {"service": "VHAGAR fire API", "console": "/console",
            "health": "/api/health", "danger": "/v1/danger", "docs": "/docs"}
