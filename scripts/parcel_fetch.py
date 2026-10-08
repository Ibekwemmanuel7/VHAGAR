"""Download the public source data for the real-data parcel suitability study.

Standard library only, so it runs in any Python 3.11+ on Windows, macOS or Linux with
no installs. Everything lands under ``data/parcel/raw/`` (git-ignored). Every file is
recorded in ``data/parcel/raw/manifest.json`` with its source URL, byte size, SHA-256,
fetch time, vintage, and licence, so the study is reproducible and auditable.

Usage (from the repository root)::

    python scripts/parcel_fetch.py stage1     # labels, parcels, infrastructure, rasters
    python scripts/parcel_fetch.py stage2     # floodplain + elevation for sampled areas
    python scripts/parcel_fetch.py status     # what is downloaded so far

Stage 2 reads ``data/parcel/work/fetch_plan.json``, written by the processing step
(``scripts/parcel_build.py``) after the parcel sample is drawn, so floodplain polygons
and elevation tiles are fetched only where the study needs them.

The script is resumable: finished files are skipped (size and checksum recorded), partial
downloads are written to ``*.part`` and renamed only on success. A full log is written to
``data/parcel/raw/fetch_log.txt``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "parcel" / "raw"
WORK = ROOT / "data" / "parcel" / "work"
MANIFEST = RAW / "manifest.json"
LOG = RAW / "fetch_log.txt"
UA = "VHAGAR-parcel-study/1.0 (research; contact via github.com/Ibekwemmanuel7/VHAGAR)"
# Some hosts (TxGIO's CloudFront/S3 front door) reject non-browser user agents with 403.
# For those, retry with ordinary browser headers and the site's own referer.
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"),
    "Accept": "*/*",
}
REFERERS = {"data.geographic.texas.gov": "https://data.geographic.texas.gov/",
            "api.tnris.org": "https://data.geographic.texas.gov/"}


def _headers(url: str, browser: bool) -> dict[str, str]:
    host = urllib.parse.urlparse(url).netloc
    if not browser and host not in REFERERS:
        return {"User-Agent": UA}
    h = dict(BROWSER_HEADERS)
    if host in REFERERS:
        h["Referer"] = REFERERS[host]
    return h

_lock = threading.Lock()

# ---------------------------------------------------------------------------- sources

TXGIO_COLLECTION = "1d33dafb-6e9c-40ac-887d-f4577909ed9b"  # StratMap Land Parcels, 2026-09-25
TXGIO_RESOURCES = ("https://api.tnris.org/api/v1/resources/?collection_id="
                   f"{TXGIO_COLLECTION}&limit=1000")

STAGE1 = [
    # (key, url or list of candidate urls, relative path, vintage, licence, source name)
    ("uspvdb_geojson", "https://energy.usgs.gov/uspvdb/assets/data/uspvdbGeoJSON.zip",
     "uspvdb/uspvdbGeoJSON.zip", "v4.0 (2026-04-14)", "public domain (USGS)",
     "USGS/LBNL U.S. Large-Scale Solar Photovoltaic Database"),
    ("uspvdb_csv", "https://energy.usgs.gov/uspvdb/assets/data/uspvdbCSV.zip",
     "uspvdb/uspvdbCSV.zip", "v4.0 (2026-04-14)", "public domain (USGS)",
     "USGS/LBNL U.S. Large-Scale Solar Photovoltaic Database"),
    ("uspvdb_meta", "https://energy.usgs.gov/uspvdb/assets/data/uspvdb_v4_0_20260414.xml",
     "uspvdb/uspvdb_v4_0_20260414.xml", "v4.0 (2026-04-14)", "public domain (USGS)",
     "USPVDB FGDC metadata"),
    ("hifld_transmission",
     "https://data.source.coop/seerai/hifld/transmission-lines/transmission-lines/"
     "transmission-lines.parquet/part-00000-tid-2544231510869868098-1f58cb0d-7488-4777-"
     "823f-5046fbe7c5a9-163-1-c000.zstd.parquet",
     "hifld/transmission_lines.parquet", "HIFLD snapshot archived 2025-11 (portal closed 2025-08-25)",
     "CC BY 4.0 (archive by SeerAI on Source Cooperative)", "HIFLD Transmission Lines"),
    ("tiger_roads_2016",
     "https://www2.census.gov/geo/tiger/TIGER2016/PRISECROADS/tl_2016_48_prisecroads.zip",
     "tiger/tl_2016_48_prisecroads.zip", "2016", "public domain (US Census Bureau)",
     "TIGER/Line primary and secondary roads, Texas"),
    ("padus_tx",
     "https://www.sciencebase.gov/catalog/file/get/62226321d34ee0c6b38b6be3"
     "?name=PADUS3_0_State_TX_SHP.zip",
     "padus/PADUS3_0_State_TX_SHP.zip", "PAD-US 3.0 (2022, ver. 2.0 March 2023)",
     "public domain (USGS GAP)", "Protected Areas Database of the United States, Texas"),
    ("whp_2018", "https://www.fs.usda.gov/rds/archive/products/RDS-2015-0047-2/RDS-2015-0047-2.zip",
     "whp/RDS-2015-0047-2.zip", "2018", "public domain (USDA Forest Service RDS)",
     "Wildfire Hazard Potential for the conterminous US (270 m)"),
    ("nlcd_2016",
     ["https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
      "Annual_NLCD_LndCov_2016_CU_C1V1.zip",
      "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
      "Annual_NLCD_LndCov_2016_CU_C1V0.zip"],
     "nlcd/Annual_NLCD_LndCov_2016_CU.zip", "2016 (Annual NLCD Collection 1)",
     "public domain (USGS/MRLC)", "Annual NLCD land cover, CONUS, 30 m"),
]

POWER_URL = ("https://power.larc.nasa.gov/api/temporal/climatology/point?"
             "parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&format=JSON")
TX_BBOX = (-106.75, 25.75, -93.25, 36.75)  # lon_min, lat_min, lon_max, lat_max

NFHL_QUERY = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/{layer}/query"
TNM_DEM = ("https://tnmaccess.nationalmap.gov/api/v1/products?datasets="
           "National%20Elevation%20Dataset%20(NED)%201%20arc-second"
           "&bbox={bbox}&max=500&outputFormat=JSON")


# ---------------------------------------------------------------------------- utilities


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with _lock:
        print(line, flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"created": _now(), "files": {}}


def save_manifest(man: dict) -> None:
    with _lock:
        tmp = MANIFEST.with_suffix(".tmp")
        tmp.write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(MANIFEST)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, *, timeout: float = 120, tries: int = 5) -> bytes:
    """GET a URL into memory with retries (for small API responses)."""
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=_headers(url, attempt > 1))
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"GET failed after {tries} tries: {url} ({last})")


def download(key: str, urls, rel: str, man: dict, *, meta: dict | None = None,
             expect_bytes: int | None = None) -> bool:
    """Stream a file to RAW/rel with retries and resume-by-skip. Returns True on success."""
    dest = RAW / rel
    rec = man["files"].get(rel)
    if dest.exists() and rec and rec.get("bytes") == dest.stat().st_size:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for url in [urls] if isinstance(urls, str) else list(urls):
        browser = False
        for attempt in range(1, 6):
            try:
                req = urllib.request.Request(url, headers=_headers(url, browser))
                with urllib.request.urlopen(req, timeout=300) as r, part.open("wb") as fh:
                    declared = r.headers.get("Content-Length")
                    shutil.copyfileobj(r, fh, length=1 << 20)
                size = part.stat().st_size
                # A complete transfer is judged by the server's Content-Length. Catalog
                # sizes (expect_bytes) can be stale, so a mismatch there is only recorded.
                if declared is not None and size != int(declared):
                    raise OSError(f"truncated: {size} of {declared} bytes")
                catalog_note = None
                if expect_bytes and size != expect_bytes:
                    catalog_note = f"catalog lists {expect_bytes} bytes; server sent {size}"
                    log(f"NOTE {rel}: {catalog_note}")
                part.replace(dest)
                entry = {"key": key, "url": url, "bytes": size, "sha256": _sha256(dest),
                         "fetched_utc": _now(), **(meta or {})}
                if catalog_note:
                    entry["catalog_size_note"] = catalog_note
                with _lock:
                    man["files"][rel] = entry
                log(f"OK   {rel}  {size / 1e6:.1f} MB")
                return True
            except urllib.error.HTTPError as exc:
                log(f"HTTP {exc.code} {url} (attempt {attempt})")
                if exc.code == 403 and not browser:
                    browser = True  # retry once with browser headers
                    continue
                if exc.code in (400, 401, 403, 404, 410):
                    break  # try the next candidate URL
                time.sleep(min(60, 2 ** attempt))
            except Exception as exc:  # network hiccup: retry
                log(f"RETRY {rel}: {type(exc).__name__}: {exc} (attempt {attempt})")
                time.sleep(min(60, 2 ** attempt))
    part.unlink(missing_ok=True)
    log(f"FAIL {rel}")
    return False


def check_space(gb_needed: float) -> None:
    free = shutil.disk_usage(ROOT).free / 1e9
    log(f"free disk: {free:.1f} GB (need about {gb_needed:.0f} GB)")
    if free < gb_needed * 1.2:
        sys.exit(f"Not enough disk space: {free:.1f} GB free, need about {gb_needed:.0f} GB.")


# ---------------------------------------------------------------------------- stage 1


def stage1(workers: int) -> int:
    check_space(20)
    man = load_manifest()
    failures = 0

    for key, urls, rel, vintage, lic, name in STAGE1:
        ok = download(key, urls, rel, man, meta={"vintage": vintage, "license": lic,
                                                 "source": name})
        failures += not ok
        save_manifest(man)

    # TxGIO parcel resource list (saved for provenance), then all county zips.
    res = json.loads(http_get(TXGIO_RESOURCES))
    (RAW / "txgio").mkdir(parents=True, exist_ok=True)
    (RAW / "txgio" / "resources.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    # County files only: the statewide bundle (..._48_lp.zip) duplicates them (2.7 GB).
    items = [r for r in res["results"] if r.get("resource")
             and not r["resource"].endswith("landparcels_48_lp.zip")]
    download("txgio_report",
             f"https://data.geographic.texas.gov/{TXGIO_COLLECTION}/assets/"
             "land-parcels-supplemental-report.zip",
             "txgio/land-parcels-supplemental-report.zip", man,
             meta={"vintage": "collection 2026-09-25", "license": "public (TxGIO)",
                   "source": "TxGIO StratMap Land Parcels supplemental report"})
    log(f"TxGIO parcels: {len(items)} county files, "
        f"{sum(int(r.get('filesize') or 0) for r in items) / 1e9:.2f} GB")

    def one(r: dict) -> bool:
        fname = r["resource"].rsplit("/", 1)[-1]
        return download("txgio_parcels", r["resource"], f"txgio/{fname}", man,
                        meta={"vintage": "StratMap Land Parcels, collection 2026-09-25",
                              "license": "public; TxGIO and county appraisal districts, "
                                         "provided as-is", "source": "TxGIO StratMap Land Parcels",
                              "county": r.get("area_type_name")},
                        expect_bytes=int(r["filesize"]) if r.get("filesize") else None)

    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for ok in ex.map(one, items):
            done += 1
            failures += not ok
            if done % 10 == 0:
                save_manifest(man)
                log(f"parcels {done}/{len(items)}")
    save_manifest(man)

    # NASA POWER climatological irradiance on the native 0.5 degree grid over Texas.
    out = RAW / "power" / "ghi_climatology_tx.json"
    if not out.exists():
        pts = []
        lon0, lat0, lon1, lat1 = TX_BBOX
        lat = lat0
        while lat <= lat1 + 1e-9:
            lon = lon0
            while lon <= lon1 + 1e-9:
                pts.append((round(lon, 2), round(lat, 2)))
                lon += 0.5
            lat += 0.5

        def pw(pt):
            lon, lat = pt
            try:
                j = json.loads(http_get(POWER_URL.format(lon=lon, lat=lat), timeout=60))
                return {"lon": lon, "lat": lat,
                        "ghi_kwh_m2_day": j["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]["ANN"]}
            except Exception as exc:
                return {"lon": lon, "lat": lat, "error": str(exc)}

        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            vals = list(ex.map(pw, pts))
        bad = sum("error" in v for v in vals)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"source": "NASA POWER climatology API, ALLSKY_SFC_SW_DWN "
                                   "(annual mean, kWh/m2/day), 0.5 x 0.625 deg native grid",
                                   "fetched_utc": _now(), "points": vals}), encoding="utf-8")
        man["files"]["power/ghi_climatology_tx.json"] = {
            "key": "nasa_power_ghi", "url": POWER_URL, "bytes": out.stat().st_size,
            "sha256": _sha256(out), "fetched_utc": _now(),
            "vintage": "POWER climatology (multi-year mean)", "license": "public (NASA)",
            "source": "NASA POWER", "points": len(vals), "errors": bad}
        log(f"NASA POWER: {len(vals)} grid points, {bad} errors")
        failures += bad > 0
        save_manifest(man)

    log(f"STAGE 1 DONE with {failures} failure(s). Tell Claude it finished.")
    return failures


# ---------------------------------------------------------------------------- stage 2


def _arcgis_pages(layer: int, where: str, bbox: tuple, out_fields: str,
                  page: int = 1000, simplify: bool = False) -> list[dict]:
    feats: list[dict] = []
    offset = 0
    while True:
        params = {"where": where, "geometry": ",".join(f"{v:.5f}" for v in bbox),
                  "geometryType": "esriGeometryEnvelope", "inSR": "4326",
                  "spatialRel": "esriSpatialRelIntersects", "outFields": out_fields,
                  "outSR": "4326", "returnGeometry": "true", "resultOffset": str(offset),
                  "resultRecordCount": str(page), "f": "geojson"}
        if simplify:  # ~2 m generalisation and 6-decimal coordinates: far below 30 m pixels
            params.update({"maxAllowableOffset": "0.00002", "geometryPrecision": "6"})
        url = NFHL_QUERY.format(layer=layer) + "?" + urllib.parse.urlencode(params)
        j = json.loads(http_get(url, timeout=180))
        if "error" in j:
            raise RuntimeError(j["error"])
        batch = j.get("features", [])
        feats += batch
        if len(batch) < page and not j.get("properties", {}).get("exceededTransferLimit"):
            return feats
        offset += len(batch)


def _sfha_robust(b: tuple, depth: int = 0) -> list[dict]:
    """Flood zones in ``b``. A server error on a dense cell is retried with simplified
    geometry and smaller pages, then by splitting the cell into quarters (up to 3 levels).
    Polygons that straddle sub-cells come back more than once; the processing step drops
    exact duplicates."""
    try:
        return _arcgis_pages(28, "SFHA_TF='T'", b, "FLD_ZONE,SFHA_TF,DFIRM_ID",
                             page=1000 if depth == 0 else 250, simplify=depth > 0)
    except Exception as exc:
        if depth >= 3:
            raise
        log(f"NFHL split {b} (depth {depth + 1}) after: {str(exc)[:80]}")
        x0, y0, x1, y1 = b
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        out: list[dict] = []
        for q in ((x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)):
            out += _sfha_robust(q, depth + 1)
        return out


def stage2(workers: int) -> int:
    plan_path = WORK / "fetch_plan.json"
    if not plan_path.exists():
        sys.exit("No data/parcel/work/fetch_plan.json yet. Run stage1, then ask Claude to "
                 "build the sample, then run stage2.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    check_space(8)
    man = load_manifest()
    failures = 0

    cells = plan["nfhl_cells"]  # [[lon0, lat0, lon1, lat1], ...]
    ndir = RAW / "nfhl"
    ndir.mkdir(parents=True, exist_ok=True)

    def cell(b):
        name = f"cell_{b[0]:.3f}_{b[1]:.3f}.geojson".replace("-", "m")
        dest = ndir / name
        if dest.exists():
            return True
        try:
            sfha = _sfha_robust(tuple(b))
            avail = _arcgis_pages(0, "1=1", tuple(b), "STUDY_ID", simplify=True)
            dest.write_text(json.dumps({"bbox": b, "fetched_utc": _now(),
                                        "sfha": {"type": "FeatureCollection", "features": sfha},
                                        "availability": {"type": "FeatureCollection",
                                                         "features": avail}}),
                            encoding="utf-8")
            return True
        except Exception as exc:
            log(f"NFHL FAIL {b}: {exc}")
            return False

    done = 0
    with cf.ThreadPoolExecutor(max_workers=min(workers, 4)) as ex:
        for ok in ex.map(cell, cells):
            done += 1
            failures += not ok
            if done % 50 == 0:
                log(f"NFHL cells {done}/{len(cells)}")
    man["files"]["nfhl/"] = {"key": "fema_nfhl", "url": NFHL_QUERY.format(layer=28),
                             "cells": len(cells), "fetched_utc": _now(),
                             "vintage": "NFHL effective as of fetch date",
                             "license": "public domain (FEMA)",
                             "source": "FEMA National Flood Hazard Layer (layers 28 and 0)"}
    save_manifest(man)

    # Elevation: newest 1 arc-second tile for each needed 1-degree cell.
    want = set(plan["dem_tiles"])  # e.g. "n32w103"
    j = json.loads(http_get(TNM_DEM.format(bbox="-106.75,25.75,-93.25,36.75")))
    best: dict[str, dict] = {}
    for it in j.get("items", []):
        u = it.get("downloadURL") or ""
        tag = next((t for t in want if f"_{t}_" in u or f"/{t}/" in u), None)
        if tag and (tag not in best or it.get("publicationDate", "") > best[tag].get(
                "publicationDate", "")):
            best[tag] = it
    missing = sorted(want - set(best))
    if missing:
        log(f"DEM: no tile listed for {missing}")

    def dem(tag):
        it = best[tag]
        return download("usgs_3dep_1as", it["downloadURL"], f"dem/{tag}.tif", man,
                        meta={"vintage": it.get("publicationDate"), "license":
                              "public domain (USGS 3DEP)", "source": "USGS 3DEP 1 arc-second DEM",
                              "title": it.get("title")})

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for ok in ex.map(dem, sorted(best)):
            failures += not ok
    save_manifest(man)
    failures += extract_rasters()
    log(f"STAGE 2 DONE with {failures} failure(s). Tell Claude it finished.")
    return failures


def _unzip(zpath: Path, members: list[str], dest_dir: Path) -> None:
    """Extract members atomically with a size check (partial files never survive)."""
    import zipfile

    with zipfile.ZipFile(zpath) as zf:
        for name in members:
            size = zf.getinfo(name).file_size
            out = dest_dir / name.split("/", 2)[-1] if name.startswith("Data/") else dest_dir / name
            if out.exists() and out.stat().st_size == size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.name + ".part")
            with zf.open(name) as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 24)
            if part.stat().st_size != size:
                raise OSError(f"bad extract {out}")
            part.replace(out)


def extract_rasters() -> int:
    """Unzip the land-cover GeoTIFF and the wildfire grid next to their archives, so the
    processing step reads them directly (much faster than reading inside a zip)."""
    import zipfile

    fails = 0
    jobs = [(RAW / "nlcd" / "Annual_NLCD_LndCov_2016_CU.zip", lambda n: n.endswith(".tif"),
             RAW / "nlcd"),
            (RAW / "whp" / "RDS-2015-0047-2.zip",
             lambda n: n.startswith("Data/whp_2018_continuous/") and not n.endswith("/"),
             RAW / "whp")]
    for zpath, pick, dest in jobs:
        try:
            with zipfile.ZipFile(zpath) as zf:
                names = [n for n in zf.namelist() if pick(n)]
            _unzip(zpath, names, dest)
            log(f"OK   extracted {len(names)} file(s) from {zpath.name}")
        except Exception as exc:
            fails += 1
            log(f"FAIL extract {zpath.name}: {exc}")
    return fails


def status() -> None:
    man = load_manifest()
    by: dict[str, list] = {}
    for rec in man["files"].values():
        by.setdefault(rec.get("key", "?"), []).append(rec.get("bytes") or 0)
    for k, v in sorted(by.items()):
        print(f"{k:<22} {len(v):>4} file(s)  {sum(v) / 1e9:7.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["stage1", "stage2", "status"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.stage == "status":
        status()
        return
    log(f"=== {a.stage} start (python {sys.version.split()[0]}) ===")
    fails = stage1(a.workers) if a.stage == "stage1" else stage2(a.workers)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
