"""Generate a dated Wildfire Event Evidence Pack HTML report for a customer portfolio.

Ties the pieces together into the market-entry deliverable: it intersects a portfolio
against the live (or a supplied) fire-event feed, optionally samples a burn-severity
raster to add a per-location damage screen (xView2/xBD classes), and writes a single
self-contained HTML file with an affected-location table, an evidence map, provenance
and freshness, and the honest disclosure. Optionally exports a PDF for a claims file.

    python scripts/evidence_pack_report.py portfolio.csv --events events.geojson \
        --name "ACME Portfolio" --out pack.html
    python scripts/evidence_pack_report.py portfolio.csv \
        --base-url https://vhagar-console.onrender.com --region california --days 3 \
        --severity-tif mtbs_extract/mtbs_CONUS_2021.tif --severity-source "MTBS 2021" \
        --out pack.html --pdf
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
from evidence_pack import _load_events, _read_portfolio  # noqa: E402

from vhagar.eval.damage_screen import (  # noqa: E402
    SEVERITY_CLASS_NAMES,
    severity_to_class_index,
)
from vhagar.intersect import intersect_portfolio  # noqa: E402
from vhagar.report import render_evidence_pack  # noqa: E402


def _attach_damage(res: dict, severity_tif: str) -> int:
    """Sample the burn-severity raster at each affected location and attach a
    burn_severity code and an xView2/xBD damage_class. Returns the count sampled."""
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds
    aff = res.get("affected", [])
    if not aff:
        return 0
    lon = np.array([r["lon"] for r in aff], float)
    lat = np.array([r["lat"] for r in aff], float)
    ds = rasterio.open(severity_tif)
    xs, ys = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform(lon, lat)
    pad = 2000.0
    win = from_bounds(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad,
                      ds.transform).round_offsets().round_lengths()
    arr = ds.read(1, window=win)
    inv = ~ds.window_transform(win)
    cols, rows = inv * (np.asarray(xs), np.asarray(ys))
    rows = np.clip(np.round(rows).astype(int), 0, arr.shape[0] - 1)
    cols = np.clip(np.round(cols).astype(int), 0, arr.shape[1] - 1)
    sev = arr[rows, cols].astype(int)
    idx = severity_to_class_index(sev)
    for r, sv, ix in zip(aff, sev, idx, strict=False):
        r["burn_severity"] = int(sv)
        r["damage_idx"] = int(ix)
        r["damage_class"] = SEVERITY_CLASS_NAMES[int(ix)]
    return len(aff)


def _to_pdf(html_path: str) -> tuple[bool, str]:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return False, "no PDF engine found; open the HTML and Print to PDF from your browser"
    try:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir",
                        str(Path(html_path).parent), html_path],
                       check=True, capture_output=True, timeout=180)
        return True, str(Path(html_path).with_suffix(".pdf"))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return False, f"soffice conversion failed: {type(exc).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("portfolio", help="CSV of locations (lat/lon columns, optional id)")
    ap.add_argument("--events", help="local events GeoJSON file")
    ap.add_argument("--base-url", default="https://vhagar-console.onrender.com")
    ap.add_argument("--region", default="california")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--buffer-m", type=float, default=1000.0)
    ap.add_argument("--severity-tif", help="burn-severity raster to add a per-location damage screen")
    ap.add_argument("--severity-source", help="label for the severity source shown in provenance")
    ap.add_argument("--name", default=None, help="portfolio name shown in the report header")
    ap.add_argument("--out", default="evidence_pack.html")
    ap.add_argument("--pdf", action="store_true", help="also write a PDF (best effort via LibreOffice)")
    args = ap.parse_args()

    portfolio = _read_portfolio(args.portfolio)
    fc = _load_events(args)
    res = intersect_portfolio(portfolio, fc, buffer_m=args.buffer_m)
    meta = fc.get("metadata", {})
    res["metadata"] = {"region": args.region, "days": args.days, "mode": meta.get("mode"),
                       "event_count": meta.get("event_count"),
                       "window_end_utc": meta.get("window_end_utc")}
    if args.severity_tif:
        n = _attach_damage(res, args.severity_tif)
        res["damage_source"] = args.severity_source or Path(args.severity_tif).name
        print(f"damage screen: sampled {n} affected locations from {res['damage_source']}")

    gen = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    html = render_evidence_pack(res, fc, portfolio_name=args.name, generated=gen)
    Path(args.out).write_text(html, encoding="utf-8")
    s = res["summary"]
    print(f"portfolio {s['portfolio_size']}  affected {s['affected_count']} "
          f"(inside {s['inside_footprint_count']}, buffer {s['within_buffer_count']})")
    print("wrote", args.out)
    if args.pdf:
        ok, info = _to_pdf(args.out)
        print("wrote", info) if ok else print("PDF not written:", info)
        if ok:
            print("note: for best fidelity (dark theme, map), open the HTML and Print to PDF from a browser")


if __name__ == "__main__":
    main()
