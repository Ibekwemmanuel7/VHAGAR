"""Generate a dated Wildfire Event Evidence Pack HTML report for a customer portfolio.

Ties the pieces together into the market-entry deliverable: it intersects a portfolio
against the live (or a supplied) fire-event feed and writes a single self-contained
HTML file, with an affected-location table, an evidence map, provenance and freshness,
and the honest disclosure. Reuses the loaders from ``evidence_pack.py`` and the
renderer from ``vhagar.report``.

    python scripts/evidence_pack_report.py portfolio.csv --events events.geojson \
        --name "ACME Portfolio" --out pack.html
    python scripts/evidence_pack_report.py portfolio.csv \
        --base-url https://vhagar-console.onrender.com --region california --days 3 \
        --out pack.html
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
from evidence_pack import _load_events, _read_portfolio  # noqa: E402

from vhagar.intersect import intersect_portfolio  # noqa: E402
from vhagar.report import render_evidence_pack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("portfolio", help="CSV of locations (lat/lon columns, optional id)")
    ap.add_argument("--events", help="local events GeoJSON file")
    ap.add_argument("--base-url", default="https://vhagar-console.onrender.com")
    ap.add_argument("--region", default="california")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--buffer-m", type=float, default=1000.0)
    ap.add_argument("--name", default=None, help="portfolio name shown in the report header")
    ap.add_argument("--out", default="evidence_pack.html")
    args = ap.parse_args()

    portfolio = _read_portfolio(args.portfolio)
    fc = _load_events(args)
    res = intersect_portfolio(portfolio, fc, buffer_m=args.buffer_m)
    meta = fc.get("metadata", {})
    res["metadata"] = {"region": args.region, "days": args.days, "mode": meta.get("mode"),
                       "event_count": meta.get("event_count"),
                       "window_end_utc": meta.get("window_end_utc")}
    gen = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    html = render_evidence_pack(res, fc, portfolio_name=args.name, generated=gen)
    Path(args.out).write_text(html, encoding="utf-8")
    s = res["summary"]
    print(f"portfolio {s['portfolio_size']}  affected {s['affected_count']} "
          f"(inside {s['inside_footprint_count']}, buffer {s['within_buffer_count']})")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
