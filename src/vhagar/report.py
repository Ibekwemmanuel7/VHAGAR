"""Render a Wildfire Event Evidence Pack as a self-contained HTML report.

This turns the portfolio-to-event intersection (``vhagar.intersect``) into the actual
deliverable from ``docs/21_PRODUCT_POSITIONING.md``: a dated, inspectable report a
claims or exposure team can receive, with an affected-location table, an evidence map,
provenance and freshness, and the honest disclosure that a footprint is a detection
hull, not an agency perimeter.

Pure stdlib (``html.escape`` only), so it runs in the core CI env and is unit-tested.
The map is inline SVG built from the event footprints and the portfolio points; there
are no external assets, so the file opens anywhere and can be emailed as-is.
"""
from __future__ import annotations

import html
from datetime import UTC, datetime

__all__ = ["render_evidence_pack"]

_C = {"inside_footprint": "#c62828", "within_buffer": "#ef6c00",
      "clear": "#5b6b7a", "no_events_in_view": "#5b6b7a"}
_LABEL = {"inside_footprint": "Inside footprint", "within_buffer": "Within buffer",
          "clear": "Clear", "no_events_in_view": "No events in view"}


def _bbox(pts, rings):
    xs = [p[0] for p in pts] + [c[0] for r in rings for c in r]
    ys = [p[1] for p in pts] + [c[1] for r in rings for c in r]
    if not xs:
        return -120.0, 39.0, -119.9, 39.1
    w, e, s, n = min(xs), max(xs), min(ys), max(ys)
    if e - w < 1e-4:
        w, e = w - 0.02, e + 0.02
    if n - s < 1e-4:
        s, n = s - 0.02, n + 0.02
    return w, s, e, n


def _svg_map(result, events, width=760, height=420, pad=24) -> str:
    rings = []
    for f in (events.get("features", []) if isinstance(events, dict) else (events or [])):
        geom = (f or {}).get("geometry") or {}
        if geom.get("type") == "Polygon" and geom.get("coordinates"):
            rings.append(geom["coordinates"][0])
    pts = [(r["lon"], r["lat"], r["status"]) for r in result.get("affected", [])]
    pts += [(r["lon"], r["lat"], r.get("status", "clear")) for r in result.get("clear", [])]
    w, s, e, n = _bbox([(x, y) for x, y, _ in pts], rings)
    scale = min((width - 2 * pad) / (e - w), (height - 2 * pad) / (n - s))
    ox = pad + ((width - 2 * pad) - (e - w) * scale) / 2
    oy = pad + ((height - 2 * pad) - (n - s) * scale) / 2

    def X(lon):
        return ox + (lon - w) * scale

    def Y(lat):
        return height - (oy + (lat - s) * scale)      # flip: north is up

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="background:#0e1117;border-radius:8px" xmlns="http://www.w3.org/2000/svg">']
    for r in rings:
        d = " ".join(f"{X(c[0]):.1f},{Y(c[1]):.1f}" for c in r)
        parts.append(f'<polygon points="{d}" fill="rgba(255,138,0,.14)" '
                     f'stroke="#ff8a00" stroke-width="1.2"/>')
    for x, y, st in pts:
        parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.1" '
                     f'fill="{_C.get(st, "#5b6b7a")}" fill-opacity="0.9"/>')
    # legend
    ly = height - 14
    for i, (st, lab) in enumerate([("inside_footprint", "Inside footprint"),
                                   ("within_buffer", "Within buffer"), ("clear", "Clear")]):
        lx = 14 + i * 150
        parts.append(f'<circle cx="{lx}" cy="{ly}" r="4" fill="{_C[st]}"/>'
                     f'<text x="{lx + 9}" y="{ly + 4}" fill="#c8d0da" font-size="11" '
                     f'font-family="system-ui,sans-serif">{lab}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _esc(v) -> str:
    return html.escape("" if v is None else str(v))


_DC = {"No Damage": "#5b6b7a", "Minor": "#c9a227", "Major": "#ef6c00", "Destroyed": "#c62828"}


def _rows(affected, show_damage: bool, show_overlap: bool) -> str:
    out = []
    for r in affected:
        dmg = ""
        if show_damage:
            dc = r.get("damage_class")
            cell = (f"<span class='pill' style='background:{_DC.get(dc, '#39424e')}'>{_esc(dc)}</span>"
                    if dc else "<span class='none'>n/a</span>")
            dmg = f"<td>{cell}</td>"
        ov = ""
        if show_overlap:
            pct = r.get("overlap_pct")
            ov = f"<td class='num'>{(_esc(pct) + '%') if pct is not None else '&mdash;'}</td>"
        out.append(
            "<tr>"
            f"<td>{_esc(r.get('id'))}</td>"
            f"<td><span class='pill' style='background:{_C.get(r['status'], '#5b6b7a')}'>"
            f"{_LABEL.get(r['status'], r['status'])}</span></td>"
            f"{ov}"
            f"<td class='num'>{_esc(r.get('distance_m'))}</td>"
            f"<td>{_esc(r.get('event_label') or r.get('event_id'))}</td>"
            f"{dmg}"
            f"<td>{_esc(r.get('last_seen_utc'))}</td>"
            f"<td class='num'>{_esc(r.get('data_age_hours'))}</td>"
            f"<td>{_esc(', '.join(r.get('sensors') or []))}</td>"
            f"<td>{_esc(r.get('confidence'))}</td>"
            "</tr>")
    return "".join(out)


def render_evidence_pack(result, events=None, *, title="Wildfire Event Evidence Pack",
                         portfolio_name=None, generated=None, meta=None) -> str:
    """Return a full self-contained HTML string for the intersection ``result``.

    ``result`` is the dict from ``vhagar.intersect.intersect_portfolio``. ``events`` is
    the event FeatureCollection used (for the map). ``meta`` may carry region, days,
    mode, and window_end_utc for the provenance line."""
    gen = generated or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    s = result.get("summary", {})
    meta = meta or result.get("metadata", {}) or {}
    disclosure = result.get("disclosure", "")
    prov = " · ".join(x for x in [
        f"region {meta['region']}" if meta.get("region") else None,
        f"window ≤ {meta['days']}d" if meta.get("days") else None,
        f"feed {meta['mode']}" if meta.get("mode") else None,
        f"events {meta['event_count']}" if meta.get("event_count") is not None else None,
        f"latest event obs {meta['window_end_utc']}" if meta.get("window_end_utc") else None,
    ] if x)
    affected = result.get("affected", [])
    show_damage = any(r.get("damage_class") for r in affected)
    cards = [
        ("Portfolio", s.get("portfolio_size", 0)),
        ("Affected", s.get("affected_count", 0)),
        ("Inside footprint", s.get("inside_footprint_count", 0)),
        ("Within buffer", s.get("within_buffer_count", 0)),
        ("Clear", (s.get("portfolio_size", 0) - s.get("affected_count", 0))),
    ]
    if show_damage:
        cards.append(("Screened destroyed",
                      sum(1 for r in affected if r.get("damage_class") == "Destroyed")))
    card_html = "".join(
        f"<div class='card'><div class='k'>{_esc(k)}</div><div class='v'>{_esc(v)}</div></div>"
        for k, v in cards)
    show_overlap = any("overlap_pct" in r for r in affected)
    dmg_th = "<th>Damage screen</th>" if show_damage else ""
    ov_th = "<th>In fire %</th>" if show_overlap else ""
    table = (f"<table><thead><tr><th>Location</th><th>Status</th>{ov_th}<th>Distance (m)</th>"
             f"<th>Event</th>{dmg_th}<th>Last seen (UTC)</th><th>Age (h)</th><th>Sensors</th>"
             f"<th>Confidence</th></tr></thead><tbody>{_rows(affected, show_damage, show_overlap)}</tbody></table>"
             if affected else "<p class='none'>No portfolio locations were affected in this window.</p>")
    dmg_disclosure = (" A damage screen column is present: it maps a burn-severity raster "
                      "to an xView2/xBD damage class per location, a screen for inspection "
                      "targeting, not a per-structure inspection or a proof of loss."
                      if show_damage else "")
    dmg_source = result.get("damage_source")
    if show_damage and dmg_source:
        prov = (prov + " · " if prov else "") + f"damage screen {dmg_source}"
    sub = _esc(portfolio_name) + " · " if portfolio_name else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>
:root{{color-scheme:dark}}
body{{margin:0;background:#0b0f16;color:#e6eef8;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.5}}
.wrap{{max-width:1000px;margin:0 auto;padding:28px 22px 60px}}
h1{{font-size:1.35rem;margin:0 0 2px}}
.meta{{color:#93a1b3;font-size:.82rem;margin-bottom:18px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}}
.card{{flex:1;min-width:130px;background:#131a24;border:1px solid #223;border-radius:10px;padding:12px 14px}}
.card .k{{color:#93a1b3;font-size:.68rem;text-transform:uppercase;letter-spacing:.06em}}
.card .v{{font-size:1.5rem;font-weight:700;margin-top:3px}}
table{{width:100%;border-collapse:collapse;margin-top:8px;font-size:.82rem}}
th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid #1c2430}}
th{{color:#93a1b3;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.pill{{display:inline-block;padding:2px 8px;border-radius:999px;color:#fff;font-size:.72rem;font-weight:600}}
.disclosure{{margin-top:22px;background:#1a1206;border:1px solid #3a2a12;border-radius:10px;padding:12px 15px;color:#f0d9b5;font-size:.82rem}}
.foot{{margin-top:22px;color:#6b7a8d;font-size:.74rem}}
.none{{color:#93a1b3}}
h2{{font-size:.95rem;color:#cbd5e2;margin:24px 0 6px}}
@media print{{
  :root{{color-scheme:light}}
  body{{background:#fff;color:#111}}
  .card{{background:#f4f6f9;border-color:#ccd3dc}}
  .card .k,.meta,.foot,.none{{color:#555}}
  th{{color:#555}}
  th,td{{border-bottom-color:#dde3ea}}
  .disclosure{{background:#fdf6e6;border-color:#e6cf9a;color:#5b4708}}
  tr,svg,.disclosure{{break-inside:avoid}}
  h2{{color:#222}}
}}
</style></head><body><div class="wrap">
<h1>{_esc(title)}</h1>
<div class="meta">{sub}Generated {_esc(gen)}{(' · ' + _esc(prov)) if prov else ''}</div>
<div class="cards">{card_html}</div>
<h2>Evidence map</h2>
{_svg_map(result, events or {})}
<h2>Affected locations</h2>
{table}
<div class="disclosure"><b>Disclosure.</b> {_esc(disclosure)}{_esc(dmg_disclosure)}</div>
<div class="foot">Wildfire Event Evidence Pack, generated by VHAGAR. Detection footprints
are the convex hull of observed GOES/VIIRS/MODIS thermal detections. This report is
screening evidence for inspection and triage, not a proof of loss, a claims
adjudication, or an agency fire perimeter.</div>
</div></body></html>"""
