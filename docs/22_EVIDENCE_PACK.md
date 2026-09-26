# Wildfire Event Evidence Pack (portfolio-to-event intersection)

The market-entry product from `docs/21_PRODUCT_POSITIONING.md`, built on the tiers
that already run: a customer supplies a portfolio of locations, and VHAGAR returns,
per location, the nearest fire event, whether it falls inside the detection footprint
or within a screening buffer, the first and last detection times, the sensors that
saw it, a corroboration confidence, and the data age.

This is deliberately narrow. It is sellable now because it uses only real T1 fused
detection and the event footprints the console already serves, and it is the thing
that generates the exposure, claims, and validation data the heavier products need.

## Honest scope

An event footprint here is the convex hull of observed thermal detections, an
evidence extent, not a legal or agency fire perimeter and not a burned-area
measurement. "Inside footprint" is a screening signal for inspection targeting, not
a proof of loss. Confidence reflects sensor corroboration and detection count, not a
probability of damage. Every output carries this disclosure inline.

## Pieces

- `src/vhagar/intersect.py`: the core, pure numpy and stdlib so it runs in the core
  CI env. `intersect_portfolio(portfolio, events, buffer_m, now)` returns `affected`,
  `clear`, a `summary`, and a `disclosure`. Point-in-polygon is ray casting;
  distance to a footprint is point-to-segment in a local metric plane; confidence is
  a corroboration tier from sensor diversity and detection count.
- `scripts/evidence_pack.py`: the CLI. Reads a portfolio CSV (latitude and longitude
  columns, optional id), loads events from a local GeoJSON (`--events`) or the live
  API (`--base-url` with `--region`/`--days`), and writes a JSON report and an
  affected-location CSV.
- `POST /api/intersect` in `serve/vhagar_api.py`: body
  `{"portfolio": [{"id","lat","lon"}, ...], "region", "days", "buffer_m"}`, returns
  the same structure against the live event feed.
- `tests/test_intersect.py`: CI-safe unit tests (synthetic square footprint).

## Run it

```bash
# against a local events GeoJSON
python scripts/evidence_pack.py portfolio.csv --events events.geojson \
    --buffer-m 1000 --out pack.json --csv affected.csv

# against the live feed
python scripts/evidence_pack.py portfolio.csv \
    --base-url https://vhagar-console.onrender.com --region california --days 3
```

Portfolio CSV: a header row with latitude and longitude columns (any of
lat/latitude/y, lon/lng/longitude/x, case-insensitive) and an optional id/name column.

## Output, per affected location

`status` (inside_footprint or within_buffer), `distance_m` to the footprint,
`event_id` and `event_label`, `first_seen_utc` and `last_seen_utc`, `data_age_hours`,
`sensors`, `n_detections`, `max_frp_mw`, and `confidence` (confirmed_multi_sensor,
single_sensor_repeated, or single_sensor_sparse). Locations with no event within the
buffer are returned under `clear`, so the whole portfolio is accounted for.

## The dated HTML report (the deliverable)

`src/vhagar/report.py` renders the intersection result into a single self-contained
HTML file, the artifact a claims or exposure team actually receives. It carries a dated
header with provenance and freshness, summary cards (portfolio, affected, inside
footprint, within buffer, clear), an inline SVG evidence map (event footprints plus
portfolio points colored by status, no external assets), the affected-location table,
and the honest disclosure. `render_evidence_pack(result, events, ...)` is pure stdlib
and unit-tested (`tests/test_report.py`), including HTML escaping so a hostile portfolio
id cannot inject markup.

```bash
python scripts/evidence_pack_report.py portfolio.csv --events events.geojson \
    --name "ACME Portfolio" --out pack.html
python scripts/evidence_pack_report.py portfolio.csv \
    --base-url https://vhagar-console.onrender.com --region california --days 3 --out pack.html
```

The report opens in any browser and can be emailed as-is; it is screening evidence for
inspection and triage, not a proof of loss, a claims adjudication, or an agency
perimeter, and it says so.

### Optional damage screen and PDF

Pass `--severity-tif <burn-severity raster>` and the report samples the severity at each
affected location and adds a **Damage screen** column with the xView2/xBD class (No
Damage, Minor, Major, Destroyed), a "Screened destroyed" summary card, and the severity
source in the provenance line. Two schemes:

- `--severity-scheme rbr` (VHAGAR-native): the raster is VHAGAR's own **T2 scaled-RBR
  severity product** (the optical Stage-0 output, RBR reprojected to the EPSG:5070
  reference grid). Classified with VHAGAR's Key-Benson-derived thresholds
  (`vhagar.eval.damage_screen.rbr_to_class_index`, breakpoints 100/440/660). This is
  what makes the whole pack run on VHAGAR outputs, not a third-party product.
- `--severity-scheme mtbs` (default, a validation reference): the raster is an MTBS
  thematic severity mosaic, mapped by `severity_to_class_index`. Useful for historical
  fires and for validating the RBR screen against MTBS.

The scoring core and the honest scope (a screen, not an adjudication) are in
`docs/23_POSTFIRE_DAMAGE.md`. The severity raster window is padded in the raster's own
units (5 pixels), so both metre-grid and degree rasters sample correctly.

`--pdf` writes a PDF via LibreOffice as a convenience. The report also carries an
`@media print` stylesheet (light background, page-break-avoid on rows and the map), so
the highest-fidelity claims-file PDF is produced by opening the HTML and using the
browser's Print to PDF.

```bash
python scripts/evidence_pack_report.py portfolio.csv --events events.geojson \
    --severity-tif mtbs_extract/mtbs_CONUS_2021.tif --severity-source "MTBS 2021" \
    --name "ACME Portfolio" --out pack.html --pdf
```

## Building-polygon portfolios (footprint-to-footprint)

The pack works on real building footprints, not just points. A portfolio entry may carry
a `footprint` (a lon/lat ring, GeoJSON Polygon coordinates, or a geometry dict), or the
CLI can be handed a GeoJSON of Polygon features directly (`portfolio.geojson`, with an
`id`/`name`/`policy_id` property per feature). Footprint entries are matched
polygon-to-polygon: because an event footprint is a convex hull, the intersection is
exact via Sutherland-Hodgman clipping (`vhagar.intersect.footprint_overlap` /
`clip_polygon`, pure stdlib, CI-tested), and each affected building carries an
`overlap_pct`, the share of the building inside the detection hull, shown as an
"In fire %" column in the report. Point entries keep the point-in-hull / distance
behaviour unchanged.

```bash
python scripts/evidence_pack_report.py buildings.geojson --events events.geojson \
    --severity-tif mtbs_extract/mtbs_CONUS_2021.tif --severity-source "MTBS 2021" \
    --name "ACME building portfolio" --out pack.html
```

## Next steps toward a paid pilot

- Wire the live-event path to VHAGAR's own T2 burned-area product as the severity
  source, so the pack runs end to end on VHAGAR outputs rather than MTBS.
