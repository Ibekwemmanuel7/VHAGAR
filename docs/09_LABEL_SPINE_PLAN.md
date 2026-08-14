# Label spine plan (Step 3)

Status: draft for review, not yet implemented. This is the shape of the work, so
we agree on sources and interfaces before writing code.

## Why this is next

The archive (Tier A detections, Tier B climatology) gives inputs. Nothing can be
trained or honestly evaluated without labels. The label spine is what unblocks
Step 4, the first real number, a T2 burned-area Stage-0 baseline, which the
architecture deliberately sequences before T1 modelling because the labels are
better and the Olofsson area machinery already exists (`eval/area_estimation.py`).

## What already exists, and the gap

Two pieces are done and should not be rebuilt:

- `vhagar/labels/registry.py`: `FireEventRecord`, `LabelQuality`, `LabelSource`,
  the source-to-quality map, the `EVALUATION_ONLY` reservation (Copernicus EMS,
  NIROPS, CBI held out as test), the `assert_trainable` guard that refuses a
  perimeter with no interior mask, and `EventRegistry` with `to_split_units()`.
- `vhagar/eval/splits.py`: `SplitUnit`, `SplitManifest`, `spatial_block_split`,
  `leave_year_out`, `leave_one_group_out`, `verify_no_overlap`, JSON
  serialisation, and a `vhagar splits build` CLI. `random_split` deliberately
  raises.

So the vocabulary and the splitting are built. The gap is the middle:

1. **Ingestion adapters.** Nothing reads a real MTBS / NIFC / EFFIS file and
   normalises it into `FireEventRecord`. This is the bulk of the work.
2. **Registry persistence.** The registry is in-memory only. It needs to save and
   load as GeoParquet, the versioned artifact the architecture calls for.
3. **Tile assignment.** `FireEventRecord.tile_ids` is empty. Events must be
   located on the analysis grid so a tile-blocked split and per-tile training
   reads are possible.
4. **The manifest pipeline.** A single path from raw source files to a versioned
   set of split manifests, plus a registry summary.

## Module shape

New module `vhagar/labels/ingest.py`, with a thin adapter per source and a pure
normalisation core:

- `normalize_mtbs(rows) -> list[FireEventRecord]`, and one `normalize_*` per
  source. Each maps that source's fields to the common record: event id, dates,
  area, representative point, geometry and severity paths, ecoregion, cause,
  fire type, and the `LabelSource` (which fixes `LabelQuality`).
- The `normalize_*` functions take already-parsed rows (mappings), so the field
  mapping is pure and unit-testable with a handful of synthetic rows. A thin
  `read_*` wrapper does the file IO (GeoParquet or shapefile via geopandas or
  pyogrio, lazily imported) and hands rows to the normaliser. Network and heavy
  geo dependencies stay at the edge; the logic that can be wrong stays testable.

New module `vhagar/labels/tiles.py`:

- `assign_tiles(record, grid)`: project the event's representative point (and,
  when a geometry is present, its bounding box corners) into the region CRS and
  return the analysis-grid tile ids it falls in. Reuses `vhagar.grid`. Point
  assignment first; full polygon coverage (every tile a perimeter touches) is a
  follow-up, flagged below as a decision.

Registry persistence, added to `registry.py`:

- `EventRegistry.to_geoparquet(path)` / `from_geoparquet(path)`. One row per
  event, the representative point as the geometry column, all record fields as
  columns. This is the versioned label artifact; it is what `splits build`
  should consume instead of an ad-hoc JSON.

## The pipeline and CLI

A `vhagar labels build` command:

1. read each configured source file through its adapter,
2. normalise into `FireEventRecord`, assign tiles,
3. add to the registry, refusing duplicates and (optionally) records that fail
   `assert_trainable`,
4. write the registry GeoParquet and print the `summary()` (counts by
   region/source and by quality).

Then `vhagar splits build` consumes the registry GeoParquet, calls
`to_split_units()`, and materialises the T2 manifests the protocol requires:
leave-one-fire-out, leave-one-ecoregion-out, and the headline
leave-one-continent-out (train MTBS, test EMSR). `verify_no_overlap` gates each.

## Quality and trainability, enforced not assumed

The registry already encodes the rules; the spine must honour them at ingest:

- Copernicus EMS, NIROPS and CBI are ingested but marked evaluation-only, so they
  can never leak into training. The leave-one-continent-out split is exactly
  train-on-MTBS, test-on-EMSR.
- A perimeter-only record (NIFC/WFIGS extent with no interior severity mask) is
  ingested and flagged. It is usable for event-level tasks, and
  `assert_trainable` refuses it for pixel training until an interior mask exists.
  MTBS is the primary training source precisely because it carries the
  continuous dNBR/RBR severity raster, not just a boundary.

## Testing, all offline

Mirror the archive tests: unit-test each `normalize_*` on a few synthetic rows
(correct field mapping, quality assignment, evaluation-only flagging,
perimeter-without-mask flagging), test `assign_tiles` against known grid
coordinates, test the GeoParquet round-trip, and test that the registry produces
split units that `spatial_block_split` and `leave_year_out` accept and
`verify_no_overlap` passes. The real file reads are exercised by the user
pointing the CLI at a downloaded MTBS extract.

## Staged delivery

1. Registry GeoParquet persistence plus `assign_tiles`, with tests.
2. The MTBS adapter (CONUS severity, the T2 training source) plus tests.
3. `vhagar labels build` wiring the above, and `splits build` reading the
   registry GeoParquet.
4. Additional adapters as needed: NIFC/WFIGS perimeters (extent, flagged),
   EFFIS (Europe), Copernicus EMS (the held-out European test), FPA-FOD (points,
   for T3 later).

Steps 1 to 3 give a real, versioned CONUS label spine and are the natural first
PR. Step 4 is per-source and can follow.

## Open decisions, need a call before coding

1. **Which source first? DECIDED: MTBS.** The T2 training source in the
   architecture (continuous dNBR/RBR severity, analyst-QC), and what the first
   honest T2 number needs. NIFC/WFIGS perimeters (extent-only, flagged
   non-trainable for pixels) and EFFIS (Europe) follow in step 4.
2. **Geometry dependency.** Reading real perimeter files needs geopandas or
   pyogrio plus shapely. Recommendation: adopt pyogrio for reads, keep it at the
   IO edge and lazily imported, so the normalisation and tile logic stay
   dependency-light and testable. Pin it alongside GDAL/PROJ as the architecture
   already requires.
3. **Tile assignment granularity.** Representative point first (one or a few
   tiles per event), or full polygon coverage (every tile a perimeter touches)
   now. Recommendation: point first. Polygon coverage matters for per-tile
   training reads and is a clean follow-up once a geometry reader is in.

Decision 1 sets the first adapter and blocks step 2. The rest can be settled as
we reach them.
