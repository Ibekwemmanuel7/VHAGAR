# VHAGAR progress tracker

Last updated: 2026-08-13. Keep this file current. It is the single place to look
before starting a session, and the place to update before ending one.

## Where things stand right now

- Package `vhagar`, CLI `vhagar`. Base was v0.11 (240 tests). After this
  session the offline suite reports **222 passed, 2 skipped, ruff clean** on the
  sandbox interpreter. Re-run on your own machine to confirm (see below).
- Interpreter on this machine: **Python 3.12.10**, which satisfies the `>=3.11`
  requirement. No upgrade needed.
- Repo at `C:\Users\taylo\VHAGAR`. It is not yet a git repo on disk and nothing
  has ever been pushed to `https://github.com/Ibekwemmanuel7/VHAGAR`.

## How to verify on your machine

```powershell
cd C:\Users\taylo\VHAGAR
pip install -e ".[archive]"
pip install pytest ruff hypothesis
pytest -q
ruff check src tests
```

Expect 222 passed, 2 skipped, ruff clean. The `.[archive]` extra pulls s3fs,
xarray, h5netcdf, h5py, pyproj, pyarrow. h5py is listed explicitly because
`pip install h5netcdf` does not pull it on Windows and xarray then fails with a
bare ImportError.

## Done this session: section 10 fixes

All six items from the handoff brief's section 10, in order.

- [x] **10.1 Navigation cache.** `_fixed_grid_navigation` in
  `src/vhagar/io/goes_reader.py` caches lat, lon, view-zenith and pixel-area,
  keyed by projection parameters plus `x.tobytes()`/`y.tobytes()`. Double-checked
  lookup under a `threading.Lock`, compute held inside the lock so cold-start
  workers produce one miss not sixteen. Cached arrays are read-only because they
  are shared across every granule on the grid. Cache cap is 2; cap 0 disables it
  and is the benchmark toggle.
- [x] **10.2 Per-day timing.** `backfill()` in
  `src/vhagar/archive/backfill.py` now times each day and sets
  `day_result.elapsed_s` before `progress()`, so the per-day granules/s line no
  longer prints 0.0.
- [x] **10.3 Coverage command.** New `coverage_gaps()` and `failed_records()` in
  `backfill.py`, plus a `vhagar coverage <dir>` CLI command in
  `src/vhagar/cli.py`. It prints observed intervals, every hole with start, end
  and duration, and the failed-granule list. `coverage_intervals` was left
  unchanged on purpose: its logic is correct, so a multi-interval report means a
  real hole in the data, and this command is what shows where.
- [x] **10.4 Probe defaults.** `probe_workers` now warms the navigation cache
  before timing, so the default full-domain probe measures steady-state
  per-granule cost, not the one-off grid build. `bbox=None` kept as default
  because that is what a real CONUS backfill reads.
- [x] **10.5 Planner wall clock.** `DEFAULT_SECONDS_PER_GRANULE` is now 14.7,
  derived from the real 7-day run (2015 granules in 30.7 min at 16 workers is
  0.914 s wall per granule, about 14.7 single-worker-equivalent seconds). The
  fictional 0.8 that predicted 6 hours for an 80-hour job is gone. Docstring says
  what it measured and that it must be re-measured on the target machine after
  the cache. The 0.78 literals in the FDC plans and the stale "40 hours"/"six
  hours" prose were corrected to match.
- [x] **10.6 Detection rate.** `DEFAULT_DETECTION_RATE` recomputed to 2.5e-5
  from 188,639 / (2015 x 2500 x 1500). Documented as a peak-August CONUS figure,
  so it is an upper bound on the annual mean, which is the safe direction for
  sizing disk.

## Corrupt scan-start guard and the day-215 repair

Running `vhagar coverage data\detections` explained the "2 intervals": one M4
granule on day 215 (Aug 3, 15:10 UTC),
`OR_ABI-L2-FDCC-M4_G18_s20262151510224_...`, decoded its `t` field to
2000-01-01 (the ABI epoch), which split coverage into two intervals 26 years
apart and stamped 60 detection rows with the year 2000. It was a bad timestamp,
not a real outage; the actual data was one continuous block.

- [x] **Reader guard.** `_validated_scan_start` in `goes_reader.py` recovers the
  scan time from the filename (`parse_goes_key`) whenever the decoded `t`
  predates GOES-R first light (2017), and logs it by name. Wired into
  `open_fdc`. Three tests added.
- [x] **Archive repair.** Corrected the 60 poisoned rows in place across 31
  `year=2026/.../part-20260803.parquet` files (t set to 2026-08-03T15:10:22, the
  key's true time) and fixed the one manifest line. Verified: total rows
  unchanged at 188,639, zero rows before 2017, and coverage now reports **1
  interval, no holes**. No files were deleted; each parquet was rewritten via
  write-to-temp then atomic replace.

Offline suite after this work: **225 passed, 2 skipped, ruff clean**.

## Measured effect of the navigation cache

Controlled single-worker backfill, 8 granules, CONUS-scale grid
(2500 x 1500 = 3.75M points), S3 fetch stubbed so the delta isolates the
navigation change, same code path both sides, on the Linux sandbox:

```
BEFORE (cache OFF): 1.28 granules/s | 0.783 s/granule | nav misses = 8
AFTER  (cache ON):  9.19 granules/s | 0.109 s/granule | nav miss = 1, hits = 7
speedup 7.2x, detections identical (400 both ways)
```

At the decode level, cold decode over 3.75M points was 1.46 s, warm was 2.9 ms.

Read this carefully: the numbers above are a sandbox measurement with S3 stubbed
and a single worker. They are **not comparable** to the real 1.09 granules/s from
the 16-worker S3 run, and the fact that the "before" is near 1.28 is a
coincidence. What they measure honestly is the navigation cost the cache removes,
because both runs walk the identical decode path and differ only in the cache.
The definitive re-measurement must run on this machine against real S3 with
`vhagar archive-plan --measure` and `vhagar probe-workers`.

## Tests added this session

In `tests/test_goes_reader.py`: navigation computed once per grid and reused,
cached arrays are read-only, a different grid is a separate entry, computed once
under 8-way concurrency (misses == 1), and cap 0 disables caching.

In `tests/test_backfill.py`: each day records its own elapsed time,
`coverage_gaps` names the hole between two intervals, a continuous run has no
gaps, a single dropped granule is not a gap, and `failed_records` lists only
failures.

## CMIP decoder (Tier B), step 1 done

Plan is in `docs/08_CMIP_DECODER_PLAN.md`. Decision settled: read CMIP CMI
(brightness temperature in kelvin), derive radiance via planck where FRP needs
it.

- [x] **Step 1: single-channel decoder.** `src/vhagar/io/cmip_reader.py` mirrors
      the FDC reader: `decode_cmip`, `open_cmip`, `list_cmip_granules`,
      `cmip_key_prefix`, and a `CMIPChannel` dataclass. Reuses
      `_fixed_grid_navigation` unchanged, so geometry is computed once and shared
      with FDC and across channels. CMI is treated as BT, Ch7 saturation (>=400 K)
      is censored not passed through, fill and out-of-range DQF become NaN. Ten
      offline tests in `tests/test_cmip_reader.py`, including proof the nav cache
      is shared with FDC (one miss, two hits) and that CMIP and FDC co-register
      (same array objects). Shared fixture `_synthetic_cmip` added.
- [x] **Step 2: multi-channel stack.** `CMIPStack` plus `stack_channels`,
      `group_cmip_keys_by_timestamp`, and `open_cmip_stack`. Grouping pairs the
      per-channel files of one timestep within a 2-minute tolerance and drops any
      incomplete timestep so no stack is built with a missing band. `stack_channels`
      validates all channels share the grid (identity check on the cached nav
      array, corner-value fallback) and holds geometry once. `bt_difference`
      gives the co-registered C07 minus C14 contextual signal as a plain
      subtraction. Eight offline tests, network stubbed for the open path.
- [x] **Step 3: make CMIP measurable (code).** `plan.measure_granule` now times
      the full CMIP decode via `open_cmip` instead of a bare byte fetch, warms
      the nav cache first, and defaults both products to full domain, so the FDC
      and CMIP figures are finally comparable (the retracted "bytes barely
      matter" mistake came from comparing a decode against a fetch). The CLI
      `archive-plan --measure` note is updated to match.
      - [x] **Measured on this machine** with `vhagar archive-plan --measure`,
            GOES-18, warm cache, full decode: FDC 0.32 MB / 0.33 s, CMIP 4.48 MB
            / 0.78 s. CMIP is 14x the bytes and 2.4x the decode time of FDC.
            `DEFAULT_GRANULE_MB` set to 4.48 (confirmed). Added
            `MEASURED_SINGLE_WORKER_DECODE_S = {"FDC": 0.33, "CMIP": 0.78}`.
      - Note: `DEFAULT_SECONDS_PER_GRANULE` stays 14.7, NOT 0.33. The 0.33 is
        single-worker decode; the backfill is I/O/S3-bound and does not scale
        linearly, so 14.7 (calibrated to reproduce the real 16-worker run) is the
        planner figure. Dividing 0.33 by workers would predict ~2 h for a 3-year
        FDC backfill against the ~80 h reality. CMIP's true multi-worker wall
        clock is still unmeasured, pending a real Tier B probe.
- [x] **Step 4: climatology reducer.** `src/vhagar/archive/climatology.py`,
      `DiurnalClimatology`: streaming per-pixel, per-hour mean and variance via
      vectorised NaN-aware Welford, so the cube is never held. Bins by UTC hour,
      which for geostationary GOES is a per-pixel local-time diurnal cycle
      (fixed longitude per pixel). `merge` combines shards with parallel Welford
      for concurrent reduction; `save`/`load` to npz. Ten offline tests: stats
      match numpy nan-aware, per-pixel counts exclude NaN, merge equals a single
      pass, round-trip, and the CMIPStack path. Sized per tile (48x48), not full
      CONUS.
- [x] **Step 5: Tier B backfill.** `src/vhagar/archive/climatology_backfill.py`,
      `backfill_climatology`, plus a `vhagar climatology-backfill` CLI command.
      Reads the thermal channels over a window, groups into complete stacks,
      thins to the cadence, and folds each into a `DiurnalClimatology` on the
      native ABI grid (decision from the plan). Resumable without double
      counting: the checkpoint npz carries the Welford state AND the watermark of
      processed timestep ids, written with an atomic replace, so a crash never
      leaves a frame both on disk and out of the watermark. Reads run
      concurrently, the fold stays single-threaded (no lock). Manifest and
      coverage reuse the Tier A machinery. Seven offline tests including a
      resume-equals-one-pass numerical check; network stubbed.
      - [x] **First live Tier B run**, GOES-18, California bbox
            (-124,36,-118,42), 2026-08-03, 8 workers: 96 frames (24 h at 15-min),
            0 failed, 2.9 min. Throughput 0.54 five-channel frames/s = 2.76 CMIP
            granule-reads/s at 8 workers, single-worker-equivalent about 2.9 s per
            granule. At this rate a year of 15-min climatology over a
            California-sized region is roughly 18 h at 8 workers. Scope: full
            granule bytes fetched, cropped decode, so representative for regional
            Tier B sizing but not the full-CONUS single-granule figure.
      - **Validation:** the output is physically correct, not just non-empty. The
        C07 centre-pixel diurnal cycle bottoms at 289.7 K in UTC bin 13 (pre-dawn
        local) and peaks at 317.6 K in bin 21 (early afternoon local), confirming
        the UTC-hour binning recovers a real per-pixel diurnal cycle. All 24 bins
        hold 4 samples each; BTs are in sensible ranges with C07 hottest.
      - Storage note: the checkpoint is 206 MB for this one bbox (237x302), which
        extrapolates to about 11 GB for full CONUS. Chunk the climatology per
        region rather than holding CONUS in one accumulator.

All five CMIP decoder steps are done. The decoder, stacking, measurement,
climatology reducer and Tier B backfill are in and tested offline.

## Label spine (Step 3), MTBS first

Plan in `docs/09_LABEL_SPINE_PLAN.md`. Decision settled: ingest MTBS first (US
severity, the T2 training source). The registry vocabulary and the splitting
machinery already existed; this built the middle layer that connects them.

- [x] **Registry persistence + tile assignment.** `EventRegistry.to_parquet` /
      `from_parquet` (the versioned label artifact; lon/lat columns, no geopandas
      dependency, GeoParquet geometry column is an additive upgrade later).
      `vhagar/labels/tiles.py` `assign_tiles` projects an event to the region CRS
      and reads off analysis-grid tile ids, point or bbox, reusing `vhagar.grid`.
- [x] **MTBS adapter.** `vhagar/labels/ingest.py`: pure `normalize_mtbs` (field
      mapping, date-format variants, acres->hectares, Incid_Type to
      wildland/prescribed, dNBR severity path so records are trainable) plus a
      thin `read_mtbs` pyogrio wrapper at the IO edge.
- [x] **Pipeline + CLI.** `vhagar labels build` (ingest -> assign tiles -> write
      registry -> summary) and `vhagar splits build --registry` to materialise
      leakage-proof manifests from the registry. Ten offline tests including the
      full registry -> split-units -> spatial-block/leave-year-out -> no-overlap
      path.
- [ ] **Run on real MTBS** (needs the download + pyogrio): point
      `vhagar labels build --source mtbs --path <mtbs.shp> --severity-dir <...>`
      at a real extract, then `vhagar splits build --registry registry.parquet
      --scheme leave_one_ecoregion_out`.
- [ ] Further adapters (step 4): NIFC/WFIGS perimeters (extent, flagged), EFFIS
      (Europe), Copernicus EMS (held-out European test), FPA-FOD (points, T3).

This unblocks Step 4, the first honest T2 burned-area Stage-0 number.

## T2 burned-area Stage-0 (Step 4), the first honest number

Plan in `docs/10_T2_STAGE0_PLAN.md`. Decision settled: MTBS dNBR predictor first
(lineage-shared, flagged), swap to independent composites later. The algorithms
already existed (`rbr`/`dnbr`, `tune_threshold`/`threshold_baseline`, Olofsson
`estimate_areas` + `allocate_samples`, siamese U-Net, Dice/Combo losses); this
wired them.

- [x] **Dataset builder.** `vhagar/datasets/burned_area.py`: `T2Sample`
      (predictor, reference, valid), `mtbs_burned_mask` (thematic classes ->
      burned/valid), `make_sample` with nodata propagation into the valid mask
      (the classic silent-EO bug, tested), and a `read_mtbs_sample` rasterio
      edge. MTBS dNBR and thematic share a grid, so no regridding for the first
      number.
- [x] **Stage-0 driver.** `vhagar/eval/t2_stage0.py`: per fold, calibrate the
      dNBR threshold on train fires, apply to test, report F1/IoU and an
      **Olofsson error-adjusted burned area with a 95% CI** (stratified reference
      sample, burned class floored, seeded so the CI is reproducible). Per-fold
      results plus mean/std. Eight offline tests on separable synthetic fires:
      the adjusted area lands within ~2% of mapped with a realistic ~7% CI, and a
      perfectly separable map correctly yields a zero CI.
- [x] **First real T2 number, perimeter-vs-severity, CONUS 2021.** The annual
      MTBS mosaic turned out to be thematic-severity only (uint8 classes, no
      continuous dNBR), so instead of the calibrated-threshold baseline (which
      needs per-fire dNBR) we did the perimeter-vs-severity commission analysis
      the architecture explicitly asks for. New `vhagar/eval/t2_perimeter.py`
      (`perimeter_vs_severity`, `class_histogram`) and a `vhagar t2-perimeter`
      CLI. Streamed the whole 14.8 GB CONUS 2021 mosaic in 23 s.
      **Result (burned = classes 2,3,4):** rasterised-perimeter area 3,205,462 ha,
      severity-classified burned 2,622,517 ha, so a rasterised MTBS perimeter
      overstates burned area by **18.2%** (582,944 ha of unburned-to-low and
      increased-greenness islands inside the perimeters). With the lenient
      definition (burned = 1,2,3,4) it drops to 0.4%: the whole commission is
      class-1 "unburned to low", which is the honest, load-bearing caveat. This
      is a census, exact w.r.t. the MTBS severity product and lineage-shared with
      the perimeter. Four offline tests on synthetic histograms.
- [x] **Independent optical Stage-0 pipeline (SOTA path), built and tested.**
      Rather than the lineage-shared MTBS dNBR, the predictor is now Sentinel-2
      RBR computed independently, so calibrating on it and testing against MTBS is
      a real accuracy claim. New `io/optical.py` (SCL cloud mask, masked temporal
      mean composite, NBR, and a STAC + WarpedVRT edge that reads each scene
      straight onto the fire's MTBS 30 m Albers window, folding reprojection and
      windowing into one step) and `datasets/t2_optical.py` (per-fire window
      geometry from area, MTBS reference warped to the same grid, sample
      assembly). CLI `vhagar t2-stage0` runs it leave-one-fire-out through the
      existing driver and reports F1/IoU + Olofsson adjusted area with 95% CI and
      per-fold std. 16 offline tests (masking, compositing, RBR separation, window
      geometry, stubbed assembly); network/rasterio only at the edge.
      - [ ] **Run it** (open network + pystac-client + rasterio, on this machine):
            `vhagar t2-stage0 --registry data\labels\registry.parquet
            --mosaic mtbs_extract\mtbs_CONUS_2021.tif --min-area-ha 5000
            --max-fires 12`. Produces the first independent, leakage-proof,
            CI-bearing T2 accuracy number.
      - Note: the sandbox proxy blocks STAC and the imagery S3 buckets, so the
        Sentinel-2 pull was built and unit-tested but must run on the open network
        here. Everything up to the network edge is verified.
- [ ] Plain U-Net companion baseline (Dice/Combo loss), same eval.
- [ ] Swap predictor to independent S2/Landsat composites for the report number.
- [ ] Leave-one-continent-out (MTBS train, EMSR test) once EFFIS/EMSR ingested.

## Still open

From section 10.6 and the roadmap, not started:

- [ ] Re-measure `DEFAULT_SECONDS_PER_GRANULE` on this machine after the cache,
      via `vhagar archive-plan --measure`. The 14.7 is a conservative pre-cache
      upper bound.
- [x] DEM parallax term. `geo_leo_tolerance_m` now accepts per-pixel elevation
      arrays (the `float()` cast that blocked them is gone) and treats NaN as
      unknown, falling back to the placeholder. New `vhagar/harmonize/dem.py`:
      a `DEM` bilinear sampler in the region CRS (so detections sample by x/y with
      no reprojection), `from_rasterio` loader, and `attach_elevation` to fill
      detections. `Detection` gained an optional `elevation_m` used in
      `tolerance_m`. Nine offline tests.
- [ ] CMIP decoder. Without it the radiance tier cannot be built and its wall
      clock is unmeasured.
- [x] Parquet small-file compaction. `vhagar/archive/compaction.py`,
      `compact_detections`, plus a `vhagar compact` CLI. Merges each tile's
      per-day files into one file per year. Safe (verify merged row count and the
      on-disk count before deleting any original, atomic replace) and idempotent
      and incremental (folds new day files into the compacted one). Six offline
      tests including row-preservation, idempotency, incremental, and dry-run.
- [x] Initialise git and push to GitHub. Done: v0.12 pushed to
      github.com/Ibekwemmanuel7/VHAGAR, main tracking origin/main. The old v0.4
      snapshot and its `_to_delete/` zips were replaced by the clean tree.
- [x] `_write_day` partial-day resume risk fixed. It now reads back each day
      file, drops only the rows of the granules being written this call (so a
      re-read replaces its own rows and does not duplicate), keeps every other
      granule's rows, and writes via a temp file plus atomic replace. Three
      tests: partial-day resume preserves earlier granules, cross-run retry does
      not duplicate, same-day idempotency still holds. (Noticed while planning
      the day-215 repair.)

## Roadmap after the fixes (section 11 of the brief)

1. Step 2 Tier A at scale: multi-year FDC backfill, resumable.
2. Step 2 Tier B radiance: needs the CMIP decoder first.
3. Step 3: label spine (MTBS, NIFC), event registry, split manifests.
4. Step 4: Stage 0 baseline on T2 burned area (first honest number).
5. Step 5: T1 Stage 0 once the archive has about a year of depth.

## External questions someone needs to answer

- LP DAAC: does a NOAA-20 or NOAA-21 burned area product replace `VNP64A1`?
- JRC: has EFFIS/GWIS migrated off S-NPP before 2026-11-01?
- S-NPP NRT delivery ceases 2026-11-01. Pull what you want into the corpus
  before then.

## Gotchas to remember

- No em dashes in writing or code. Commas, colons, semicolons, full stops.
- Every measurement must state what it measured. Never compare two numbers
  unless they walked the same code path. This is what the retracted latency
  claim got wrong.
- `pip install h5netcdf` does not pull `h5py` on Windows.
- Never print only `type(exc).__name__` in an error handler.
- xarray auto-decodes CF time, so ABI `t` arrives as datetime64. Adding it to
  the ABI epoch as raw seconds overflows timedelta.
- If you ever strip dashes with a script, replace dash characters only. A past
  script destroyed Python's `...` Ellipsis across nine files.
- A benchmark that finishes in half a second has measured nothing.
- `setx` on Windows needs a new shell. Use `$env:VAR = "..."` for the current one.
