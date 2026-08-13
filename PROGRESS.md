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
- [ ] **Step 3: measure the wall clock.** Update `plan.measure_granule` to decode
      CMIP for real, then set the CMIP granule size and seconds-per-granule in
      `plan.py` from a measurement on this machine.
- [ ] **Step 4: climatology reducer.** Welford mean and variance per pixel and
      per local hour.
- [ ] **Step 5: Tier B backfill** reusing the manifest and coverage machinery.

## Still open

From section 10.6 and the roadmap, not started:

- [ ] Re-measure `DEFAULT_SECONDS_PER_GRANULE` on this machine after the cache,
      via `vhagar archive-plan --measure`. The 14.7 is a conservative pre-cache
      upper bound.
- [ ] DEM to replace the 1000 m elevation placeholder in `geo_leo_tolerance_m`
      (parallax term).
- [ ] CMIP decoder. Without it the radiance tier cannot be built and its wall
      clock is unmeasured.
- [ ] Parquet small-file compaction step. Will matter after a year of data.
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
