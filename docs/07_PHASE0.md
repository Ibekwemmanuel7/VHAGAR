# VHAGAR: Phase 0 Execution Plan

**From "designed" to "first honest number."**
Roughly 6-8 weeks. Everything before this is apparatus.

---

## Where you are

Done: architecture, physics (Planck / atmosphere / Wooster FRP / Dozier /
geometry), the evaluation contract, the sensor registry, event fusion, 169 tests.

Not done: a single real observation in the system, one harmonised label, one
split manifest built from real fire events, one trained model, one number you
could defend.

Phase 0 closes that gap. It is deliberately ordered by *dependency*, not by
interest.

---

## Step 0: Accounts and keys (30 minutes, do it first)

| What | Where | Cost | Needed for |
|---|---|---|---|
| **FIRMS map key** | https://firms.modaps.eosdis.nasa.gov/api/map_key/ | free | VIIRS detections, step 1 |
| **Earthdata Login** | https://urs.earthdata.nasa.gov/ | free | MTBS, VIIRS archive, LAADS |
| **Copernicus Data Space** | https://dataspace.copernicus.eu/ | free | Sentinel-2, Sentinel-3 SLSTR FRP |
| **Earth Engine project** | https://code.earthengine.google.com/ | free non-commercial | label generation, compositing |
| Nothing at all | `s3://noaa-goes18`, `s3://noaa-goes19` | free, no credentials | **all GOES work** |

Windows terminal:

```bat
setx FIRMS_MAP_KEY your_key_here
```

⚠️ Earth Engine's free tier is **non-commercial only**. If VHAGAR is ever
operated commercially, price your EECU profile before letting a GEE dependency
reach the product path. Nothing in Phase 0 depends on GEE.

---

## Step 1: First light ✅ *built, ready to run*

**Goal: one real GOES detection and one real VIIRS detection in the same event
object.** ~1 hour.

```bash
pip install -e ".[dev]"
pip install s3fs xarray h5netcdf pyproj

python scripts/step1_first_light.py --bbox -124 36 -118 42 --satellite 18 --hours 6
```

Or run `notebooks/01_first_light.ipynb` in Colab, which adds plots.

**What it does.** Lists GOES ABI L2 FDC granules from NOAA's public S3, crops
each to your bbox *in scan-angle space before decoding*, converts the ABI fixed
grid to lat/lon with the GOES-R PUG algorithm, attaches view zenith and true
pixel area to every detection, pulls VIIRS from FIRMS, and fuses both through
the parallax-aware clusterer.

**Four things to look at, in order of how much they will change your plan:**

1. **The mask breakdown.** What fraction are low-probability (code 15) versus
   good (10/30)? That ratio is your precision/recall dial, and you set it.
2. **GEO/LEO separation.** Does the 2 km parallax tolerance cover the bulk of
   the distribution? Naive nearest-pixel matching reports 26-36 % apparent
   false alarms; a 3×3 buffer drops it to 7-15 %.
3. **Single-sensor events.** Your false-alarm suspect pool, and the population
   Stage-2 persistence features exist to classify.
4. **View zenith range.** Above ~60°, uncorrected FRP is low by >2×.

**Normal outcomes that are not failures:** zero detections (no fire in the box. 
check https://firms.modaps.eosdis.nasa.gov/map/ first), or a few skipped
granules (S3 hiccups; the reader logs and continues).

---

## Step 2: Tile archive backfill 🔴 *start this week*

**Goal: 3+ years of dense GOES time series over selected tiles.** Runs
unattended and parallelises with everything else. Measured in v0.9 at roughly
20 hours at 12 workers, not the weeks originally scoped.

**The scoping insight that makes this tractable:** you do not need the whole
archive. Stage-2 persistence features and the thermal foundation model both need
*dense time series over small areas*, not full-disk coverage. Full CONUS FDC for
5 years is terabytes; **a few hundred 256×256 tiles at 5-minute cadence is tens
of gigabytes**, which fits a laptop disk and a Colab session.

**The second scoping insight, measured in v0.9:** FDC and radiance are not the
same download problem and should not be one tier.

Both figures below are **measured** on Chidi's connection, not assumed.

| | granule size | read time | 500 tiles × 3 yr × 5 min | wire |
|---|---|---|---|---|
| **FDC** (mask, power, area, temp) | **0.32 MB** | 0.78 s | 0.6 GB as detection rows | **0.10 TB** |
| **CMIP** (radiance, one file per band) | **4.41 MB** | 0.75 s | 182 GB as int16 raster | 2.4 TB |

Three things fall out of that table.

**FDC is 14x smaller on the wire** because it is a sparse product: nearly the
whole grid is fill and compresses away.

**FDC is also sparse on disk**, so storing detections as rows rather than as a
raster takes the same coverage from 182 GB to 0.6 GB. The cost is that you no
longer have an explicit negative field, which is why the coverage record below
is not optional.

**Read time barely moves between a 0.32 MB granule and a 4.41 MB one.** Do not
read that as "bytes do not matter", which is the conclusion I drew and then had
to withdraw: the two timings are not comparable. The FDC number includes HDF5
parse and navigation, the CMIP number is a raw byte read with no decoder
attached. A bare S3 read of an FDC granule is about **0.12 s**; the full path
the backfill walks is about **0.75 s**. So roughly six sevenths of the
per-granule cost is not the network, and it will track your CPU rather than
your connection. Size concurrency with `vhagar probe-workers --mode full`,
never on a fetch-only proxy.

A note on the 4.41 MB: that is a **2 km** channel at CONUS extent. All five
bands VHAGAR needs (C07, C11, C13, C14, C15) are 2 km, which is the only
reason the radiance tier is affordable. C02 at 0.5 km would be 16x the pixels,
and Full Disk is roughly 6x CONUS. If you ever add a visible band for smoke
context, re-run the sizing first.

The whole three-tier backfill now lands near **3.5 TB and 20 hours at 12
workers**. That is a long weekend. Sequencing below is therefore about getting
a usable dataset early, not about affording one at all.

**Tier A, detection history. Built in v0.10, start it now.**

```bash
pip install -e ".[archive]"

# 1. Find where concurrency stops helping. Takes a few minutes, saves hours.
#    Defaults to timing fetch+decode, which is what the backfill actually does.
#    If it finishes in under a few seconds it has measured noise: raise
#    --n-granules until it does not.
vhagar probe-workers --candidates 1,4,8,16,32,64

# 2. A short trial run first. Check the numbers before committing to years.
vhagar backfill data/detections --start 2026-08-01 --end 2026-08-07 --workers 16

# 3. The real thing. Safe to interrupt; re-run the identical command to resume.
vhagar backfill data/detections --start 2023-01-01 --end 2026-08-01 --workers 16
```

What it writes:

- `detections/year=YYYY/tile=conus_xNNNN_yNNNN/part-YYYYMMDD.parquet`, one file
  per tile per day rather than per granule, because five-minute cadence would
  otherwise leave 288 tiny files per tile per day and the Parquet metadata
  would outweigh the data.
- `_manifest.jsonl`, **the coverage record**, appended after every granule
  whether it succeeded or failed. This is not bookkeeping, it is half the
  dataset. A tile with no rows at 14:35 means either "observed, nothing
  burning" or "that granule was never read", and those are opposite facts.
  `coverage_intervals()` turns the manifest back into the observed periods a
  loader needs to mine honest negatives.
- `_config.json`, a fingerprint of the settings. Running a second backfill with
  a different bbox into the same directory is refused, because the manifest
  would then claim coverage the rows do not support.

Order matters inside the run: rows are written **before** the manifest line. A
crash between the two causes the granule to be re-read and its rows rewritten
to the same path, which is idempotent. The other order would record coverage
for rows that are not on disk, and that lie is unrecoverable.

Detections are stored at native ABI pixel centres with a tile ID attached, not
resampled onto the 375 m analysis grid. Resampling a point detection onto a
finer grid invents precision the instrument never had.

This tier alone unblocks persistence features, diurnal detection statistics and
the event history that T3 and T4 both need.

**Tier B, radiance. The bulk of the wire cost, so scope it from Tier A.**

- Select ~200-500 tiles: stratify across biome × fire-frequency × view-zenith
  band, and deliberately include **clean negatives**, hot desert, solar farms,
  gas flares, coastlines, sun-glint water. Those are your operational failure
  modes and they must be in the corpus. Tier A tells you which tiles actually
  burn, so run it first and let it inform the stratification.
- For each tile pull ABI C07 (3.9 µm), C11, C13, C14, C15, the bands the
  temporal anomaly model needs. Remember these ship **one file per channel**,
  so 5 bands is 5 S3 reads per timestep, not one.
- Store as **int16 brightness temperature** with per-band scale/offset in
  chunked Zarr. Never float32; you would double storage for precision that
  does not matter.
- 15-minute cadence for the wide climatology, 5-minute only for a narrow
  fire-season slice. Cadence is the steepest cost gradient in the whole plan.

Tier B must be resumable the same way Tier A already is. It will be
interrupted.

**Where to run it:** Colab sessions time out, so use Colab for interactive work
and your Windows box (or a small cheap VM) for the long backfill. If you ever
move to AWS `us-east-1`, GOES reads become free and same-region fast. At 3.5 TB
that is now a question of whether you want the 20 hours to be 3, not a necessity.

⚠️ Also this week, unrelated to compute: **pull whatever S-NPP data you want in
the corpus.** NRT delivery ceases 2026-11-01. The LAADS archive should persist,
but verify rather than assume.

---

## Step 3: Label spine and the first real split manifest 🔑 *gates everything*

**Goal: an event registry and a versioned, leakage-proof split built from real
fires.** ~2 weeks. **Nothing can be trained or measured before this exists.**

1. Ingest MTBS (continuous dNBR/RBR. **not** the thematic class, which is set
   per fire by analyst review and is not comparable across fires) plus
   NIFC/WFIGS perimeters, for 2 or 3 recent seasons.
2. Normalise into `labels.registry.FireEventRecord` with an explicit
   `LabelQuality`. Hold **Copernicus EMS, NIROPS and CBI plots out entirely**. 
   they are the highest-quality geometries you have and worth far more as an
   unseen test set.
3. Build split manifests and commit them:

```bash
vhagar splits build --records events.json --scheme leave_one_group_out --out splits/
vhagar splits build --records events.json --scheme spatial_block --out splits/ --n-folds 5
vhagar splits verify splits/leave_one_group_out.json
```

4. **Run the relabel audit.** This is the single most valuable practice
   available to you, and the prior-art platform proved its worth: it found
   ~6,610 mislabelled pixels and took a headline false-alarm precision from
   99.86 % to an honest 90.6 %. Score your labels against evidence that shares
   no features with them, and publish the number that makes you look worse.

**Do not** train a pixel model on rasterised perimeters without an interior
severity mask. ~9 % of the area inside a typical perimeter is unburned islands,
and the error is spatially structured. `assert_trainable()` will refuse.

---

## Step 4: Stage 0 baseline: T2 burned area 🎯 *your first honest number*

**Goal: one defensible, CI-bounded burned-area product.** ~2-3 weeks.

T2 first, not T1, best labels, no latency engineering, the Olofsson machinery
is already coded, and Prithvi-BurnScars is free performance. T1 needs the Step 2
archive to exist before its best features do.

In order:

1. **Calibrated RBR threshold**, tuned on training folds only
   (`eval.baselines.tune_threshold`). This is your permanent baseline and a
   large fraction of published deep learning never beats it.
2. **Plain U-Net** on Sentinel-2 pre/post mean composites with the
   unburned-buffer offset correction. Use **Dice or combo loss, not
   cross-entropy**, in the one published geostationary fire benchmark that
   swap took a U-Net's fire IoU from 0.022 to 0.272, a 12× gain that dwarfs
   what pretraining bought anyone.
3. **Fine-tune Prithvi-EO-2.0-300M-BurnScars** (Apache-2.0, 87.5 % IoU out of
   the box). If it does not beat the U-Net, that is a result worth reporting. 
   on PANGAEA a plain U-Net beat Prithvi on HLS Burn Scars, 84.51 vs 83.62 mIoU.
4. **Report properly**: per-fold, with all three split schemes, mandatory
   baselines, and burned area as an **Olofsson error-adjusted estimate with a
   95 % CI**, never a pixel count.

**Exit criterion:** a table of the form
`task | model | split | n folds | metric ± fold sd | baseline | Δ`
where the Δ survives leave-one-continent-out (train MTBS → test Copernicus EMS).
Expect F1 around 0.735 ± 0.186 on that transfer. The ±0.19 *is* the finding.

---

## Step 5: T1 Stage 0, once the archive has depth

**Goal: median detection latency against reported ignition, beating the
operational product at equal false-alarm rate.** Starts when Step 2 has ~1 year.

1. Physics features from `features.physics_features`, 34 of them, with the
   coordinate guard on.
2. Causal persistence features from the archive (Step 2's whole purpose).
3. **LightGBM event classifier**, cause-agnostic first.
4. Only then the temporal anomaly model on the GOES 3.9 µm series.

Primary metric is **median minutes from agency-reported ignition to first
alert** at a fixed acceptable false-alarm rate, not pixel IoU.

---

## Two blocking queries to send now (not code)

1. **LP DAAC:** does a NOAA-20/21 VIIRS burned-area product replace the
   S-NPP-based `VNP64A1`? If not, the NASA standard burned-area record ends in
   November 2026 and your T2 label strategy needs a European fallback
   (Fire_cci Sentinel-3 SYN).
2. **JRC:** has EFFIS/GWIS migrated off S-NPP? Their page still names it. If
   not, the European operational feed silently degrades on 2026-11-01.

---

## Sequencing at a glance

```
week   1    2    3    4    5    6    7    8
Step 0 ▓
Step 1 ▓▓
Step 2 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← unattended, starts week 1
Step 3      ▓▓▓▓▓▓▓▓▓
Step 4                ▓▓▓▓▓▓▓▓▓▓▓▓▓
Step 5                                    ▓▓▓▓ →
```

Step 2 starting in week 1 is the only scheduling decision that matters. Everything
else can slip; the archive cannot be compressed later.

---

## What "done with Phase 0" looks like

- [ ] `step1_first_light.py` has run and you have looked at all four diagnostics
- [ ] Tile archive is accumulating, resumably, with a completion manifest
- [ ] Event registry holds ≥2 seasons of MTBS + NIFC, with quality labels
- [ ] Committed split manifests that pass `vhagar splits verify`
- [ ] Relabel audit run, with its uncomfortable number written down
- [ ] One burned-area model with per-fold metrics, mandatory baselines, and an
      Olofsson area estimate with a confidence interval
- [ ] Both blocking queries answered
