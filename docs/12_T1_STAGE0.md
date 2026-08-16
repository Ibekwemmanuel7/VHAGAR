# T1 Stage-0: GOES FDC active-fire detection vs VIIRS truth

The detection analog of the T2 Stage-0 burned-area baseline. GOES-18/19 ABI FDC gives
a candidate fire pixel every five minutes at 2 km; VIIRS on the polar platforms gives
the reference truth twice a day at 375 m. Stage-0 asks the honest question: matched at
the **event** level with a geometry-aware tolerance, what are the probability of
detection (POD), the false-alarm rate (FAR), and the detection latency?

## First real result (GOES-18 CONUS, 2026-08-01..07, vs VIIRS NOAA-20 + S-NPP)

Detection-level coincidence: a VIIRS fire detection counts as seen by GOES when a GOES
detection lies in the same spatial cell (8-neighbour) within +/-30 min, restricted to
the GOES sector. 104,772 VIIRS detections in domain.

| matching | cell | POD | median gap |
|---|---|---|---|
| naive | 2 km | 0.376 | 2 min |
| **parallax-aware** | **4 km** | **0.499** | **2 min** |

And the precision / false-alarm side, scored only on GOES detections VIIRS was actually
overhead for (a GOES detection is *evaluable* when some VIIRS detection is within ~50 km
and +/-30 min, so a real fire between the twice-daily overpasses is not miscounted as a
false alarm). 30,800 of 188,639 GOES detections are evaluable:

| matching | cell | precision | FAR |
|---|---|---|---|
| naive | 2 km | 0.843 | 0.157 |
| **parallax-aware** | **4 km** | **0.944** | **0.056** |

**This is the architecture's headline geometry number, reproduced.** The apparent FAR
drops from 15.7% (naive 2 km) to 5.6% (parallax 4 km), landing squarely in the published
"naive 26-36% -> parallax 7-15%" range. The 10-point drop is footprint quantisation plus
terrain parallax, not model error: a too-tight cell mislabels an offset-but-real GOES/
VIIRS match as a false alarm. Conditioning on VIIRS coincidence is what makes precision
interpretable at all for a GEO/LEO pair.

Two things to read from the POD side:

- **GOES FDC detects about half of VIIRS's fire pixels** (POD ~0.50 at the parallax
  scale), near-simultaneously (median gap 2 min). That is the credible, literature-
  consistent number: GOES at 2 km misses the small/cool fires VIIRS's 375 m catches, so
  a POD around 0.4-0.6 is exactly what a geostationary detector scores against a polar
  one, not a bug.
- **The +0.12 POD from 2 km to 4 km is geometry, not model quality.** A nominal 2 km ABI
  pixel covers >13 km2 at high view zenith (effective side ~3.6 km) and terrain parallax
  displaces an elevated fire by ``h * tan(vza)``, so the GOES detection sits offset from
  the VIIRS location. Matching at the footprint+parallax scale recovers 12 points of POD
  that naive 2 km matching throws away. This is the T1 twin of the T2 lesson: a naive
  default (there, the discarded-unburned reference; here, a too-tight match) manufactures
  a worse number than the sensor deserves.

### A broken first metric, and why it was caught

The first cut clustered detections into events and matched event **centroids** with a
3.6 km tolerance, and pulled VIIRS over a huge bbox (down to Guatemala). It reported
POD 0.047, which is absurd for two fire sensors over the same week. The naive-baseline
instinct from T2 caught it: a 4.7% match rate is a broken metric, not a real detection
rate. Diagnosis: (1) VIIRS spanned tropical/agricultural fires outside the GOES-CONUS
sector, counted as misses; (2) matching 50 km-cell cluster centroids at 3.6 km
guarantees misses (median nearest-centroid distance was 276 km); (3) an int-unit bug in
an exploratory grid check gave a spurious 0.0. The fix is detection-level coincidence in
space **and** time, domain-restricted, above. Precision/FAR then need one more
correction, conditioning on VIIRS overpass coincidence (only score a GOES detection when
VIIRS was overhead), which is the precision/FAR table above; without that conditioning a
real fire between overpasses would be miscounted as a false alarm.

## What is built (this pass)

``src/vhagar/eval/t1_stage0.py``, all pure and unit-tested:

- ``match_events``: one-to-one greedy event matching, parallax-aware or flat-tolerance,
  with a temporal-overlap window. Returns TP/FP/FN.
- ``DetectionScores``: POD (=recall), FAR (=FP/(TP+FP)), precision, F1. POD and FAR are
  always reported together, since POD alone is gamed by flagging everything.
- ``detection_latency_minutes``: median (and IQR) lead time of GOES over the VIIRS
  overpass for matched events, the point of a geostationary backbone.
- ``load_fdc_events`` (thins the 5-min repeats to a ~2 km/hourly grid, then clusters in
  coarse spatial cells so a week runs in ~1 min without a spatial index) /
  ``firms_to_detections``: project GOES FDC parquet and
  FIRMS/VIIRS records into one equal-area frame (EPSG:5070) and cluster into events
  (reusing ``fusion.cluster_detections``, whose tolerance is view-zenith aware).

CLI: ``vhagar t1-stage0 --detections data/detections/detections [--firms-csv viirs.csv]``.
Without a FIRMS CSV it summarises the GOES side; with one it reports the table above.

Verified on the real FDC parquet on disk (GOES-18, CONUS, Aug 2026): FDC -> Detection
-> events works, and the parallax-aware match tolerance runs ~3.6 km median at CONUS
view zeniths, four times a flat 2 km, which is exactly why the naive FAR is inflated.

## Runbook: first real POD/FAR/latency

```
# 1. free FIRMS map key: https://firms.modaps.eosdis.nasa.gov/api/map_key/
$env:FIRMS_MAP_KEY = "<your key>"
# 2. pull the VIIRS truth for exactly the GOES window (reads dates/bbox from the FDC parquet)
vhagar firms-fetch --detections data\detections\detections --out viirs_truth.csv
# 3. score
vhagar t1-stage0 --detections data\detections\detections --firms-csv viirs_truth.csv
```

``firms-fetch`` reads the FDC window (here 2026-08-01 to 2026-08-07, CONUS+HI bbox) and
pulls the matching VIIRS in <=10-day chunks; ``t1-stage0`` then reports the parallax-
aware vs naive-2 km table above. The FDC window is one week, so the ball-tree is not
needed for this first run; per-tile clustering handles it.

## Stage-2 preview: does raw lat/lon leak? (`t1-classify`)

The architecture's central T1 warning: in a published FIRMS classification, raw
coordinates gave ~89% of a classifier's gain while *harming* out-of-region transfer,
F1 0.985 (random) -> 0.767 (event-aware) -> 0.627 (5-degree block). ``t1-classify``
reproduces the phenomenon on our GOES-18 FDC + VIIRS week: each GOES detection is a
sample, labelled 1 when VIIRS coincides with it in space and time, and a gradient-
boosted classifier is trained with and without raw lon/lat under three splits.

| split | physical F1 | + lat/lon | lat/lon gain |
|---|---|---|---|
| random | 0.767 | 0.790 | +0.023 |
| cell-grouped (event-aware) | 0.752 | 0.778 | +0.026 |
| 5-degree spatial block | 0.642 | **0.602** | **-0.040** |

Two things reproduce, qualitatively:

- **The generalisation gap.** F1 falls from 0.767 (random) to 0.642 (spatial block),
  the same shape as the published 0.985 -> 0.627: a classifier that looks good when it
  can see nearby locations in training is worse when whole regions are held out.
- **Raw lon/lat leaks.** Its gain is positive in-region (+0.03) and turns **negative**
  out-of-region (-0.04): the coordinates memorise where fires are confirmed, which helps
  on a random split and *hurts* transfer to a new 5-degree block. That is precisely why
  production T1 features (``fusion.event_features``) exclude raw coordinates.

Honest caveat on magnitude. Our effect (a ~0.07 swing in the lat/lon gain) is far
smaller than the published 89%-of-gain, because one week of CONUS FDC with a VIIRS-
coincidence label is a weak, timing-influenced proxy (VIIRS-confirmed rate is only 3%),
not the balanced, multi-region wildfire/non-wildfire dataset the published study used.
The *direction* is the finding; the magnitude needs more data and a cleaner label. A
synthetic unit test (``test_latlon_leakage_helps_on_random_and_collapses_out_of_region``)
confirms the framework registers a large leak when one is present, so the modest real
number is the data's, not the tool's.

## Stage-1 differentiator: temporal-anomaly early detection (`t1-temporal`)

The contextual algorithms (GOES FDC, SLSTR FRP) flag a fire when its brightness
temperature crosses an *absolute* threshold. That threshold has to sit above the midday
diurnal peak to avoid false alarms, so a fire, especially a night fire starting cold, is
caught late. The architecture's one learned Stage-1 component forecasts the *expected*
per-pixel BT from recent history plus solar geometry and flags **residual** excursions;
because the residual is measured against each pixel's own diurnal baseline, a fire stands
out as soon as it lifts BT above that baseline, not when it crosses a global cut.

``t1-temporal`` makes the mechanism concrete on a synthetic 3.9 um series with a night
fire injected, comparing the residual detector to an absolute-BT threshold **calibrated
to the same false-alarm rate** (the only fair comparison):

| target FAR | residual detects (min after onset) | absolute detects | lead |
|---|---|---|---|
| 0.05 | 0 | 75 | **+75 min** |
| 0.01 | 5 | 75 | **+70 min** |
| 0.002 | 15 | 85 | **+70 min** |

The residual detector flags the night fire within 0-15 minutes of onset while the
absolute cut waits ~75-85 minutes for it to cross the contextual threshold, a **~70
minute lead at equal FAR**. That reproduces the published evidence (porting a rapid-scan
temporal algorithm to 5-min geostationary data roughly doubled mean lead time, 35 -> 65
min) and its mechanism. The magnitude here is synthetic and tunable (it scales with the
fire ramp rate and diurnal amplitude); the *direction and cause*, diurnal-baseline
removal buys lead time, are the finding.

### Grounded in the real 3.9 um climatology

The synthetic magnitude is tunable, but the *reason* it works is a real, measurable
quantity: how far the 3.9 um brightness temperature actually swings between night and
day. That gap is what an absolute threshold has to clear, so it is the sensitivity a
diurnal-baseline detector recovers. Measured on the on-disk per-pixel, per-UTC-hour C07
(3.9 um) climatology (``DiurnalClimatology``, GOES-18, N. California, 71,574 pixels):

| C07 diurnal amplitude (max-hour mean - min-hour mean) | value |
|---|---|
| median | **32.9 K** |
| p25 / p90 | 23.4 / 45.5 K |

So the real 3.9 um diurnal amplitude is about **33 K** (median), rising to ~46 K at the
90th percentile. An absolute contextual threshold must sit roughly that far above each
pixel's night baseline to avoid firing on the midday peak, meaning it is ~33 K less
sensitive to a cold-start night fire than a residual detector that removes the diurnal
cycle first. That is the mechanism of the synthetic lead-time table above, expressed as a
real number rather than a chosen ``diurnal_amp``. ``vhagar t1-temporal --climatology
data/climatology/climatology.npz`` prints it (``climatology_diurnal_amplitude``). Honest
caveat: the per-hour sigma the climatology reports (~0.5 K) is thin, each UTC-hour bin
holds only ~4 samples in this backfill window, so the amplitude (a difference of hourly
means, robust to sample count) is the number to trust, not the amplitude-in-sigmas.

What is built: the numpy pieces (``DiurnalForecaster``, matched-FAR calibration, the
lead-time experiment, ``climatology_diurnal_amplitude``) run and are unit-tested
anywhere; the production forecaster is
``models.TemporalAnomalyNet`` (a 3D-conv TCN forecasting the next BT frame, trained on
clear-sky history, no fire labels), wired by ``train_temporal_net`` for real data. Only
the forecaster changes; the residual / matched-FAR / lead-time protocol is identical.

To run it for real, pull GOES ABI CMIP band 7 (3.9 um) at 5-min cadence over a region and
period as a ``[T, H, W]`` cube, train ``TemporalAnomalyNet`` on the clear-sky span, and
feed its residuals to ``early_detection_experiment`` against the FDC first-detection
times. Needs torch and the CMIP pull; that is the one heavy piece left for this component.

## Open items

- **Reference pull** is now one command (``firms-fetch`` above); it just needs the free
  FIRMS map key and network. Until it is run, POD/FAR/latency are framework, not numbers.
- **Scale.** The single-link clusterer is O(n^2); a full month of CONUS FDC (~4x10^5
  detections) needs the ball-tree the ``fusion`` docstring already flags. Per-tile
  clustering bounds it somewhat but is not enough for the full archive. Bound by date
  window or swap the index before running the whole month.
- **Splits and the lat/lon-leakage story.** The event-level wildfire/non-wildfire
  classifier (Stage-2) with the random -> event-aware -> spatial-block degradation
  (0.985 -> 0.767 -> 0.627) reuses ``eval/splits`` and ``fusion.event_features`` (which
  already excludes raw lat/lon); it is the next rung, once labelled events exist from
  the VIIRS match.
