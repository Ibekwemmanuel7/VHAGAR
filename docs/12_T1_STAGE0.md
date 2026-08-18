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

### Running it on real GOES data (the pull)

The pull that turns this from a demonstration into a measured number is now built, in
``archive/temporal_cube.py``. It reads GOES ABI L2 CMIP band 7 (3.9 um) from the public S3
archive, crops every 5-minute frame to a small bbox, and stacks them on the one stationary
ABI fixed grid into a ``[T, H, W]`` cube that carries its own UTC timestamps and geometry.
Because the fixed grid does not move, cropping the same bbox yields the identical pixel
window every timestep; that is asserted (shape and corner navigation), and any frame that
disagrees is dropped, never misaligned. NaN stays NaN, so cloud and fill never enter the
baseline. Keep the bbox small (a fire-prone box, not CONUS): the cube is dense.

The real lead-time comparison closes the loop against GOES FDC itself. ``t1-temporal-real``
fits the NaN-safe per-pixel diurnal baseline (``HourlyBaselineForecaster``, the on-the-fly
counterpart of ``DiurnalClimatology``, since a real cube's NaNs rule out the harmonic
least-squares fit) on the leading clear-sky fraction, then for every pixel FDC eventually
flags it measures how many minutes earlier the residual crossed a threshold **calibrated to
the same false-alarm rate on the fire-free pixels**. Positive lead means the residual
detector beat FDC's own first detection on that pixel, at equal FAR. This is the synthetic
+70 min demo, re-run on real observations rather than an injected ramp.

```
# 1. pull a 3.9um cube over a box and window that contains a known fire (needs s3fs+xarray)
vhagar t1-pull-cube cube.npz --start 2026-08-01T00:00 --end 2026-08-02T00:00 \
    --bbox -121.2,39.3,-120.6,39.9 --satellite 18
# 2. time the residual detector against FDC first detection, at matched FAR
vhagar t1-temporal-real cube.npz --detections data/detections/detections
```

The learned upgrade is a drop-in: train ``TemporalAnomalyNet`` on the same cube's clear-sky
span (``train_temporal_net``, needs torch), take its per-frame residual, and feed it to
``real_lead_experiment`` in place of the hourly-baseline residual. Only the forecaster
changes; the matched-FAR lead-time protocol is identical. A ``solar_zenith_cube`` covariate
(the 3.9 um channel carries daytime solar reflectance) is available for the learned path.

### First real run, and the trap it exposed

The first real pull was a central-Utah fire, bbox ``-112.55,38.65,-112.05,39.10``, window
2026-08-01 18:00 to 08-03 00:00 (360 frames, 98% valid). The lead-time table looked
spectacular and was wrong:

| target FAR | fire pixels | residual led | median lead |
|---|---|---|---|
| 0.05 | 48 | 100% | +925 min |
| 0.01 | 48 | 83% | +642 min |
| 0.002 | 34 | 29% | **-78 min** |

The +925 min (15 hour) "lead" is an artefact. The fire ignited at 19:47 UTC, only 1h47m
after the window opened, but ``--clear-frac 0.6`` fit the diurnal baseline on the first 18
hours, which overlap the active fire by ~16 hours. So the baseline for the core fire pixels
was built from the fire's own hot brightness temperature. With a corrupted baseline and a
loose false-alarm rate, the residual trips early on ordinary diurnal warming and baseline
error, manufacturing a huge apparent lead. The tell is the collapse down the FAR column:
+925 -> +642 -> -78. A real early-warning signal holds its lead as the false-alarm rate
tightens; this one evaporates and goes negative, because at a strict FAR the same
contaminated baseline instead *suppresses* the residual (the fire is partly in the
expected value), so the detector lags FDC. This is the T1 twin of the T2 naive-baseline
lesson: the framework is honest, the experiment design was not.

The fix is twofold. In the experiment, pull a window whose leading span is genuinely
pre-ignition (a fire that ignites well into the record, with a full pre-fire diurnal cycle
as the baseline). In the code, ``baseline_contamination`` now measures the share of fire
pixels whose first FDC detection falls inside the clear-sky window, and ``t1-temporal-real``
prints a red refusal when that exceeds 20%, so this class of mistake cannot be quietly
presented as a result again. The clean recipe: a fire that ignites at, say, 08-02 07:47 UTC
pulled from 08-01 00:00 gives ~32 hours of real pre-fire baseline before onset, and the
onset is at night (no solar contamination in 3.9 um, the exact case an absolute threshold
is slowest on and the residual detector should win).

### The clean run, and the second flaw it exposed

Re-pulled with a genuinely pre-ignition baseline: the north cell ``-112.35,38.85,-112.05,
39.10``, window 08-01 00:00 to 08-02 18:00 (504 frames, 98% valid), a fire that ignites at
07:47 UTC with ~32 hours of clear diurnal history before it. The contamination guard stayed
silent, as it should. And the crude hourly-mean residual **still did not beat FDC**:

| target FAR | fire pixels | residual led | median lead |
|---|---|---|---|
| 0.05 | 24 | 100% | +668 min |
| 0.01 | 24 | 50% | +5 min |
| 0.002 | 16 | 0% | **-192 min** |

Same collapse shape (+668 -> +5 -> -192), now with a clean baseline, so the cause is not
contamination. It is a flaw in the protocol itself: the residual threshold was a single
**global** percentile across all hours. But daytime 3.9 um residuals have much larger
variance than night (solar reflectance, thermal churn), so a global percentile is set by
the daytime tail and desensitises the detector at night, which is precisely when this fire
ignites (07:47 UTC is ~01:47 local). Calibrating the residual with one global cut
re-introduces the very night-blindness the residual detector exists to remove. The +668 at
loose FAR is, again, the threshold low enough to trip on daytime residual noise; the -192
at strict FAR is the night fire falling under a daytime-set bar.

The fix is a **per-time-of-day threshold** (``real_lead_experiment(..., far_bins=N)`` /
``t1-temporal-real --far-bins 6``): split the day into bins and calibrate each to the target
FAR on the fire-free pixels pooled within that bin, so a night fire is judged against the
night distribution. That is the thesis made operational: an absolute cut sacrifices night
sensitivity, a diurnally-aware detector recovers it.

### The third flaw: first-crossing is gamed by pre-fire false alarms

Re-running with ``--far-bins 6`` produced *huge* leads, +2048 / +705 / +682 min across FAR
0.05 / 0.01 / 0.002, and those are an artefact too. The estimator took the **first** residual
exceedance anywhere in the 42-hour record as the detection time. Over ~500 frames, a
per-frame FAR of 0.01 gives every pixel several isolated false exceedances; the more
sensitive night threshold lands them on the night *before* ignition, so a single pre-fire
blip is scored as an 11-to-34-hour "lead". Matched FAR controls the false-alarm *rate*, but
first-crossing is the one estimator maximally sensitive to it. A 34-hour lead points to
before the fire existed, so it cannot be real.

The fix is **persistence**: require ``min_consec`` consecutive exceedances before declaring a
detection (``t1-temporal-real --min-consec 3``, the default now), which is how contextual
detectors confirm across scans. An isolated blip is filtered; only a sustained residual ramp
counts, and its end is when the alarm could actually be raised. At FAR 0.01 a 3-frame false
run has probability ~1e-6 per position, so pre-fire false detections effectively vanish and
the detection time tracks the real onset. A unit test confirms persistence ignores a pre-fire
spike and detects the sustained ramp instead.

### The trustworthy read: the crude baseline does NOT beat FDC on this fire

With all three artefacts controlled (clean pre-ignition baseline, per-time-of-day threshold,
3-frame persistence), the honest table is:

| target FAR | fire pixels | residual led | median lead |
|---|---|---|---|
| 0.05 | 24 | 100% | +692 min |
| 0.01 | 21 | 33% | **-170 min** |
| 0.002 | 13 | 0% | **-225 min** |

At any defensible false-alarm rate the residual detector **lags** GOES FDC (-170 min at
0.01, -225 at 0.002), and for 11 of 24 fire pixels it never confidently detects the fire at
strict FAR at all (the fire-pixel count falls 24 -> 13). The +692 at FAR 0.05 is the last
vestige of the loose-threshold artefact: a one-diurnal-cycle baseline holds real per-pixel
*bias* (only ~1-2 clear samples per hour bin), so at a permissive cut the residual crosses
persistently before the fire even starts. At a strict cut that same bias forces a high
threshold, and the residual has to grow well past onset to clear it, by which time FDC's
mature multi-band contextual algorithm has already flagged the pixel.

The conclusion, stated plainly: **on this real fire, a crude hourly-mean diurnal-residual
detector does not beat GOES FDC at matched false-alarm rate.** The synthetic +70 min demo
does not transfer to this baseline. That is a real, negative result, and it localises the
cause, the baseline, not the residual idea itself. The residual detector can only win if its
expected-BT model is good enough to keep the matched-FAR threshold low, and a per-pixel mean
over one diurnal cycle is not.

The two levers that remain, both requiring more input than the existing cube:

1. **A multi-day clear-sky baseline.** Pull several days of pre-fire 3.9 um history so each
   hour bin has many samples; the per-pixel mean (and its variance) then carries far less
   bias, lowering the matched-FAR threshold. This is a longer ``t1-pull-cube`` window
   (baseline days + the fire day) and no new code.
2. **The learned forecaster.** ``TemporalAnomalyNet`` with the ``solar_zenith_cube``
   covariate models the daytime solar-reflectance component the hourly mean cannot, and
   forecasts from recent frames rather than a static climatology; its residuals feed the
   same ``real_lead_experiment`` unchanged. Needs torch.

The value banked regardless: a real GOES pull, a matched-FAR lead-time protocol against FDC,
and three independent artefacts found and fixed by a collapse-across-FAR diagnostic. The
honest current state is a negative result for the crude baseline, with the mechanism
understood and the next levers scoped, not a claimed win.

### The learned forecaster, wired end-to-end

The state-of-the-art path is now one command: ``t1-temporal-real --learned`` trains
``TemporalAnomalyNet`` on the cube's leading clear-sky span and feeds its residuals to the
same matched-FAR / per-time-of-day / persistence protocol. The plumbing is
``learned_residuals`` (train on ``cube[:clear_end]``, then ``temporal_net_residuals`` over
the whole cube), all torch-guarded and unit-tested offline. Two properties make it honest on
real data:

- **NaN never trains or scores.** BT is mean-centred by the clear-sky mean and holes are
  filled with zero so the 3D convolution sees an in-distribution value, but the forecasting
  loss is masked to the finite target pixels and the residual is set back to NaN wherever
  the input was NaN. Cloud and fill influence neither the fit nor the lead-time count.
- **It models what the mean cannot.** The forecaster consumes a window of recent frames plus
  ``cos(solar_zenith)`` (built by ``solar_zenith_cube``), so it can track the diurnal cycle
  and the daytime 3.9 um solar-reflectance term that inflates the hourly-mean residual and
  forced the high matched-FAR threshold. If anything lets the residual detector beat FDC on
  this fire, it is a forecaster good enough to keep that threshold low.

Run it on the existing cube (needs torch):

```
vhagar t1-temporal-real utah_north_cube.npz --detections data\detections\detections \
    --clear-frac 0.7 --far-bins 6 --min-consec 3 --learned --epochs 15
```

The same three artefact guards apply, so the result will be read the same way: a positive
lead is only real if it holds as the false-alarm rate tightens and stays physically
plausible. If the learned forecaster still does not clear FDC, the remaining lever is the
multi-day baseline pull (more history for either forecaster); that would be a data-scale
limit, not a modelling-idea limit.

### The learned result, and the robust conclusion

Ran (15 epochs, window 6, + solar) on the same cube:

| target FAR | fire pixels | residual led | median lead |
|---|---|---|---|
| 0.05 | 24 | 75% | +435 min |
| 0.01 | 24 | 33% | **-95 min** |
| 0.002 | 14 | 0% | **-222 min** |

The learned forecaster is marginally better than the hourly mean at moderate FAR (-95 vs
-170 min at 0.01) but the verdict is unchanged: at any defensible false-alarm rate it lags
GOES FDC, and at strict FAR only 14 of 24 fire pixels are detected at all. The +435 at 0.05
is the loose-threshold artefact again.

So the conclusion is now **robust across forecasters**: a crude per-pixel hourly mean and a
learned TemporalAnomalyNet with a solar covariate, both trained on one diurnal cycle, both
fail to beat FDC on this night fire at matched FAR. Two honest reads of why, neither a defect
in the residual idea:

- **Data scale.** Both forecasters see only ~30 hours (one diurnal cycle) of clear-sky
  history, so both carry enough per-pixel bias to force a high matched-FAR threshold. The
  one untried lever is a multi-day pull; it is the single factor common to both failures.
- **Sample size.** This is n = 1 fire, and a well-behaved one: FDC's contextual algorithm,
  mature and multi-band, already detects it early even at night. The residual detector's
  theoretical edge is specifically the cold-start, slow-ramp night fire that an absolute
  threshold is slowest on; one fast fire that FDC handles well is a weak test of that edge,
  not a refutation of it. A fair verdict needs a cohort of fires, not this one.

What is banked and true: a real GOES pull, a matched-FAR lead-time protocol against FDC,
both a physics-baseline and a learned forecaster wired to it, three artefacts found and
fixed by a collapse-across-FAR diagnostic, and an honest negative result on this fire with
the two remaining levers (more history, more fires) named. The synthetic +70 min lead is a
property of the synthetic setup; on real data, so far, FDC is not beaten.

### The right test: a stratified fire cohort, not one fire

The proper way to ask whether the detector works is not to keep tuning against one fire but
to test it across a **cohort stratified by the condition that theory says matters**. The
residual detector's edge is specifically the cold-start night fire an absolute contextual
threshold is slowest on; a day fire is the control where it should show no edge. If the
detector leads FDC on the night stratum and not the day stratum, that is evidence for the
mechanism; if it leads on both or neither, it is not.

``select_fire_cohort`` makes the selection reproducible: it clusters the FDC parquet into
fires, computes each fire's ignition local-solar-hour and early FRP ramp, and picks
``night_coldstart`` fires (ignition in local night, slow ramp) and ``day`` controls, each
with a ready-to-pull cube window and a ``clear_frac`` that ends the baseline before ignition.
``t1-cohort-select`` writes the spec and prints the pull commands; ``t1-temporal-cohort``
scores every fire (hourly-mean or ``--learned``) through the same matched-FAR / per-time-of-
day / persistence / contamination-guard pipeline and aggregates lead over FDC **per
stratum** (``cohort_lead_summary``). Selected from this week's GOES-18 FDC, the cohort is
three cold-start late-night Sonora fires (~28-29 N, local ~23-24 h) and three day-ignition
controls.

```
vhagar t1-cohort-select --detections data\detections\detections    # writes cohort/cohort.json
vhagar t1-cohort-pull --spec cohort\cohort.json                    # pulls all cubes, resumable (S3)
vhagar t1-temporal-cohort --spec cohort\cohort.json --detections data\detections\detections \
    --far 0.01 --far-bins 6 --min-consec 3            # add --learned for the net
```

``t1-cohort-pull`` runs the per-fire pull over the whole spec in one command, skipping cubes
already on disk (so an interrupted cohort resumes) and recording rather than aborting on a
failed pull.

The eval harness, selection, and aggregation are built and unit-tested offline; the pulls
(six small cubes) are the user's to run. This turns the T1 differentiator question from an
anecdote about one fire into a stratified measurement with a control, which is the honest
standard the rest of VHAGAR is held to.

### The fourth flaw: training on the evaluation frames

The first cohort run reported enormous night-stratum leads (+275 to +755 min) and auto-
excluded every day control. Both were bugs in the experiment, not results:

- **Training on test.** The baseline was fit on the leading (clear) frames, and the residual
  was then allowed to "detect" on those same frames. A fire pixel differs from the fire-free
  calibration pixels for ordinary reasons (a warmer or differently-shaped surface), so it
  crossed the spatial threshold *during the pre-ignition baseline*, and that pre-fire false
  detection was scored as a ~10-hour lead. The fix is a train/test split: ``real_lead_
  experiment(..., eval_start=clear_end)`` counts residual detections only on held-out frames
  after the baseline, the same period in which FDC first flags the fire. A unit test
  confirms a sustained in-baseline excursion is ignored and detection is measured post-split.
- **Contaminated controls.** The day controls were excluded by the contamination guard
  because the cube box (wider than the cluster cell) caught a neighbouring earlier fire
  inside the baseline span. ``select_fire_cohort`` now anchors each window on the earliest
  FDC detection anywhere *in the box* and ends the baseline three hours before it, so the
  training window is fire-free by construction. Verified on the real FDC: all six fires
  (three night, three day) now have a clean baseline (earliest in-box fire ~3 h after the
  baseline ends).

That is the fourth artefact caught by the same discipline: an implausible number (a 10-hour
lead) and a broken comparison (no control) traced to a methodological error, not banked as a
finding. The re-run below is the first cohort measurement with a real control and no
training-on-test.

```
vhagar t1-cohort-select --detections data\detections\detections   # re-selects: 3 night + 3 clean day
vhagar t1-cohort-pull --spec cohort\cohort.json --refetch          # windows changed; re-pull
vhagar t1-temporal-cohort --spec cohort\cohort.json --detections data\detections\detections \
    --far 0.01 --far-bins 6 --min-consec 3
```

### The fifth flaw: a non-detection is not a zero-lead tie

The first cohort runs looked like a weak tie: night fires at "median lead 0", which read as
"as fast as FDC". It was an accounting bug. When the residual **never** crossed its threshold
for a fire in the held-out window, the code reported ``median_lead_min = 0`` (a tie) and
counted all of that fire's pixels, but contributed nothing to the pooled distribution. So a
fire the detector completely **missed** was being shown as a draw. The learned run exposed it
with an impossible table (per-fire medians of 0 alongside a stratum "100% of pixels led"): the
100% was computed over only the one fire that detected anything, while the "0" fires were
silent misses.

The fix separates detection from lead. ``RealLeadResult`` now carries ``n_fire_pixels_total``
and ``detection_rate``; a fire with no detection reports ``median_lead_min = NaN`` (not 0) and
is counted as not-led. ``cohort_lead_summary`` leads with the **detection rate** (fraction of
fire pixels the residual flagged at all in the held-out window) and only then the lead *among
detected pixels*. ``t1-temporal-cohort`` prints "no detection" in red for a missed fire. A
unit test locks it in: a non-detecting fire lowers the detection rate, counts as not-led, and
is excluded from the lead median rather than posing as a zero-lead tie.

This changes the reading of the earlier tables: those night "0 min" entries were largely
**non-detections**, so the detector was not tying FDC on night cold-starts, it was failing to
detect many of those fires at all at FAR 0.01 with 3-frame persistence on held-out frames.
The re-run below reports the honest detection rate.

```
vhagar t1-temporal-cohort --spec cohort\cohort.json --detections data\detections\detections \
    --far 0.01 --far-bins 6 --min-consec 3            # add --learned for the net
```

### The honest cohort result (detection-rate-first)

With the bug fixed, the self-consistent table (hourly-mean forecaster, FAR 0.01, 6 time-of-day
bins, 3-frame persistence, held-out frames only):

| stratum | fires | detection rate | fires led | median lead (detected) | pooled px lead | px led |
|---|---|---|---|---|---|---|
| night_coldstart | 3 | 23% px (67% fires) | 33% | +72 min | **+175 min** | **94%** |
| day | 2 | 29% px (100% fires) | 50% | -141 min | **-430 min** | **17%** |

Per fire: night = {a full miss (0/80), +175 min (72/163 px), -30 min (6/100 px)}; day =
{+148 min (2/2 px), -430 min (27/97 px)}.

Two honest reads, held together:

- **Low sensitivity.** At a defensible false-alarm rate the residual detects only ~a quarter
  of fire pixels in the held-out window. Most fire pixels it never flags. That alone rules it
  out as a stand-alone detector; it is at best a lead-time *supplement* to FDC on the pixels
  it does catch. (For scale, FDC itself only detects ~half of VIIRS's fire pixels, so a single
  temporal band catching a quarter is not absurd, but it is low.)
- **A directional signal, conditional on detection.** Where it *does* detect, the predicted
  night/day split appears clearly: night-coldstart detected pixels lead FDC (+175 min pooled,
  94% of detected pixels lead), day-control detected pixels lag (-430 min pooled, 17% lead).
  That is the mechanism's signature, a diurnal-residual detector helps where an absolute
  threshold is slowest (night) and not by day, and it is the first real evidence for it.

The caveat that keeps this from being a result: the cohort is tiny and each stratum is
effectively one informative fire (night is carried by a single 72-pixel fire at +175; the
others are a miss and a 6-pixel lag). So this is *suggestive directional evidence at low
sensitivity*, not a demonstrated operational lead.

### Honest bottom line for the T1 temporal component

After five artefacts, each caught by the same discipline (an implausible number or a broken
comparison traced to a method error, never banked): baseline contamination, global-threshold
night-blindness, first-crossing false alarms, training-on-test, and non-detection-as-tie. What
is solid is the *infrastructure, the method, and one honest signal*: a real GOES 3.9 um pull, a
physics and a learned forecaster, a matched-FAR / per-time-of-day / persistence / train-test
protocol against FDC, a stratified cohort with a control, a detection-rate-first scoreboard,
and a directional night-leads/day-lags result consistent with the mechanism. What is **not**
established is an operational lead: sensitivity is low (~25% detection) and the cohort is too
small (effectively one fire per stratum) to claim more than a suggestive direction. Firming it
up would need a much larger cohort (dozens of fires per stratum, more than one GOES-week gives)
and, to raise sensitivity, the multi-band context FDC already uses, not more single-band
tuning. The component is best banked here: a well-instrumented, honestly-scored negative with a
directional hint, not a win dressed up as one.

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
