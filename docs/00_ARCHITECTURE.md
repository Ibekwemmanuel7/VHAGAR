# VHAGAR: System Architecture & Methodology

**Multi-sensor wildfire intelligence: detection, mapping, danger, and spread**
Coverage: CONUS/USA · Canada · Europe
Version 0.1 · August 2026

---

## 0. Executive summary

VHAGAR is four models and one platform. The four models answer four different
questions that operational users conflate at their peril:

| Task | Question | Unit of prediction | Primary sensors |
|---|---|---|---|
| **T1. Detection** | *Is something burning right now, and where?* | fire pixel / event, minutes | GOES-19/18 ABI, MTG-I FCI, VIIRS 375 m, S3 SLSTR |
| **T2. Burned area & severity** | *What burned, and how badly?* | 20-30 m polygon + continuous severity | Sentinel-2, Landsat 8/9, Sentinel-1 |
| **T3. Fire danger** | *Where is ignition likely, and how will fire behave if it starts?* | 1-4 km cell × day, probability | ERA5/HRRR/ECMWF + fuels + human drivers |
| **T4. Spread** | *Where does this fire go in the next 12-72 h?* | 375 m burn probability field | Fused arrival-time state + NWP + fuels |

The single most important design decision in this document: **VHAGAR does not
try to replace the operational physics with end-to-end deep learning.** The
literature is unambiguous that the marginal return on the tenth U-Net variant
is near zero, while the return on better fuel state, better labels, better
assimilation, and honest validation is large. Machine learning is placed where
it demonstrably wins, event-level classification, state estimation from noisy
multi-sensor observations, simulator emulation, and residual correction, and
physical algorithms are kept where decades of tuning cannot be cheaply beaten.

The second most important decision: **the validation protocol is a first-class
artifact, versioned and enforced in CI.** Almost every published wildfire ML
result that looks too good is a leakage artifact. Section 6 defines the split
contract; `src/vhagar/eval/splits.py` implements it; `tests/` enforces it.

---

## 1. Design principles

1. **Physics core, ML at the boundaries.** Contextual thermal detection,
   the FWI/NFDRS fuel-moisture codes, and level-set fire propagation are
   retained as first-class components. ML augments them.
2. **Physics is structure, not a feature.** The Planck function, the Dozier
   mixture, the Wooster FRP equation and atmospheric transmittance are built
   into the model as constraints and differentiable layers, not learned from
   data that cannot possibly contain enough of them. See `docs/03_PHYSICS.md`.
3. **Every number ships with an uncertainty.** Burned area is reported as an
   Olofsson error-adjusted estimate with a 95 % CI, never as a pixel count.
   Danger and spread ship calibrated probabilities, and calibration
   (reliability, Brier, ECE) is a release gate, not an afterthought.
4. **No leakage, by construction.** Splits are event-blocked, space-blocked,
   and time-blocked. Random splits are physically impossible to request
   through the public API of `vhagar.eval.splits`.
5. **Baselines are permanent.** Persistence, persistence-plus-buffer, a
   calibrated spectral-index threshold, and a physics simulator run with the
   same inputs are reported next to every model, in every experiment, forever.
6. **One analysis grid.** All raster fusion happens on a single equal-area
   375 m grid per region. Sensor-native processing happens before regridding,
   never after.
7. **Latency tiers are products, not failures.** A 0-day provisional burned
   area and a 45-day consolidated one are both shipped, with the accuracy gap
   published.
8. **Cloud-native, resumable, and reproducible.** Zarr + Icechunk for arrays,
   GeoParquet for vectors, pinned GDAL/PROJ, versioned spatial splits.

---

## 2. Data architecture

### 2.1 The analysis grid

A single **equal-area 375 m grid** per region, matching VIIRS I-band native
resolution:

| Region | CRS | Notes |
|---|---|---|
| CONUS | EPSG:5070 (NAD83 / Conus Albers) | operational US standard |
| Canada | EPSG:3979 (NAD83(CSRS) / Canada Atlas Lambert) | |
| Europe | EPSG:3035 (ETRS89-extended / LAEA Europe) | EEA reference grid |

Tiles are 256 × 256 cells (96 km × 96 km) with a 32-cell halo for
convolutional context. Tile IDs are stable, versioned, and stored as
GeoParquet, they *are* the spatial-split primitive.

Rationale for rejecting alternatives: a DGGS (H3/S2/rHEALPix) does not tile
into rectangular tensors and resamples continuous radiance for no modelling
gain. H3 is used, but only for **point aggregation** (detections, alerts) in
the serving database, where its indexed joins are genuinely valuable.
EASE-Grid 2.0 is adopted only where SMAP interoperability requires it.

### 2.2 Resampling policy: by quantity type

Getting this wrong silently corrupts fire energy accounting, so it is fixed
policy, not per-script choice:

| Quantity | Method | Tool |
|---|---|---|
| Swath radiance/BT (VIIRS, MODIS) | nearest-neighbour with swath geometry | `pyresample` |
| Gridded continuous (reflectance, LST, DEM) | bilinear / cubic | `odc-geo.xr_reproject` |
| **Flux-like (FRP, precipitation, burned area fraction)** | **conservative** | `xesmf`, weights cached to disk |
| Categorical (fuel model, land cover) | mode / nearest | `odc-geo` |
| Polygon → grid statistics | exact area-weighted | `exactextract` |

`tests/test_regrid.py` asserts mass conservation for the conservative path.
Total FRP before and after regridding must match to float tolerance.

### 2.3 Storage layers

```
                       ┌─ raw (immutable, provider-native) ────────────┐
  S3 / CDSE / GEE  ──▶ │  GOES NetCDF · S2 COG · Landsat COG · ERA5     │
                       └──────────────────┬────────────────────────────┘
                                          │  VirtualiZarr (no copy)
                       ┌──────────────────▼────────────────────────────┐
                       │  virtual. Zarr v3 manifests over raw files   │
                       └──────────────────┬────────────────────────────┘
                                          │  harmonize → 375 m grid
                       ┌──────────────────▼────────────────────────────┐
                       │  analysis. Icechunk-versioned Zarr datacubes  │
                       │  (tile, time, band) chunked for training reads │
                       └──────────────────┬────────────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────┐
        ▼                                 ▼                             ▼
  training chips (Zarr)         PostGIS/TimescaleDB            COG + PMTiles
  + GeoParquet splits            (events, detections)           (serving)
```

The GOES archive is the case study for why this matters: cloud-optimising
seven years of GOES-16 NetCDF as ~7.1 billion virtual Zarr chunks costs a
few hundred dollars of one-time compute and single-digit dollars per month of
manifest storage, versus thousands per month to duplicate the archive.

### 2.4 Where Google Earth Engine belongs: and does not

GEE is used for: **label generation** over sparse points, decade-scale
temporal compositing, the AlphaEarth / Satellite Embedding V1 feature layer
(64 bands @ 10 m, CC-BY-4.0), and rapid prototyping.

GEE is **not** in the NRT hot path. Interactive requests are capped at ~5
minutes of compute and tens of megabytes; batch tasks queue behind other
tenants; the free noncommercial tiers (150-1,000 EECU-hr/month) will not carry
production. Anything with a latency SLA reads directly from S3.

Patch export pattern (implemented in `vhagar/io/gee.py`):
`ee.data.computePixels(..., format="NUMPY_NDARRAY")` against the
**high-volume endpoint**, a process pool of ~25 workers, exponential-backoff
retry, results repacked into chunked Zarr. TFRecord export is rejected: it
coerces everything to float32 (2× inflation on uint16 reflectance), is not
randomly seekable for shuffling, and is a TensorFlow-shaped format in a
PyTorch stack.

> **Commercial licensing is a blocking question, not a detail.** If VHAGAR
> is ever operated commercially, GEE requires a paid Cloud plan. Get a quote
> for your EECU profile before any GEE dependency reaches the product path.

---

## 3. Task T1: Active fire detection

### 3.1 Sensor plan (revised: see `docs/06_CONSTELLATION.md`)

**Terminology first, because it drives the plan.** VIIRS is an *instrument*.
S-NPP, NOAA-20, NOAA-21 and the future NOAA-22 (JPSS-4) all *carry* VIIRS.
"Move from VIIRS to NOAA-21" is a **platform identifier change**
(`VIIRS_SNPP_NRT` → `VIIRS_NOAA21_NRT`), not a sensor change. Encoded in
`vhagar.io.sensors`.

| Source | Res. | Cadence / local time | Role |
|---|---|---|---|
| **GOES-19 ABI FDC** (East) | 2 km | **5 min** CONUS / 10 min FD | temporal backbone, Americas |
| **GOES-18 ABI FDC** (West) | 2 km | 5 min | temporal backbone, western CONUS |
| **MSG SEVIRI FRP-Pixel** (LSA-502) | 3 km | 15 min | **operational** European GEO feed |
| MTG-I FCI FRP-Pixel (LSA-509) | 1 km | 10 min | emerging European GEO. ⚠ *demonstration status* |
| **VIIRS / NOAA-21** | 375 m | ~13:30 & ~01:30 | **primary polar**, best daytime small-fire sensitivity |
| **VIIRS / NOAA-20** | 375 m | ~13:30 & ~01:30 | secondary polar (past design life) |
| **Sentinel-3 SLSTR FRP** | 1 km / 500 m | ~10:00 & ~22:00 | **primary for Europe and the morning/night band** |
| Metop-SG-A1 METimage | 500 m | ~09:30 | morning orbit. ⚠ *no fire L2 product confirmed* |
| Landsat 8/9, Sentinel-2 | 20-30 m | 5-16 d | validation and perimeter refinement only |

**Promoting Sentinel-3 SLSTR to primary is right, with one hard caveat.**
S7 (3.74 µm) **saturates at ~311 K**, and the FRP algorithm cannot process
daytime scenes where >1 % of *background* pixels are saturated. That removes
much of peak-summer daytime Mediterranean and southern-CONUS coverage. The F1
high-gain channel exists for this but the fully F1-based daytime path is still
future work. SLSTR's published **<5 MW detection floor is a night-time result**;
do not quote it for daytime. Verdict: excellent night/morning instrument,
poor peak-summer daytime detector. **complement to VIIRS, not a substitute.**
SLSTR's S6 (2.25 µm) is also what makes the gas-flare colour-temperature
discriminant possible.

**Hard scheduling constraints.**
* **S-NPP data delivery ceases 2026-11-01 13:00 UTC**, including HRD direct
  broadcast, which means S-NPP ultra-real-time (<60 s) dies with it.
* CONUS inter-satellite interval goes **~25 min → ~50 min**. Re-tune alert
  dedup windows and fire growth-rate estimators.
* **NOAA-21 carries a ~+10 % FRP bias** (M13 spectral response shift). Apply
  `sensors.frp_to_reference_scale` or your time series gains a fake step.
* **There is no NOAA-21 collection in Google Earth Engine.** If your pipeline
  runs in GEE you lose S-NPP and are left with NOAA-20 only, or must ingest
  FIRMS API output into your own asset.
* **MODIS ends late 2026 / early 2027.** Nothing replaces Terra's 10:30
  morning overpass in the US programme, all JPSS platforms share one plane at
  ~13:25 LTAN. **The morning orbit now belongs to Europe** (SLSTR 10:00,
  METimage 09:30). There is also a ~15:00-18:00 polar gap; only geostationary
  covers the late-afternoon fire peak.
* No GOES imager replacement until GeoXO ~2032.

`vhagar.io.sensors.active_platforms(date)` encodes all of this, so
2026-11-01 changes pipeline behaviour rather than silently opening a hole.

### 3.2 Architecture: two stages

**Stage 1: sensor-native candidate generation (physics retained).**
Ingest the operational contextual products as-is (VIIRS 375 m, GOES FDC,
FCI FIR, SLSTR FRP). These encode decades of hand-tuned false-alarm rejection
over deserts, glinting water, gas flares and solar farms; a from-scratch model
on a realistic label budget will not beat them.

One learned component is added here, where the gain is measurable: a
**temporal anomaly model on the geostationary 3.9 µm time series**, a TCN or
lightweight temporal transformer that predicts expected per-pixel brightness
temperature from the previous 24-72 h plus solar geometry and flags residual
excursions. Evidence that sub-threshold signal exists before the contextual
algorithm fires: porting a rapid-scan temporal algorithm to 5-minute
geostationary data raised detections from 11 to 27 of 189 recorded incidents
and roughly doubled mean lead time (35 → 65 min) ahead of official reporting.

Target: **detect 10-30 minutes earlier than FDCA at equal false-alarm rate.**

**Stage 2: event-level fusion and classification (where ML clearly wins).**
Candidates from all sensors are clustered into spatiotemporal *fire events*
(DBSCAN-style, land-cover-dependent buffers) with an explicit
**parallax-aware ≈3×3-pixel spatial tolerance** for GEO↔LEO matching. Naive
nearest-pixel matching between GOES and VIIRS produces 26-36 % apparent false
alarm rate; a 3×3 buffer drops it to 7-15 %. That difference is geometry, not
model quality.

A gradient-boosted classifier then scores each *event* (not pixel) on:
FRP trajectory and growth rate, multi-sensor detection agreement, persistence,
ΔT(MIR−TIR), SWIR ratio where S2/Landsat coincide, land cover, distance to the
FIRMS static thermal-anomaly mask, and view zenith angle.

**Raw latitude and longitude are excluded as features.** In a published
FIRMS wildfire/non-wildfire classification, raw coordinates accounted for
~89 % of model gain while *harming* out-of-region transfer; F1 fell from
0.985 (random split) to 0.767 (event-aware) to 0.627 (5° spatial block).
VHAGAR reports all three numbers on every release.

### 3.3 On foundation models: train our own for thermal, fine-tune for optical

The decision is **task-specific, not ideological**. Full analysis in
`docs/04_FOUNDATION_MODEL.md`.

**T1 detection (thermal): train our own, staged.** Optical-pretrained GeoFMs do
not transfer to 3.9 µm physics, in the one published geostationary thermal
benchmark, ImageNet pretraining scored *worse* than a from-scratch ViT
(0.856 vs 0.883 balanced accuracy), and Sentinel-pretrained GeoFMs lost
decisively to a thermal-native model. Thermal→thermal transfer, by contrast,
works well (79.6 % macro-F1 zero-shot across two very different thermal
sensors). Target ~90M parameters and **under $25k**, not 300-600M.

**T2 burned area (optical): fine-tune Prithvi-EO-2.0-300M-BurnScars.** Apache-2.0,
87.5 % burned-area IoU out of the box. Training our own here would be spending
money to reproduce something free.

**Before either: fix the loss.** In that same benchmark, switching from
cross-entropy to Dice took a from-scratch U-Net's fire IoU from **0.022 to
0.272**, a 12× gain that dwarfs everything pretraining bought anyone. And on
PANGAEA, a plain U-Net beat Prithvi on HLS Burn Scars (84.51 % vs 83.62 % mIoU).
Stage 0 is a tuned supervised baseline, and it may well be the product.

### 3.4 Success metric

Not pixel IoU. **Median minutes from agency-reported ignition (NIFC / CWFIS /
EFFIS) to first system alert, at a fixed operationally acceptable false-alarm
rate** (e.g. ≤1 false incident per 10,000 km² per day), stratified by
day/night, land cover, fire size decile, and view zenith angle.

---

## 4. Task T2: Burned area & severity

### 4.1 Labels

| Region | Primary | Role |
|---|---|---|
| USA | MTBS (30 m, 1984-present). **continuous dNBR/RBR, not the thematic class** | training severity |
| USA | Landsat C2 Level-3 Burned Area (CONUS only, no Alaska) | training extent |
| Canada | NBAC (hybrid agency + 30 m Landsat, 1972-present) | training extent |
| Europe | EFFIS burnt area (Sentinel-2 20 m since 2018) | training extent |
| Europe | **Copernicus EMS Rapid Mapping (EMSR), held out entirely as test** | evaluation |

MTBS thematic severity classes are derived per fire by analyst review, so they
are **not comparable across fires**; the continuous index is the label.

**Never train a pixel model on rasterised perimeters** (WFIGS, EFFIS, CNFDB)
without an interior severity mask. Roughly 9 % of the area inside a typical
VIIRS-derived perimeter is unburned islands, and the error is spatially
structured, you would inject a large commission-error prior.

### 4.2 Spectral method

- **NBR** = (NIR − SWIR2)/(NIR + SWIR2). Sentinel-2 uses **B8A**, not B8, to
  match SWIR bandpass and resolution.
- **RBR = dNBR / (NBR_pre + 1.001)** is the primary continuous severity metric.
  Against 1,681 CBI field plots it beat RdNBR and dNBR (pooled R² 0.705 vs
  0.677 vs 0.646), avoids RdNBR's singularity as NBR_pre → 0, and preserves
  sign. dNBR is retained for MTBS/EFFIS comparability.
- Class breakpoints are derived **per ecoregion from CBI regression**, not by
  importing fixed Key & Benson constants.
- **Mean compositing** over pre-fire (−90 to −15 d) and post-fire (+15 to +75 d)
  windows, with an **unburned-buffer offset correction**, replaces manual
  scene selection and removes phenological drift.

### 4.3 Model

A **bi-temporal siamese encoder-decoder**, not a single-date segmenter. 
burned area is intrinsically a change-detection problem. Shared-weight encoder
initialised from a geospatial foundation model (TerraMind base or
Prithvi-EO-2.0-300M-BurnScars, both Apache-2.0), multi-scale feature
differencing in the decoder, weighted BCE + Dice loss.

Cloud/smoke resilience: Sentinel-1 GRD as a second modality with
mixture-of-experts routing under cloud, or TerraMind's modality-generation to
synthesise the missing optical layer.

**Permanent baselines:** a plain U-Net and a *calibrated RBR threshold*. A
large fraction of published DL burned-area papers never beat a well-tuned
spectral baseline; several do not report one.

### 4.4 Latency tiers

Modelled on published NRT Sentinel-2 burned-area work: R0 (provisional,
VIIRS-seeded) → R10 (refined) → NTC (45-day consolidated). Expect roughly a
4-5 point Dice gap between provisional and consolidated. **Publish the gap.**

### 4.5 Agricultural vs wildland fire

Hard-gate on land cover (CORINE in Europe, NLCD + CDL in the US, AAFC ACI in
Canada) plus compactness, field-boundary alignment, and single-overpass
duration features. Emit a `fire_type` field; do not silently drop ag burns.

---

## 5. Task T3: Fire danger & ignition probability

### 5.1 Three quantities, never collapsed into one "risk" number

- **Fire danger** (FWI, ERC, BI): a *conditional* intensity, how fire behaves
  *if* it starts. Contains no ignition information, which is exactly why FWI
  over-forecasts in fuel-limited deserts.
- **Ignition probability**: P(≥1 ignition | cell, day). A rare-event binary
  with base rates of 10⁻⁵-10⁻⁷ at 1 km.
- **Expected burned area**: E[BA] = P(ignition) × E[BA | ignition]. The
  conditional distribution is heavy-tailed, so MSE training is dominated by a
  handful of events. Fit in log space or with a Tweedie/GPD tail; evaluate
  with CRPS and quantile loss, never RMSE.

The UI surfaces all three separately.

### 5.2 Layer 1: deterministic physical indices

Run **FWI1987** and **FWI2025** in parallel across all regions, plus
NFDRS2016 ERC/BI/SC/IC for CONUS (ingested from gridMET's `erc`/`bi`/`fm100`/
`fm1000` rather than reimplemented).

> ⚠ **Do not feed FWI2025 into the FBP1992 rate-of-spread equations.** NRCan's
> reference repository carries a February 2026 notice that the two are not yet
> compatible; an interim `iFBP2025` module is in development.

Serve percentile rank and anomaly-vs-climatology alongside raw values, this is
what makes an index comparable across Mediterranean, boreal, and Great Basin
fuels. Average **DSR = 0.0272 · FWI^1.77**, never raw FWI.

### 5.3 Layer 2: ML ignition probability, cause-stratified

Two gradient-boosted heads (human-caused, lightning-caused) at 1-4 km daily.
Their covariate structure is nearly orthogonal, and stratifying markedly
improves both.

Features: FWI + NFDRS components as engineered inputs (use them, do not make
the network relearn 50 years of fuel-moisture physics); **VPD computed from
Tmax/RHmin, never from daily means** (time-averaging materially changes the
inferred VPD-burned-area relationship); fm100/fm1000; SPEI/EDDI; SMAP L4
root-zone soil moisture; live fuel moisture (L-band SMAP-derived preferred);
fuel type (LANDFIRE FBFM40 / Canadian FBP 30 m / FirEUrisk 1 km);
WorldPop + GHSL; distance to road, rail, transmission line; WUI class; and
lightning density with a **0-14 day holdover lag stack**.

The strongest published evidence supports this shape: ECMWF's operational
Probability-of-Fire uses XGBoost on 19 predictors and found that **XGBoost beat
both simpler and more complex neural architectures**, that input data quality
mattered more than architectural complexity, and that **modelled fuel state was
the single most important predictor globally**. Spend the effort on fuels.

### 5.4 Layer 3: deep learning as a challenger, in shadow mode

A ConvLSTM or U-Net-3+ trained with Fractions Skill Score loss and evaluated
at 40/80/120 km neighbourhoods (pixel-exact verification is the wrong response
design for rare point-like events). Promoted only if it beats gradient boosting
on blocked, base-rate-preserving AUPRC *and* Brier. Expect it to earn its place
at seasonal lead times, not daily.

### 5.5 Forcing strategy

| Lead | Source |
|---|---|
| Days 0-2 | HRRR 3 km (CONUS) / ECMWF HRES 9 km (EU, Canada), anchored to RTMA/URMA |
| Days 3-10 | ECMWF ENS + **AIFS v2** as an additional cheap ensemble |
| Days 10-45 | CEMS seasonal fire danger |

Propagate the ensemble through FWI to produce **probabilistic danger**
(P(FWI > class threshold)), this is what users actually need.

Bias-correct NWP to reanalysis climatology **before** running the fuel-moisture
codes: DMC and DC integrate precipitation over 12-52 days, so precipitation
bias accumulates. AI NWP is usable at days 3-15 for smooth fields but should
not be trusted at days 0-2 for the terrain-forced downslope wind events
(Santa Ana, Diablo, foehn, bora) that dominate catastrophic fire. ~28 km
cannot resolve them, and AI models systematically under-disperse extremes.

### 5.6 Sampling design: the trap that ruins most ignition models

Ignition databases are presence-only with strong **reporting bias**: detection
probability rises with population density and road proximity, which are the
very covariates being modelled. A naive model learns *where people report
fires*, not where fires start.

Mitigations, all implemented in `vhagar/datasets/danger.py`:
(a) target-group background sampling, draw pseudo-absences from the same
reporting process; (b) stratify negatives to match the positive land-cover
distribution; (c) prefer satellite-detected active fire over agency reports
where detection is more spatially uniform, accepting the ~1 ha size floor;
(d) if you downsample to balance classes, **apply the rare-event prior
correction**, otherwise your probabilities are calibrated to a fictional base
rate and the reported accuracy is meaningless operationally.

---

## 6. Task T4: Spread forecasting

### 6.1 What is realistically achievable: stated up front

For 24-hour next-day burned-mask prediction at 375 m, expect **average
precision in the 0.35-0.45 range**. The field has been stuck there since 2023,
and the most careful study to date showed that **doubling the historical
training data yielded almost no accuracy gain**, cross-year domain shift
dominates. For 12-24 h perimeter forecasts with assimilation on
well-instrumented incidents, expect Sørensen/IoU of roughly **0.6-0.8 on
wind-driven fires and materially worse on plume-dominated ones**, with rate-of-
spread MAPE around 40-60 % in timber.

The binding constraints are label quality (VIIRS-derived perimeters score
0.71-0.93 F1 against agency perimeters, that is your ceiling), fuel-map error,
wind downscaling, and unmodelled suppression. Not model architecture.

Any claim of ≫0.5 AP on next-day spread, or >0.9 IoU on real perimeters, is
almost certainly a leaky split or cumulative-rather-than-incremental burned
area. Budget accordingly.

### 6.2 Architecture: four components

**(1) State estimation, highest return on investment.**
Fuse VIIRS 375 m, GOES/FCI 5-10 min fire detection, and Sentinel-2/3 into a
continuous **fire arrival-time field**. The strongest published result in the
ML fire literature is exactly this: a conditional GAN trained on coupled
atmosphere-fire simulations, inferring arrival time from satellite active fire,
validated against airborne IR perimeters at **Sørensen 0.81, false alarm ratio
0.23, ignition time error ~32 min**. Anchor to NIROPS airborne IR where
available and to VIIRS-derived 12-h perimeters elsewhere.

**(2) Propagation.** A level-set simulator (ELMFIRE is the best-documented
open option: GPU, Monte Carlo, permissive licence) driven by HRRR/RRFS in
CONUS, HRDPS in Canada, and convection-permitting models in Europe, with wind
downscaling. Canadian FBP fuel logic for Canada, Rothermel/LANDFIRE for CONUS.
**For Europe, fuel mapping will be the dominant error term**, there is no
LANDFIRE equivalent, only a 1 km harmonised map.

**(3) Assimilation.** Assimilate the arrival-time analysis on *every satellite
pass*, not just at perimeter delivery. Classical EnKF on perimeter vertices is
known to produce self-intersecting polygons; a diffusion-model-based ensemble
score filter has been shown to give lower RMSE, ~5× faster, with stable
geometry. Re-calibrate per-fire ROS adjustment factors online, per-fire
calibration is where hybrid ML demonstrably earns its keep (IoU > 0.6 sustained
to 72 h after a calibration window, in published neural-CA work).

**(4) ML as accelerator and corrector.** Train a diffusion or neural-operator
surrogate on *your own* simulator ensemble for 100-500× speedup on large
ensembles and what-if queries. Train a residual corrector (UTAE/Swin-UNet
class, ImageNet-pretrained, pretraining is worth ~+0.04 AP) on the difference
between simulated and observed 12-h growth. Expect real but modest gains.

### 6.3 Uncertainty

Ship burn-probability contours from the ensemble as the primary product. 
operators already read them. Add per-pixel calibration diagnostics and
**front-localised** UQ evaluation (metrics computed near the fire front, not
over 99 %-negative background, where global UQ scores are dominated by
trivially confident empty space). Treat conformal prediction as an R&D track:
exchangeability fails under cross-year domain shift, and pixel-wise conformal
sets give no coverage guarantee on the perimeter as a set-valued object.

---

## 7. Validation protocol (binding)

This section is a contract. `src/vhagar/eval/` implements it; CI enforces it.

### 7.1 Splits

| Task | Required blocking |
|---|---|
| T1 detection | event-aware **and** spatial-block (≥5° or ≥500 km) **and** leave-year-out |
| T2 burned area | leave-one-fire-out, leave-one-ecoregion-out, **leave-one-continent-out** (train MTBS → test EMSR) |
| T3 danger | spatial block (size from residual variogram range) × leave-one-season-out |
| T4 spread | leave-one-fire-out × leave-year-out (12-fold year permutation) |

Report **per-fold** results. Fold standard deviation on spread tasks is
±0.08-0.10 AP, comparable to the entire spread of model rankings. A claimed
0.02 AP improvement without fold-wise reporting is noise.

Random splits are not available through the API.

### 7.2 Metrics by task

- **T1**: POD, FAR, precision, F1 at event level; FRP bias/RMSE; **median
  detection latency vs reported ignition**. Stratify by day/night, land cover,
  size decile, view zenith.
- **T2**: Dice/IoU + **Olofsson error-adjusted area with 95 % CI**. Never
  report a pixel count as an area. 50-100 reference samples in the burned
  stratum per reporting region. Severity: R² and RMSE against CBI, per ecoregion.
- **T3**: AUPRC, Brier with Murphy decomposition, log loss, ECE, reliability
  diagrams, BSS against a pixel × day-of-year climatology. Tune only on
  **proper** scores. F1 and CSI are improper and their optimum depends on an
  arbitrary threshold. Report CSI/POD/FAR at operational thresholds as decision
  diagnostics only.
- **T4**: AP + IoU@τ + **burned-area ratio** (predicted/observed, which exposes
  bias that IoU hides) + arrival-time MAE, per fold and per fire, stratified by
  wind-driven vs plume-dominated regime.

### 7.3 Mandatory baselines

Persistence · persistence + calibrated isotropic buffer · calibrated spectral
threshold (T2) · climatology (T3) · physics simulator with identical forcing
(T4). All reported in every experiment.

### 7.4 The Olofsson estimator (T2)

Map pixel counts are biased estimates of area because classifiers make
asymmetric commission/omission errors. Under stratified random sampling with
map classes as strata:

```
p̂_k   = Σ_i W_i · (n_ik / n_i)                     # error-adjusted proportion
Â_k   = A_total · p̂_k                              # adjusted area
S(p̂_k)= sqrt( Σ_i W_i² · p̂_ik(1−p̂_ik) / (n_i − 1) )
95% CI = Â_k ± 1.96 · A_total · S(p̂_k)
```

Implemented in `vhagar/eval/area_estimation.py` with sample-size allocation.
Burned area is <1 % of the landscape, simple random sampling finds almost
nothing, so stratification is not optional.

---

## 8. Platform architecture

### 8.1 Two paths, deliberately separated

```
HOT PATH (target < 60 s, SLA 5-15 min)
  GOES-19 ABI L2 FDC lands in s3://noaa-goes19
    └─▶ SNS NewGOES19Object
        └─▶ SQS (prefix filter, DLQ)          ← queues rather than drops
            └─▶ Lambda/Fargate: decode → temporal anomaly model → fuse
                └─▶ PostGIS/TimescaleDB  ─▶ Redis pub/sub ─▶ SSE ─▶ browser

COLD PATH (Dagster, space × time partitioned assets)
  backfills · harmonised cube materialisation · retraining ·
  reconciliation of NRT stream against later authoritative products
```

Dagster is chosen over Prefect/Airflow for one specific reason: the pipeline is
a graph of assets partitioned by (space × time), and Dagster's
multi-dimensional partitions plus **freshness policies** let "the GOES asset
must be no more than 15 minutes stale" be declared as a first-class SLA with
built-in alerting. Freshness *is* the product requirement.

### 8.2 Serving

FastAPI + PostGIS/TimescaleDB (hypertable on time, partitioned by H3 cell,
continuous aggregates for dashboard queries) · TiTiler for dynamic COG **and**
Zarr tiling · Martin for live PostGIS vector tiles, PMTiles on object storage
for static layers (fuels, WUI, historical perimeters) · MapLibre GL JS base map
with a Deck.gl overlay in the same WebGL context · **SSE, not WebSockets**, for
alerts (one-directional, auto-reconnecting, proxy-friendly).

### 8.3 Competitive positioning

NASA FIRMS delivers ultra-real-time detections in **under 60 seconds** over
CONUS via direct broadcast. Watch Duty's moat is human editorial trust.
ALERTCalifornia and Pano AI operate camera networks with human verification.

**Do not compete on polar-orbiter detection latency, consume FIRMS.** All four
incumbents are detection-centric and largely single-modality. The open gap is
productised **multi-sensor fusion + short-horizon spread forecasting** with
fuels, terrain and NWP winds. Position there.

---

## 9. Engineering stack

| Layer | Choice |
|---|---|
| Bulk archive | direct S3 (`sentinel-cogs`, `usgs-landsat`, `noaa-goes19`) via `odc-stac` + `pystac-client` |
| GEE | `earthengine-api` + `xee`, high-volume endpoint, labels & embeddings only |
| Arrays | Zarr v3 on **Icechunk** (versioning + transactions); VirtualiZarr over legacy NetCDF |
| Vectors | GeoParquet with native Parquet GEOMETRY types, queried by DuckDB |
| Regrid | `odc-geo` (bulk) · `pyresample` (swath) · `xesmf` cached weights (flux) · `exactextract` (zonal) |
| DL | `torchgeo` ≥0.9 (native `(T,C,H,W)` time series) + PyTorch Lightning, bf16-mixed |
| Weights | Prithvi-EO-2.0-300M-BurnScars, TerraMind-1.0-base (both Apache-2.0) |
| Tracking | **MLflow, self-hosted** (VPC-deployable, real registry, no per-seat cost) |
| Versioning | Icechunk for arrays + DVC for labels/splits |
| Config | Hydra + OmegaConf |
| Serving | ONNX Runtime behind FastAPI; Triton only for multi-model GPU sharing |
| Orchestration | Dagster (cold) + SNS→SQS→Lambda (hot) |
| Validation | `pandera` schemas in-pipeline (git-versioned beside the code) |
| Testing | `pytest` + `hypothesis` |

**Explicitly rejected:** `stackstac` (2 years stale → use `odc-stac`),
`mmsegmentation` (unreleased since 2023), `raster-vision` (2 years idle),
`TorchServe` (unmaintained), `aim` (15 months stale), TFRecord for patch
export, DGGS as a raster substrate.

### 9.1 Reproducibility specifics for geospatial ML

Seeding `torch`/`numpy`/`random` is necessary and insufficient. Also version:
the exact tile IDs in each split (as GeoParquet), the CRS, the **GDAL and PROJ
versions** (a PROJ minor release can move your pixels via grid-shift updates),
and the STAC item IDs of every scene used. Pin GDAL and `pyproj` in the
container; test upgrades explicitly.

### 9.2 Geospatial tests that catch silent failures

1. CRS round-trip invariance (A→B→A within tolerance), catches PROJ regressions
2. Conservative-regrid **mass conservation** on FRP, catches weight-cache corruption
3. Hypothesis property tests on geometry ops. GEOS edge cases are real
4. **Nodata/mask propagation**, the most common silent EO bug is nodata
   becoming 0 and then becoming "cold ground"
5. Georeferencing golden test, a known historical fire must be detected within
   N metres of ground truth
6. Determinism, same seed + input → bitwise-identical output, asserted in CI

---

## 10. Roadmap

**Phase 0. Foundations (weeks 1-4).** Grid and tiling. Icechunk/Zarr analysis
store. GEE and STAC ingestion. Event registry and label harmonisation across
MTBS/NBAC/EFFIS/EMSR. Split machinery + CI enforcement. *Exit: a versioned
datacube for one region-year and a leakage-proof split you can defend.*

**Phase 1. T2 burned area (weeks 5-10).** The best-labelled task; start here.
Calibrated RBR baseline → U-Net → siamese foundation-model encoder. Olofsson
area estimation with CIs. Leave-one-continent-out (MTBS → EMSR) as the headline
generalisation number. *Exit: an honest, CI-bounded burned-area product.*

**Phase 2. T1 detection (weeks 8-14).** GOES/FCI ingest on the hot path.
Temporal anomaly model. Event fusion with parallax tolerance. Event classifier.
*Exit: median-latency-to-ignition beats FDCA at equal FAR.*

**Phase 3. T3 danger (weeks 12-20).** FWI1987/2025 + NFDRS engine. Driver
stack. Cause-stratified GBDT. Calibration gates. Ensemble-propagated
probabilistic danger. *Exit: calibrated daily probability beating FWI-threshold
and climatology on blocked splits.*

**Phase 4. T4 spread (weeks 18-30).** Arrival-time state estimation. ELMFIRE
integration. Assimilation. Surrogate + residual corrector. *Exit: AP in the
0.35-0.45 band with fold-wise reporting, beating persistence + buffer.*

**Phase 5. Platform (parallel from week 10).** Hot/cold pipelines, PostGIS,
tiles, map UI, alerting, model registry and shadow deployment.

### 10.1 Known open questions to resolve before committing

- GEE commercial pricing for the actual EECU profile (blocking if commercial).
- FireSat / OroraTech data licensing, latency and API, architect ingest so a
  new point-detection source is a plugin, not a rewrite.
- Timeline for `iFBP2025` and the CFFDRS Fire Occurrence Prediction module.
- Whether a public LANDFIRE 2024+ fuel product is released.
- European fuel mapping, the largest single work item for T4 in Europe.

---

## 11. Bibliography of primary sources

Full annotated source lists, with URLs and the specific claim each supports,
are in the four research briefs under `docs/research/`:

- `docs/research/T1_detection.md`
- `docs/research/T2_burned_area.md`
- `docs/research/T3_danger.md`
- `docs/research/T4_spread.md`
- `docs/research/T5_engineering.md`

Later additions:

- `docs/03_PHYSICS.md`, radiative transfer, atmospheric correction, emissivity,
  false-alarm physics, and the staged physics-aware ML plan
- `docs/04_FOUNDATION_MODEL.md`, train-your-own vs fine-tune, with costs
- `docs/05_PRIOR_ART.md`, what to borrow and avoid from the FirePerim platform
- `docs/06_CONSTELLATION.md`, the 2026-2030 sensor plan and risk register

Claims in this document that the research phase could not verify against a
primary source are flagged in those files under "Flagged uncertainties" and
must be re-checked before they drive an implementation decision.
