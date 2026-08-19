# VHAGAR progress tracker

Last updated: 2026-08-18. Keep this file current. It is the single place to look
before starting a session, and the place to update before ending one.

## Latest: Phase 5 -- console rebuilt on Mapbox GL (FirePerim layout) + blue ice-dragon rebrand (2026-08-18)

Rebrand: replaced the gold/green dragon emblem with the user's blue ice-dragon art (icedragon-full.jpg);
brand/ regenerated (circular mark 256/128/64/32 + favicon, ice-blue ring), console accent shifted
green-flame -> ice-blue (--ice #5cc8ff). VHAGAR identity kept.

Console rebuilt on Mapbox GL (user chose it for FirePerim's satellite-streets rendering), mirroring
FirePerim's operator layout while keeping the blue VHAGAR identity and every existing feature.
vhagar_console.html now: Mapbox GL v3 map (Satellite = satellite-streets-v12, Dark = dark-v11,
Terrain = outdoors-v12), FirePerim-style layers (FRP glow + dot circle layers with zoom x FRP radius
interpolation, event fill+line colored by peak FRP, a highlighted selected-event line, wind-arrow
symbol layer), a per-satellite Sensor tab row (All / GOES-18 / GOES-19) that filters map + feed +
KPIs, collapsible legend, click popups, prioritised incident feed, schema-aware KPIs/drawer, the T3
danger strip, weather/spread-risk display when enriched, and GeoJSON + KMZ export. Token: the /console
route injects VHAGAR_MAPBOX_TOKEN (env); without it a "token needed" panel shows and the rest still
works. Verified: /console 200 with token injected + placeholder gone, Mapbox GL loaded, emblem
embedded; console JS node-checked, no em dashes; ruff clean. Files: vhagar_console.html (full rewrite),
serve/vhagar_api.py (/console token injection), serve/README, brand/, icedragon-full.jpg.

## Phase 5 -- operational parity with FirePerim (weather + spread-risk + KMZ) (2026-08-18)

Closed the three quick operational gaps a FirePerim review flagged, so the console reaches parity on
the operational product. New package modules (dependency-light, pure-tested; the live fetch is a
user-machine step): vhagar/weather/open_meteo.py (current wind/RH/temp via Open-Meteo, stdlib urllib,
batched + cached, graceful fallback, parse pure), vhagar/features/spread_risk.py (transparent 0-100
weather-driven spread-risk score wind 55% / dryness 30% / heat 15% + Low/Moderate/High/Extreme class;
an operational triage signal, NOT the calibrated T3 danger), vhagar/export/kmz.py (hand-rolled KML ->
KMZ, risk-styled event perimeters, no simplekml dep). Wired into serve/vhagar_api.py: GET
/api/export/kmz (verified: 200, 89 placemarks over the CA events); optional weather+risk enrichment on
/api/events gated by VHAGAR_WEATHER (attaches wind/rh/temp + spread_risk + risk_class; labelled
"current conditions", coincident in NRT mode, present-day for archived data; off by default -> events
stay honest FDC-only, metadata.weather null). Console: KMZ export link + a weather/spread-risk block
in the event drawer shown only when enriched. Tests tests/test_operational.py (weather parse, risk
monotone/bands, KMZ roundtrip) green; console JS node-checked, no em dashes; ruff clean.

Still NOT closed (the big FirePerim gap): the airborne/UAS IR imagery track (orthorectification,
telemetry georeferencing, multi-pass fusion, desmoke/dehaze, real FLAME3 ingest) -- a genuine new
capability area, satellite-only VHAGAR has none of it. Files: vhagar/weather/, vhagar/features/
spread_risk.py, vhagar/export/, serve/vhagar_api.py, vhagar_console.html, tests/test_operational.py.

## Phase 5 -- /v1/danger endpoint + T3 danger card in the console (2026-08-18)

Wired the three T3 danger quantities into the product. serve/vhagar_api.py gains GET /v1/danger which
returns FWI (+ class, from the Canadian Fire Weather Index on supplied weather), ignition probability
(cause-stratified GBDT, King-Zeng prior-corrected) and expected burned area E[BA] = P(ig) x E[BA|ig]
(quantile-GBDT + GPD tail), KEPT SEPARATE, never collapsed into one risk number. Danger models train
lazily on VHAGAR's synthetic danger scenarios and cache (first call ~2s, then instant); labelled
schema="t3-danger-demo" with an honest note to wire real fuels/weather/occurrence for operational
values. Verified monotone: hot/dry (FWI 26.7 Very-high, ign 18%, E[BA] 154 ha) >> cool/wet (FWI 0.2
Low, 0.5%, 0.5 ha). vhagar_console.html gains a "Fire danger, T3 (demo)" strip below the KPIs showing
the three quantities, fetched from /v1/danger, distinct from the FDC detection product and hidden if
the endpoint is absent. Test tests/test_serve_danger.py (three quantities present + monotone) green;
console JS node-checked, no em dashes; ruff clean. This connects the Phase-3 science to the Phase-5
serving contract. Files: serve/vhagar_api.py (+/v1/danger), vhagar_console.html (+danger card),
tests/test_serve_danger.py.

## Phase 4 -- T4 generative arrival-time model (conditional GAN, physics-anchored) (2026-08-18)

Built the published SOTA for spread state estimation (docs/00 6.2): a conditional GAN that infers the
arrival-time field from active fire. models/arrival_gan.py: a U-Net generator conditioned on the
observed perimeter + a normalised detection-time channel + mapped covariates, a PatchGAN discriminator,
and losses = LSGAN adversarial + L1 reconstruction + an EIKONAL-CONSISTENCY term ||grad T| - 1/ROS|
that ties the generated field to the level-set physics so it cannot hallucinate an impossible front.
It sits on top of the physics-anchored estimator (state_estimation.py gives the calibrated prior; the
GAN learns the residual a single per-fire scale cannot). The conditioning + normalisation builders
(build_conditioning, normalize_arrival, make_training_pair) are pure numpy and unit-tested (data
contract verified without torch); generator/discriminator/Eikonal-loss/training loop are torch-guarded
(GPU box). Tests: tests/test_arrival_gan.py (4; torch shapes via importorskip), pure ones green, ruff
clean. Files: models/arrival_gan.py, tests/test_arrival_gan.py, docs/15.

## Phase 4 -- T4 anisotropic (wind-driven) elliptical spread (2026-08-18)

Made the level-set physically faithful: wind-driven fires spread as elongated ellipses, not circles.
models/spread.py adds the elliptical wavelet (Richards) directional ROS(psi)=head_ros*(1-e)/(1-e*cos psi)
with eccentricity from a wind-driven length-to-breadth ratio (length_to_breadth + eccentricity_from_lb),
and anisotropic_arrival, an 8-connected least-cost-path (Dijkstra) anisotropic arrival-time solver (the
rigorous continuous counterpart is the Ordered Upwind Method; 8-connectivity is an adequate simple
approximation). front_length_breadth measures the resulting ellipse. CLI t4-aniso.

VERIFIED (vhagar t4-aniso): zero wind = a symmetric circle (measured LB 1.00, downwind ext == upwind);
wind stretches it downwind with the head far outrunning the back (wind 0.3: measured LB 2.12 vs
prescribed 1.90, downwind 60 vs upwind 4; wind 0.6: LB 3.46 vs 2.80; wind 0.9: 3.86 vs 3.70). Measured
length-to-breadth tracks the prescribed value; a calibrated FBP/Alexander LB drops in for real fires.
Tests: tests/test_anisotropic.py (LB/eccentricity monotone, zero-wind circle + symmetry, wind elongates
downwind head>>back and LB tracks prescribed), all green, ruff clean.

T4 now has: isotropic + ANISOTROPIC fast propagation, honest incremental validation, and arrival-time
state estimation with online ROS calibration + assimilation. Remaining T4: the conditional-GAN /
diffusion arrival-time model (torch, GPU), real perimeters (NIROPS/VIIRS). Files: models/spread.py
(+anisotropic), cli.py (+t4-aniso), tests/test_anisotropic.py, docs/15.

## Phase 4 -- T4 arrival-time state estimation + online assimilation (highest-ROI piece) (2026-08-18)

Built the piece docs/00 6.2 calls the highest return on investment in spread: fuse sparse timed
satellite detections into a continuous arrival-time analysis and re-calibrate per-fire ROS online.
models/state_estimation.py: since arrival ~ 1/ROS, a robust per-fire scale k = median(prior_arrival /
observed_detection_time) aligns a prior ROS field (mapped fuel/wind pattern) to the detections; the
analysis field is prior_arrival / k (calibrate_ros_scale + estimate_arrival_field + AnalysisState).
eval/assimilation.py runs the sequential loop: after each satellite pass, re-calibrate to all
detections so far and forecast to the next pass; score the INCREMENTAL new burn (where naive
persistence has no skill) with Sorensen/FAR vs naive persistence and the uncalibrated prior. CLI
t4-assimilate.

RESULT (vhagar t4-assimilate, wind, prior ROS biased 0.6x, 6 passes): online calibration recovers the
bias, k -> ~1.73 (ideal 1.67); the analysis reconstructs the perimeter at full-Sorensen ~0.79 (near
the published conditional-GAN ~0.81); on the between-pass new burn it beats naive persistence (~0.43
vs 0.00) and the uncalibrated prior (~0.43 vs ~0.06) by a wide margin. New-burn FAR is high (~0.5-0.75)
and rises as the fire grows, the honest over-prediction from unmodelled suppression + prior spatial
error, which the generative / diffusion score-filter upgrades (torch, GPU; the U-Net in
models/ignition_conv is the machinery) are what reduce. Tests: tests/test_assimilation.py
(calibration recovers a known scale, analysis field = prior/k, assimilation beats baselines +
calibrates), all green, ruff clean.

T4 now has: level-set fast-marching propagation, honest incremental validation, AND arrival-time state
estimation with online per-fire ROS calibration + sequential assimilation. Next T4: anisotropic
wind-driven spread; the conditional-GAN / diffusion arrival-time model (torch); real perimeters
(NIROPS/VIIRS). Files: models/state_estimation.py, eval/assimilation.py, cli.py (+t4-assimilate),
tests/test_assimilation.py, docs/15.

## Phase 4 OPENED -- T4 spread: level-set propagation + honest incremental validation (2026-08-18)

Opened the last major science phase (T4 spread, docs/00 section 6). Built the physics propagation
core the architecture keeps first-class: models/spread.py solves the Eikonal front-tracking equation
|grad T|=1/ROS with a hand-rolled Fast Marching Method (no skfmm dep; O(N log N), exact upwind, no
self-intersecting polygons), plus a monotone fuel/wind/slope rate_of_spread, spread_forecast (front
-> horizon), and the mandatory persistence_buffer baseline. FMM verified: constant speed recovers
Euclidean distance (axis exact, diagonal within tolerance).

eval/spread.py is the honest validation harness, built to respect the architecture's stated ceiling
(next-day AP 0.35-0.45; "anything much above 0.5 is leaky or cumulative"). A synthetic fire is grown
to truth, then given three effects the forecaster cannot see (hidden suppression / fuel break,
spotting, fine-scale fuel heterogeneity); the forecaster propagates the t0 perimeter with a
spatially-correlated wrong ROS; scoring is on the INCREMENTAL new-burn region only (never cumulative),
with AP/IoU/Dice/burned-area-ratio/arrival-MAE, stratified wind vs plume. CLI t4-spread.

RESULT (synthetic, thin band base ~0.11): physics AP ~0.77 IoU ~0.57 Dice ~0.72 BAratio ~1.4 vs
persistence+buffer AP ~0.75 IoU ~0.45 vs persistence AP ~0.11 IoU 0. Honest read: physics beats both
mandatory baselines and IoU sits at the wind-driven band edge; absolute AP is OPTIMISTIC and flagged
as such (synthetic truth is a perturbed level-set, close to the forecaster's model class; real ceiling
is 0.35-0.45, driven by model-form error / fuel maps / wind / suppression that a synthetic cannot
reproduce); burned-area ratio >1 is the honest over-prediction from unmodelled suppression. Did NOT
crank noise to fake 0.4. Real numbers need real perimeters (NIROPS/VIIRS), a user data step.

Tests: tests/test_spread.py (FMM=distance, ROS monotone, forecast grows + probabilistic, buffer
dilation, physics beats baselines incrementally + BAratio>1), all green, ruff clean. Next T4:
anisotropic (wind-driven) spread, arrival-time state estimation from real detections + assimilation
(conditional-GAN result), diffusion/neural-operator surrogate + residual corrector (reuses the U-Net
in models/ignition_conv). Files: models/spread.py, eval/spread.py, cli.py (+t4-spread),
tests/test_spread.py, docs/15_T4_SPREAD.md.

## Phase 3 -- Layer-3 deep challenger in shadow mode (FSS + promotion gate) (2026-08-18)

Built T3's Layer-3 deep challenger the way docs/00 5.4 frames it: a spatial model admitted only as a
CHALLENGER to the gradient boosting and promoted only on blocked, proper-scored evidence. Added the
Fractions Skill Score to eval/metrics (fractions_skill_score, classic thresholded + probabilistic
variant): neighborhood verification for rare point-like events, since pixel-exact scoring punishes a
forecast that is right about where fire is likely but off by a cell. eval/danger_grid.py is the
shadow-mode harness: a gridded ignition world (ignition from clean spatially-coherent fields, model
sees noisy per-cell obs), a pointwise GBDT baseline vs a spatial challenger (booster on
neighborhood-pooled features) under leave-time-block-out CV, scored with FSS at 40/80/120 km +
base-rate-preserving AUPRC and Brier, with the promotion gate = beat baseline on AUPRC AND Brier.
models/ignition_conv.py is the real deep challenger: a compact U-Net trained with a differentiable
soft-FSS loss + BCE (torch-guarded, GPU box). CLI t3-challenger (--torch optional).

DEMO (vhagar t3-challenger, 36 days x 40x40 @ 20km, base rate ~0.05, obs noise 0.15, stable across
seeds): pointwise baseline AUPRC ~0.12 / Brier ~0.046 / FSS 0.58/0.75/0.82; spatial challenger ~0.16
/ ~0.045 / 0.64/0.81/0.88 -> PROMOTE (it denoises the observations and wins on AUPRC, Brier, and FSS
at every scale). The gate is the deliverable, not the verdict: when spatial context adds nothing the
same gate keeps the challenger in shadow, and the architecture expects the deep model to earn its
place rarely at daily lead times. Tests: tests/test_danger_grid.py (FSS perfect/neighborhood-growth/
probabilistic variant, grid scenario, pooling variance, shadow_evaluate signal + gate consistency,
torch shapes via importorskip), all green (torch test skips here), ruff clean.

T3 now has all three quantities AND the Layer-3 challenger: FWI (danger), cause-stratified ignition
probability, E[BA], and the shadow-mode deep-model gate. Files: eval/metrics.py (+FSS),
eval/danger_grid.py, models/ignition_conv.py, cli.py (+t3-challenger), tests/test_danger_grid.py,
docs/14.

## Phase 3 -- expected-burned-area head E[BA] (heavy-tailed, CRPS-scored) (2026-08-18)

Added the third of T3's three quantities (E[BA] = P(ignition) x E[BA | ignition]), the heavy-tailed
one, the way docs/00 5.1/5.4 demand. eval/burned_area.py: BurnedAreaModel fits a log-space quantile
gradient-boosting distribution (HistGradientBoostingRegressor loss=quantile per level) with a
generalised-Pareto peaks-over-threshold tail for the far quantiles; expected_burned_area combines a
per-cell P(ignition) with the predictive mean (trapezoid over quantiles). Scored with CRPS (quantile
decomposition) + pinball, NEVER RMSE. Added crps_from_quantiles + pinball_loss to eval/metrics. CLI
t3-expected-ba (--synthetic demo + --no-synthetic --fires real hook). lon/lat excluded; spatial-block
CV.

DEMO (vhagar t3-expected-ba --synthetic, heavy-tailed world median ~200 ha / p99 ~1.7k / max ~176k):
model CRPS ~235 vs climatology ~260 -> skill +0.09 (covariates carry real signal); GPD tail adds a
little at the extreme quantiles; RMSE fold std ~1100 on mean ~2400 (std/mean ~0.9) = tail-driven
instability, which is exactly why the architecture forbids selecting on RMSE. 6 new tests
(tests/test_burned_area.py: pinball known-value, CRPS rewards sharpness, heavy tail, monotone
quantiles + GPD fit, E[BA] scales with P(ig), beats-climatology + RMSE-instability regression), all
green, ruff clean. Three T3 quantities now exist: danger (FWI), ignition probability, E[BA]. Files:
eval/burned_area.py, eval/metrics.py (+pinball/crps), cli.py (+t3-expected-ba), tests/test_burned_area.py,
docs/14.

## Phase 3 OPENED -- T3 cause-stratified ignition model + the sampling-trap demo (2026-08-18)

Opened the next roadmap phase (T3 fire danger, docs/00 section 5), built the state-of-the-art way:
gradient boosting not a deep net (per ECMWF's operational Probability-of-Fire), with the effort on
the sampling design and honest scoring, which is where ignition models actually fail.

- datasets/danger.py (the "trap that ruins most ignition models", docs/00 5.6): target-group
  background sampling, negative stratification, King-Zeng rare-event prior correction, human/lightning
  cause split; all pure + unit-tested. Plus synthetic_reporting_scenario, a biased presence-only world
  so the trap can be shown, not asserted.
- eval/danger.py: cause-stratified, spatial-block-CV, proper-scored ignition eval reusing eval/metrics
  (AUPRC, Brier + Murphy decomposition, ECE, log loss, BSS vs base-rate climatology), prior-corrected,
  lon/lat excluded (the T1 leakage lesson). Permutation importance on a held-out block to expose which
  covariate the model leans on.
- CLI t3-ignition (--synthetic demo); docs/14_T3_DANGER.md.

DEMONSTRATION (vhagar t3-ignition --synthetic, spatial-block CV, prior-corrected): in a world where
true ignition depends only on weather/fuel but REPORTING depends on the human footprint, naive random
background inflates AUPRC (~0.33) and makes people/roads a top feature (human-footprint importance
+0.043); target-group background collapses that reliance (+0.011) and reveals the honest, lower skill
(~0.29). The architecture's warning, shown on data not asserted. (BSS negative by design: the
synthetic signal is deliberately weak; the point is the sampling effect.)

Tests: tests/test_danger.py, 12 tests incl. the trap regression (target-group footprint importance <
naive), all green, ruff clean. Real-data ingest loader now built too: frames_to_records +
reporting_weight (target-group intensity from occurrence density) turn a fire-occurrence table + a
candidate cell-day pool into the model's inputs; CLI t3-ignition --no-synthetic --occurrence
--candidates --features runs the full pipeline (verified end-to-end on a generated parquet). What
stays the user's networked step is ASSEMBLING those two tables: the fire-occurrence DB (FPA-FOD /
CWFIS) + covariate stack (fuels, VPD, SMAP, WUI, roads, lightning holdover). Layer 1 FWI already
existed (features/fwi.py); NFDRS, NWP forcing, and the Layer-3 DL shadow challenger remain. Files:
datasets/danger.py, eval/danger.py, cli.py (+t3-ignition, real + synthetic paths),
tests/test_danger.py, docs/14.

## Phase 5 -- live NRT ingest path (rolling GOES pull + background refresh) (2026-08-18)

Built the hot-path ingest so the console can be a live feed, not just the cached August window.
serve/ingest.py reuses VHAGAR's own resumable archive builder (vhagar.archive.backfill) to poll the
newest GOES-18/19 ABI L2 FDC granules from the public NOAA S3 bucket (anonymous), decode via the
existing io.goes_reader, append to a rolling store, and prune partitions past a retention window so
clustering stays fast; once and loop modes. The API was refactored from a one-shot lru_cache into a
swappable in-memory snapshot with a background refresher: VHAGAR_REFRESH_SEC>0 rebuilds the snapshot
on a cadence so new granules appear without a restart and requests never block on the ~30s clustering
(they read the last-good snapshot until the new one is ready). /api/health now reports refreshes +
last_refresh.

Tested offline (S3 is blocked in-sandbox and netCDF4 absent, so the actual pull is a user-machine
step, like every prior network pull in this project): backfill import + reuse path OK; ingest --help
OK; prune() unit test (drops old part-YYYYMMDD, keeps fresh) OK; API parses; background refresher
verified via TestClient (refresh count 0 -> 2 over ~3s, endpoints stay honest, no fabricated fields
survive a refresh). Console JS unchanged, still clean.

Run live (user machine, needs internet + s3fs/xarray/h5netcdf/pyproj): terminal 1
`python -m serve.ingest --out data/detections_nrt --sat 18 --interval 300 --retention-days 3`;
terminal 2 set VHAGAR_DET_DIR=...\data\detections_nrt\detections and VHAGAR_REFRESH_SEC=300, then
`uvicorn serve.vhagar_api:app`. Honest scope: GOES-18 covers the US West; add a --sat 19 ingester for
the East. Files: serve/ingest.py; serve/vhagar_api.py refactored (state holder + refresher + health
counters); serve/README.md "Live" section. This is the operational NRT wiring; the fuller architecture
hot path (SNS->SQS->worker->PostGIS->SSE) remains the production upgrade.

## Latest: Phase 5 -- console wired to real GOES FDC data via a self-hosted VHAGAR API (2026-08-18)

Took the console off the bundled sample and onto VHAGAR's real detections. New serve/vhagar_api.py
(FastAPI) reads the cached FDC parquet (188,639 GOES-18/19 detections, 2026-08-01..07 CONUS),
clusters them into fire events with VHAGAR's own parallax-aware fusion (geo_leo_tolerance_m plus the
same single-link rule as harmonize.fusion.cluster_detections, KD-tree neighbour search so it scales
to the CONUS week), and serves /api/events + /api/detections as GeoJSON plus /console, all from one
process (no Vercel, no Mapbox). Verified end to end via TestClient: 704 events total (89 in
California), /api/export + /console return 200, first load ~45s then cached for the process.

Honesty held to the letter. FDC gives position, FRP, brightness temperature, confidence, view zenith
and time; it does NOT give spread risk, fire weather, or a validated burned area, so those fields are
ABSENT, not invented (checked: no risk_score / wind / rh / temp / false_alarm in the payload). The
event polygon is the convex hull of a cluster's detection pixels, a detection FOOTPRINT, not burned
area. The API sets schema="fdc" and the console relabels accordingly: KPIs become detection footprint
/ cumulative FRP / peak FRP / GOES sensors; perimeters are colored by peak FRP not risk; the spread-
risk and weather panels are hidden; the incident feed sorts by peak FRP. Console JS node-checked, no
em dashes. Run: pip install -r serve/requirements.txt; uvicorn serve.vhagar_api:app; open /console.
Files: serve/vhagar_api.py, serve/README.md, serve/requirements.txt, serve/__init__.py; console made
schema-aware.

Polish (2026-08-18): (1) disk cache of the clustered events + detections keyed on the dataset
manifest fingerprint (serve/.cache, gitignored), so a warm start is 0.12 s vs ~45 s cold (measured);
VHAGAR_NO_CACHE=1 forces a rebuild. (2) Map legibility: event hull outlines now render in a pane
ABOVE the detection dots (bright, dashed, solid when selected) with a faint fill below, and the FRP
glow radius/opacity were toned down, so individual fires and their hulls read instead of one orange
mass.

## Phase 5 opened -- VHAGAR wildfire console + brand identity (product UI) (2026-08-18)

Pivoted from the science core to the platform/UI phase (roadmap Phase 5, docs/00 section 10).
Built VHAGAR's first operations UI and locked its visual identity. This is product/UI work; the
model science is unchanged since the transfer result below.

WHERE THE ROADMAP STANDS (docs/00 section 10):
- Phase 0 Foundations: DONE (grid + tiling, leakage-proof splits + CI, label spine, event registry).
- Phase 1 T2 burned area: DONE through the headline generalisation number. RBR baseline -> U-Net
  (+0.441) -> stack U-Net (+0.538) / siamese -> per-ecoregion + leave-one-continent-out -> Prithvi-
  EO-2.0 fine-tune rebalanced (+0.398), beats same-fire NBR (+0.398 vs +0.163), transfers CONUS->
  Europe beating CONUS-tuned NBR (+0.488 vs +0.372, wins 6/9). Only "make it decisive" (scale
  cohorts, user GPU/network) remains.
- Phase 2 T1 detection: done to honest Stage-0 / Stage-2 / temporal results. Detection POD 0.50,
  precision 0.94 / FAR 5.6% (parallax), latency 2 min; Stage-2 lat/lon leak reproduced; temporal-
  anomaly early detection an honest negative (does not beat FDC at matched FAR, with a night
  directional hint). NOT yet the hot-path GOES ingest / event-fusion service.
- Phase 3 T3 danger and Phase 4 T4 spread: NOT STARTED.
- Phase 5 Platform / UI: STARTED HERE. The map UI is the first deliverable. Alerting, PostGIS/
  TimescaleDB, tiles, and the model registry are still to build.

WHAT WAS BUILT (2026-08-18):
- vhagar_console.html (VHAGAR repo): a single self-contained wildfire operations dashboard. No
  build step, no Mapbox token, no external host (deliberately NOT Vercel). Leaflet with free Dark/
  Satellite/Terrain tiles. Reads a fire API's /api/events + /api/detections (GeoJSON, the FirePerim
  prior-art contract in docs/05); point it at a backend with window.VHAGAR_API, else same origin;
  bundled sample fallback so it always renders. Live/sample pill, 5-min auto-refresh, KPI strip
  (events, detections, burned area, total FRP, top spread risk, false-alarm screen %), prioritised
  incident feed (click -> flyTo + detail drawer with weather and GeoJSON/KMZ export), risk-colored
  perimeter polygons, FRP glow+dot detections, wind-spread arrows. JS node-checked, no em dashes.
- Brand identity: Vhagar dragon-head emblem (green dragonfire, gold ring) cropped from the user's
  art, saved in brand/ (square mark + circular 256/128/64/32 + favicon). Emblem logo and favicon
  embedded in the console as data URIs; green-flame accent in the wordmark; functional fire/risk
  colors left untouched.
- portal_fire.html (PipelineWatch_NG lillie): VHAGAR wired into the Lillie product suite as a per-
  AOI portal behind the same deny-by-default aoi_allowed gate as the oil/mining portals
  (_build_fire_data + dispatch branch in view_portal; demo AOI static/fire_watch/aoi/great-basin-
  demo.json).

NEXT (roadmap choices): (a) continue Phase 5 -- wire the console to real VHAGAR fire data (adapt
the lillie server, or bring a small events/detections API into the VHAGAR repo), then add alerting
and tiles; (b) return to the science and make the T2 transfer decisive (scale CONUS cohort to ~60
fires + a larger European set); or (c) open Phase 3 (T3 danger). T3 and T4 are the untouched
roadmap phases.

## TRANSFER RESULT -- CONUS-trained Prithvi generalises to Europe, beats NBR (+0.488 vs +0.372) (2026-08-16)

Ran the leave-one-continent-out transfer end to end (transfer notebook: train CONUS burn-
balanced + dice, predict 9 EU chips with that checkpoint; scored locally via t2-prithvi-transfer).
RESULT: CONUS-trained Prithvi on European fires mean skill +0.488, wins 6/9, vs CONUS-tuned NBR
threshold +0.372 (3/9). Per-fire Prithvi-NBR: +0.44,+0.05,+0.38,+0.18,+0.34,+0.15 (wins),
-0.12,-0.16,-0.21 (losses). Foundation model fine-tuned ONLY on US fires transfers across the
Atlantic and beats a spectral threshold by +0.116 mean -- genuine cross-continent generalisation.
CONUS test IoU this run 0.195 (consistent w/ the 0.21 rebalanced run). docs/13 "Transfer result".

Full T2 Prithvi arc (all same-code-path, every setback diagnosed+fixed): built -> naive under-
performs (imbalance) -> rebalanced U-Net-competitive (+0.398) -> beats same-fire NBR baseline
(+0.398 vs +0.163) -> transfers to Europe, beats CONUS-tuned NBR (+0.488 vs +0.372). Honest
scope: 9 EU fires modest, Prithvi loses 3/9; direction clear. A strong, complete positive result.

Note: transfer-notebook bug found -- predict_eu.py had files.download INSIDE the subprocess (fails,
no kernel); masks still zipped fine, download must be in-kernel. (Claude ran the transfer scoring
directly from the user's Downloads via the mount.)

PATCHED (2026-08-16): colab/prithvi_transfer_colab.ipynb now writes masks+zip in the subprocess
and downloads in a SEPARATE in-kernel cell (bug fixed). t2-prithvi-transfer output now prints
"Prithvi wins N/M" (6/9). Suite green, ruff clean.

TO MAKE RESULTS DECISIVE (user, network pulls; code is done):
- CONUS: t2-prithvi-build --max-fires 60 --select size -> t2-prithvi-export --burn-balance ->
  re-Colab (60 fires -> ~9 test fires vs 3).
- Europe: emsr-candidates --year-min 2018 -> emsr-ingest (bigger emsr.csv) -> t2-prithvi-build-emsr
  -> t2-prithvi-export-infer -> predict with CONUS ckpt -> t2-prithvi-transfer.

## wired the leave-one-continent-out transfer test (CONUS-trained Prithvi vs Europe) (2026-08-16)

European fires ARE on disk (emsr.csv + EMSR delineation shapefiles + old RBR emsr samples), just
not in the main registry (0 europe records). Built the transfer-test enablement (all pure/tested
except the pull): t2-prithvi-build-emsr (6-band European samples, EMS delineation reference, reuses
read_emsr + read_emsr_reference_on_grid + build_prithvi_sample); export_inference_chips / t2-
prithvi-export-infer (chip a cohort flat, no split, for predicting with a model trained elsewhere);
nbr_threshold_transfer (tune NBR cut on CONUS, score Europe -- spectral analogue of the transfer);
t2-prithvi-transfer (stitch European Prithvi preds, score vs EMS, compare to CONUS-tuned NBR). +2
tests. Full suite green, ruff clean. docs/13 "Leave-one-continent-out transfer test".

Workflow (user): t2-prithvi-build-emsr (S2 pull, ~9 EU fires) -> t2-prithvi-export-infer -> Colab:
predict EU chips with the CONUS checkpoint -> t2-prithvi-transfer. Read: if CONUS-Prithvi keeps a
skill margin over CONUS-tuned NBR on European fires, pretraining bought cross-continent transfer.

## same-fire baseline -- Prithvi (+0.398) BEATS post-NBR threshold (+0.163) on identical fires (2026-08-16)

Added the first strict same-code-path comparison. nbr_threshold_baseline() fits one post-fire
NBR cut on train fires, scores the IDENTICAL 3 test fires with the SAME skill-over-naive metric
(_post_nbr uses bands 3=nir08, 5=swir22). CLI t2-prithvi-baseline (pure numpy, no GPU). Ran on
real cache: NBR-threshold mean skill +0.163 (MN -0.047, WA +0.320, WA +0.218) vs rebalanced
Prithvi +0.398 (MN +0.042, WA +0.520, WA +0.634). Prithvi beats the spectral threshold by
+0.235 mean AND wins all 3 fires individually. This is the honest apples-to-apples the +0.54
(U-Net's own CV) couldn't give: the foundation model earns its keep over a pointwise cut here.
+1 test. Full suite green, ruff clean. docs/13 "Same-fire baseline".

Open items unchanged (both local, no GPU): more CONUS fires + European set (leave-one-continent-
out) for a full verdict; optional strict t2-unet on these exact 3 fires (needs torch).

## REBALANCED Prithvi fine-tune is U-Net-competitive (+0.398 skill, was +0.054) (2026-08-16)

Re-ran on Colab T4 with --burn-balance (318 train chips, 63% burn) + LOSS=dice, 60 epochs. The
fix worked, big jump: test burn-IoU 0.10->0.21, burn recall 11%->50%, per-fire mean skill
+0.054 -> +0.398 (ALL 3 fires positive now): MN +0.042 (was a total miss -0.098), WA +0.520
(F1 0.82), WA +0.634 (F1 0.82). vs U-Net +0.54: two of three fires at/above U-Net level; the
small MN fire pulls the mean. So a rebalanced foundation-model fine-tune is COMPETITIVE with the
small model here -- complete turnaround, vindicates the imbalance diagnosis.

Complete honest arc for T2 Prithvi: pipeline built + validated end-to-end on GPU; naive fine-
tune underperforms (+0.054) by class-imbalance under-detection; diagnosed; fixed (burn-balanced
chips + dice); recovered to U-Net-competitive (+0.398). docs/13 "The rebalanced result".

STILL FOR A VERDICT (both local, no GPU): (1) strict same-fire comparison -- run t2-unet /
t2-stage0 on these EXACT 3 test fires (the +0.54 was U-Net's own CV; these fires' naive F1s
0.10-0.30 are a specific small sample). (2) SCALE -- more CONUS fires + European set for leave-
one-continent-out transfer. Note: watch Colab download versioning -- browser saves
prithvi_preds (N).zip, don't score a stale one.

## first Prithvi fine-tune RAN on Colab T4 -- honest underperformance; added rebalancing (2026-08-16)

Full Prithvi pipeline ran end-to-end on a Colab T4. 20 fires -> 470/75/138 chips -> fit (ce,
~29 ep, early-stopped) -> test -> 138 preds -> download -> t2-prithvi-score. Fixes that got it
running: torchgeo==0.7.1 pin; removed predict_* config keys; no-restart notebook running
fit/test/predict as SUBPROCESSES (dodges pip's numpy-under-kernel conflict); session-upload data.

RESULT (honest, underwhelming): test burn-IoU 0.10, burn recall 11%, not-burned 98%; per-fire
mean skill +0.054 (per fire: miss -0.098 / +0.013 / +0.246) vs U-Net +0.54. Straight fine-tune
badly UNDER-DETECTS burned area = CLASS IMBALANCE (wide windows -> mostly-unburned pixels/chips,
plain CE predicts "not burned"), NOT a foundation-model defect. Benchmark 87.5 used curated
burn-balanced chips; ours aren't.

FIX built + tested: (1) chip_sample(burn_balance=True) / t2-prithvi-export --burn-balance:
denser stride, keep all burn chips, cap background to max_bg_ratio x burn; TRAIN split only
(val/test faithful). Real data: train burn-chip frac 42%->65% (199->713 burn chips). (2) Colab
config now has a LOSS param (default 'dice', imbalance-robust) instead of ce. +1 test. Full
suite green, ruff clean. Notebook regenerated. docs/13 "First real fine-tune result".

NEXT (user): re-export --burn-balance, re-run Colab with LOSS='dice' for the fair test. Also
run t2-unet on the SAME 3 test fires for strict apples-to-apples. Then scale fires + Europe.

## terratorch validated config+chips; wired per-chip->per-fire stitch; no local GPU (2026-08-16)

terratorch fit RAN on the user's machine (after pinning torchgeo==0.7.1 -- terratorch's
torchgeo constraint is currently commented out/un-pinned, so pip pulled an incompatible newer
one that dropped torchgeo.trainers.utils): loaded prithvi_burnscars_vhagar.yaml, downloaded
Prithvi-EO-2.0-300M weights, built the 324M model, started training. So the config + chip
dataset are validated against terratorch itself. BUT GPU available: False -- the box has no
NVIDIA GPU (nvidia-smi not found; cu121 torch still reports cuda False). Real fine-tune ->
cloud GPU; everything transfers as-is.

Closed the last integration gap: terratorch predicts PER CHIP, scoring needs PER FIRE.
export_prithvi_chips now writes _chips.json (stem -> event_id,y0,x0,H,W); new
stitch_chip_predictions reassembles per-chip preds into per-fire masks (offset placement,
edge-clip, burned-wins overlap). t2-prithvi-score --chips-manifest stitches then scores via
the same skill-over-naive as RBR/U-Net. +2 tests (stitch reassembles quadrants; clips edge
overhang). Full suite green, ruff clean.

Remaining: run terratorch fit on a cloud GPU (or capped-CPU smoke test) -> predict test chips
-> t2-prithvi-score --chips-manifest -> compare mean skill to U-Net +0.54 on same test fires.
Note: 20 fires = 15/3/3 split (only 3 test fires) -> enough to validate, too few to settle;
scale fires + add European fires for leave-one-continent-out before claiming a result.

## Prithvi six-band pull confirmed on real fires; terratorch config generated (2026-08-16)

Ran the real six-band pull end-to-end (user's machine): 20 CONUS-2021 fires cached at 30 m
(first run hit transient earth-search APIErrors; resumable re-run got all 20), then export ->
470/75/138 train/val/test chips. Verified on disk: [6,224,224] float32 reflectance in [0,~0.9],
int16 labels {-1,0,1}, split by whole fire (15/3/3), image/label counts matched. The whole
VHAGAR side of the Prithvi pipeline is now confirmed on real data, not just unit tests.

Fetched the authoritative burn_scars_config.yaml and aligned the export to its exact layout:
single out_dir/data with {stem}_merged.tif + {stem}.mask.tif + out_dir/splits/{train,val,
test}.txt (was separate images/labels dirs). Generated prithvi_burnscars_vhagar.yaml at repo
root: the published config with only data paths + per-band means/stds changed; the means/stds
were COMPUTED from this dataset's 15 training fires (19.6M valid px, no val/test leakage) so
they match Sentinel-2 L2A reflectance not HLS. means=[0.0498,0.0687,0.0765,0.2146,0.2049,
0.1462], stds=[0.0311,0.0363,0.0496,0.0850,0.0943,0.0826].

Suite green, ruff clean, YAML parses. docs/13 updated. User must RE-EXPORT (layout changed):
vhagar t2-prithvi-export --cache-dir data\t2_prithvi --out-dir data\t2_prithvi_chips --chip 224,
then pip install terratorch + terratorch fit -c prithvi_burnscars_vhagar.yaml (GPU), then
vhagar t2-prithvi-score vs U-Net +0.54 on the same test fires.

## Prithvi pipeline complete VHAGAR-side (chip export + fair scoring bridge) (2026-08-16)

Completed the VHAGAR side of the Prithvi fine-tune so only network-pull + GPU-fit remain.
eval/t2_prithvi.py (all pure/tested except the rasterio write):
- grouped_split: whole fires -> train/val/test, leakage-proof (no fire in two splits).
- chip_sample: tile a 6-band T2Sample into chip image [6,c,c] + label [c,c] int8 in HLS Burn
  Scars convention (0 unburned / 1 burned / -1 nodata); pads small samples up to one chip.
- write_chip_geotiffs / export_prithvi_chips: paired 6-band image + signed-label GeoTIFFs under
  out_dir/{split}/{images,labels} + _split.json (rasterio-guarded). CLI t2-prithvi-export.
- score_masks / summarise_scores: predicted burned masks -> F1/IoU + skill-over-naive per
  held-out fire via the SAME confusion_counts as RBR/U-Net, so the head-to-head is one code
  path. CLI t2-prithvi-score (reads {event_id}.tif preds, prints per-fire + mean skill).
+4 tests (grouped_split partition; chip_sample 6-band signed labels + small-sample pad; score
matches confusion/naive). Full suite green, ruff clean. docs/13 updated (build->export->fit->score).

Only user's-machine steps remain: t2-prithvi-build (S2 network pull) and terratorch fit (HF
weights + GPU). Everything else assembled + scored leakage-proof, unit-tested.

## pivoted to T2 Prithvi-EO-2.0; built the six-band re-pull (SOTA foundation model) (2026-08-16)

Banked T1 temporal (honest negative + directional hint) and pivoted to the other frontier item:
fine-tune Prithvi-EO-2.0 (NASA/IBM geospatial foundation model) for T2 burned area, to try to
beat the U-Net's +0.54 skill. Researched the real recipe (model card + terratorch): Prithvi-EO-
2.0-300M + UNet decoder, 6 HLS bands (Blue/Green/Red/narrow-NIR/SWIR1/SWIR2), single timestamp,
weighted CE + LoRA, `terratorch fit`. 87.5 IoU on HLS Burn Scars benchmark.

Built + tested offline the one new prerequisite, the SIX-BAND RE-PULL (RBR used only 2 bands):
- io.optical.sentinel2_bands6 + stream_band_composite: post-fire 6-band cloud-masked surface-
  reflectance composite [6,H,W] in Prithvi band order, streaming/flat memory. PRITHVI_BAND_ASSETS.
- datasets.t2_optical.build_prithvi_sample: pairs the 6-band stack with the MTBS mask on the
  SAME grid/window/reference as the RBR sample (so Prithvi is scorable head-to-head vs RBR/U-Net).
  Cached T2Sample .npz (tag p6). Stubbed-pull unit test.
- CLI t2-prithvi-build: select fires + build/cache the 6-band dataset.
- docs/13_T2_PRITHVI.md: full runbook (build -> export terratorch chips on grouped folds ->
  pip install terratorch + HF weights -> terratorch fit -> score via skill-over-naive on the
  same folds). Honest caveats: S2-vs-HLS domain shift; only fair vs RBR/U-Net on same fires.

+2 tests (stream_band_composite masks/scales; build_prithvi_sample stacks 6 bands vs MTBS).
Full suite green, ruff clean. STILL USER'S TO RUN (network/GPU): the 6-band pull, terratorch
chip export, the fine-tune, fold-wise scoring. Next code increment: chip export + scoring bridge.

## 5th artefact -- non-detection masked as zero-lead tie; now report DETECTION RATE (2026-08-16)

The learned cohort table was internally impossible (per-fire medians of 0 next to stratum
"100% px led"), which exposed an accounting bug: when the residual NEVER crossed threshold for
a fire in the held-out window, real_lead_experiment returned median_lead_min=0 (a TIE) counting
all fire px but adding nothing to the pooled distribution -- a MISS shown as a draw. So the
earlier "night median 0" (both hourly + learned) was largely NON-DETECTIONS, not ties: the
detector was failing to detect many night fires at all at FAR 0.01 / 3-frame / held-out, not
matching FDC.

Fix: RealLeadResult carries n_fire_pixels_total + detection_rate; a non-detecting fire reports
median_lead_min=NaN (not 0) and counts as not-led. cohort_lead_summary leads with detection
rate (px flagged at all in held-out window), then lead AMONG DETECTED px. CLI prints "no
detection" in red per missed fire; table leads with detection rate. +1 test (miss lowers det
rate, counts not-led, excluded from lead median). Suite green, ruff clean.

HONEST BOTTOM LINE (5 artefacts fixed: contamination, global-threshold night-blindness,
first-crossing false alarms, training-on-test, non-detection-as-tie): infra + method are solid
(real pull, physics+learned forecasters, matched-FAR/per-tod/persistence/train-test protocol,
stratified cohort w/ control, detection-rate-first scoreboard). NO lead over FDC established: a
single-band 3.9um temporal-residual detector, scored honestly, does not beat FDC and often
doesn't detect fires before it at a defensible FAR. Best banked at this honest negative.
docs/12 "The fifth flaw" + "Honest bottom line for the T1 temporal component".

HONEST COHORT NUMBERS (hourly mean, FAR 0.01, 6 bins, 3-confirm, held-out):
  night_coldstart: 3 fires, det 23% px (67% fires), fires led 33%, median lead(det) +72,
                   pooled px +175, px led 94%. Per-fire: miss(0/80), +175(72/163), -30(6/100).
  day:             2 fires, det 29% px (100% fires), fires led 50%, median lead(det) -141,
                   pooled px -430, px led 17%. Per-fire: +148(2/2), -430(27/97).
Read: (1) LOW SENSITIVITY -- detects only ~25% of fire px at defensible FAR; rules it out as a
stand-alone detector, at best a lead-time supplement on px it catches. (2) DIRECTIONAL SIGNAL
conditional on detection -- night detected px LEAD FDC (+175 pooled, 94%), day detected px LAG
(-430 pooled, 17%): the mechanism's signature (helps at night where absolute threshold slowest,
not by day). Caveat: tiny cohort, effectively 1 informative fire/stratum -> suggestive, not
demonstrated. Best banked here: well-instrumented honest negative WITH a directional hint.
docs/12 "The honest cohort result" + "Honest bottom line". Suite green, ruff clean.

## first trustworthy cohort result -- ties FDC on night, trails on day (directional, underpowered) (2026-08-16)

With train/test split + clean controls, leads are physically plausible for the first time.
Fire-level medians (hourly mean, FAR 0.01): night_coldstart 3 fires, 33% led, median 0 min;
day 2 fires, 50% led, median -141 min (1 day fire skipped, no in-box FDC). DIRECTION is right
(night 0 > day -141: residual relatively better where an absolute threshold is slowest) but
night is a TIE not a win, and n=5 with per-fire pixel counts 2..80 is noisy. Added pooled-
pixel aggregation (cohort_lead_summary now reports pooled_pixel_median_lead + frac, robust to
tiny fires; RealLeadResult carries per-pixel leads_min) + CLI columns. +updated test.

HONEST BOTTOM LINE (after 4 artefact fixes, real GOES pull, learned+physics forecasters,
stratified cohort w/ control): a single-band 3.9um temporal-residual detector does NOT beat
GOES FDC on this week's fires; it ties on night cold-starts, trails on day. Directional
signal consistent with the mechanism, far short of a demonstrated lead, underpowered at n=5.
Matching a mature multi-band operational product with one band + one GOES-week is a
reasonable, honest landing. A real lead would need a much larger cohort + multi-band context,
not more single-band tuning.

Suite green, ruff clean. docs/12 "The first trustworthy cohort result". Next: optional
--learned run on the SAME clean cohort (reuses cubes, torch) to see if it lifts night tie->
slight lead; then bank the temporal component. Other frontier item remains: Prithvi-EO
fine-tune for T2.

## 4th artefact (training-on-test) fixed with a train/test split; clean day controls (2026-08-16)

First cohort run gave enormous night leads (+275..+755 min) and auto-excluded ALL day
controls -- both bugs, not results. (1) TRAINING ON TEST: the residual detected on the same
frames the baseline was fit on, and a fire pixel differs from fire-free calibration pixels
for ordinary reasons (warmer surface), so it crossed the spatial threshold DURING the
pre-ignition baseline -> a ~10h pre-fire false detection scored as lead. Fix: real_lead_
experiment(..., eval_start=clear_end) counts detections only on held-out post-baseline frames
(same period FDC first flags the fire). CLI passes eval_start=clear_end in both t1-temporal-
real and t1-temporal-cohort. (2) CONTAMINATED CONTROLS: the box (wider than the cluster cell)
caught a neighbouring earlier fire in the baseline span. Fix: select_fire_cohort anchors each
window on the earliest FDC detection IN THE BOX and ends the baseline 3h before it -> fire-
free training by construction. Verified on real FDC: all 6 fires (3 night + 3 day) now clean.

+1 test (eval_start ignores a sustained in-baseline excursion, detects post-split). 4th
artefact caught by the same discipline (implausible number + broken comparison -> method
error, not banked). Suite green (18 temporal-file tests), ruff clean. docs/12 "The fourth
flaw: training on the evaluation frames".

Re-run (windows changed -> re-pull): vhagar t1-cohort-select -> vhagar t1-cohort-pull --spec
cohort\cohort.json --refetch -> vhagar t1-temporal-cohort --spec cohort\cohort.json
--detections data\detections\detections --far 0.01 --far-bins 6 --min-consec 3 [--learned].
This is the first cohort measurement with a real control and no training-on-test.

## stratified fire-cohort eval harness (the right SOTA test, n>1 with a control) (2026-08-16)

n=1 can't answer whether the temporal detector works. Built the proper test: a cohort
stratified by the condition theory says matters. select_fire_cohort clusters FDC into fires,
computes ignition local-solar-hour + early FRP ramp, picks night_coldstart fires (the
residual's edge) + day controls, each with a ready pull window + pre-ignition clear_frac.
CLI t1-cohort-select (writes spec JSON + prints pull cmds) and t1-temporal-cohort (scores
every fire through the SAME matched-FAR/far-bins/persistence/contamination pipeline, hourly-
mean or --learned, aggregates lead over FDC PER STRATUM via cohort_lead_summary). Ran
selection on real GOES-18 FDC: 3 cold-start late-night Sonora fires (~28-29N, local 23-24h,
slow ramp) + 3 day controls. Scientific read: leads FDC on night stratum but not day =
evidence for the mechanism; both/neither = not.

+3 tests (cohort_lead_summary aggregation; select_fire_cohort stratifies night/day on a
synthetic parquet; selection clear-window ends before ignition). Full suite green, ruff
clean. Harness + selection + aggregation unit-tested offline; the 6 small cube pulls are the
user's to run (need S3). docs/12 "The right test: a stratified fire cohort, not one fire".

Workflow (one-shot pull added, cohort_pull + t1-cohort-pull, resumable/skip-existing, +1
test): vhagar t1-cohort-select -> vhagar t1-cohort-pull --spec cohort\cohort.json -> vhagar
t1-temporal-cohort --spec cohort\cohort.json --detections data\detections\detections --far
0.01 --far-bins 6 --min-consec 3 [--learned]. Cohort selection already reproduced on user's
machine (3 Sonora night fires + 3 day controls); pulls + score pending (need S3).

## learned forecaster ran -- robust negative, FDC not beaten on this fire (2026-08-16)

Ran --learned (TemporalAnomalyNet, 15 epochs, window 6, +solar) on the north-cell cube:
+435/-95/-222 min at FAR 0.05/0.01/0.002. Marginally better than the hourly mean at moderate
FAR (-95 vs -170 at 0.01) but SAME verdict: at any defensible FAR the residual lags FDC; at
strict FAR only 14 of 24 fire pixels detected. Conclusion now ROBUST across forecasters: a
crude mean AND a learned net w/ solar covariate, both trained on one diurnal cycle, both fail
to beat FDC on this night fire at matched FAR.

Two honest reads (neither a defect in the residual idea): (1) DATA SCALE -- both forecasters
see only ~30h (one diurnal cycle); the untried lever is a multi-day pull (common factor in
both failures, no new code, just a longer t1-pull-cube window). (2) SAMPLE SIZE -- n=1 fire,
and FDC handles it well even at night; the residual's edge is specifically cold-start slow
night fires; a fair verdict needs a cohort. The synthetic +70min is a synthetic property; on
real data so far FDC is not beaten. Honest state, banked. docs/12 "The learned result, and
the robust conclusion". Suite green, ruff clean.

Decision point for user: (a) multi-day baseline pull (more history, highest-value single
lever), (b) cohort of fires (fair test of the night-fire edge), or (c) bank the honest
negative + full hardened protocol as the T1 temporal deliverable and pivot to Prithvi/T2.

## learned TemporalAnomalyNet wired end-to-end (SOTA forecaster path) (2026-08-16)

User chose the state-of-the-art lever. Wired the learned forecaster end-to-end so it runs as
one command: t1-temporal-real --learned trains TemporalAnomalyNet on the cube's clear-sky
span and feeds residuals to the SAME matched-FAR / far-bins / min-consec protocol.
learned_residuals() = train_temporal_net(cube[:clear_end], solar covariate) then
temporal_net_residuals() over the full cube. NaN-robust: BT mean-centred + holes filled with
0 so the 3D conv sees in-distribution values, but the forecasting loss is MASKED to finite
target pixels and residuals reset to NaN where input was NaN -- cloud never trains or scores.
Solar covariate = cos(solar_zenith_cube) models the daytime 3.9um reflectance the hourly mean
can't (the thing that forced the high threshold). CLI: --learned/--baseline, --window,
--epochs, --no-solar. +1 torch-guarded test (NaN-safe learned residuals feed the experiment).

Couldn't runtime-verify in sandbox (torch won't install: CPU index proxy-blocked, default
build's CUDA deps exceed 2.4G free). Code is torch-guarded, shapes matched to
TemporalAnomalyNet (B,T,C,H,W), non-torch suite green, ruff clean; torch tests run on user's
machine. Run on existing cube (no new pull):
  vhagar t1-temporal-real utah_north_cube.npz --detections data\detections\detections --clear-frac 0.7 --far-bins 6 --min-consec 3 --learned --epochs 15
Same three artefact guards apply; read the same way (lead must hold as FAR tightens + be
physically plausible). If it still doesn't clear FDC, remaining lever is a multi-day baseline
pull (data-scale limit, not modelling-idea limit). docs/12 "The learned forecaster, wired".

## TRUSTWORTHY temporal read -- crude baseline does NOT beat FDC (honest negative) (2026-08-16)

With all three artefacts controlled (clean baseline, --far-bins 6, --min-consec 3), the
honest table on the north-cell fire: +692/-170/-225 min at FAR 0.05/0.01/0.002. At any
defensible FAR the residual detector LAGS FDC (-170 at 0.01, -225 at 0.002); for 11 of 24
fire pixels it never confidently detects at strict FAR (count 24->13). The +692 at 0.05 is
the last loose-threshold artefact: a 1-diurnal-cycle baseline holds real per-pixel BIAS
(~1-2 samples/hour), so at a permissive cut the residual crosses pre-fire; at a strict cut
that bias forces a high threshold and FDC's mature multi-band algorithm wins.

CONCLUSION (plainly): on this real fire, a crude hourly-mean diurnal-residual detector does
NOT beat GOES FDC at matched FAR. The synthetic +70min demo does not transfer to this
baseline. Real negative result; cause localised to the BASELINE, not the residual idea.

Two levers remain (both need more input than the existing cube): (1) multi-day clear-sky
baseline pull -> many samples/hour -> less bias -> lower matched-FAR threshold (just a longer
t1-pull-cube window, no new code); (2) learned TemporalAnomalyNet + solar_zenith_cube
covariate (models daytime solar reflectance the mean can't; needs torch). Banked regardless:
real GOES pull + matched-FAR lead-time protocol vs FDC + three artefacts found/fixed by a
collapse-across-FAR diagnostic. docs/12 "The trustworthy read". Suite green, ruff clean.
Awaiting user's choice of lever.

## 3rd temporal artefact (first-crossing false alarms); added persistence confirm (2026-08-16)

--far-bins 6 produced huge leads (+2048/+705/+682 min) -- another artefact. The estimator
took the FIRST residual exceedance anywhere in the 42h record as detection time; over ~500
frames a per-frame FAR of 0.01 gives every pixel several isolated false exceedances, and the
sensitive night threshold lands them the night BEFORE ignition, so a single pre-fire blip
scores as an 11-34h "lead". Matched FAR controls the rate, but first-crossing is maximally
gamed by it. A 34h lead points to before the fire existed -> not real.

Fix: persistence. real_lead_experiment(..., min_consec=k) via _first_persistent() requires k
consecutive exceedances before declaring detection (how contextual detectors confirm across
scans); isolated blips filtered, only a sustained ramp counts, alarm at run end. CLI
--min-consec (default 3 = 15min). At FAR 0.01 a 3-frame false run is ~1e-6/position, so
pre-fire false detections vanish. +1 test (persistence ignores a pre-fire spike, detects the
ramp). Suite green (13 temporal tests), ruff clean. docs/12 "The third flaw".

Three artefacts now, each caught by the same collapse-across-FAR / physical-implausibility
diagnostic, not a headline: (1) baseline contamination [guard], (2) global-threshold night-
blindness [per-time-of-day threshold], (3) first-crossing false alarms [persistence]. The
next run is the first TRUSTWORTHY read:
  vhagar t1-temporal-real utah_north_cube.npz --detections data\detections\detections --clear-frac 0.7 --far-bins 6 --min-consec 3
If a positive strict-FAR lead holds at a physically plausible margin, the differentiator is
real on this fire. If the ~1-diurnal-cycle baseline is too thin, levers: multi-day baseline
pull or learned TemporalAnomalyNet + solar covariate. Deliverable is as much the hardened,
self-skeptical protocol as any single number.

## clean temporal run exposed a 2nd flaw; added per-time-of-day FAR threshold (2026-08-16)

Re-pulled with a genuinely pre-ignition baseline (north cell -112.35,38.85,-112.05,39.10,
08-01 00:00..08-02 18:00, 504 frames, fire ignites 07:47 UTC with ~32h clear history).
Contamination guard silent (good). But the crude hourly-mean residual STILL did not beat
FDC: +668/+5/-192 min at FAR 0.05/0.01/0.002 -- same collapse, clean baseline, so a
different cause. It's a flaw in my own protocol: the residual threshold was a single GLOBAL
percentile across all hours, but daytime 3.9um residual variance is far larger (solar
reflectance), so a global cut is set by daytime and desensitises night -- exactly when this
fire ignites (~01:47 local). I re-introduced the night-blindness the residual detector is
meant to cure.

Fix: per-time-of-day threshold. real_lead_experiment(..., far_bins=N) / t1-temporal-real
--far-bins 6 splits the day into bins and calibrates each to target FAR on fire-free pixels
pooled in that bin, so a night fire is judged against the night distribution. +1 test (per-
time-of-day threshold detects a night excursion a global threshold misses). Suite green (12
temporal tests), ruff clean. docs/12 "The clean run, and the second flaw it exposed".

Next measurement (reuses existing cube, NO new pull): re-run
  vhagar t1-temporal-real utah_north_cube.npz --detections data\detections\detections --clear-frac 0.7 --far-bins 6
If the strict-FAR lead flips positive, the differentiator holds; if the ~1 diurnal cycle
baseline is still too thin (1-2 samples/hour), next lever is the learned TemporalAnomalyNet
+ solar covariate, or a multi-day baseline pull. Method note: every flaw here was surfaced
by the collapse-across-FAR diagnostic, not a headline number.

## first real temporal pull ran; caught a baseline-contamination artefact (2026-08-16)

Ran the real pull for the first time (user's machine, S3). Central-Utah fire, bbox
-112.55,38.65,-112.05,39.10, 08-01 18:00..08-03 00:00: cube (360,23,31), 98% valid. The
lead-time table read +925/+642/-78 min across FAR 0.05/0.01/0.002 -- and the +925 is an
ARTEFACT, not a win. The fire ignited 19:47 UTC, 1h47m after the window opened, but
clear-frac 0.6 fit the diurnal baseline on the first 18h, overlapping the active fire by
~16h. Contaminated baseline + loose FAR -> residual trips on diurnal/baseline noise ->
fake 15h lead. The tell: lead collapses as FAR tightens (+925 -> +642 -> -78); a real
signal holds its lead. At the only strict FAR the residual LAGS FDC. Honest null result,
correctly surfaced by the framework (the T1 twin of the T2 naive-baseline lesson).

Hardened the code so this can't be presented as a result again: baseline_contamination()
measures the share of fire pixels first-detected inside the clear window; t1-temporal-real
prints a red refusal above 20%. Silenced the benign empty-hour-bin nanmean warning. +1
test. Suite green, ruff clean. docs/12 "First real run, and the trap it exposed".

Clean re-run recipe (still to run): a fire that ignites mid-window with a full pre-fire
diurnal baseline, e.g. north cell bbox -112.35,38.85,-112.05,39.10 (ignites 08-02 07:47,
zero detections before 08-02), pulled from 08-01 00:00 with --clear-frac ~0.7 so the
baseline ends before onset. Onset is at night: the ideal case for the residual detector.

## T1 temporal-anomaly real-data pull + FDC lead-time eval (2026-08-16)

Built the real input and the real comparison for the temporal detector, the heavy piece
that was outstanding. archive/temporal_cube.py: pull_bt_cube reads GOES ABI L2 CMIP band 7
(3.9um) from public S3, crops each 5-min frame to a small bbox, stacks onto the one
stationary ABI fixed grid into a [T,H,W] cube carrying its own UTC timestamps+geometry.
Grid alignment asserted (shape+corner nav; mismatched frames dropped, never misaligned);
NaN stays NaN. CLI `t1-pull-cube`. Plus solar_zenith_cube covariate, load/save_bt_cube.

Closed the loop against GOES FDC: `t1-temporal-real` fits HourlyBaselineForecaster (NaN-safe
per-pixel per-hour mean, the on-the-fly DiurnalClimatology; real cubes' NaNs rule out the
harmonic lstsq) on the leading clear-sky fraction, then real_lead_experiment times the
residual's first threshold crossing vs FDC first-detection per pixel, threshold calibrated
to matched FAR on the fire-free pixels. Positive lead = residual beat FDC at equal FAR. This
is the synthetic +70min demo re-run on real observations. fdc_first_detection_grid maps FDC
parquet detections to cube pixels (nearest, <=3km reject). Learned upgrade is drop-in: swap
TemporalAnomalyNet residuals into real_lead_experiment; protocol unchanged.

4 new offline tests (cube assemble/align drops mismatched grid; solar zenith ~small at local
noon; hourly baseline NaN-safe+removes diurnal; real_lead_experiment residual leads a late
FDC). End-to-end smoke test on a synthetic cube+fake FDC parquet: residual leads +50 min at
FAR 0.01. Suite green, ruff clean. docs/12 "Running it on real GOES data (the pull)".

Still the user's to run: the S3 pull itself (network + a bbox/window with a real fire) and,
if wanted, the torch TemporalAnomalyNet training. All alignment/matched-FAR code is in place.

## T1 temporal-anomaly grounded in real 3.9um climatology (2026-08-16)

Grounded the temporal-anomaly differentiator in real data. The synthetic lead magnitude
is tunable, but its CAUSE is a measurable quantity: the actual 3.9um night->day BT swing,
which is exactly what an absolute threshold must clear and a diurnal-baseline detector
recovers. Measured on the on-disk per-pixel per-UTC-hour C07 climatology (DiurnalClimato-
logy, GOES-18, N. California, 71,574 pixels): real C07 diurnal amplitude = 32.9 K median
(p25 23.4, p90 45.5). So an absolute contextual threshold is ~33 K less sensitive to a
cold-start night fire than the residual detector. `climatology_diurnal_amplitude` +
`vhagar t1-temporal --climatology data/climatology/climatology.npz` print it; +1 test.
Honest caveat: per-hour sigma (~0.5 K) is thin (~4 samples/bin in this backfill), so trust
the amplitude (a difference of hourly means), not amplitude-in-sigmas. Suite green, ruff
clean. docs/12 "Grounded in the real 3.9 um climatology".

## T1 temporal-anomaly early detection built + demonstrated (2026-08-16)

Built the T1 differentiator (Stage-1): temporal-anomaly early detection. src/vhagar/eval/
t1_temporal.py + CLI `vhagar t1-temporal`. Forecast expected per-pixel BT from a diurnal
harmonic (clear-sky history), flag RESIDUAL excursions; compare to an absolute-BT
threshold calibrated to the SAME false-alarm rate. Synthetic night-fire demonstration:

  target FAR   residual(min after onset)   absolute(min)   lead
  0.05                 0                        75          +75
  0.01                 5                        75          +70
  0.002               15                        85          +70

~70 min lead at equal FAR: residual-vs-diurnal-baseline catches the night fire as it
lifts above its own baseline; the absolute cut must wait for the global threshold. Repro-
duces the published "doubled lead time 35->65 min" mechanism (magnitude is synthetic/
tunable; direction+cause is the finding). Numpy core (DiurnalForecaster, matched-FAR,
lead-time) runs+tested anywhere; production forecaster is TemporalAnomalyNet (3D-conv TCN,
forecast-then-residual, no fire labels), wired via train_temporal_net (torch). 5 tests.
Full suite green, ruff clean. docs/12 "Stage-1 differentiator".

Remaining heavy piece for this component (your machine): pull GOES ABI CMIP band 7 (3.9um)
5-min cubes, train TemporalAnomalyNet on clear-sky, feed residuals to the lead-time eval
vs FDC first-detection times.

## T1 Stage-0 completed with precision/FAR (2026-08-16)

Finished the T1 detection metric: added precision_far_scores() (conditioned on VIIRS
overpass coincidence, so a fire between overpasses is not miscounted as a false alarm)
to eval/t1_stage0.py, wired into the t1-stage0 CLI as a second table. Real result:

  naive 2km:     precision 0.843, FAR 0.157
  parallax 4km:  precision 0.944, FAR 0.056   (30,800 of 188,639 GOES evaluable)

This REPRODUCES the architecture's headline geometry number on our data: apparent FAR
15.7% (naive) -> 5.6% (parallax), squarely in the published "26-36% -> 7-15%" range. The
10-point drop is footprint quantisation + terrain parallax, not model error. T1 Stage-0
is now a complete detection result: POD 0.50, precision 0.94 / FAR 0.06, latency 2 min,
with the GEO/LEO geometry effect measured on BOTH POD (+0.12) and FAR (-0.10). 1 test.
Full suite 356 passed, 5 skipped, ruff clean. docs/12 updated.

## T1 Stage-2 preview, lat/lon leakage demonstrated (2026-08-16)

Built the Stage-2 event-classifier leakage experiment (src/vhagar/eval/t1_classifier.py,
CLI `vhagar t1-classify`), reproducing the architecture's central T1 warning on our data.
Each GOES detection labelled by VIIRS coincidence; gradient-boosted classifier (sklearn)
trained with vs without raw lon/lat under random / cell-grouped / 5-deg-block splits:

  split               physical  +lat/lon   gain
  random                 0.767    0.790   +0.023
  cell_grouped           0.752    0.778   +0.026
  spatial_block_5deg     0.642    0.602   -0.040

Two findings reproduce qualitatively: (1) F1 falls random->spatial-block (0.767->0.642),
same shape as the published 0.985->0.627 generalisation gap; (2) raw lon/lat gain is
POSITIVE in-region (+0.03) and NEGATIVE out-of-region (-0.04), the leak, which is why
production features exclude coordinates. Honest caveat: magnitude is far smaller than the
published 89%-of-gain because 1 week of CONUS FDC + a VIIRS-coincidence label (3% positive,
timing-influenced) is a weak proxy, not a balanced multi-region wildfire/non-wildfire set.
A synthetic test confirms the framework registers a LARGE leak when present, so the modest
real number is the data's, not the tool's. Tried a persistence (flare-vs-fire) label too;
too few persistent sources in 1 week (4 cells) to balance. Installed scikit-learn (sandbox).
docs/12 has the Stage-2 preview. Full suite green.

## T1 Stage-0 RAN, first real detection number (2026-08-16)

First real T1 result on GOES-18 CONUS 2026-08-01..07 vs VIIRS (NOAA-20 + S-NPP, pulled
via firms-fetch, 107k detections). Detection-level coincidence (space cell + /-30 min,
GOES-domain restricted), 104,772 VIIRS in domain:
  naive 2km:      POD 0.376 (median gap 2 min)
  parallax 4km:   POD 0.499 (median gap 2 min)
GOES FDC detects ~half of VIIRS fire pixels near-simultaneously (credible: GOES 2km
misses small fires VIIRS 375m catches). The +0.12 POD from 2->4km cell is GEO/LEO
geometry (footprint quantisation + terrain parallax), not model quality, the T1 twin of
the T2 naive-baseline lesson.

IMPORTANT metric fix: the first cut (event-centroid matching, huge VIIRS bbox) reported
POD 0.047, an artefact. Caught by the same "suspicious number -> diagnose" instinct as
the T2 reference bug. Fixes: detection-level coincidence in space+time (not event
centroids), restrict VIIRS to GOES sector, correct POSIX time handling. New
coincidence_scores() is the metric; CLI reports naive-2km vs parallax-4km POD.
Precision/FAR deferred (need VIIRS swath geometry to be interpretable). 9 T1 tests, full
suite 353 passed, 5 skipped, ruff clean. docs/12 rewritten with the real result + the
broken-metric post-mortem.

## T1 Stage-0 detection framework opened (2026-08-15)

Opened the second pillar: T1 active-fire detection (GOES FDC vs VIIRS truth). Built
src/vhagar/eval/t1_stage0.py + CLI `vhagar t1-stage0`, reusing the existing fusion
infra (Detection, FireEvent, cluster_detections, geo_leo_tolerance_m) and eval/splits.

- match_events: parallax-aware one-to-one GEO/LEO event matching -> TP/FP/FN.
- DetectionScores: POD (recall), FAR (FP/(TP+FP)), precision, F1 (POD+FAR always together).
- detection_latency_minutes: median GOES lead over the VIIRS overpass (the GEO backbone point).
- load_fdc_events_by_tile / firms_to_detections: project to EPSG:5070, cluster to events.
- CLI reports parallax-aware vs naive-2km FAR side by side (the 26-36% -> 7-15% geometry
  finding: footprint quantisation + terrain parallax, not model error).
- 6 pure unit tests; verified FDC->Detection->events on the real parquet (median match
  tolerance 3.6 km at CONUS view zenith, 4x a flat 2 km). Full suite 350 passed, 5 skipped.

Also closed the Csb loop: a robust per-stratum threshold (median of per-fire cuts) does
NOT cleanly rescue Csb (+0.15) and HURTS Csa/Cfa, reconfirming the model (U-Net +0.54) is
the answer, not a better cut.

Added `firms-fetch`: reads the FDC window (2026-08-01..08-07, CONUS+HI) from the parquet
and pulls the matching VIIRS truth to a CSV in one command (FirmsClient.area_csv, <=10d
chunks). fdc_window helper + test. So the runbook is now 3 commands (docs/12):
  set FIRMS_MAP_KEY -> vhagar firms-fetch -> vhagar t1-stage0 --firms-csv viirs_truth.csv
Full suite 351 passed, 5 skipped, ruff clean.

Recommended next: RUN the firms-fetch + t1-stage0 (needs free FIRMS map key) for the
first real POD/FAR/latency and the naive-vs-parallax FAR number. Then ball-tree for full
month, then the Stage-2 event classifier with the random->event-aware->spatial-block
leakage story (0.985->0.767->0.627).

## deep-model ladder RAN, inputs > architecture (2026-08-15)

The full three-way comparison ran on the SAME 32 stack fires, same 5 folds (RBR U-Net
re-run on _w15bgs so it shares exact fires/folds):

  global threshold:           skill +0.000
  RBR U-Net (1 channel):      skill +0.448  (29/32)
  stack U-Net (pre/post+dnbr): skill +0.538 (30/32)
  Siamese change model:       skill +0.533  (30/32)

Findings (all same-path): (1) the spatial model is the big lever, any U-Net crushes the
collapsed threshold; (2) richer inputs buy a real +0.09, stack U-Net +0.538 vs RBR
U-Net +0.448 (~20% relative), so pre/post NBR carry signal a pre-differenced RBR loses;
(3) Siamese ~ plain multi-channel U-Net (+0.533 vs +0.538), the bi-temporal change
architecture earns NOTHING over stacking pre/post as channels. Inputs > architecture,
spatial model most of all. All three models agree on the hard fires (MT47702, AR36076,
OK36688). Bar for the foundation-model fine-tune: +0.54, and it must beat a plain
multi-channel U-Net, not just the threshold. docs/11 "Result: inputs matter more" done.

## 9-fire EU run, Csb fails (threshold-transfer, not signal) (2026-08-15)

Ran continent-out with the 2 new large Csb Portugal fires (9 EU fires now). Per-fire
skill: Csa Attika +0.661, Syria +0.345, Evia +0.057; Dfb Albania +0.559, Poland -0.004
(tiny); Cfa Montenegro +0.451; Csb Arouca +0.000, Estrela +0.000, West Spain +0.000.

Key finding: Csb FAILS even on the two large clean Portuguese fires (8% and 18% burned,
not degenerate), yet their per-fire oracle is +0.589/+0.587 -> RBR separates Portuguese
Csb strongly; the transferred THRESHOLD fails, not the signal. Diagnosed: the 7 US Csb
fires have wildly heterogeneous per-fire Youden cuts (2,26,27,55,65,99, one pathological
-941218), so pooling their pixels collapses the Csb threshold to -3787 (predict-all-
burned); plus an absolute-scale offset (US Csb unburned RBR ~-30 vs Portuguese ~+30).
This is the pointwise threshold's failure mode, and points straight to the U-Net (+0.441)
as the scale-adaptive fix. Per-stratum helps only when a stratum's US fires share the EU
fire's RBR scale (Csa/Cfa/Dfb here). docs/11 "Nine-fire EU generalisation" rewritten.
Pooled row +0.112. Possible future: robust per-stratum aggregate (median of per-fire
cuts) would rescue some Csb, but the model is the durable answer.

## multi-channel U-Net + Siamese change model (t2-deep) (2026-08-15)

Built the architecture's intended T2 direction: pre/post NBR bands and a Siamese change
model, as the next rung above the RBR U-Net (which scored +0.441 vs threshold +0.096).

- sentinel2_stack: returns [pre_nbr, post_nbr, dnbr] from the SAME imagery pull as RBR
  (no extra scenes). T2Sample gains optional `stack` + `features` accessor; save/load
  round-trip it. build_optical_sample(with_stack=True) and t2-stage0 --with-stack cache
  it under a distinct `_w15bgs` tag (never collides with the RBR-only cache).
- src/vhagar/eval/t2_deep.py + CLI `vhagar t2-deep --model siamese|unet`: multi-channel
  U-Net over the stack, and SiameseChangeNet (shared encoder on pre/post NBR, fused as
  [|f_post-f_pre|, f_post]). Same leakage-proof grouped folds, per-channel standardiser
  (train-only), masked BCE+Dice, skill-over-naive, RBR threshold on the identical fold.
- 6 tests (numpy core + stack round-trip + torch smoke for both models via importorskip).
  Full suite 344 passed, 5 skipped, ruff clean.

Needs a stack re-pull on your machine (torch + network), the RBR-only cache can't feed
the deep models:
  vhagar t2-stage0 --min-area-ha 2000 --max-fires 34 --select size --res-m 100 ^
    --objective youden --with-stack
  vhagar t2-deep --model siamese --folds 5 --epochs 20
  vhagar t2-deep --model unet    --folds 5 --epochs 20
Question: do pre/post bands + the change formulation beat the +0.441 RBR U-Net on the
same folds? Recorded once the run completes. docs/11 "Richer inputs" has the writeup.

## EMS batch, added 2 large summer Csb fires (2026-08-15)

Downloaded (browser) burnt-area delineations for 2 large summer Csb wildfires from
Portugal, joining the set (emsr_extract/, ingest gives 9 delineations now):
  EMSR831 Beiras e Serra da Estrela (2790 burn polygons, big), EMSR824 Arouca (105).
These strengthen Csb, which previously rested on the cloud-thinned West Spain window
(0.9% burned). Both validated with burnt geometries; added to emsr_candidates_starter.csv.

Catalog limitation found: the public rapidmapping API only lists ~2024-2026 (EMSR765+),
and Europe's big fires there are Mediterranean, so LARGE Dfb/BSk activations are scarce
(central Europe reads as Cfb; the 2021 Turkey/BSk fires and EMSR527-era events are not
in this API). Dfb rests on Albania (mountain, +0.214) and tiny Poland; BSk still has no
usable delineation (EMSR826 was grading-only). To fill Dfb/BSk would need an older-
catalog source or non-EMS burnt-area data.

Re-run to test the expanded Csb (needs S2 pull on your machine for EMSR824/831; rest
cached):
  vhagar emsr-ingest emsr_extract --dates emsr_candidates_starter.csv --out emsr.csv
  vhagar t2-continent-out --registry data\labels\registry.parquet ^
    --mosaic mtbs_extract\mtbs_CONUS_2021.tif --emsr-manifest emsr.csv ^
    --stratify-raster koppen_extract\1991_2020\koppen_geiger_0p00833333.tif ^
    --min-area-ha 2000 --max-fires 34 --res-m 100 --objective youden

## U-Net companion baseline built (t2-unet) (2026-08-15)

Built the plain-U-Net companion baseline the protocol asks for, as a fair head-to-head
against the RBR threshold. New module src/vhagar/eval/t2_unet.py + CLI `vhagar t2-unet`:

- Single-channel U-Net over the RBR field (same input the threshold sees).
- Leakage-proof grouped k-fold (each fire tests once, no fire in both sides).
- Masked ComboLoss (weighted BCE + soft Dice over valid pixels only); pos_weight and
  input standardisation (median/MAD) fit on train folds only; burned-biased crops.
- Reports per-held-out-fire skill over naive next to the threshold's skill on the same
  fold; summary says how often the U-Net beats the threshold.
- Numpy core (crops, standardiser, folds) is unit-tested in-sandbox; the torch train
  loop is guarded and runs on your machine (sandbox proxy blocks the torch wheel).

Fixed two subtle correctness issues while writing it: masking valid pixels into a 1-D
vector breaks the Dice term (each pixel becomes its own image), so the loss masks the
maps and keeps spatial structure; and eval now runs the full window in one pass with a
pad-to-multiple-of-8 (three encoder poolings) instead of tiling, which avoided tiny
remainder tiles collapsing. 6 tests (5 numpy + 1 torch smoke, importorskip).

RESULT (ran on your machine, 43 fires, 5-fold): U-Net mean skill +0.441 vs global
threshold +0.096, U-Net wins 39/43. That +0.441 is near the per-fire ORACLE ceiling
(+0.464) and far above per-stratum (+0.097): a spatial model over the same single RBR
channel recovers almost all the transferable skill a pointwise cut throws away, WITHOUT
the Koppen raster. Credible (no positional shortcut: plain conv U-Net is translation-
equivariant, no coord channels; the 4 losses are the degenerate RBR-can't-separate
fires). Caveat: U-Net used 5-fold, the oracle/per-stratum numbers used LOFO, so that
cross-reference is indicative; the global-threshold comparison is same-path (identical
folds). Reframes "threshold transfer is the limitation": it's a limitation of a
POINTWISE threshold; a spatial model largely dissolves it. Sets the bar the foundation-
model fine-tune must clear (+0.441). docs/11 "Companion baseline" has the writeup.

Next inputs: pre/post NBR bands + Siamese change model (should raise the ceiling);
degenerate fires want better imagery not a bigger head.

## 7-fire EU generalisation, corrected + CLI fixes (2026-08-15)

Ran the expanded continent-out and, on scrutiny, corrected two things in the first
writeup. Canonical CLI result (size-stratified US training, burnt-centroid strata,
per-stratum, Youden), per-fire skill over naive:

  Csa: Attika +0.732, Syria +0.581, Evia +0.050 (low-ceiling fire)
  Cfa: Montenegro +0.451
  Dfb: Albania +0.214, Poland +0.000 (0.2% burned, single-class)
  Csb: Spain +0.000 (0.9% burned, cloud-thinned)
  pooled row: skill +0.26.

Transfer is positive across THREE zones (Csa, Cfa, Dfb), not Mediterranean-specific.

Two corrections vs the first pass: (1) Albania is Dfb at its burnt centroid (mountain
fire), not Csa, and transfers +0.214, not +0.51; (2) the exact skills shifted because
the reproducible CLI path differs from the one-off script.

Two CLI fixes (this is why the first CLI run looked weak, +0.046 pooled):
- t2-continent-out now prints a PER-FIRE skill table before the pooled row.
- t2-continent-out now defaults to --select size. It used largest-N, which clusters
  in western US zones (Dsb, BSk) with NO Cfa fire, so Montenegro fell back to the
  global predict-all-burned threshold (+0.000). Size stratification pulls in the small
  fires that carry Cfa/other zones. Real methodological point: per-ecoregion transfer
  needs the training set to span the ecoregions. docs/11 updated.

Next: more clean Csb/Dfb/BSk activations; the per-fire oracle column in the CLI.

## EU fire set expanded to 4 Koppen zones, ready to run (2026-08-14)

Downloaded (via the browser) burnt-area delineations for 5 new EMS wildfire
activations and extracted them into emsr_extract/, joining the 2 original Greek
fires. The EU test set now spans FOUR Koppen zones instead of one:

  Csa (Mediterranean): EMSR527 Greece x2, EMSR816 Albania, EMSR811 Syria
  Csb (warm Mediterranean): EMSR837 West Spain
  Cfa (humid subtropical): EMSR836 Montenegro
  Dfb (humid continental): EMSR801 Poland

All 7 validated with real burnt geometries (read_emsr_burned_geometries: 2 to 1882
polygons each, EPSG:4326). Skipped EMSR826 (BSk) because it only had a Grading
product, no observedEventA; BSk still needs a delineation activation.

To run the generalisation test on your machine (needs the Sentinel-2 pull for the 5
new EU fires; reuses the US _w15bg cache):

  vhagar emsr-ingest emsr_extract --dates emsr_candidates_starter.csv --out emsr.csv
  vhagar t2-continent-out --registry data\labels\registry.parquet ^
    --mosaic mtbs_extract\mtbs_CONUS_2021.tif --emsr-manifest emsr.csv ^
    --stratify-raster koppen_extract\1991_2020\koppen_geiger_0p00833333.tif ^
    --min-area-ha 2000 --max-fires 34 --res-m 100 --objective youden

This is the test of whether per-stratum transfer holds across Csa/Csb/Cfa/Dfb, not
just Greek Csa. Cleanup: leftover *.zip in emsr_extract and EMSR837_products.zip
(2.1 GB) in Downloads can be deleted.

## EMSR batch-pull tooling to generalise the cross-continent result (2026-08-14)

To turn the +0.115 Csa->Csa transfer into a general claim we need more European fires
across more Koppen zones. Built two tools (src/vhagar/labels/emsr_fetch.py, CLI):

- `vhagar emsr-candidates --koppen <tif> --out emsr_candidates.csv`: queries the
  public CEMS Rapid Mapping API (no login) for wildfire activations, tags each with
  its Koppen zone from the raster, writes a climate-diverse candidate table. Runs on
  your machine (network).
- `vhagar emsr-ingest <folder> --dates emsr_candidates.csv --out emsr.csv`: scans a
  folder of downloaded EMS vector packages, finds each AOI's burnt-area
  observedEventA (latest monitoring step), writes the t2-continent-out manifest. No
  network, no manual CSV editing. Tested against the EMSR527 folders (picks MONIT03).

Runbook:
  1. vhagar emsr-candidates --koppen koppen_extract/1991_2020/koppen_geiger_0p00833333.tif
  2. From the portal (rapidmapping.emergency.copernicus.eu/EMSR<code>/download), download
     the vector package for each chosen code into a folder, e.g. emsr_extract/. A curated
     starter set spanning Csa/Csb/BSh/BSk/Cfa/Cfb/Dfb is in emsr_candidates_starter.csv.
  3. vhagar emsr-ingest emsr_extract --dates emsr_candidates_starter.csv --out emsr.csv
  4. vhagar t2-continent-out ... --emsr-manifest emsr.csv --stratify-raster <koppen> --objective youden
Then per-stratum should be testable across several shared US<->EU climate zones, not
just Csa. Suite 334 passed, 2 skipped, ruff clean.

## reference bug found and fixed; the predictor is actually good (2026-08-14)

The big one. The wide-window re-pull did NOT drop the burned fraction (still 57-95%)
because the confound was never the window: read_mtbs_reference_on_grid marked only
MTBS classes 1-5 (inside the perimeter) valid and DISCARDED class 0 (unburned
background). So the eval measured severity-within-a-fire (~90% burned), not burned-
area detection. The predictor was finite over the whole window all along (one 2,030
ha fire: 90,601 finite RBR px, only 2,042 scored).

Fix: mtbs_burned_mask(..., include_background=True), now default in the sample
builder; counts class 0 as unburned, excludes only class 6. Rebuilt samples locally
from cached predictors (no re-pull), tagged _w15bg. On the corrected reference (29
fires, realistic 0.10 burned fraction), leave-one-fire-out, balanced objective:

  global one-threshold:   skill +0.000  (9/29 beat naive)
  per-stratum Koppen:     skill +0.097  (17/29)   <- climate matching now HELPS
  per-fire oracle ceiling:skill +0.464  (29/29)   <- predictor separates every fire

So: the predictor is good (oracle 29/29), a single global threshold captures none of
it (RBR scale varies fire-to-fire), and per-ecoregion calibration recovers ~a fifth
of the ceiling. This vindicates the architecture's per-ecoregion thesis and REVERSES
the earlier "stratification hurts" finding, which was a reference-bug artefact.
docs/11 has a new "Corrected reference" section that supersedes the stratification-
negative and objective sections (kept for audit trail). Added include_background,
threaded through read_mtbs_reference_on_grid and build_optical_sample (bg cache tag),
plus a regression test.

Corrected continent-out (US bg -> EU), the capstone: global one-threshold +0.002
(collapses, RBR scale heterogeneity), per-stratum Koppen (Csa->Csa) +0.115, EU oracle
ceiling +0.123. So climate stratification recovers ~93% of the achievable cross-
continent skill, taking transfer from nothing to nearly oracle. Strongest evidence yet
for the per-ecoregion thesis; rests on 2 US Csa + 2 EU fires, needs more to generalise.

Coastal-window caveat: MTBS mosaic uses 0 for both background and nodata, so ocean
counts as unburned; fine for these interior fires, flag for coastal ones.

Next levers: more EU EMS fires (generalise the +0.115), more US fires per Koppen
stratum (push within-CONUS per-stratum toward its +0.464 oracle ceiling), and per-fire
adaptive calibration as the eventual production path.

## within-CONUS F1 has no skill over predict-all-burned (2026-08-14, superseded above)

The decisive check. On the narrow per-fire windows, the trivial predict-all-burned
baseline scores F1 0.896 (large fires) to 0.911 (all 34), and the calibrated RBR
threshold does NOT beat it on a single fold (0/34, skill -0.01). The within-CONUS
0.865/0.900 "accuracy" numbers are window artefacts, not skill. The predictor shows
real skill in exactly one place: the balanced EU continent-out test (naive 0.485),
where Youden-objective RBR scores 0.573, a +0.088 margin. That modest cross-continent
margin is the only defensible accuracy signal so far.

Baked the no-skill baseline into the eval so this can never be hidden: FoldResult now
carries naive_f1 and skill_f1, summarise_stage0 reports skill_f1_mean and
folds_beating_naive, and both CLI tables print a naive-F1 column and a red/green skill
margin. docs/11 headline rewritten to retract the accuracy claim and record the skill
finding. Suite 327 passed, 2 skipped, ruff clean. This makes the wide-window re-pull
essential (only balanced windows can measure predictor skill), not just a refinement.

## Koppen climate stratification, a mostly-negative result (2026-08-14)

Downloaded the 1 km Koppen-Geiger present-day raster (Beck et al., 1991-2020) and
tested whether matching US-to-EU climate zones lifts continent-out transfer. First
pass looked like a win (per-stratum Koppen F1 0.609 vs global 0.488 on the fixed
34-fire pool), but three follow-up checks show the gain is mostly an artifact:

1. Within-CONUS leave-one-fire-out, per-stratum HURTS: 0.827 vs 0.909 global, with
   catastrophic folds. If stratification were real it would help here too.
2. The global -1706 threshold is a degenerate-window artifact: 26/34 US windows are
   >80% burned, so F1-tuning rewards "predict all burned" (-1706), which is
   catastrophic on the 32%-burned Greek windows. The Csa per-stratum threshold (99)
   helps only because its one fire (AZ32635) has a balanced window.
3. Re-tuning the global threshold with a balanced objective (Youden's J) recovers
   EU F1 to 0.573 with NO climate matching. Koppen (0.609) sits only 0.036 above a
   properly-tuned global, and that residual rests on a single fire.

Conclusion: the tight per-fire window is the dominant confound, not climate. The
real fix is wider analysis windows (needs an imagery re-pull); the balanced
objective is a cheap partial fix already in hand. Full writeup in
`docs/11_T2_STAGE0_RESULTS.md`. Data: `koppen_geiger_tif.zip` at repo root,
extracted to `koppen_extract/` (both gitignored). Do not compare any of this to
the earlier 0.582 (different 6-fire pool, no Csa, per-stratum inert).

Two follow-ups worth doing: (a) add `objective="youden"` as a named option in
`tune_threshold` and expose it on the CLI, so the balanced baseline is reproducible
without a script; (b) wider windows on a re-pull to remove the all-burned confound.

Both are now done in code (committed 2afe461 for (a)). For (b): `target_grid_for_fire`
now sizes the half-window as `max(radius * 2.5, 15 km)` capped at 30 km (was
`radius * 1.6`, 5 km floor), so small fires get a wide unburned ring instead of a
~90%-burned window. The sample cache key now encodes the window floor
(`{event}_r{res}_w{km}.npz`, e.g. `_r100_w15`), so the new wide-window samples do
not collide with or get pooled against the old narrow ones. Suite still 324 passed,
2 skipped, ruff clean.

### Re-pull the wide-window cache (needs network, run on your Windows machine)

The sandbox cannot reach Sentinel-2 (STAC/S3 blocked by the proxy), so build the
new samples where the network is open. This creates new `_w15` npz files and leaves
the old `_r100` ones untouched.

```powershell
cd C:\Users\taylo\VHAGAR
# US CONUS 2021, size-stratified, wide windows, balanced objective:
vhagar t2-stage0 --registry data/labels/registry.parquet `
  --mosaic mtbs_extract/mtbs_CONUS_2021.tif `
  --min-area-ha 2000 --max-fires 34 --select size --res-m 100 --objective youden
# Leave-one-continent-out, wide windows, balanced objective:
vhagar t2-continent-out --registry data/labels/registry.parquet `
  --mosaic mtbs_extract/mtbs_CONUS_2021.tif --emsr-manifest emsr.csv `
  --min-area-ha 2000 --max-fires 34 --res-m 100 --objective youden
```

Then compare the new wide-window numbers against the narrow-window ones in docs/11.
Expect the small-fire burned fractions to drop from ~90% toward 30-60%, the
Olofsson area to become computable on more than 2 of 34 folds, and the F1 gap
between within-CONUS and continent-out to be more meaningful (less inflated by
all-burned windows). Paste the tables back and I will write up the comparison.

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
      - [x] **RAN on real Sentinel-2 + MTBS. First accuracy number.**
            **F1 0.865 ± 0.056, IoU 0.765 ± 0.084** over 5 leave-one-fire-out
            folds on the largest 2021 CONUS fires (Dixie, Bootleg, Caldor, ...),
            independent RBR vs MTBS, with per-fire Olofsson adjusted areas and 95%
            CIs. Full table and reading in `docs/11_T2_STAGE0_RESULTS.md`. One
            fully-clouded fire dropped, disclosed. Samples cached in
            `data/t2_cache/` so widening does not re-pull.
      - Robustness added during the run: per-fire sample caching, degenerate-fire
        filtering (all-cloud or single-class), folds skip rather than crash, and
        a coarse-res + scene-cap + streaming compositor so large fires do not blow
        memory.
      - [x] **Leave-one-continent-out capability built (MTBS -> EMSR headline).**
            EMS ingest (`build_emsr_record`/`read_emsr`, europe/EPSG:3035,
            evaluation-only), a rasterised burnt-area reference
            (`rasterize_burned_on_grid`/`read_emsr_reference_on_grid`, reprojects
            then burns polygons onto the fire window), the sample builder
            generalised to any reference source, and a `vhagar t2-continent-out`
            CLI that trains the threshold on the cached US fires and tests on
            European EMS fires (single honest cross-continent fold). Also fixed a
            pixel-area bug: area now derived from `--res-m` (was hardcoded 0.09).
            Four offline geometry tests. Needs the user to download a few EMS
            delineation shapefiles and run.
      - [x] **RAN leave-one-continent-out on real EMS fires. The headline.**
            Train threshold on US MTBS, test on EMSR527 Evia + Attika (Greece,
            Aug 2021). **Within-CONUS F1 0.87 -> cross-continent F1 0.58**
            (IoU 0.41), a ~0.28 generalisation gap: a US-calibrated RBR cutoff
            transfers poorly to Greek Mediterranean fuels. Olofsson adjusted
            33,452 +/- 8,449 ha. Clean diagnostics, not degenerate. In
            `docs/11_T2_STAGE0_RESULTS.md`.
      - [x] **Adaptive Otsu companion baseline added and measured (negative
            result).** Hypothesised a per-fire adaptive threshold would transfer
            better than a global one; it does not. Calibrated global beats Otsu
            at both scales: CONUS 0.865 vs 0.713, continent-out 0.582 vs 0.552.
            RBR's heavy tails and weak window-scale bimodality make Otsu
            under-detect. `otsu_threshold` (outlier-robust) in `eval/baselines.py`,
            `--method global|otsu` on both CLIs, computed directly from the cache.
            Reporting the negative result is the permanent-baselines rule in
            action.
      - [x] **Per-fire standardization tested (another negative for transfer).**
            Recenter/scale each fire's RBR then apply a global threshold: helps
            CONUS slightly (0.865 -> 0.876) but hurts continent-out (0.582 ->
            0.535). Three methods now mapped; calibrated raw-RBR global is best
            for transfer. The US->EU gap is genuine domain shift, not scaling.
      - [x] **Size-stratified fire selection** (`select_fires`, `--select size`):
            sample fires across the area distribution instead of only the
            largest, so a scaled evaluation is distribution-representative rather
            than megafire-biased.
      - [x] **Per-stratum (Köppen climate) thresholds built.** The transfer fix
            is a GLOBAL stratum both continents share, not US-only ecoregions:
            California and Greece are both Köppen Csa (Mediterranean), so a
            threshold learned on US Mediterranean fires can apply to Greek ones.
            `datasets/strata.py` samples any global class raster at each fire;
            `evaluate_fold(method="perstratum")` calibrates per stratum with a
            global fallback; both CLIs take `--stratify-raster koppen.tif`. Tested:
            per-stratum beats global when strata have different severity scales.
      - [x] **Scaled to 34 size-stratified CONUS fires: global F1 0.900 ± 0.083.**
            Higher than 5-fire 0.865 but partly an artifact: small fires' windows
            are 80-96% burned (easy per-pixel F1), and the Olofsson area is only
            estimable on 2 of 34 folds (rest single-class). Honest caveat in the
            results doc. Also fixed a real crash: a single-class map broke the
            Olofsson allocator (OverflowError); now the area is skipped cleanly and
            regression-tested.
      - [ ] Next: perstratum continent-out with a Köppen raster (the climate-match
            hypothesis, no imagery re-pull); OR widen the window to add unburned
            context so small fires are informative and areas measurable (re-pull).
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
