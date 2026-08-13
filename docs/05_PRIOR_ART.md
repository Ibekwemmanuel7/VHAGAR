# VHAGAR: Lessons from the FirePerim / FireSight Platform

A study of `C:\Users\taylo\Coulson_group\Geospatial Platform Engineer`. 
135 git-tracked files, 37 commits (Jun-Aug 2026), ~5,870 lines of Python, a
React frontend, and 7.1 GB of FLAME 3 imagery.

**Purpose of this document:** decide what VHAGAR should reimplement from
scratch, what it should deliberately not repeat, and which ideas survive the
airborne → satellite gap.

---

## 1. What that platform is

Two systems, one repo:

- **FireSight-IR** (research, separate repo), a four-branch physics-informed
  network, 202,228 params, trained on **1,149,722 VIIRS fire pixels**, western
  CONUS 2018-2022, with 2023 (76,084 pixels) as a strict temporal holdout. Its
  scientific payload is an **ablation table**: removing the surface-context
  branch collapses false-alarm precision **97.83 % → 35.33 %**, while removing
  the physics loss costs <1 pt and removing ERA5 *improves* results slightly.
- **FirePerim Live** (production, this repo). Streamlit app, FastAPI backend on
  Render, React 18 + Mapbox GL console on Vercel, all over one pipeline core:
  FIRMS/GOES ingest → DBSCAN → alpha-shape perimeters → weather → risk → export.

They join at **FireSight-Context**: the ablation operationalised as a
HistGradientBoosting student on 19 live-computable features, trained in 10 s on
CPU, which on the label-referenced 2023 holdout **beats** the 13-GPU-hour PINN
(acc 0.9632 vs 0.9584; FA precision 0.9986 vs 0.9783).

It is a job-application artifact, engineered to a high standard, and it should
be read as one, the thesis "swap the ingestion layer and you have ORBIT" is a
hiring pitch, not an architecture requirement.

---

## 2. Borrow: reimplement these from scratch

**1. The `DetectionSource` ABC and one frozen detection schema.**
The best idea in the repo, and it transfers completely. FIRMS, GOES ABI, a
synthetic airborne frame and an orthorectified oblique frame all reach the same
clustering code with zero downstream changes, because there is exactly one
12-key `DETECTION_COLUMNS` dict, a separate `OPTIONAL_COLUMNS`, a typed
`empty()`, and a `validate()` at the seam.
→ *VHAGAR: `harmonize.fusion.Detection` is this. Keep the `validate()`
discipline as ingest adapters are added.*

**2. A second seam for the ML screen.**
`DetectionClassifier` with `heuristic | model | heavy` tiers selected by env var,
instance-cached, degrading loudly in logs but silently in behaviour. VHAGAR
will hit the same "the good model needs L1B data we don't have at this latency"
problem.
→ *VHAGAR: `pipelines.nrt.NRTPipeline` takes an optional `classifier` and falls
back to a transparent, documented rule. Formalise the tiers.*

**3. Cluster before you filter.**
`api/pipeline.py` assigns `event_id` on the full scored set and only then drops
flagged detections. Event IDs stay stable across an operator toggle, and
filtered perimeters are *more* accurate. Small decision, real operational value.

**4. Occlusion-based explainability.**
`ContextModelScorer.explain()` NaNs each feature in turn and re-predicts. ~19
model calls, zero extra dependencies, exact for what it claims. For a tree model
with native NaN support this is strictly better than bolting on SHAP.

**5. The ground-truth relabel audit as a permanent artifact.**
`scripts/relabel_audit.py` is the single most scientifically valuable thing in
the repo, and the pattern VHAGAR most needs. It pulls NIFC/WFIGS 2023
perimeters, buffers 750 m, spatially joins the 76,084 validation detections, and
defines ground truth from evidence that shares no features with the model.

Result: the label rule scores wildfire P/R 0.870/0.956 but **false-alarm P/R
0.907/0.386**, and the deployed student reproduces it almost exactly. The stated
conclusion: the labels are precise but blind, **~6,610 real static sources are
labelled "wildfire" in the training data**, and the headline 99.86 %
label-referenced FA precision becomes **90.6 %** against independent evidence.

That is an author auditing their own headline number until it gets worse. Copy
the practice, not just the script.

**6. Robust geometry fallbacks with a `method` column.**
Alpha shape → convex hull → point buffer, with `perimeter_method` written into
every output feature. Any real feed produces degenerate clusters; recording
*which* fallback fired is what makes output auditable.

**7. Time-aware, outlier-rejecting multi-look stacking.**
Recency weight `exp(−Δt/τ)` with 3·1.4826·MAD rejection and a Gaussian-feathered
"sharp front" blend at a shorter τ, so the moving fire edge stays crisp while
background is multi-look averaged. Conceptually correct answer to "the fire
moved between observations," and it generalises to satellite compositing.

**8. The interaction-term false-alarm heuristic.**
`prob = 0.62·(persistence × isolation) + 0.20·low_conf + 0.18·weak_frp`. The
*product* term is the insight, a flare is persistent **and** isolated. Well
reasoned; a good shape for VHAGAR's Stage-2 persistence features.

---

## 3. Avoid: specifics

**1. Decorative georeferencing.** All five `web/public/ir/*/bounds.json` are
byte-identical, placing five different products on one 1.25 km tile at Sycan
Marsh, Oregon, while the rasters themselves are tagged **UTM zone 11N**
(California) and the build script hardcodes Yosemite coordinates. The overlay
and the raster's own CRS disagree by ~500 km. Two commits did this deliberately
for visual composition. **Georeferencing that is decorative is worse than none.**

**2. The asymmetric morphological close.** `poly.buffer(150).buffer(-75)` in
`perimeter._finalize()`, and its twin in `airborne.extract_perimeter`. Every
perimeter is inflated by ~75 m; for a 100 ha event ~1.1 km across that is
**+15-20 % area**, and `area_ha`, the platform's headline number, inherits it
silently. A symmetric close does what the docstring claims.

**3. The spread-risk score.** `0.55·wind + 0.30·dryness + 0.15·heat`, with no
calibration, no validation, no fuel, no slope, no fuel moisture, and no relation
to NFDRS/Fosberg/Haines, rendered on the map as a four-band colour ramp that
reads as authoritative.
→ *VHAGAR: `features.fwi` implements the real thing and is regression-tested
against the Van Wagner worked example.*

**4. Business logic duplicated across Python and JavaScript.** `risk.py` and
`web/src/weather.js` implement the same formula twice with no shared test.

**5. CI that cannot fail on what matters.** `ruff check ... || true`, plus 24 of
58 tests behind `importorskip` on dependencies deliberately excluded from
`requirements.txt`. **The entire raster/fusion/ortho surface, the most complex
maths in the repo, is untested in CI.**
→ *VHAGAR: one dependency set per deployable; `pytest.importorskip` used only
for genuinely optional extras that CI installs in a separate job.*

**6. Dead modules presented as features.** `processing/fusion.py` (cross-sensor
confirmation) and `ingest/goes.py` are written, documented and partly tested but
never invoked. The UI's "All sensors (fused)" tab is a query-string parameter,
not that module.

**7. Unreproducible headline artifacts.** No committed script regenerates
`FLAME3_RealFusion_*`, `Georef_Telemetry_*`, `Optical_Restore_*`, or three of
the web overlays. The four metrics JSONs at repo root cannot be rebuilt from the
repo.

**8. Metric selection that flatters.** Full fusion scores SSIM **0.813** against
naive averaging's **0.866**, the one-pager quotes only the gain over a single
pass. "×17 sharper" is Laplacian variance after an unsharp mask, which rewards
amplified noise. And `frp_mw = (BT − 300)/12` is a fabricated quantity with no
Stefan-Boltzmann, no pixel area and no background subtraction, which then flows
into `total_frp_mw` and is displayed in megawatts.
→ *VHAGAR: `physics.frp` is the real Wooster method, warns when transmittance
is missing, and refuses to silently invent a constant for an unknown sensor.*

**9. Equirectangular projection at one mean latitude** (`_local_xy` in
`false_alarm.py`, `fusion.py`). Over the CONUS bbox (24.4-49.4 °N) the longitude
scale factor varies by ~28 %, so cell sizes and radii are wrong at the edges. 
while `cluster.py` next door already reprojects to EPSG:5070 correctly.

**10. Demo-scoped infrastructure.** In-memory GeoDataFrames, a dict cache, no
database, no persistence, no history, `allow_origins=["*"]`, free-tier host that
sleeps.

---

## 4. Gaps VHAGAR must fill

- **No persistence layer at all** → no perimeter history, no growth tracking, no
  rate of spread. The platform can only describe *now*.
- **No real tiling.** COGs are written but the map consumes static PNGs with
  hardcoded corners; no TiTiler/rio-tiler/XYZ path despite the README.
- **No provenance, versioning or lineage** on outputs.
- **No DEM anywhere**, every "orthorectification" uses a flat plane, correctly
  identified by the author as the binding constraint.
- **No cloud/smoke masking** on the satellite side.
- **No uncertainty on the perimeter geometry itself.**
- **No perimeter validation.** The WFIGS audit validates *detection
  classification*; perimeter IoU is never measured.
- **US-West only**; the model is explicitly out of distribution elsewhere.
- **No queue or worker model**; single-region, single-process.

---

## 5. The airborne → satellite gap

**Transfers directly.** The source-agnostic detection schema and both ABC seams
, they were *designed* for the swap and prove it in both directions. DBSCAN in a
metric CRS with a physical `eps`. Alpha-shape-with-fallbacks perimeters (they
consume points; they do not care what produced them). The FA-screening tier
pattern. Cluster-before-filter and stable event IDs. The relabel-audit
methodology. Time-aware recency weighting. The GOES ABI fixed-grid navigation
code and fire-mask confidence table are directly reusable satellite assets.
GeoJSON/KMZ agency export.

**And the headline scientific finding transfers wholesale**, because it *is* a
satellite finding, validated on 1.15M VIIRS pixels: **surface context beats
thermal signal for false-alarm rejection**, removing that branch took FA
precision from 97.83 % to 35.33 %. VHAGAR inherits it, including the caveat
that the labels behind it are 61 % blind on the FA class.

> This is the strongest independent corroboration of VHAGAR's Stage-1/Stage-2
> design in `docs/03_PHYSICS.md`. Context and persistence beat radiometry. It is
> also a caution: that platform's context stack was built by interpolating
> distances *from the training parquets themselves* because Overpass was
> blocked, self-consistent by construction, not an independent geodata product.
> VHAGAR must source real context layers.

**Transfers with modification.** Multi-look fusion: destripe → radiometric
normalise → co-register → robust stack applies to satellite time series, but the
failure modes differ. BRDF, view-angle and bow-tie effects, cloud, orbital
resampling, not line-scanner fixed-pattern noise. Phase correlation and ECC on
375 m VIIRS pixels behave nothing like on 5 cm drone frames.

Optical restoration: dark-channel-prior dehazing is a *near-field* atmosphere
method. At satellite scale you want radiative-transfer atmospheric correction
(`docs/03_PHYSICS.md` §2) and cloud/smoke masking, not a DCP.

**Does not transfer.** Everything in `raster/georef.py`. DJI EXIF/XMP parsing,
laser-rangefinder boresight (a genuinely clever idea: the camera→LRF-target
vector supplies the yaw DJI omits, giving **median 9.6 m ground residual from
telemetry alone, no GCPs**), gimbal-roll assumption, pinhole intrinsics.
Satellites are georeferenced by the provider via RPCs and geolocation arrays;
the VHAGAR equivalent is RPC/GLT handling, not pose recovery. Same for
`raster/ortho.py`'s collinearity demo and the FLAME 3 dataset itself (55 m AGL,
3-5 s cadence).

**Keep the seam; discard the thesis.** Designing a satellite platform around an
imagined airborne ingest is how `frp_mw = (BT − 300)/12` ends up in production
code.

---

## 6. Actions taken in VHAGAR

| Lesson | VHAGAR response |
|---|---|
| Surface context beats thermal signal for FA rejection | `features.physics_features`, 34 features, context and persistence weighted heavily; `docs/03_PHYSICS.md` Stages 1-2 |
| Labels are precise but blind (~6,610 mislabelled) | `labels.registry.LabelQuality`; `EVALUATION_ONLY` sources; relabel audit is a Phase-1 deliverable |
| Asymmetric close inflates area silently | Symmetric operations only; `eval.area_estimation` reports error-adjusted area with a CI, never a pixel count |
| Fabricated FRP proxy | `physics.frp` implements Wooster properly, warns on missing transmittance, refuses unknown sensor constants in strict mode |
| Uncalibrated hand-weighted risk score | `features.fwi`, the real FWI System, regression-tested against Van Wagner |
| Equirectangular at one mean latitude | `grid.py`, equal-area CRS per region, always |
| CI cannot fail on the hard maths | One dependency set per extra; mass-conservation, CRS round-trip and nodata-propagation tests in CI |
| Metric selection that flatters | `docs/02_VALIDATION.md`, mandatory baselines, per-fold reporting, proper scoring rules only for tuning |
| Decorative georeferencing | Golden georeferencing test in the CI list; provenance required on every product |
| No persistence, no history | PostGIS + TimescaleDB in the architecture from the start; perimeter history is what makes T4 possible at all |
