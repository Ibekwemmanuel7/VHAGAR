# VHAGAR

**Multi-sensor wildfire intelligence, detection, burned area, danger, spread.**
Coverage: CONUS/USA · Canada · Europe.

VHAGAR is four models and one platform:

| Task | Question | Prediction unit | Primary sensors |
|---|---|---|---|
| **T1 detection** | What is burning right now? | fire event, minutes | GOES-19/18 ABI, MTG-I FCI, VIIRS 375 m, SLSTR |
| **T2 burned area** | What burned, and how badly? | 20-30 m polygon + severity | Sentinel-2, Landsat 8/9, Sentinel-1 |
| **T3 danger** | Where might it start, and how will it behave? | 1-4 km cell × day, probability | ERA5/HRRR/ECMWF + fuels + human drivers |
| **T4 spread** | Where does this fire go next? | 375 m burn probability | fused arrival-time state + NWP + fuels |

| Doc | What it covers |
|---|---|
| **[`docs/00_ARCHITECTURE.md`](docs/00_ARCHITECTURE.md)** | full system design |
| [`docs/01_DATA_CATALOG.md`](docs/01_DATA_CATALOG.md) | every dataset, ID, licence, caveat |
| [`docs/02_VALIDATION.md`](docs/02_VALIDATION.md) | the binding evaluation contract |
| **[`docs/03_PHYSICS.md`](docs/03_PHYSICS.md)** | radiative transfer, atmosphere, emissivity, false-alarm physics |
| **[`docs/04_FOUNDATION_MODEL.md`](docs/04_FOUNDATION_MODEL.md)** | train your own vs fine-tune, with costs |
| [`docs/05_PRIOR_ART.md`](docs/05_PRIOR_ART.md) | lessons borrowed from the FirePerim platform |
| **[`docs/06_CONSTELLATION.md`](docs/06_CONSTELLATION.md)** | 2026-2030 sensor plan and risk register |
| **[`docs/07_PHASE0.md`](docs/07_PHASE0.md)** | **start here to build**, week-by-week execution plan |

---

## The two decisions that shape everything

**0. Physics is structure, not a feature.** The Planck function, the Dozier
mixture, the Wooster FRP equation and atmospheric transmittance are built in as
constraints and differentiable layers. `ConstrainedFRPHead` keeps the Wooster
equation and learns only a bounded multiplicative correction, initialised at
exactly 1.0. The model starts as pure physics and must earn every departure.

**1. Physics core, ML at the boundaries.** VHAGAR does not replace operational
contextual fire detection, the FWI/NFDRS fuel-moisture codes, or level-set fire
propagation with end-to-end deep learning. The evidence says the marginal
return on the tenth U-Net variant is near zero, while the return on better fuel
state, better labels, better assimilation, and honest validation is large. ML
goes where it demonstrably wins: event-level classification, state estimation
from noisy multi-sensor observations, simulator emulation, and residual
correction.

**2. The validation protocol is code, not prose.** Almost every wildfire ML
result that looks too good is a leakage artifact. So:

```python
>>> from vhagar.eval.splits import random_split
>>> random_split(units)
NotImplementedError: Random splits are not supported. Fire data is spatially
and temporally autocorrelated; a random split inflates metrics by 0.2-0.4 F1.
Use spatial_block_split(), leave_one_group_out(), or leave_year_out().
```

That is not a joke. On a published FIRMS classification task, F1 fell from
**0.985** (random split) to **0.767** (event-aware) to **0.627** (5° spatial
block), and raw lat/lon supplied ~89 % of model gain while *harming*
out-of-region transfer.

---

## Install

```bash
pip install -e ".[dev]"          # core + tests, no GDAL or torch needed
pip install -e ".[all]"          # everything
```

Extras: `geo` (xarray/rasterio/odc-stac/zarr) · `gee` (earthengine-api, xee) ·
`torch` · `gbdt` · `serve` (FastAPI/PostGIS).

The core package is deliberately dependency-light so metrics, splits, spectral
indices and the FWI engine run in CI without a geospatial stack.

## Start here

```bash
pip install -e ".[dev]" && pip install s3fs xarray h5netcdf pyproj

# Phase 0 Step 1: first real bytes: GOES from S3 + VIIRS from FIRMS, fused
python scripts/step1_first_light.py --bbox -124 36 -118 42 --satellite 18 --hours 6
```

Then `docs/07_PHASE0.md`. Colab users: `notebooks/01_first_light.ipynb`.

## Try it offline

```bash
python scripts/demo_end_to_end.py   # full tour, no network needed
vhagar sensors                     # what's flying, and what stops when
vhagar frp --bt-mir 355 --view-zenith 55
vhagar grid info --region conus
vhagar fwi --temp 28 --rh 25 --wind 30 --days 10
vhagar area-estimate --confusion 97,3,10,90 --areas 200000,20000 --names unburned,burned
vhagar splits verify splits/leave_year_out.json
pytest -q
```

---

## Layout

```
docs/
  00_ARCHITECTURE.md      full system design and methodology
  01_DATA_CATALOG.md      every dataset, ID, licence, caveat
  02_VALIDATION.md        the binding evaluation contract
configs/                  Hydra configs (data / model / task / trainer)
src/vhagar/
  grid.py                 equal-area 375 m analysis grid + tiling
  io/                     goes_reader (real FDC ingest) · abi_grid (navigation)
                          gee · firms · sensors (registry)
  harmonize/              regrid (resampling policy) · fusion (event clustering)
  features/               indices (NBR/dNBR/RBR) · fwi (Van Wagner 1987) · physics_features
  labels/                 MTBS · NBAC · EFFIS · FEDS harmonisation
  datasets/               split-manifest-driven dataset construction
  physics/                planck · atmosphere · frp · dozier · geometry
  models/                 UNet · SiameseChangeNet · TemporalAnomalyNet · physics_heads
  train/                  losses · Lightning module · training entrypoint
  eval/                   splits · metrics · area_estimation · baselines
  serve/                  FastAPI + tile serving
  pipelines/              NRT hot path, Dagster cold path
tests/                    leakage, mass conservation, FWI + Planck regression,
                          nodata, sensor lifecycle, physics-head constraints
```

---

## What is already implemented

- **`grid.py`**, deterministic equal-area tiling for all three regions, halo
  handling, stable tile IDs
- **`eval/splits.py`**, spatial-block, leave-one-group-out (fire / ecoregion /
  continent / tile), leave-year-out; versioned, fingerprinted manifests; CI
  leakage assertions
- **`eval/metrics.py`**, average precision, IoU/Dice, POD/FAR, burned-area
  ratio, Brier with Murphy decomposition, reliability curves, ECE, F1 with an
  explicit spatial tolerance
- **`eval/area_estimation.py`**, full Olofsson error-adjusted area with 95 %
  CIs, stratified sample sizing and allocation
- **`eval/baselines.py`**, persistence, persistence + calibrated buffer,
  climatology, tuned spectral threshold
- **`features/fwi.py`**, the complete Canadian FWI System, vectorised, tested
  against the Van Wagner & Pickett worked example
- **`features/indices.py`**. NBR, dNBR, RdNBR, RBR, NDVI, NDMI, NDWI with
  guarded denominators (NaN, never inf)
- **`physics/`**. Planck radiometry, atmospheric transmittance, the Wooster FRP
  method with sensor constants and uncertainty propagation, Dozier sub-pixel
  retrieval **with its condition number attached**, and observation geometry
  (solar/view zenith, glint angle, pixel-area growth)
- **`features/physics_features.py`**, 34 physics features with a hard
  `FORBIDDEN_FEATURES` guard against coordinate leakage
- **`models/physics_heads.py`**, constrained FRP head, differentiable
  Planck-mixture decoder for learned inversion, censored loss for saturation
- **`io/sensors.py`**, the platform/instrument registry, with decommission
  dates that change pipeline behaviour rather than living in a comment
- **`harmonize/fusion.py`**, parallax-aware multi-sensor event clustering and
  coordinate-free event features
- **`harmonize/regrid.py`**, resampling policy by quantity type; exact
  conservative regridding with mass-conservation assertions
- **`io/gee.py`, `io/goes.py`, `io/firms.py`**. GEE high-volume patch export,
  GOES S3 hot path, FIRMS client
- **`models/`**, **`train/losses.py`**. U-Net baseline, siamese change net,
  geostationary temporal anomaly net, combo/Tversky/focal losses

## What is scaffolded but not yet filled in

Label harmonisation adapters (`labels/`), the Lightning training loop,
ELMFIRE integration and assimilation for T4, the serving layer, and the
Dagster cold-path assets. See the roadmap in `docs/00_ARCHITECTURE.md` §10.

---

## Numbers to keep you honest

| Claim | Reality |
|---|---|
| Next-day spread AP | **0.35-0.45** is state of the art; persistence is 0.19. Doubling training data does not help, cross-year domain shift dominates. |
| Perimeter forecast IoU | 0.6-0.8 on wind-driven fires, materially worse plume-dominated; ROS MAPE 40-60 % in timber |
| Label ceiling | VIIRS-derived perimeters score 0.71-0.93 F1 vs agency perimeters, and ~9 % of their interior is unburned islands |
| Fold variance | ±0.08-0.10 AP on spread benchmarks, a 0.02 improvement without per-fold numbers is noise |
| Uncorrected FRP | biased **~31 % low** at nadir, **>50 % low** at 60° view zenith. `transmittance=1.0` warns. |
| Dozier (p, T_f) | GOES fire-area correlated with reference at **r = −0.22**. Report FRP; gate (p, T_f) on its condition number. |
| Loss function vs pretraining | Cross-entropy → Dice took a from-scratch U-Net's fire IoU from **0.022 → 0.272**. That 12× dwarfs what pretraining bought anyone. |

Anything claiming ≫0.5 AP on next-day spread or >0.9 IoU on real perimeters is
almost certainly a leaky split or cumulative-rather-than-incremental area.

## Licence

Apache-2.0. Third-party data licences are catalogued in `docs/01_DATA_CATALOG.md`.
