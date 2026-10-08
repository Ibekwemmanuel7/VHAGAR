# 28. Real-data parcel study: where Texas utility-scale solar was built

Status: design fixed before any result was computed (2026-10-08). Results: see
"Results" (filled from `outputs/parcel_study/results.json`, which is the only source of
every number quoted).

## Question

Using only information available at a 2016 snapshot, how well can a parcel-level screen
rank Texas land that later hosted a ground-mounted utility-scale solar facility (installed
2017 to 2025) above land that did not?

This is a *revealed-siting* question. A built facility shows that a site cleared
physical, grid, financial, permitting, and landowner hurdles; an unbuilt parcel is not
proof of unsuitability. The labels are therefore biased toward land that was offered,
financed, and connected first. The study measures how well a screen anticipates where
development actually went, which is the deployment question for a siting product, and it
states that limitation wherever a number is quoted.

## Data (all public; vintages and checksums in `data/parcel/raw/manifest.json`)

| Layer | Source | Vintage | Use |
|---|---|---|---|
| Parcels | TxGIO StratMap Land Parcels, 253 county files | 2026-09 | population, geometry |
| Labels | USGS/LBNL USPVDB v4.0, ground-mounted footprints | 2026-04 | cases, install year |
| Land cover | Annual NLCD, Collection 1 | 2016 | land-cover fractions |
| Roads | TIGER/Line primary and secondary roads | 2016 | distance to road |
| Wildfire | USFS Wildfire Hazard Potential (continuous, 270 m) | 2018 | mean hazard |
| Terrain | USGS 3DEP 1 arc-second DEM | tile dates | slope statistics |
| Irradiance | NASA POWER climatology (GHI) | multi-year | interpolated GHI |
| Transmission | HIFLD Transmission Lines (archive of the portal closed 2025-08-25) | 2025 | distance to line |
| Protected land | PAD-US 3.0 Fee and Easement, Texas | 2022 | overlap fraction |
| Floodplain | FEMA NFHL flood hazard zones (SFHA) and availability | effective at fetch | overlap, mapping status |

## Population, labels, and sampling

* **Population:** every TxGIO parcel of at least 2 ha after repairing geometry, projecting
  to EPSG:5070, and removing stacked duplicate polygons (appraisal rolls often repeat one
  polygon for several accounts, especially mineral interests). Parcels already covered by
  a facility built in or before 2016 are excluded.
* **Cases:** a facility installed 2017 to 2025 covers at least 2 ha or 25% of the parcel.
  Parcels with a smaller, sub-threshold overlap are ambiguous (edge slivers, digitising
  offsets) and are excluded from both groups.
* **Controls:** a simple random sample of 30,000 non-case parcels with a fixed seed. Each
  control carries the inverse sampling fraction as its weight, so every precision, average
  precision, and land-area figure describes the Texas parcel population, not the sample.

## Features and leakage control

Features describe the land at the snapshot wherever a historical layer exists. Each
feature's leakage risk relative to 2016 is recorded in `FEATURE_SPECS`
(`src/vhagar/parcel/study/features.py`) and in `results.json`.

* **Transmission is the main leakage risk.** The only public line inventory is the 2025
  HIFLD archive, which includes generator tie lines built *for* later facilities. The main
  models use `dist_tx_robust_km`, which drops lines shorter than 15 km that end within
  2 km of any facility footprint (probable generator ties). That is conservative: it can
  also drop genuine pre-existing lines near facilities. The evaluation reports the model
  with the raw distance (optimistic), the robust distance (main), and no transmission
  features at all, so the effect of this leak is measured rather than hidden.
* **No raw coordinates.** In VHAGAR's fire-detection work, raw lat/lon supplied most of a
  model's gain under a random split and harmed transfer under a spatial holdout
  (docs/02_VALIDATION.md). Here location enters only through physical layers; a
  coordinates-only model is reported as a diagnostic of spatial memorisation.
* **No assessor values or land-use codes.** The 2026 appraisal attributes are recorded
  after construction (a solar lease changes assessed value and use codes) and are not used.
* **Floodplain mapping gaps stay missing.** Where NFHL has no effective mapping, the
  floodplain fraction is missing, not zero. The rules engine then returns
  "insufficient evidence"; the learned models treat missingness as information.

## Models

| Model | Kind | Notes |
|---|---|---|
| Null (prevalence) | baseline | every parcel tied; traces random screening |
| Nearest transmission | baseline | single feature, closer ranks higher |
| Irradiance | baseline | single feature |
| Rules engine v0.2 | transparent rules | `vhagar.parcel` solar config, not fitted to any label; exclusions and evidence gates as designed |
| Spline logistic (GAM) | learned | splines per feature, median imputation with missing indicators |
| Gradient boosting, monotone | learned (main) | monotone constraints encode prior physical knowledge (for example, steeper slope never raises the score) |
| Gradient boosting, unconstrained | learned | shows what the constraints cost or save |
| Coordinates only | diagnostic | memorises where solar already is |

Hyperparameters were fixed before evaluation (`modeling.make_hgb`); no tuning was done on
test data.

## Evaluation

* **Temporal holdout (primary):** train on cases installed 2017 to 2020 plus a random half
  of the controls; test on cases installed 2021 to 2025 plus the other half. This is the
  prospective question: would a screen built on earlier projects have pointed at the
  next ones?
* **Spatial holdout:** 5 folds of 2-degree blocks (`vhagar.eval.splits.spatial_block_split`);
  cases are blocked by their facility's location so no facility straddles train and test.
* **Metrics (population-weighted):** average precision and its lift over prevalence,
  ROC AUC, and the screening metric: the share of later solar parcels captured when the
  highest-scored parcels covering 1%, 5%, 10%, and 20% of Texas parcel land are screened.
  Ties are credited at their expected value, so random screening traces the diagonal.
* **Calibration:** probabilities from case-control fits are mapped to the population with
  the King and Zeng (2001) prior correction; reported with Brier skill and ECE.
* **Uncertainty:** 95% intervals from a cluster bootstrap that resamples whole facilities
  (parcels of one facility are not independent) and controls independently, including
  intervals for differences between models.
* **Explanation:** permutation importance (drop in weighted average precision) and
  partial dependence for the main model.

## Reproduce

```bash
python scripts/parcel_fetch.py stage1          # on a machine with internet
python scripts/parcel_build.py population
python scripts/parcel_build.py sample
python scripts/parcel_fetch.py stage2
python scripts/parcel_build.py features --part vector
python scripts/parcel_build.py features --part nfhl
python scripts/parcel_build.py features --part rasters   # resumable, rerun until done
python scripts/parcel_build.py features --part slope     # resumable, rerun until done
python scripts/parcel_build.py features --part merge
python scripts/parcel_build.py evaluate
```

## Results

(Filled after the evaluation run.)
