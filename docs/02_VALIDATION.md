# VHAGAR: Validation Contract

This is binding. `src/vhagar/eval/` implements it, `tests/` enforces it, and
CI fails the build when it is violated. A model that has not been evaluated
this way is not promotable, regardless of how good its numbers look.

---

## 1. Why this document exists

Three published results, all on wildfire tasks:

| Situation | Result |
|---|---|
| FIRMS wildfire vs non-wildfire classification | F1 **0.985** random split → **0.767** event-aware → **0.627** 5° spatial block. Raw lat/lon supplied ~89 % of model gain while *harming* out-of-region transfer. |
| Burned-area severity regression | R² **0.679** leave-one-cluster-out → **0.616** leave-one-fire-out. |
| Burned-area segmentation, MTBS → Copernicus EMS | F1 **0.735 ± 0.186** across 17 European sites. The ±0.19 standard deviation *is* the finding. |

And two on evaluation protocol itself:

* Fold-to-fold standard deviation on next-day spread benchmarks is
  **±0.08-0.10 AP**, comparable to the entire spread of model rankings.
* The same frozen model can score ~0.05 % F1 under exact pixel matching and
  7-30 % F1 under an 8-cell spatial tolerance.

Every one of these differences is larger than the improvement a typical paper
claims. The protocol is therefore not a formality.

---

## 2. Splits

### 2.1 Random splits are unavailable

```python
>>> from vhagar.eval.splits import random_split
>>> random_split(units)
NotImplementedError: Random splits are not supported. ...
```

There is no flag to turn this off.

### 2.2 Required blocking, by task

| Task | Required |
|---|---|
| **T1 detection** | event-aware **and** spatial block (≥5° or ≥500 km) **and** leave-year-out |
| **T2 burned area** | leave-one-fire-out, leave-one-ecoregion-out, **leave-one-continent-out** (train MTBS → test Copernicus EMS) |
| **T3 danger** | spatial block sized from the residual variogram range × leave-one-season-out |
| **T4 spread** | leave-one-fire-out × leave-year-out (year-permutation folds) |

Report **all** applicable schemes, not the most flattering one. The gap
between them is diagnostic information about generalisation, not noise to be
suppressed.

### 2.3 The split unit

Never a pixel, never a chip. A `SplitUnit` is a fire event, a tile, or a
(tile, year) pair, something whose members are *not* exchangeable with
members of another unit.

### 2.4 Manifests

Every split is serialised as a `SplitManifest` with a stable fingerprint and
versioned next to the model artifact. Datasets are constructed **only** via
`VhagarDataset.from_fold(records, manifest, fold, subset)`, which raises if
any manifest unit lacks a chip record, refusing to silently train on a
different set than the manifest describes.

### 2.5 Normalisation statistics

Channel means and standard deviations are computed on **training folds only**.
Computing them over the whole dataset is a subtle but real leak: test-fold
radiometry then influences what the model sees during training.

---

## 3. Metrics

### 3.1 Tune only on proper scoring rules

| Purpose | Use | Never use |
|---|---|---|
| Model selection, early stopping, hyperparameter search | log loss, Brier, CRPS | F1, CSI, IoU, accuracy |
| Decision-support reporting at an operational threshold | POD, FAR, CSI, F1, IoU |. |

F1, CSI and IoU are **improper**: their optimum depends on the threshold you
happen to pick, so tuning on them optimises a threshold choice rather than the
model. Report them; do not train against them.

### 3.2 Per task

**T1 detection**
POD · FAR · precision · F1 at *event* level · FRP bias and RMSE ·
**median minutes from agency-reported ignition to first alert** (the primary
metric) at a fixed acceptable false-alarm rate.
Stratify by: day/night, land cover, fire size decile, view zenith angle.

**T2 burned area**
Dice/IoU · **Olofsson error-adjusted area with 95 % CI**, never a pixel count.
50-100 reference samples in the burned stratum per reporting region.
Severity: R² and RMSE against CBI plots, reported **per ecoregion**.

**T3 danger**
AUPRC · Brier with Murphy decomposition (reliability / resolution /
uncertainty) · log loss · ECE with a stated binning scheme · reliability
diagram · **BSS against a pixel × day-of-year climatology** (never against a
constant base rate, which makes seasonality look like skill).

**T4 spread**
Average precision · IoU at a stated threshold · **burned-area ratio**
(predicted/observed. IoU hides systematic area bias, this exposes it) ·
arrival-time MAE. Per fold **and** per fire. Stratify by wind-driven vs
plume-dominated regime.

### 3.3 Base-rate dependence

AP shifts with the positive base rate. It is comparable **only** between
models evaluated on identical splits. Never compare an AP from one benchmark
to an AP from another.

### 3.4 Spatial tolerance must be stated

`f1_with_tolerance(y_true, y_pred, tolerance_cells)` requires the tolerance
argument explicitly. Reporting F1 without it makes the number meaningless,
because label geolocation error is itself 1-2 cells.

---

## 4. Mandatory baselines

Reported in every experiment, forever:

| Task | Baselines |
|---|---|
| T1 | the sensor's own operational contextual product |
| T2 | calibrated spectral-index (RBR) threshold; plain U-Net |
| T3 | FWI threshold; pixel × day-of-year climatology |
| T4 | persistence; **persistence + calibrated isotropic buffer**; physics simulator with identical forcing |

The persistence-plus-buffer baseline is the one the literature almost never
reports and which closes much of the apparent gap to deep models. Calibrate
the buffer radius on training folds against median observed daily growth.

---

## 5. Calibration gate

Any model whose output is served as a probability must pass, on the held-out
test fold with the base rate preserved:

* expected calibration error ≤ 0.05 (quantile bins, stated `n_bins`)
* reliability diagram monotone within sampling noise
* Brier reliability term < 10 % of total Brier

Dice/IoU-trained segmentation models are systematically miscalibrated. Fit
isotonic or temperature recalibration on a held-out, **base-rate-preserving**
set, never on downsampled data.

---

## 6. Reproducibility record

Every run logs, and every promoted model ships:

1. resolved config (Hydra)
2. split manifest **fingerprint** and file
3. `numpy`, `torch`, **GDAL and PROJ** versions (a PROJ minor release can move
   your pixels via grid-shift updates)
4. STAC item IDs / S3 keys of every scene consumed
5. random seeds, and `torch.use_deterministic_algorithms(True)`
6. training-fold normalisation statistics

---

## 7. Geospatial tests in CI

| Test | Catches |
|---|---|
| CRS round-trip invariance | PROJ grid-shift regressions on upgrade |
| Conservative-regrid **mass conservation** on FRP | resampling a flux with bilinear; corrupted weight caches |
| Hypothesis property tests on geometry ops | GEOS edge cases |
| **Nodata / mask propagation** | the most common silent EO bug: nodata → 0 → "cold ground" |
| Guarded denominators yield NaN, not inf | index blowups masquerading as extreme severity |
| Georeferencing golden test | a known historical fire must be detected within N metres |
| Determinism | same seed + input → bitwise-identical output |
| `verify_no_overlap` on every manifest | train/test contamination |
| Coordinate-free event features | spatial memorisation in the T1 classifier |

---

## 8. Reporting template

Every result table must contain, at minimum:

```
task | model | split scheme | n folds | metric ± fold sd | baseline | Δ vs baseline
```

A single number with no fold standard deviation and no baseline is not a
result. On spread tasks specifically, an improvement below ~0.08 AP is within
fold noise and must be described as such.
