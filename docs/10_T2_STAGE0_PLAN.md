# T2 burned-area Stage-0 plan (the first honest number)

Status: draft for review, not yet implemented.

## Goal

The first end-to-end, leakage-proof, uncertainty-bearing number the project can
defend. Per the architecture (§3.3, §4.3), Stage 0 is **a tuned supervised
baseline, and it may well be the product**: a **calibrated RBR/dNBR threshold**,
evaluated with **Olofsson error-adjusted area and a 95% CI** on the required T2
blocked splits, reported next to the permanent baselines. Not a foundation model.
The U-Net comes second, as a companion baseline, not as the headline.

## What already exists (do not rebuild)

The algorithms are in place. This is the pleasant surprise: Stage 0 is wiring.

- Spectral method: `features/indices.py` has `nbr`, `dnbr`, `rbr`,
  `classify_severity`. RBR is already the primary metric.
- Baselines: `eval/baselines.py` has `threshold_baseline`, `tune_threshold`,
  `persistence`, `persistence_with_buffer`. The calibrated-threshold Stage 0 is
  `tune_threshold` on a train fold, `threshold_baseline` on the test fold.
- Uncertainty: `eval/area_estimation.py` has `estimate_areas` (Olofsson with
  95% CI), `sample_size_stratified`, `allocate_samples`. The whole point of T2.
- Labels: `labels/registry.py` + the MTBS adapter give trainable events with a
  dNBR severity path.
- Splits: `eval/splits.py` gives leave-one-fire-out, spatial block, leave-year;
  the T2 protocol (§7.1) also wants leave-one-ecoregion-out and the headline
  leave-one-continent-out (MTBS -> EMSR).
- Model + loss: `models/segmentation.py` has a siamese pre/post encoder-decoder;
  `train/losses.py` has `DiceLoss`, `FocalLoss`, `TverskyLoss`, `ComboLoss`.
  Note the architecture's own finding: switching CE -> Dice moved fire IoU from
  0.022 to 0.272, a bigger jump than pretraining ever bought. Use Dice/Combo.

## What is missing: the data plumbing

1. **Pre/post optical predictor per fire.** RBR needs pre-fire and post-fire NBR.
   Either read MTBS's own dNBR raster, or composite Sentinel-2/Landsat over the
   -90..-15 d and +15..+75 d windows (mean compositing with the unburned-buffer
   offset correction, per §4.2). This is the pivotal decision below.
2. **A T2 dataset builder.** Align, per fire and per tile: the predictor index
   (RBR/dNBR) and the reference label (the MTBS burn boundary / severity), on the
   analysis grid, halo included. Nodata/mask propagation is the classic silent
   bug (§9.2), so it is a tested invariant here.
3. **A Stage-0 experiment driver.** For each fold: calibrate the RBR threshold on
   train fires, predict burned/unburned on test fires, build the confusion
   matrix against reference samples, run Olofsson to get error-adjusted area with
   CI, and report per fold alongside persistence and the U-Net. Per-fold
   reporting is mandatory (§7.1): fold std on these tasks rivals model spread.
4. **The Olofsson reference sample.** Stratified random points (map class as
   stratum, burned area is <1% so stratification is not optional), labelled from
   the highest-quality geometry available. Sample allocation already exists.

## Module shape

- `vhagar/datasets/burned_area.py`: a `T2Sample` (tile, pre index, post index,
  reference mask, fold key) and a builder that turns registry records plus
  imagery into samples. Pure alignment/masking logic tested on synthetic arrays;
  the imagery read is the lazily-imported edge.
- `vhagar/eval/t2_stage0.py`: the experiment driver. Consumes a split manifest
  and the samples, returns per-fold `AreaEstimate`s and baseline comparisons.
- Reuse everything else. No new model or loss code for Stage 0.

## Evaluation contract (binding, from §7)

- Report **Olofsson adjusted area with 95% CI**, never a pixel count.
- Metrics: Dice/IoU for the map, plus the adjusted-area estimate and its CI;
  severity R²/RMSE against CBI is a later addition, not Stage 0.
- Splits: leave-one-fire-out and spatial block first; leave-one-ecoregion-out
  once ecoregions are attached; leave-one-continent-out (MTBS -> EMSR) is the
  headline generalisation number and needs the EFFIS/EMSR adapters (label spine
  step 4).
- Permanent baselines reported every time: persistence, persistence+buffer, and
  the calibrated RBR threshold itself is the Stage-0 "model".

## Staged delivery

1. T2 dataset builder plus tests (alignment, nodata propagation, fold keying).
2. Stage-0 driver: threshold calibration -> Olofsson area + CI -> per-fold and
   baseline reporting. Tested on synthetic fires with a known burned fraction.
3. Plain U-Net companion baseline (torch-gated, Dice/Combo loss), same eval.
4. Leave-one-continent-out once EMSR is ingested.

Steps 1 to 2 produce the first defensible number. Steps 3 to 4 are the honest
companions and the headline generalisation test.

## Open decisions, need a call before coding

1. **Predictor imagery: MTBS dNBR raster, or independent S2/Landsat composites?**
   - MTBS dNBR is fast (one raster per fire, already computed) and gets a first
     number quickly, but MTBS computes that dNBR itself, so calibrating a
     threshold on it and evaluating against MTBS geometry shares lineage. It is a
     legitimate *pipeline* baseline, not an independent accuracy claim.
   - Independent S2/Landsat composites are the architecture's intended path and
     give a defensible number, at the cost of a real imagery pull (GEE or S3
     COGs) and the compositing code.
   - **DECIDED: MTBS dNBR first, then swap.** Stand the pipeline up and get a
     per-fold Olofsson number fast, clearly labelled as lineage-shared, then swap
     in independent S2/Landsat composites for the number that goes in a report.
     The driver is identical; only the predictor source changes, so the swap is a
     new input adapter, not a rewrite.
2. **First scope.** One region-year (e.g. California 2020) to keep the imagery
   pull and reference labelling small, or a wider CONUS-year set. Recommendation:
   one region-year for the first pass.
3. **Reference-sample labelling.** For a genuinely independent Olofsson number
   the reference points need a source better than the map being evaluated (VHR
   imagery review, or CBI where available). For the first lineage-shared pass,
   MTBS thematic burn boundary is the reference. Decide the independent source
   before quoting the number externally.

Decision 1 sets the dataset builder's input path. The rest can be settled as we
reach their step.
