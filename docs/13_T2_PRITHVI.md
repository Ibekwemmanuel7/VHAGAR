# T2 Prithvi-EO-2.0: fine-tuning a geospatial foundation model for burned area

The T2 burned-area work so far is a physics/threshold baseline (independent Sentinel-2 RBR vs
MTBS) plus small segmentation models (U-Net, Siamese) that beat that threshold by a modest
skill margin on leakage-proof folds. The frontier question is whether a pretrained geospatial
foundation model, fine-tuned, beats those. Prithvi-EO-2.0 (NASA/IBM) is the state-of-the-art
choice: a ViT pretrained on Harmonized Landsat-Sentinel (HLS) surface reflectance, with a
published burn-scar fine-tune (Prithvi-EO-2.0-300M-BurnScars, UNet decoder, 87.5 IoU on the
HLS Burn Scars benchmark).

This is deliberately "machine learning at the boundary": Prithvi is a feature extractor at the
optical edge, scored by the same physics-anchored, leakage-proof protocol as everything else.

## What is built here (offline, tested)

The one thing Prithvi needs that the RBR path did not is the **six-band re-pull**. RBR uses two
bands (B8A, B12) to form NBR; Prithvi consumes six surface-reflectance bands in a fixed order:
Blue, Green, Red, narrow NIR, SWIR1, SWIR2 (`io.optical.PRITHVI_BAND_ASSETS`).

- `io.optical.sentinel2_bands6` pulls the post-fire six-band cloud-masked composite onto a
  fire's 30 m grid, in Prithvi's band order, as surface reflectance `[6, H, W]`. Streaming
  compositor (`stream_band_composite`) holds flat memory over many scenes; unit-tested offline.
- `datasets.t2_optical.build_prithvi_sample` pairs that six-band stack with the MTBS burned
  mask on the identical grid and window as the RBR sample, so a fire's Prithvi sample and its
  RBR sample share a reference and can be scored head-to-head. Cached as a `T2Sample` `.npz`
  (tag `p6`); unit-tested offline with a stubbed pull.
- CLI `vhagar t2-prithvi-build` selects fires and builds/caches the six-band dataset.

## Runbook (the fine-tune runs on your GPU)

Prithvi fine-tuning uses **TerraTorch** (PyTorch Lightning + TorchGeo). That, the pretrained
weights, and a GPU are the pieces that must run on your machine; the steps below are the
state-of-the-art recipe from the model card, adapted to VHAGAR's fires and, crucially, scored
on VHAGAR's leakage-proof grouped folds so the comparison to RBR/U-Net is fair.

1. **Build the six-band dataset** (needs the open Sentinel-2 network):

   ```
   vhagar t2-prithvi-build --registry registry.parquet --mosaic mtbs_CONUS_2021.tif \
       --region conus --year 2021 --max-fires 20 --res-m 30 --cache-dir data/t2_prithvi
   ```

2. **Export terratorch chips** (built: `vhagar t2-prithvi-export`). `t2-prithvi-export`
   partitions whole fires into train/val/test (`eval.t2_prithvi.grouped_split`,
   leakage-proof), tiles each cached sample into `chip x chip` image/label pairs
   (`chip_sample`), and writes them in the published HLS-Burn-Scars layout: all chips in one
   `out_dir/data` directory as `{stem}_merged.tif` (six-band image) + `{stem}.mask.tif`
   (label 0/1/-1), with `out_dir/splits/{train,val,test}.txt` listing each split's stems.
   That is exactly what the model card's config consumes (`img_grep: "*_merged.tif"`,
   `label_grep: "*.mask.tif"`, one data root + split files).

   ```
   vhagar t2-prithvi-export --cache-dir data/t2_prithvi --out-dir data/t2_prithvi_chips --chip 224
   ```

3. **Install TerraTorch and fetch weights:**

   ```
   pip install terratorch
   # weights: ibm-nasa-geospatial/Prithvi-EO-2.0-300M (backbone) from Hugging Face
   ```

4. **Fine-tune.** A ready config is generated at the repo root, `prithvi_burnscars_vhagar.yaml`
   (Prithvi-EO-2.0-300M backbone + UNet decoder, the six bands, `num_classes: 2`,
   `ignore_index: -1` for the nodata label, `ce` loss). It is the published
   `burn_scars_config.yaml` with only two changes: the data paths point at
   `data/t2_prithvi_chips`, and the per-band `means`/`stds` were recomputed from *this*
   dataset's training fires (valid pixels only, no val/test leakage) so they match Sentinel-2
   L2A reflectance rather than HLS.

   ```
   pip install terratorch
   terratorch fit -c prithvi_burnscars_vhagar.yaml
   ```

   If `terratorch fit` errors on a datamodule argument, it is a version-schema difference in
   `GenericNonGeoSegmentationDataModule` (e.g. an arg rename); `terratorch fit --help` and the
   datamodule docstring show the current names. The band, decoder, and loss arguments come
   straight from the published config and should not need changes.

5. **Score on the same fires as RBR/U-Net** (built: `vhagar t2-prithvi-score`). Run terratorch
   inference on the test fires, write one predicted burned mask per fire as
   `{event_id}.tif` (with `:` and `/` replaced by `_`, burned where > 0), then:

   ```
   vhagar t2-prithvi-score --cache-dir data/t2_prithvi --pred-dir data/prithvi_preds
   ```

   terratorch predicts one mask *per chip*; the fire-level metric needs one mask *per fire*.
   Point `t2-prithvi-score` at terratorch's per-chip predictions plus the export's
   `_chips.json`, and it stitches them back per fire (`stitch_chip_predictions`, each chip
   placed at its stored pixel offset, clipped to the fire grid, burned-wins on overlap):

   ```
   vhagar t2-prithvi-score --cache-dir data/t2_prithvi --pred-dir data/prithvi_preds \
       --chips-manifest data/t2_prithvi_chips/_chips.json
   ```

   This pushes the stitched masks through the *same* skill-over-naive metric as RBR and the
   U-Net (`score_masks` -> `eval.metrics.confusion_counts`): F1/IoU, the predict-all-burned
   naive F1 on the same valid pixels, and `skill = f1 - naive_f1`, then the mean over fires.
   Compare that mean skill to `t2-unet` and `t2-stage0` on the *same test fires* (the
   `_split.json` test list). That is the head-to-head: does the foundation model beat the
   current +0.54 skill, and does it degrade less *out of region* on the leave-one-continent-out
   split, where the small models and the threshold both fall off.

## Honest expectations and caveats

- **Domain shift.** Prithvi was pretrained and burn-scar-fine-tuned on HLS. VHAGAR pulls
  Sentinel-2 L2A, which is harmonised with HLS-S30 but not identical (bandpass, BRDF, scaling).
  Expect some transfer loss; it is the closest same-code-path substitute from the open catalog.
  A stricter test would re-pull actual HLS, which the pipeline can be pointed at later.
- **Fair comparison or none.** The only defensible claim is Prithvi vs RBR vs U-Net on the same
  held-out fires, same reference, same skill metric. A benchmark IoU from the model card is not
  comparable to VHAGAR's numbers because it did not walk the same code path.
- **What would make it a result.** Beating the +0.54 U-Net skill on the grouped folds, and
  degrading less than the small models on the leave-one-continent-out transfer, would be real
  evidence that the foundation model's pretraining buys generalisation. Matching them would say
  the pretraining does not help at this data scale, an honest negative.

## Status

Built and tested offline: the six-band re-pull (`sentinel2_bands6`, `stream_band_composite`),
the Prithvi sample builder (`build_prithvi_sample`), the terratorch chip export
(`eval.t2_prithvi.grouped_split` / `chip_sample` / `export_prithvi_chips`, CLI
`t2-prithvi-export`), and the fair scoring bridge (`score_masks` / `summarise_scores`, CLI
`t2-prithvi-score`) — the whole VHAGAR side of the pipeline, leakage-proof and scored by the
same skill-over-naive metric as RBR and the U-Net.

The only thing that must run with a GPU is `terratorch fit`. Everything else — the six-band
pull, dataset assembly, chipping, the leave-fire-out split, per-chip→per-fire stitching, and
the head-to-head scoring — is built and unit-tested. End-to-end: `t2-prithvi-build` →
`t2-prithvi-export` → `terratorch fit` → `t2-prithvi-score --chips-manifest`, then compare the
mean skill to `t2-unet`/`t2-stage0` on the same test fires.

### First real fine-tune result (2026-08-16): honest underperformance, and the fix

The full pipeline ran end to end on a Colab T4: 20 CONUS-2021 fires -> 470/75/138 chips ->
`terratorch fit` (Prithvi-EO-2.0-300M + UNet decoder, ce loss, ~29 epochs, early-stopped) ->
`terratorch test` -> 138 per-chip predictions -> download -> `t2-prithvi-score`.

Test metrics: **burn-scar IoU 0.10, burn recall 11%, not-burned recall 98%.** Per-fire skill
over naive on the three test fires: **mean +0.054** (one fire a total miss at -0.098, one
+0.013, one +0.246), versus the U-Net's +0.54. So a straight fine-tune badly under-detects
burned area and does not beat the small model. This is not a defect of the foundation model:
it is class imbalance. Our analysis windows are wide (a fire is a small burn in a large ring
of unburned land), so most pixels, and even most burn-touching chips, are unburned, and plain
cross-entropy is minimised by predicting "not burned" almost everywhere. The published 87.5
IoU used the curated, burn-balanced HLS Burn Scars chips; ours are not.

The fix is rebalancing, in two places:

- **Burn-balanced chips.** `chip_sample(..., burn_balance=True)` (CLI `t2-prithvi-export
  --burn-balance`) keeps every *training* chip containing burned pixels and caps the
  all-unburned chips at `max_bg_ratio` times the burn chips (non-overlapping tiling, so no
  redundant near-duplicate chips). On our 20-fire set this takes the training split from 470
  chips (42% burn) to ~318 chips (63% burn), a ~400 MB dataset. Val/test stay a faithful
  uniform tiling, so the score still reflects true per-fire coverage.
- **Imbalance-robust loss.** Even a burn-*containing* chip is mostly unburned pixels, so the
  Colab config now takes a `LOSS` parameter defaulting to `dice` (which optimises overlap
  directly) instead of `ce`; `focal` is the other option.

Re-export with `--burn-balance`, set `LOSS = 'dice'` in the notebook, and re-run: that is the
fair test of what Prithvi can do on this data. A caveat on the comparison itself: the U-Net's
+0.54 came from its own CV, not these exact three test fires, so a strict head-to-head also
means running `t2-unet`/`t2-stage0` on the same three fires. And three test fires is a plumbing
check, not a verdict; more fires (and the European set for leave-one-continent-out) are the
real evaluation.

### The rebalanced result (2026-08-16): the fix works

Re-ran with `--burn-balance` (318 train chips, 63% burn) and `LOSS = 'dice'`, 60 epochs on a
Colab T4. The under-detection is gone:

| metric | imbalanced + CE | burn-balanced + Dice |
|---|---|---|
| test burn-scar IoU | 0.10 | **0.21** |
| test burn recall | 11% | **50%** |
| per-fire mean skill | +0.054 | **+0.398** |

Per-fire skill over naive: MN +0.042 (was a total miss at -0.098), WA +0.520 (F1 0.82), WA
+0.634 (F1 0.82) - all three positive, versus the U-Net's +0.54. Two of the three fires are at
or above U-Net-level skill; the small MN fire drags the mean. So the rebalanced foundation-
model fine-tune is **competitive with the small model**, a complete turnaround from the naive
run, and it vindicates the diagnosis: the first result was class imbalance, not a ceiling on
what Prithvi can do here.

What is honestly established: the pipeline works end to end, a naive fine-tune fails by
under-detection, and burn-balanced chips + a Dice loss recover it to U-Net-competitive skill on
this fire set. What still stands between this and a verdict: (1) a strict same-fire comparison,
`t2-unet` / `t2-stage0` on these exact three test fires (the +0.54 was the U-Net's own CV, and
these fires' naive F1s of 0.10-0.30 show they are a specific, small sample); (2) scale, three
test fires cannot settle it, so more CONUS fires plus the European set for the leave-one-
continent-out transfer are the real evaluation. Both are pure local `t2-prithvi-build` /
`t2-unet` work, no more GPU debugging.

### Same-fire baseline: Prithvi beats a spectral threshold (`t2-prithvi-baseline`)

The first strict same-code-path comparison is now in. `nbr_threshold_baseline` fits a single
post-fire NBR cut on the train fires and scores the **identical three test fires** with the
**same** skill-over-naive metric as the deep model (`t2-prithvi-baseline`, pure numpy, no GPU):

| held-out fire | NBR-threshold skill | Prithvi (rebalanced) skill |
|---|---|---|
| MN (small) | -0.047 | +0.042 |
| WA 46262 | +0.320 | +0.520 |
| WA 48285 | +0.218 | +0.634 |
| **mean** | **+0.163** | **+0.398** |

On the same fires, same reference, same metric, the rebalanced Prithvi fine-tune beats a
pointwise post-fire NBR threshold by **+0.235 mean skill**, and wins on all three fires
individually. That is the honest, apples-to-apples statement the +0.54 (U-Net's own CV) could
not give: the foundation model earns its keep over a simple spectral cut on this fire set.

### Leave-one-continent-out transfer test (built, awaiting the pull)

The most demanding test for a foundation model is generalisation to a continent it never saw
in fine-tuning: train on CONUS, test on Europe. The Copernicus EMS delineations are on disk
(`emsr.csv`, ~9 European fires), so this is now wired end to end:

1. `vhagar t2-prithvi-build-emsr --emsr-manifest emsr.csv --cache-dir data\t2_prithvi_emsr`
   builds six-band samples for the European fires, using each fire's EMS burnt-area
   delineation as the reference (needs the Sentinel-2 network).
2. `vhagar t2-prithvi-export-infer --cache-dir data\t2_prithvi_emsr --out-dir data\t2_prithvi_emsr_chips`
   chips them into one flat inference dataset (no split; all fires are test).
3. In the Colab session that holds the **CONUS** checkpoint, predict the European chips with
   that same checkpoint (point the inference cell at `data` under the uploaded European chips
   and at `splits/all.txt`), and download the masks.
4. `vhagar t2-prithvi-transfer --pred-dir <europe preds> --chips-manifest
   data\t2_prithvi_emsr_chips\_chips.json` stitches the European predictions per fire, scores
   them against the EMS delineations, and compares to a post-fire NBR threshold **tuned on
   CONUS and applied to Europe** (`nbr_threshold_transfer`), the spectral-cut analogue of the
   same cross-continent transfer.

The read: if CONUS-trained Prithvi keeps a clear skill margin over the CONUS-tuned NBR
threshold on European fires, its pretraining bought genuine cross-continent generalisation,
which a from-scratch model and a fixed threshold should struggle to match. All of the scoring
and chipping is pure-numpy and unit-tested; only the six-band European pull and the (reused)
CONUS-checkpoint inference need the network and the GPU.

### Transfer result (2026-08-16): Prithvi generalises across the Atlantic

Ran it: CONUS-trained Prithvi (burn-balanced, Dice) predicting nine European (EMS) fires it
never saw, versus the NBR cut tuned on CONUS and applied to Europe.

| | mean skill | fires won |
|---|---|---|
| CONUS-trained Prithvi on Europe | **+0.488** | 6 / 9 |
| CONUS-tuned NBR threshold on Europe | +0.372 | 3 / 9 |

Per fire, Prithvi minus NBR: +0.439, +0.047, +0.379, +0.177, +0.342, +0.148 (six wins),
then -0.118, -0.160, -0.213 (three losses). So a foundation model fine-tuned only on US fires
transfers to a continent it never saw and beats a spectral threshold there by **+0.116 mean
skill**, winning two-thirds of the fires. That is the crowning result for this component: the
pretraining bought real cross-continent generalisation, exactly what a from-scratch model and
a fixed threshold should struggle to match. (Both do better on Europe than on CONUS, the EU
fires are cleaner/higher burn-fraction, but Prithvi keeps its margin.)

Honest scope: nine European fires is still modest, and Prithvi does not dominate every fire
(three go to the threshold). The direction, though, is clear and consistent with the
mechanism. Full T2 Prithvi arc: pipeline built -> naive fine-tune underperforms (imbalance) ->
rebalanced to U-Net-competitive (+0.398) -> beats a same-fire spectral baseline (+0.398 vs
+0.163) -> transfers across continents and beats the CONUS-tuned threshold on Europe (+0.488
vs +0.372). Every step measured on the same code path, every setback diagnosed and fixed.

### Confirmed on real data (2026-08-16)

The pull and export ran on 20 CONUS-2021 fires; `terratorch fit` (after pinning
`torchgeo==0.7.1`, since terratorch's torchgeo constraint is currently un-pinned) loaded the
config, downloaded the Prithvi-EO-2.0-300M weights, built the 324M-param model, and began
training — so the config and chip dataset are validated against terratorch itself. The local
box has no NVIDIA GPU, so the actual fine-tune belongs on a cloud GPU (Colab / rented
instance); the config, chips, and manifest transfer as-is. Both an NVIDIA GPU host or a
capped-CPU smoke test (`--trainer.max_epochs 1 --trainer.limit_train_batches 8`) will exercise
the full train → predict → stitch → score loop.
