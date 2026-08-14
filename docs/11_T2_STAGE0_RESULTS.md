# T2 Stage-0 results: independent RBR vs MTBS, CONUS 2021

First defensible accuracy number for the project. An independent Sentinel-2 RBR
predictor, calibrated leave-one-fire-out and evaluated against the MTBS thematic
severity, on the largest 2021 CONUS wildfires.

## Headline

**F1 0.865 ± 0.056, IoU 0.765 ± 0.084** across 5 leave-one-fire-out folds.

This is an accuracy claim, not a self-comparison: the predictor (Sentinel-2 RBR)
and the reference (MTBS) share no lineage. It sits squarely in the burned-area
literature's honest range (~0.7 to 0.85), and it is not a suspicious ~0.99, which
would signal leakage.

## Per-fold table

Run: `vhagar t2-stage0 --min-area-ha 10000 --max-fires 6 --max-scenes 4 --res-m 100`,
analysis at 100 m, up to 4 least-cloudy Sentinel-2 scenes per pre/post window.

| held-out fire | threshold | F1 | IoU | mapped ha | adjusted ha | 95% CI |
|---|---|---|---|---|---|---|
| AZ33212… | 25.2  | 0.779 | 0.638 |  1,801 |  1,085 | ±201  |
| CA38586… | 25.2  | 0.924 | 0.858 |    115 |     98 | ±10   |
| CA39876… (Dixie) | 111.3 | 0.891 | 0.803 | 14,761 | 17,112 | ±687  |
| CA41142… | 25.2  | 0.885 | 0.794 |  2,198 |  2,138 | ±164  |
| OR42616… (Bootleg) | -10.9 | 0.845 | 0.731 | 12,286 |  9,694 | ±1,147 |

## What the numbers say

- **Threshold transfer is the real limitation, and it is exposed honestly.** The
  calibrated RBR cutoff ranges from -10.9 to +111 across fires. A single global
  threshold does not transfer across fuel types and fire regimes; leave-one-fire-out
  makes that visible rather than hiding it behind a pooled split. This is the
  motivation for the architecture's per-ecoregion breakpoints.
- **The Olofsson adjustment is working, bidirectionally.** Bootleg: mapped 12,286
  ha adjusted down to 9,694 ± 1,147 (the threshold over-predicted). Dixie: mapped
  14,761 adjusted up to 17,112 ± 687 (under-caught). An estimator that only ever
  moved one direction would be suspect; this one corrects both ways with sensible
  intervals.
- **Coverage is disclosed.** One of the six fires (CA40752…) returned zero valid
  pixels, a fully cloud-covered post window, and was dropped rather than faked.
  Cloud-thinned windows also explain the small valid-pixel counts and wider CIs on
  a couple of fires.

## Honest caveats

- Five fires, all large (>10,000 ha), western CONUS, one year. Not a
  distribution-representative sample; it is a first number, not the final one.
- 100 m analysis resolution and 4 scenes per window were chosen for speed. Both
  can be tightened (finer resolution, more scenes) for a sharper number at more
  compute; the samples are cached, so widening does not re-pull existing fires.
- A single global RBR threshold is the Stage-0 baseline by design. Per-ecoregion
  calibration and a plain U-Net companion are the next comparisons the protocol
  asks for.
- MTBS thematic is the reference; the independent, cross-continent number is the
  leave-one-continent-out test against Copernicus EMS, still to come.

## Correction: pixel area

The area columns above were computed with a hardcoded 0.09 ha/pixel (30 m), but
the run was at 100 m (1 ha/pixel), so **the ha figures are understated by ~11x**;
multiply by 1/0.09 for the corrected values (Dixie mapped ~164,000 ha, not
14,761). F1 and IoU are per-pixel ratios and are **unaffected**. The CLI now
derives pixel area from `--res-m`, so a fresh run (instant, cached) prints
correct areas.

## Leave-one-continent-out (the headline generalisation number)

Train the RBR threshold on the US MTBS fires, test on European Copernicus EMS
fires (EMSR527, Evia and Attika, Greece, August 2021). The threshold never sees
Europe. Run: `vhagar t2-continent-out --emsr-manifest emsr.csv`.

| test | US fires | EU fires | threshold | F1 | IoU | adjusted ha | 95% CI |
|---|---|---|---|---|---|---|---|
| EU EMS | 5 | 2 | 25.2 | **0.582** | **0.411** | 33,452 | ±8,449 |

**Within-CONUS F1 0.87 drops to 0.58 across continents.** That ~0.28 gap is the
result: a burn-severity threshold calibrated on Californian conifer and chaparral
fuels is mis-set for Greek Mediterranean pine and scrub, so it transfers poorly.
The predictor still carries signal (RBR separates burned from unburned in Greece);
it is the fixed cutoff that does not transfer. This is exactly the fuel-and-regime
dependence the architecture warns about, measured honestly rather than hidden by a
pooled split. A result near 0.87 here would have signalled leakage; the drop is
the credible outcome.

Diagnostics were clean: both delineations rasterised sensibly (Evia 74,524 valid
px at 33% burned in-window, Attika 23,728 at 29%), and the Olofsson estimate has a
real CI. Caveat: only two European fires; more EMS activations would tighten it,
and per-ecoregion thresholds are the obvious next improvement.

## Baseline comparison: calibrated global vs adaptive Otsu

The continent-out gap suggested an adaptive, per-fire threshold (Otsu) might
transfer better than a single calibrated cutoff. Measured on the cached samples,
it does not. This is a real negative result, and reporting it is the point of
the permanent-baselines rule.

| method | CONUS leave-one-fire-out F1 | continent-out F1 |
|---|---|---|
| **global (calibrated)** | 0.865 ± 0.056 | **0.582** |
| global + per-fire robust standardization | **0.876 ± 0.055** | 0.535 |
| otsu (adaptive, per fire) | 0.713 ± 0.075 | 0.552 |

The calibrated raw-RBR threshold wins for **transfer** (continent-out), which is
what matters. Per-fire robust standardization (recenter by median, scale by MAD)
nudges the within-CONUS number up but *hurts* cross-continent transfer: aligning
each fire's RBR scale does not fix the US-to-EU domain shift, so the difference is
genuine (fuel and regime), not a scaling offset. Otsu is worst. Otsu (even outlier-robust, with a
percentile-clipped histogram) picks a per-fire cut around RBR 245 for the Greek
fires, well above the transferable global cut of 25, and under-detects. RBR's
heavy tails and the window-scale class balance make its per-fire distribution
only weakly bimodal, so the mode-splitting assumption underperforms. The
takeaway: for RBR burned area, a calibrated global threshold is the stronger
Stage-0 baseline, and adaptive thresholding is not a free transfer win. Run
either with `--method global|otsu`.

## Reproduce

Samples are cached under `data/t2_cache/`, so a re-run is instant:

```
vhagar t2-stage0 --registry data/labels/registry.parquet \
  --mosaic mtbs_extract/mtbs_CONUS_2021.tif \
  --min-area-ha 10000 --max-fires 6 --max-scenes 4 --res-m 100
```
