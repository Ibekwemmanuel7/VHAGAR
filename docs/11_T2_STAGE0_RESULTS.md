# T2 Stage-0 results: independent RBR vs MTBS, CONUS 2021

An independent Sentinel-2 RBR predictor, calibrated leave-one-fire-out and
evaluated against MTBS thematic severity on 2021 CONUS wildfires. What follows
records how a good-looking first number was found, on scrutiny, to have no skill
over a trivial baseline, and where the predictor does show real skill.

## Headline (retracted as an accuracy claim, kept as a lesson)

The original headline was **F1 0.865 ± 0.056** (5 large fires), later 0.900 over 34
fires, presented as "the first defensible accuracy number." It is not defensible,
and the correction is the real result:

**On these per-fire windows the trivial "predict everything burned" baseline scores
F1 0.896 (large fires) to 0.911 (all 34), and the calibrated RBR threshold does not
beat it on a single fold (0 of 34). Skill over naive is -0.01.** The windows are
~90% burned, so a high F1 measures the window's class balance, not the predictor.
This is exactly what the permanent no-skill baseline exists to catch, and it was
caught only when the baseline was actually computed. Every fold now reports its
naive F1 and skill margin so this cannot recur.

**Where the predictor does show skill: the balanced cross-continent test.** On the
Greek EMS windows (32% burned, so naive all-burned F1 is only 0.485), the RBR
threshold with a balanced objective scores **0.573, a real +0.088 skill margin**
over naive. That, not the within-CONUS 0.9, is the defensible signal that the
Sentinel-2 RBR predictor carries independent burned-area information. It is modest
and rests on two fires, but it is skill rather than a window artefact.

Consequence: the within-CONUS leave-one-fire-out numbers below are reported with
their naive baseline and should be read as "no demonstrated skill on burn-heavy
windows," pending the wide-window re-pull that gives the test real unburned context.

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

## Skill over the no-skill baseline (the decisive check)

Read every F1 above against the trivial predict-all-burned baseline. The eval now
computes it per fold (`FoldResult.naive_f1`, `.skill_f1`) and the CLI prints it.

| set | model F1 | naive all-burned F1 | skill | folds beating naive |
|---|---|---|---|---|
| large fires (headline 5, here 13 usable) | 0.896 | 0.896 | -0.000 | 0 / 13 |
| all 34 CONUS fires | 0.900 | 0.911 | -0.010 | 0 / 34 |
| continent-out EU, F1 objective | 0.488 | 0.485 | +0.003 | - |
| **continent-out EU, Youden objective** | **0.573** | **0.485** | **+0.088** | - |

The within-CONUS model never beats predict-all-burned: the windows are ~90% burned,
so F1 is dominated by prevalence and the predictor adds nothing measurable. Only on
the balanced EU windows, and only with a balanced objective, does the RBR threshold
clear the naive baseline (+0.088). Everything else in this document should be read
through that lens: the transfer discussion, the Olofsson areas, and the threshold
spread are all still informative about mechanism, but the within-CONUS F1 is not an
accuracy claim.

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

## Scaled to 34 size-stratified fires (a caution, not a triumph)

Running 34 CONUS 2021 fires sampled across the size distribution gives global
F1 **0.900 ± 0.083** (IoU 0.827 ± 0.122), higher than the 5-fire 0.865. But this
is partly an artifact: the size-stratified set is mostly small fires whose
analysis windows are 80-96% burned, and a nearly-all-burned window is *easy* on
per-pixel F1. The tell is that the Olofsson adjusted area is computable on only
**2 of 34 folds**, the rest are single-class (no unburned pixels in-window to
stratify against). Lesson about the evaluation design: tight per-fire windows
make small fires trivially easy and their area unmeasurable. The comparable, hard
numbers are the large-fire and continent-out results, not the inflated scaled
mean. A wider window (more unburned context) would make small fires informative
and restore the area estimate; that is the next refinement.

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

## Climate stratification: does matching Koppen zones help transfer?

The architecture's thesis is that thresholds should be calibrated per climate or
fuel regime, not globally. The two Greek fires are Koppen class 8 (Csa, hot-summer
Mediterranean), sampled from a 1 km Koppen-Geiger raster (Beck et al., 1991-2020
present-day climatology) at each fire centroid. The clean test: on a **fixed
training pool** of all 34 usable US MTBS fires, calibrate the Greek fires only on
US fires that share their Csa stratum, and compare against one global threshold.
At first glance stratifying looks like a win:

| method (34-fire US pool, test 2 EU fires) | threshold | continent-out F1 | IoU |
|---|---|---|---|
| global, F1-tuned (all 34 pooled) | -1706 | 0.488 | 0.323 |
| per-stratum, raw Koppen class | 99.1 | 0.609 | 0.438 |
| per-stratum, Mediterranean group (Cs*) | -1.05 | 0.559 | 0.388 |

But two follow-up checks show the +0.12 is **mostly an artifact of a broken
objective, not evidence that climate matching works.** Reporting that honestly is
the point of the permanent-baselines rule.

**Check 1: within-CONUS, stratifying hurts.** Run the same per-stratum idea as a
leave-one-fire-out over the 34 US fires (each fire's threshold calibrated on its
same-Koppen training fires vs one global threshold). Per-stratum is worse, not
better: mean F1 **0.827 vs 0.909** global, with several catastrophic folds (an
Idaho BSk fire drops 0.982 to 0.116). Shrinking the calibration set to one
stratum overfits that stratum's idiosyncratic RBR scale. If stratification were a
real mechanism it would help here too; it does not.

**Check 2: the global -1706 is a degenerate-window artifact.** 26 of the 34
in-window US fires are >80% burned, so the pooled distribution is 82% burned.
Maximising **F1** on that pool rewards predicting everything burned: the tuned
threshold runs off to -1706 (predicts 99% of pixels burned) and still scores
F1 0.901 on the burn-heavy US pool. That same "predict all burned" threshold is
catastrophic on the balanced Greek windows (truly 32% burned), hence 0.488. The
Csa per-stratum threshold of 99 helps only because the one Csa fire (AZ32635,
55% burned) has a balanced window, so its threshold is not collapsed. It is a
non-degenerate threshold by luck of window balance, not by climate content.

**Check 3: fixing the objective recovers the gain without any climate matching.**
Re-tune the single global threshold with a balanced objective (Youden's J,
TPR - FPR) instead of F1. It picks 205.8 (predicts 62% burned) and lifts EU
transfer to **F1 0.573** with zero stratification. Koppen per-stratum (0.609) then
sits only 0.036 above a properly-tuned global baseline, and that residual still
rests on a single fire.

| global threshold objective | threshold | EU transfer F1 |
|---|---|---|
| F1 (rewards predict-all-burned on 82%-burned pool) | -1706 | 0.488 |
| Youden's J (balanced) | 205.8 | 0.573 |
| Koppen per-stratum (for reference) | 99.1 | 0.609 |

Conclusion: the dominant confound is the tight per-fire window, which makes small
fires ~90% burned and lets an F1-tuned threshold collapse to "everything burned."
That inflates within-CONUS F1 (US test windows are also mostly burned) and destroys
EU transfer. Climate stratification did not beat a properly-tuned global baseline;
it mostly rediscovered a non-degenerate threshold. The real fix is wider analysis
windows with genuine unburned context, which needs an imagery re-pull; the balanced
objective is a cheap partial fix already in hand. Do not compare any of these to
the earlier 0.582 (a different, 6-fire pool with no Csa fire, so per-stratum there
was inert); only the within-pool numbers here walk the same code path.

## The objective x window interaction, decomposed

The window fix and the objective fix are two separate levers, so measure them
separately. Holding the current narrow window fixed and varying only the objective
(same code path, same cache) shows the objective's effect is **opposite on the two
axes**, which is itself the clearest evidence that the window, not the objective, is
the root problem:

| axis (narrow window, global threshold) | F1 objective | Youden objective |
|---|---|---|
| within-CONUS leave-one-fire-out (34 fires) | 0.900 (std 0.083) | 0.588 (std 0.333) |
| continent-out (US to EU) | 0.488 | 0.573 |

Within CONUS the test windows are themselves ~90% burned, so an F1 threshold that
predicts everything burned matches them and scores 0.900; the balanced threshold is
more selective, under-predicts on those mostly-burned windows, and falls to 0.588.
Across continents the EU windows are 32% burned, so the same predict-all-burned
threshold is catastrophic (0.488) while the balanced one transfers (0.573). Neither
objective is "correct" on narrow windows; the F1 number in particular is inflated by
the shared ~90%-burned balance of train and test windows.

Two distinct effects, not one. The high within-CONUS F1 (0.900) is a window artefact
and should deflate once test windows carry unburned context. But the balanced
objective's large fold-to-fold spread (std 0.33) is a **different** effect: it does
not track window balance (correlation between a fold's burned fraction and its Youden
F1 is only +0.06, and the single most-balanced fold scores among the lowest). That
spread comes from per-fire RBR scale heterogeneity: one pooled Youden cut (~205)
fits some fires' burn severity and misses others, which is the genuine fuel-and-regime
variation the architecture's per-ecoregion breakpoints are meant to absorb.

Prediction to test after the wide-window re-pull (`w15` cache): the F1-objective
within-CONUS number should drop toward the balanced one as test windows stop being
~90% burned, so the two objectives converge mainly by the inflated F1 number coming
down. The balanced objective's per-fire spread may **not** shrink, because its driver
is RBR-scale heterogeneity rather than window balance; if it persists, that is
evidence for per-stratum thresholds done properly (enough fires per stratum), not
against them. Distinguishing "window artefact" from "real per-fire scale" is the
point of the measurement.

## Reproduce

Samples are cached under `data/t2_cache/`, so a re-run is instant:

```
vhagar t2-stage0 --registry data/labels/registry.parquet \
  --mosaic mtbs_extract/mtbs_CONUS_2021.tif \
  --min-area-ha 10000 --max-fires 6 --max-scenes 4 --res-m 100
```
