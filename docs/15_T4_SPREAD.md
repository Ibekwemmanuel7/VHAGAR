# T4: Fire spread forecasting

T4 answers "where does this fire go next?" The architecture (`docs/00` section 6)
is unusually blunt about the ceiling: next-day burned-mask **average precision
in the 0.35-0.45 band**, IoU roughly **0.6-0.8 on wind-driven fires** with
assimilation, and rate-of-spread MAPE around 40-60% in timber. The binding
constraints are label quality (VIIRS-derived perimeters score 0.71-0.93 F1
against agency perimeters, that is the ceiling), fuel-map error and wind
downscaling, **not** model architecture. *Any claim of much above 0.5 AP on
next-day spread, or above 0.9 IoU on real perimeters, is almost certainly a leaky
split or cumulative-rather-than-incremental burned area.* This module is built to
respect that.

## What is built

**Physics propagation core (`models/spread.py`).** Fire spread is a
front-tracking problem: the perimeter is the level set `T(x) = t` of an
arrival-time field satisfying the Eikonal equation `|grad T| = 1 / ROS(x)`. It is
solved with the **Fast Marching Method** (Sethian), a single monotone O(N log N)
sweep, exact for the upwind scheme, with no time stepping and no
self-intersecting polygons. `rate_of_spread` is a monotone fuel/wind/slope field
(a documented Rothermel/FBP surrogate; any calibrated ROS field drops in).
`spread_forecast` propagates the current front forward to a horizon;
`persistence_buffer` is the mandatory naive baseline.

**Honest validation harness (`eval/spread.py`).** A synthetic fire is grown to
truth with the solver, then given three effects the forecaster cannot see, which
are where real skill is lost: **hidden suppression** (a fuel break / crews hold a
flank), **spotting** (embers ignite ahead), and **fine-scale fuel heterogeneity**
below map resolution. The forecaster sees the perimeter at `t0` and a
**spatially-correlated wrong ROS** (fuel-map + wind-downscaling bias) and
propagates it. Scoring is on the **incremental** new-burn region only (cells not
already burned at `t0`), so cumulative area cannot inflate the number, with AP,
IoU, Dice, burned-area ratio and arrival-time MAE, stratified wind vs plume.

## Result (synthetic, honest)

`vhagar t4-spread`, thin next-step band (base rate ~0.11):

| model | AP | IoU | Dice | burned-area ratio |
|---|---|---|---|---|
| physics level-set | ~0.77 | ~0.57 | ~0.72 | ~1.4 |
| persistence + buffer | ~0.75 | ~0.45 | ~0.62 | ~0.65 |
| persistence | ~0.11 | 0 | 0 | 0 |

The honest reading:

- The physics forecast **beats both mandatory baselines**, and IoU (~0.57-0.60)
  sits at the edge of the cited wind-driven band.
- **Absolute AP (~0.77) is optimistic**, and we say so: the synthetic truth is a
  perturbed level set, close to the forecaster's own model class, so a level-set
  forecaster does unrealistically well. On real fires the ceiling is AP 0.35-0.45
  because model-form error, fine-scale chaos and suppression dominate, which a
  synthetic cannot fully reproduce. The value here is the propagation core and
  the incremental, baseline-anchored harness, not the absolute number.
- **Burned-area ratio > 1** is the honest tell: the forecast over-predicts
  because it cannot see the suppression that holds a flank, exactly the bias IoU
  hides and the architecture asks to report.

## State estimation and assimilation (the highest-return piece)

The architecture calls state estimation the highest return on investment in
spread: fuse the sparse, timed satellite active-fire detections into a
*continuous* arrival-time field and re-calibrate the per-fire rate of spread
online. `models/state_estimation.py` is the physics-anchored estimator: a prior
ROS (from mapped fuel/wind/slope) fixes the spatial *pattern* of spread; because
scaling ROS by `k` scales all arrival times by `1/k`, a single robust per-fire
`k = median(prior_arrival / observed_time)` aligns the *rate* to the detections.
That is exactly the "per-fire ROS adjustment factor calibrated online" the doc
asks for. `eval/assimilation.py` runs the loop: after each satellite pass it
re-calibrates to all detections so far and re-forecasts to the next pass.

`vhagar t4-assimilate` (wind regime, prior ROS biased 0.6x, 6 passes):

| | value |
|---|---|
| calibrated scale k | ~1.73 (ideal 1/0.6 = 1.67) |
| full-perimeter Sorensen (analysis) | ~0.79 (published conditional-GAN ~0.81) |
| incremental Sorensen, analysis | ~0.43 |
| incremental Sorensen, uncalibrated prior | ~0.06 |
| incremental Sorensen, naive persistence | 0.00 |

The reading: online calibration **recovers the ROS bias** from sparse timed
detections (k converges to ~1.7), the analysis **reconstructs the perimeter** at
Sorensen ~0.79, near the published conditional-GAN result, and on the between-pass
**new burn** it beats naive persistence (which predicts no growth) and the
uncalibrated prior (wrong rate) by a wide margin. The new-burn false-alarm ratio
is high (~0.5-0.75) and *rises* as the fire grows, the honest over-prediction from
unmodelled suppression and prior spatial error; reducing it is precisely what the
generative model below is for.

## Generative arrival-time inference (conditional GAN)

The published state of the art for state estimation is a conditional GAN that
infers the arrival-time field from active fire (Sorensen ~0.81, ignition-time
error ~32 min). `models/arrival_gan.py` is that model: a U-Net **generator**
conditioned on the observed perimeter, a normalised detection-time channel and
the mapped covariates; a PatchGAN **discriminator**; and losses that combine
LSGAN adversarial + L1 reconstruction + an **Eikonal-consistency** term
`| |grad T| - 1/ROS |` that ties the generated field to the level-set physics so
it cannot hallucinate a geometrically impossible front. It sits on top of the
physics-anchored estimator: `state_estimation.py` gives the calibrated prior, the
GAN learns the residual structure a single per-fire scale cannot. The conditioning
and normalisation builders are pure numpy and unit-tested (the data contract is
verified without torch); the generator, discriminator, Eikonal loss and training
loop are torch-guarded and run on a GPU box.

## What is next

- **Anisotropy** (done): wind-driven elliptical spread, `anisotropic_arrival` in
  `models/spread.py`. In direction `psi` from the head the rate is
  `ROS(psi) = head_ros * (1 - e) / (1 - e * cos psi)` (Richards' elliptical
  wavelet), eccentricity `e` set by the local wind via a length-to-breadth ratio.
  Arrival time is the least-cost path on the 8-neighbour grid (a simple discrete
  anisotropic solver; the rigorous continuous counterpart is the Ordered Upwind
  Method). `vhagar t4-aniso` verifies it: zero wind is a circle (LB ~ 1), and the
  measured length-to-breadth tracks the prescribed value as wind rises (e.g. ~2.1
  at wind 0.3, ~3.5 at 0.6), with the head far outrunning the back. Plug in a
  calibrated FBP / Alexander length-to-breadth for real fires.
- **State estimation and assimilation**: infer the arrival-time field from real
  timed detections (VIIRS/GOES first-detection, NIROPS airborne IR) and
  assimilate on every satellite pass; the strongest published result is a
  conditional GAN inferring arrival time (Sorensen ~0.81, ignition-time error
  ~32 min).
- **ML at the boundaries**: a diffusion / neural-operator surrogate on the
  simulator ensemble for 100-500x speedup, and a residual corrector (the U-Net in
  `models/ignition_conv.py` is the machinery) on simulated-vs-observed growth.
- **Real numbers**: score against NIROPS / VIIRS 12-hour perimeters, a networked,
  user-machine data step. Only then are absolute AP / IoU meaningful.
