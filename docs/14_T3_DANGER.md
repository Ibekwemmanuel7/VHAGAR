# T3: Fire danger and ignition probability

T3 answers "where is ignition likely, and how will fire behave if it starts?"
The architecture (`docs/00` section 5) is emphatic that this is three separate
quantities that must never be collapsed into one "risk" number:

- **Fire danger** (FWI, ERC, BI): a conditional intensity, how fire behaves *if*
  it starts. Carries no ignition information.
- **Ignition probability**: P(>=1 ignition | cell, day), a rare-event binary.
- **Expected burned area**: E[BA] = P(ignition) x E[BA | ignition].

## What is built

**Layer 1, deterministic indices.** `features/fwi.py` implements the Canadian
FWI system end to end (FFMC, DMC, DC, ISI, BUI, FWI); `vhagar fwi` runs it. This
is the conditional-danger layer and it is complete for FWI. NFDRS2016
(ERC/BI/SC/IC from gridMET) is not yet wired.

**Layer 2, ignition probability, the state-of-the-art shape.** The evidence in
the architecture is that gradient boosting, not a deep net, is the state of the
art here: ECMWF's operational Probability-of-Fire uses XGBoost on ~19 predictors
and found input quality (fuel state above all) mattered more than architecture.
So VHAGAR fits a gradient-boosted classifier, **per cause** (human, lightning,
whose covariate structures are nearly orthogonal), and puts the effort into an
honest sampling design and honest scoring.

- `datasets/danger.py`: the sampling design.
- `eval/danger.py`: cause-stratified, spatial-block, properly-scored evaluation.
- `vhagar t3-ignition`: runs it, with a synthetic demonstration of the trap.

## The trap that ruins most ignition models

Ignition databases are presence-only with strong reporting bias: a fire is more
likely to be *recorded* where there are people and roads, which are the very
covariates the model wants to use. A naive model with randomly drawn
pseudo-absences learns *where people report fires*, not where fires start.

The four mitigations, each a pure, unit-tested function in `datasets/danger.py`:

1. **Target-group background sampling** (`target_group_background`): draw
   pseudo-absences from the same observation pool that produced the presences,
   so detection bias is shared and cancels.
2. **Negative stratification** (`stratify_negatives`): match the background
   land-cover distribution to the positives, removing a land-cover prior that is
   an artefact of the sampling.
3. **Rare-event prior correction** (`rare_event_correction`): the King and Zeng
   (2001) intercept shift maps probabilities from the down-sampled design's base
   rate back to the true one, so calibration means what it says operationally.
4. **Cause stratification**: human and lightning ignitions are carried as a
   label and modelled separately.

## The demonstration (synthetic, honest)

`synthetic_reporting_scenario` builds a world where true ignition depends only on
weather and fuel (noisily), while *reporting* depends strongly on the human
footprint. Running `vhagar t3-ignition --synthetic` on it, under spatial-block
CV with King-Zeng correction:

| sampling | AUPRC | reliability | BSS vs climo | human-footprint importance |
|---|---|---|---|---|
| naive random background | ~0.33 | ~0.015 | negative | **+0.043** (people is a top feature) |
| target-group background | ~0.29 | ~0.014 | negative | **+0.011** (collapsed) |

The reading: naive background **inflates** apparent skill and leans on the
human-footprint covariate, an observation artefact; target-group background
drawn from the same reporting process collapses that reliance and reveals the
honest, lower skill. That is the architecture's exact warning, shown on data
rather than asserted. (BSS is negative because the synthetic signal is
deliberately weak and noisy; the point of the demo is the *sampling effect*, not
a headline skill number.)

Scoring is proper by construction, reusing `eval/metrics.py`: AUPRC, Brier with
its Murphy decomposition (reliability / resolution / uncertainty), ECE with
quantile bins, log loss, and Brier skill score against a base-rate climatology.
F1 and CSI are improper and are not used for model selection. `lon`/`lat` are
excluded from the model (the T1 leakage lesson); they only build the 5-degree
spatial blocks.

## Expected burned area, the third quantity

`E[BA] = P(ignition) x E[BA | ignition]`, kept separate from ignition and from
danger. The conditional distribution of burned area given an ignition is
heavy-tailed: a handful of wind-driven events dominate the total, so squared
error and RMSE are captured by those extremes. `eval/burned_area.py` follows the
architecture's prescription:

- **Fit the distribution, not the mean** (`BurnedAreaModel`): log-space quantile
  gradient boosting gives a full predictive quantile set per cell.
- **A GPD tail** for the extremes: a peaks-over-threshold generalised-Pareto fit
  extrapolates the far quantiles (q99+) the boosting cannot see enough of.
- **Score with CRPS and pinball, never RMSE.** CRPS via its quantile
  decomposition (`crps_from_quantiles` in `eval/metrics.py`) is proper and
  stable; RMSE is reported only to *show* its instability.

`vhagar t3-expected-ba --synthetic` on a heavy-tailed world (median ~200 ha, p99
~1.7k ha, max ~176k ha):

| metric | model | climatology |
|---|---|---|
| CRPS | ~235 | ~260 (skill **+0.09**) |
| RMSE across folds | mean ~2400, **std ~1100** | |

The reading: the covariates carry real signal (CRPS beats a climatology of
constant training quantiles by ~9%), the GPD tail adds a little at the extreme
quantiles, and RMSE's fold-to-fold standard deviation is a large fraction of its
mean, tail-driven noise, which is exactly why the architecture forbids selecting
on it. `expected_burned_area` combines a per-cell `P(ignition)` with the
predictive mean (trapezoidal over the quantiles) to give `E[BA]`. Real data is
`t3-expected-ba --no-synthetic --fires per_fire.parquet --features ...`, one row
per fire with `area_ha`, `lon`, `lat`, `year`, and the covariates.

## Layer 3: the deep challenger, in shadow mode

A spatial deep model is admitted only as a *challenger* to the gradient boosting,
and promoted only on evidence. Because danger is gridded (cell x day),
verification is spatial: pixel-exact scoring punishes a forecast that is right
about *where* fire is likely but off by a cell, so the primary metric is the
**Fractions Skill Score** (`fractions_skill_score` in `eval/metrics.py`), which
compares neighborhood fractions of the observation and forecast and is reported
at 40 / 80 / 120 km.

`eval/danger_grid.py` is the harness. It builds a gridded ignition world where
ignition is driven by *clean* spatially coherent weather and fuel but the model
sees only *noisy per-cell observations*, fits a **pointwise** gradient-boosting
baseline and a **spatial** challenger (the same booster on neighborhood-pooled
features, a runnable stand-in for the ConvLSTM) under leave-time-block-out CV,
and applies the promotion gate: the challenger is promoted only if it beats the
baseline on base-rate-preserving **AUPRC and Brier**. `models/ignition_conv.py`
is the real deep challenger, a compact U-Net trained with a differentiable
**soft-FSS loss** plus BCE (torch-guarded, runs on a GPU box); the harness scores
whatever probability field a model produces.

`vhagar t3-challenger` on the synthetic world (36 days, 40x40 cells at 20 km,
base rate ~0.05, obs noise 0.15), stable across seeds:

| model | AUPRC | Brier | FSS 40 | FSS 80 | FSS 120 |
|---|---|---|---|---|---|
| pointwise baseline | ~0.12 | ~0.046 | ~0.58 | ~0.75 | ~0.82 |
| spatial challenger | ~0.16 | ~0.045 | ~0.64 | ~0.81 | ~0.88 |

Here the spatial challenger earns promotion: it beats the baseline on AUPRC,
Brier, and FSS at every scale, because neighborhood pooling denoises the
observations. That is the gate working, not a foregone conclusion, when spatial
context adds nothing the same gate keeps the challenger in shadow. The
architecture's expectation stands: on real daily ignition the deep model should
earn its place rarely, and more so at seasonal lead times; the value here is the
FSS-based spatial verification and the AUPRC-and-Brier promotion gate, so a deep
model is only ever promoted on blocked, proper-scored evidence.

## Honest scope and what is next

This increment builds and unit-tests the *method*: the sampling design, the
cause-stratified gradient-boosted model, the proper-scoring blocked evaluation,
and the real-data ingest loader, demonstrated on a synthetic biased world.

### Running on real data

`frames_to_records` turns two tables into the model's inputs, and
`t3-ignition --no-synthetic` runs the pipeline on them:

```
vhagar t3-ignition --no-synthetic \
  --occurrence occurrence.parquet \    # one row per reported ignition cell-day
  --candidates candidates.parquet \    # target-group pool of observable cell-days
  --features vpd,fm100,fm1000,fwi,erc,fuel_load,elevation,dist_road,wui \
  --tau 0.0                            # 0 = infer base rate from presences/candidates
```

Both tables carry the covariate columns plus `lon`, `lat`, `year` (and optional
`stratum`); `occurrence` also carries `cause`. If `candidates` has no `weight`
column, `reporting_weight` fills one from occurrence density, giving target-group
sampling for free. `lon`/`lat` are used only to build the 5-degree spatial
blocks, never as features.

What is still a networked, user-machine step is *assembling those two tables*:
the fire-occurrence database (FPA-FOD / CWFIS) and the covariate stack, FWI/NFDRS
components, VPD from Tmax/RHmin (never daily means), fm100/fm1000, SPEI/EDDI,
SMAP root-zone soil moisture, live fuel moisture, fuel type, WorldPop/GHSL,
distance to road/rail/transmission, WUI class, and lightning density with a
0-14 day holdover lag. The remaining modelling work, in the architecture's order:
- NFDRS2016 alongside FWI1987/FWI2025, served as percentile rank and
  anomaly-vs-climatology.
- Ensemble-propagated probabilistic danger (P(FWI > class threshold)) from
  HRRR/ECMWF/AIFS forcing, bias-corrected before the fuel-moisture codes.
- The Layer-3 deep challenger (ConvLSTM / U-Net-3+ with Fractions Skill Score
  loss) in shadow mode, promoted only if it beats the gradient boosting on
  blocked AUPRC *and* Brier.
- Calibration as a release gate (reliability, Brier, ECE), which the scoring
  here already computes.
