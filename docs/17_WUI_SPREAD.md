# T4 WUI extension — ember spotting + structure-to-structure spread

Status: **prototype, physically motivated, not yet calibrated to observed loss.** The
modules are built and unit-tested; the DINS validation harness is wired but the
model parameters have not been fit to real structure loss. Read every number
below as "reasonable prototype," not "validated."

## Why this exists

VHAGAR's T4 core (`models/spread.py`) tracks a wildland fire front as the level set
of an Eikonal arrival-time field. Two behaviours that dominate loss in the
**wildland-urban interface (WUI)** are invisible to a pure wildland front:

1. **Ember spotting.** Firebrands lofted from the flaming front land downwind and
   start fires *ahead* of the perimeter, frequently the mechanism that jumps a road
   or firebreak into a community.
2. **Structure-to-structure spread.** Once a building ignites, radiant/convective
   heat and its own embers ignite neighbours, a conflagration that travels through
   the built environment, not the vegetation.

Both are layered on top of the existing arrival-time solver rather than replacing
it, keeping the "physics core, ML at the boundaries" discipline: the transport and
propagation are explicit physical surrogates; ML belongs only in fitting their
parameters to observed loss.

## The logic, tier by tier

**Exposure** (`rasterize_structures`). Building footprints (Microsoft Global ML
Building Footprints, FEMA USA Structures) are snapped to the analysis grid as a
per-cell weight (count, or replacement value for the T5 loss tier). This is the
surface everything else reads.

**Spotting** (`spotting_kernel`, `spot_ignition_probability`). Firebrands emitted
by the current front are convolved with a **wind-oriented landing kernel**:
distance decay `exp(-d/scale)` with `scale` growing with wind (stronger wind throws
brands farther), times a downwind directional weight `clip(cos θ, 0, 1)^focus`
blended toward isotropic as wind falls. The kernel conserves brands (sums to 1,
self-cell zero). Landings are thinned by receptivity (fuel or structure presence)
through a Poisson law `p = 1 - exp(-landings · receptivity)`, giving a per-cell
spot-ignition probability. Any calibrated spotting-distance distribution (Albini,
Sardoy) drops straight into `scale`/`focus`.

**Structure-to-structure** (`structure_to_structure_spread`). Structures within a
radius of a burning structure can ignite; the pairwise probability decays with
separation and is boosted downwind, scaled by `base_p`. Each structure accumulates
an independent-OR probability from all burning neighbours and ignites past a
threshold, iterated to a fixed point on the structure graph (KD-tree neighbours).
It is **deterministic** (returns the fixed point, not a random draw) so the output
is directly scorable; for stochastic realisations, sample against the per-round
probabilities with an explicit RNG.

**Orchestration** (`wui_spread`). Runs the wildland arrival-time solver, seeds the
structures the front reaches by the horizon, adds spot ignitions where embers land
on structures ahead of the front, then propagates structure-to-structure. Returns
the wildland `arrival` field and per-structure `reached` / `spot_ignited` /
`destroyed` masks.

## Validation plan (the honest part)

The ground truth for structure loss is **CAL FIRE DINS** (Damage Inspection): every
structure in a significant California fire gets a damage class (No Damage → Affected
→ Minor → Major → Destroyed) with a point location. `eval/wui.py` provides:

- `load_dins(path)` — reads a DINS export to `(lats, lons, destroyed)`, with
  "Destroyed (>50%)" as the binary positive by default.
- `match_points(...)` — associates modelled structures to DINS points (nearest
  within tolerance) so prediction and truth refer to the same buildings.
- `score_structures(pred, truth)` — POD, FAR, precision, recall, F1 over the aligned
  structure set, the same detection-style scoring the other tiers use.

The intended experiment: pick fires with WUI loss and DINS coverage (e.g. recent
California fires), run `wui_spread` seeded from the observed ignition/perimeter,
match to DINS, and report POD/FAR/F1 on destroyed structures, then fit `base_p`,
the spotting `scale`/`focus`, and the ignition threshold to maximise held-out F1
(leave-one-fire-out, same leakage discipline as T2/T3). Until that fit is done and
reported, the parameters are physically-plausible defaults, not calibrated values,
and any figure produced must say so.

## What this is and isn't

It **is**: a working WUI layer on the arrival-time core, exposure + spotting +
structure-graph conflagration, with a real validation target and honest,
scorable outputs.

It **isn't**: calibrated, or a match for a mature operational WUI model. There is
no structure-construction/defensible-space vulnerability yet (only geometry and
wind), no fitted spotting distribution, and no DINS-fit parameters. Those are the
next steps, and they're exactly the "shape the next generation of WUI spread"
mandate that motivated building this.
