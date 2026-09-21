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

## Calibration harness (`eval/wui_calibrate.py`)

The harness turns the prototype into a validated model. A `WuiFire` bundles one
fire's inputs (ROS field, ignition seed, wind, structure set) plus its DINS-derived
`truth_destroyed`. Given a list of fires:

```python
from vhagar.eval.wui_calibrate import calibrate_lofo, default_param_grid
report = calibrate_lofo(fires, default_param_grid())
print(report["mean_heldout_f1"], report["selected"])
```

`calibrate_lofo` does **leave-one-fire-out**: for each fire it grid-searches the
spotting/structure parameters on all *other* fires, then scores the held-out fire,
so `mean_heldout_f1` measures generalisation, not fit-to-self. `selected` is the
parameter set to deploy (searched over all fires). This is the same leakage-safe
discipline as T2/T3, applied to structure loss.

### Data you need (download locally; not bundled)

1. **CAL FIRE DINS (Damage Inspection Program).** Statewide structure-damage points
   with a `DAMAGE` class and lat/lon, published on the CAL FIRE / California open
   data portal (search "DINS Damage Inspection"). Filter to the fire(s) of interest.
   `eval.wui.load_dins` reads the CSV; "Destroyed (>50%)" is the binary positive.
2. **Building footprints** for the same area, to define the structure set the model
   places and destroys, either Microsoft Global ML Building Footprints or FEMA USA
   Structures (both open). Use the footprint centroids as the structures.
3. **A fire ignition/perimeter and weather** to seed the ROS field and wind, e.g.
   from your T1 detections or an agency perimeter, plus LANDFIRE fuels for ROS.

### Assembling a `WuiFire`

For each calibration fire: build the analysis grid, compute the ROS field (LANDFIRE
fuels + wind + slope via `models.spread.rate_of_spread`), set the ignition
`burned_seed`, snap footprint centroids to grid `(struct_rows, struct_cols)`, load
DINS and associate its destroyed points to your structures with
`eval.wui.match_points`, and set `truth_destroyed` on the aligned set. Then run
`calibrate_lofo`. Pick 3+ WUI fires with good DINS coverage (e.g. recent California
fires) so the leave-one-fire-out estimate is meaningful.

Report `mean_heldout_f1`, plus per-fold POD/FAR, as the honest headline. Until that
run exists, the model's parameters remain physically-plausible defaults, and any
rendered output must say "uncalibrated prototype."

## Scoped result on real DINS (structure-network only, 2026-09-15)

`scripts/dins_wui_scoped_calibration.py` runs a deliberately partial calibration on
the real DINS export: the **structure-to-structure conflagration only**, with no
wildland front, no wind, and the ignition core proxied by the destroyed structures
nearest the cluster centroid (scored on non-seed structures; large fires
stratified-subsampled). It is not a full physical calibration, it isolates the
structure-network component to see how much of the destroyed/survived pattern
proximity alone explains.

Result over 6 large fires (leave-one-fire-out, selected radius 120 m, base_p 0.8):
**mean held-out F1 ~0.23, and strongly bimodal by fire geometry:**

| fire | POD | FAR | F1 | character |
|---|---|---|---|---|
| Eaton | 0.91 | 0.38 | **0.735** | dense compact urban WUI |
| Palisades | 0.37 | 0.37 | 0.467 | dense hillside WUI |
| Tubbs | 0.05 | 0.08 | 0.087 | jumped Hwy 101 (discontinuous) |
| North Complex | 0.03 | 0.21 | 0.051 | large rural |
| Camp | 0.02 | 0.26 | 0.030 | linear ridge burn (Paradise) |
| Valley | 0.00 | 0.60 | 0.002 | large rural |

The honest reading: proximity-based structure-to-structure spread reproduces
**compact urban conflagrations** well (Eaton 0.74, POD 0.91) but fails on
**spatially extended or discontinuous** fires, where the destroyed footprint is set
by the wildland front, wind, and long-range spotting, not building adjacency. That is
the expected, informative negative, and it quantifies exactly what the faithful
calibration (per-fire ignition/perimeter + wind + fuels driving the arrival-time
front, plus the spotting layer) is needed to add. The scoped summary is written to
`dins_wui_scoped_summary.json`; the seed heuristic and subsampling make it a
diagnostic, not a headline metric.

## Faithful calibration (real ignition + wind): the harness

`scripts/dins_wui_faithful_calibration.py` runs the real version: it drives the WUI
model with each fire's **actual ignition point and wind**, so the wind-driven
arrival-time front (`wui_spread(anisotropic=True)`) decides which structures the fire
reaches, then spotting + structure-to-structure fill in the rest. It does a
leave-one-fire-out grid search over the spotting/structure params **and** a per-fire
front-extent (horizon) multiplier, and reports held-out POD/FAR/F1. The wind-driven
arrival field is invariant to the searched params, so it is solved once per fire and
reused (`arrival=` cache), which is what makes the sweep tractable on 600x600 grids.

### The few numbers you supply per fire

A JSON config (`dins_fires_config.example.json` is a template) with, per fire:

- `ignition_lat` / `ignition_lon` - the fire's origin. Source: the CAL FIRE incident
  record / IRWIN / IR perimeter first-detection, or the well-documented origin for
  major fires.
- `wind_speed_ms`, `wind_from_deg` - a representative wind during the main run.
  Source: the nearest **RAWS** station (mesowest / RAWS archive) for the fire's peak
  run hours, or ERA5/HRRR at the ignition cell.

Structures and destroyed/survived labels come from DINS by `* Incident Name`. Then:

```
python scripts/dins_wui_faithful_calibration.py \
    POSTFIRE_MASTER_DATA_SHARE_*.csv dins_fires_config.json \
    --out dins_wui_faithful_summary.json
```

Start with the compact fires where the signal is cleanest (Eaton, Palisades), then add
extended ones (Camp, Tubbs) once the compact set validates.

### Honest tuning notes

- The example config values are **placeholders**; replace them with verified ignition
  and RAWS wind before quoting any result.
- With uniform ROS the front is a homogeneous wind-driven ellipse. At high normalised
  wind the ellipse runs narrow, so if POD comes out low, widen the front by lowering
  the wind normalisation (`wind_ref_ms` in `fire_from_points`, default 15 m/s) or lean
  on the horizon multiplier / spotting to cover off-axis structures. LANDFIRE fuels and
  slope (not yet wired) would shape the front more realistically and are the next
  refinement.
- This is the path from "prototype" to a validated, physical POD/FAR/F1; until it is
  run with verified inputs, the WUI spread numbers remain the scoped diagnostic above.

### Faithful result on four fires, real ignition + wind

Ran with verified per-fire ignition and wind, Eaton (34.205, -118.088) and Palisades
(34.0725, -118.5425), Santa Ana ~15 m/s NE; Camp (39.810, -121.437, Jarbo Gap ~18 m/s
NE); Tubbs (38.628, -122.606, Diablo ~18 m/s NE), leave-one-fire-out:

| fire | POD | FAR | F1 | destroyed frac |
|---|---|---|---|---|
| Eaton | 1.00 | 0.48 | 0.68 | ~0.51 |
| Palisades | 0.43 | 0.47 | 0.47 | ~0.56 |
| Camp | 1.00 | 0.20 | 0.89 | ~0.80 |
| Tubbs | 1.00 | 0.06 | **0.97** | ~0.94 |
| **mean held-out** | | | **0.75** | |

Selected: length-to-breadth 1.6, horizon x6, structure radius 2 cells, base_p 0.5,
spotting distance 6 / intensity 3. This is a real end-to-end run on four major fires,
driven by real ignition and wind, scored leave-one-fire-out (params fit on the *other*
three fires, so 0.75 is generalisation, not fit-to-self). The model reproduces Camp
(Paradise) and Tubbs (which jumped Highway 101 into Coffey Park, F1 0.97) essentially
completely, so the wind-driven front plus spotting carries fire across the gaps that
broke the scoped structure-only run.

**It does not beat the trivial baseline, and we do not market it as a win.** The
DINS-inspected structures on these fires are 51 to 94 percent destroyed, so simply
labelling every inspected structure "destroyed" scores a mean leave-one-fire-out F1 of
about **0.81**, above the model's 0.75. The model ties that predict-all baseline on Camp
and Tubbs and loses on Palisades. So this result is base-rate dominated: it shows the WUI
layer runs end to end on real fires with real ignition and wind, not that it has skill
over a trivial baseline. A genuine win requires per-structure discrimination against the
surviving structures (POD/FAR conditioned on survivors) and a fitted vulnerability curve,
which is the stated next step. The 0.75 is reported as an end-to-end sanity number, never
as evidence of structure-level skill.

**Honest caveat on the metric.** F1 is highest where the destroyed fraction of the
inspected structures is highest (Tubbs 0.94, Camp 0.80): those fires levelled whole
communities, so few survivors remain for false alarms, and a front that reaches the
dense burn scores well. Eaton/Palisades have more surviving structures interleaved, so
their false-alarm rate (and thus lower F1) is the more demanding test, and FAR ~0.48
there is the real headroom. The main error everywhere is over-prediction, expected from
uniform ROS and no defensible-space vulnerability; LANDFIRE fuels and a fitted
vulnerability curve are what would bring the false-alarm rate down.

**Why the front length-to-breadth is calibrated (checked, not a bug).** An initial low
POD at the nominal high-wind length-to-breadth (4) looked like a solver problem, but a
direct measurement shows the solver is correct: the near-field front tracks the
prescribed length-to-breadth (measured 3.9 vs prescribed 4.0). The real reason a lower
value (~1.6) wins is physical, the observed destroyed footprint is *broader* than a
pure 4:1 downwind ellipse, because real WUI loss is less directional than the mean wind
(local wind variability, terrain channeling, and spotting spread fire off the wind
axis). So the calibration searching `lb_max` and selecting a broader front is a genuine
result, not a workaround: with a single mean wind, the best-fit front is wider than the
textbook high-wind ellipse. LANDFIRE fuels + slope and the spotting layer are what would
let a narrower, more physical front still reach the off-axis structures, and are the
next refinements (plus extending to Camp/Tubbs, where spotting across gaps matters more).

## Fuel-aware front (LANDFIRE FBFM40): wired, to cut the false-alarm rate

The over-prediction (FAR ~0.48 on the mixed Eaton/Palisades fires) comes from the
uniform ROS: the front spreads through roads, irrigated land, and water it could not
really cross. `models/fuels.py` maps LANDFIRE **FBFM40** (Scott & Burgan) fuel codes to
a relative spread factor; non-burnable codes (NB1 urban 91, NB3 agriculture 93, NB8
water 98, NB9 barren 99) become **0**, so a fuel-aware front stops at fuel breaks and
lets the structure-to-structure + spotting layers carry fire into the built area, the
mechanism expected to bring FAR down.

`fire_from_points(..., fuel_sampler=...)` takes a callable `(lon_grid, lat_grid) ->
FBFM40 code grid` and builds the ROS with `ros_from_fuel_codes` (non-burnable -> 0).
Wired and unit-tested; the real run needs a fuel clip:

1. **Get a LANDFIRE FBFM40 clip for each fire's bounding box** (small, not the national
   raster): landfire.gov data portal or the LFPS product-request API, layer "40 Scott
   and Burgan Fire Behavior Fuel Models" (e.g. LF 2022 `FBFM40`).
2. **Pass a rasterio sampler**, e.g.:

   ```python
   import numpy as np, rasterio
   from rasterio.warp import transform as warp_transform
   ds = rasterio.open("fbfm40_clip.tif")
   def sampler(lon_g, lat_g):
       xs, ys = warp_transform("EPSG:4326", ds.crs, lon_g.ravel(), lat_g.ravel())
       codes = np.array(list(ds.sample(zip(xs, ys)))).reshape(lon_g.shape + (-1,))
       return codes[..., 0]
   fire = fire_from_points(..., fuel_sampler=sampler)
   ```

Then re-run the faithful calibration. Why this is the right design: urban (NB1) is
non-burnable to the *wildland* front on purpose, intra-town spread is the
structure-to-structure model's job, so the front reaches the town edge and the graph
carries it in, keeping POD high while cutting the false alarms in the gaps. The code
path is in place and tested; this is the next real-data run.

### Fuel-aware result (honest negative, 2026-09-15)

Ran the four fires with the real LANDFIRE **LF2025 FBFM40 CONUS** raster driving a
fuel-aware ROS, plus edge-seeding (`struct_edge_cells`, structures within a few cells of
the burned wildland are seeded, since urban cells are non-burnable). It did **not** cut
the false-alarm rate; it made things worse: mean held-out F1 fell from **0.75 (uniform
ROS) to ~0.41**, with POD collapsing (Eaton 0.18, Camp 0.00, Palisades 0.15; only Tubbs
held up at 0.75). This is a real, informative negative, and worth stating plainly:

- Destroyed structures sit in **non-burnable urban (NB1)** cells, so the wildland front
  cannot enter the town; it stops at the edge. Edge-seeding helps a little but not enough.
- Worse, real fires reach and cross these areas by **long-range spotting over non-fuel
  gaps** (river canyons, roads), Camp jumped the Feather River canyon; Tubbs jumped
  Highway 101. A fuel-blocked front with the current short-range spotting cannot make
  those jumps, so it never reaches the dense town interior where the loss occurred.
- The uniform-ROS front scored 0.75 precisely because, by ignoring fuel, it spread freely
  into the built area, physically wrong, but it happened to cover the destroyed structures.

So the fuels experiment did not improve the metric; it exposed that the model's mechanism
for getting fire **into and across the built and non-fuel environment** (edge-seeding +
short-range spotting) is underpowered once the front is fuel-constrained. The uniform-ROS
run remains the reference configuration at **F1 0.75**, but that number is *below* the
predict-all-destroyed baseline of ~0.81 (see the results section above): it is an
end-to-end sanity figure that over-predicts because it ignores fuel, not a demonstrated
win over the trivial baseline. The real next step is not "add fuels" but a
stronger built-environment spread model: long-range ember spotting calibrated to jump
non-fuel gaps, and treating the dense structure network as its own spread medium seeded
broadly at the wildland-urban interface. The fuel-aware code path (windowed FBFM40
sampler, `struct_edge_cells`) is committed and tested; it is the substrate for that work,
not a finished improvement.
