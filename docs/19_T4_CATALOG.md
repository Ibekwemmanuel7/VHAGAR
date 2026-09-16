# T4 batch event-catalog runner (catastrophe-style aggregation)

Catastrophe (CAT) modelling does not price on one fire; it runs a *catalogue* of many
events under varied drivers and aggregates them into an exceedance-probability (EP)
curve. This is the batch harness that does that on VHAGAR's own T4 spread engine, the
piece that turns a single-fire forecast into a portfolio-level loss distribution.

## Chain

    ignition catalogue (varied fuel + wind)
        -> per-event anisotropic arrival-time spread   (T4 physics core, models.spread)
        -> per-event burned area + ground-up loss       (T5 loss chain, models.loss)
        -> catalogue aggregate: AAL, OEP curve, return-period losses

Each event is a synthetic ignition with a fuel scenario (grass / shrub / timber /
mixed), a drawn wind speed and direction, an annual occurrence rate, a synthetic
FBFM40 fuel field (with fuel breaks and an urban cluster), and a set of structures
concentrated in that cluster. The event is grown with the same
`anisotropic_arrival` solver used everywhere else, burned area is read at the horizon,
per-structure hazard intensity is taken from the arrival-time field (front arrival at
the structure or its edge), and `vulnerability_damage_ratio` + `ground_up_loss` turn
that into a ground-up event loss. `oep_curve` aggregates the catalogue.

## Properties

* **Parallel, and worker-count invariant.** Events run across processes
  (`concurrent.futures`); every event is fully seeded, so `--workers 1` and
  `--workers 8` produce identical losses. A test asserts this.
* **Same code path.** The engine and loss functions are the shared modules, so a
  catalogue number and a single-fire number are comparable.
* **Fuel drives spread.** Grass out-spreads timber by ~30x mean burned area in the
  synthetic set, the expected ordering, and a test asserts grass > timber.

## Run

    python -m vhagar.cli t4-catalog --n-events 300 --workers 4 --out summary.json

Representative synthetic run (300 events, 4 workers): mean burned area ~390 ha,
per-event runtime ~0.3 s, AAL ~$13M/yr, OEP return-period losses rising from ~$6M
(10-yr) to ~$15.5M (250-yr); by fuel, grass ~884 ha mean vs timber ~26 ha.

## Real-data mode (CAL FIRE DINS + LANDFIRE)

The same harness runs on real data:

    python -m vhagar.cli t4-catalog \
      --real-dins dins_fires_config.json \
      --dins-csv POSTFIRE_MASTER_DATA_SHARE_*.csv \
      --fuel-tif LF2025_FBFM40_CONUS.tif [--no-fuel]

Each configured fire (Eaton, Palisades, Camp, Tubbs) is one **real event**: real
ignition point and RAWS wind (config), real DINS structures at their real locations,
real parcel assessed values (winsorized at the 98th percentile, the T5 outlier
handling), and real LANDFIRE FBFM40 fuels. It reports **modeled vs actual** destroyed
structures and loss, the verification-against-observed-events task, and aggregates to
an AAL over an illustrative rate.

Representative real run (`--no-fuel`, uniform ROS): modeled destroyed / loss under the
actual on the compact urban fires (total modeled ~$6.1B vs actual ~$13.1B), because
the current WUI parameters are fixed global values, not the per-fire leave-one-out
selection that reaches mean F1 ~0.75. With `--use-fuel` (LANDFIRE), the fuel-blocked
front under-predicts further (Camp collapses to zero): the documented honest negative
(the front cannot enter non-burnable urban cells or jump non-fuel gaps that real fires
cross by long-range spotting; docs/17). Reporting that gap is the point, not hiding it.

## Honesty

The default (synthetic) mode uses **SYNTHETIC** fuels and drivers, so its losses
demonstrate the pipeline, not a real portfolio. The real mode uses real ignitions,
wind, structures, values, and fuels, but the ROS field is the documented fuel-and-wind
surrogate (not full Rothermel), the WUI parameters are fixed (not per-fire calibrated
here), and the occurrence rates are assigned, not fitted to a historical frequency, so
the AAL is illustrative. The value of this harness is the batch orchestration and the
spread-to-EP aggregation with honest modeled-vs-actual validation, not the specific
dollar figures.
