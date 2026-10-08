"""Real-data parcel suitability study: utility-scale solar in Texas.

The question the study answers is a deployment question, stated before any data were
looked at: *using only information available at a 2016 snapshot, how well can a
parcel-level screen rank Texas land that later hosted a utility-scale solar facility
(2017 to 2025) above land that did not?*

Design (see ``docs/28_PARCEL_STUDY_TEXAS_SOLAR.md``):

* **Population.** Every Texas land parcel of at least 2 ha in the TxGIO 2026 StratMap
  release, after de-duplicating stacked geometries, excluding parcels already covered by
  a facility built in or before the 2016 snapshot.
* **Labels.** USGS/LBNL USPVDB v4.0 ground-mounted facility footprints. A parcel is a case
  when a facility installed 2017 to 2025 covers at least 2 ha or 25% of it.
* **Case-control sampling.** All cases plus a simple random sample of non-case parcels
  with a recorded sampling fraction, so every metric is re-weighted to the population.
* **Features at the snapshot.** 2016 land cover and roads, 2018 wildfire hazard, and
  slower-changing layers (terrain, irradiance climatology, protected areas, floodplain,
  transmission) with their vintages and leakage risks recorded per feature.
* **Evaluation.** Temporal holdout (train on 2017 to 2020 cases, test on 2021 to 2025) and
  spatial-block holdout, against null, single-feature, and transparent rules baselines,
  with facility-clustered bootstrap intervals.

Modules: ``paths`` (constants and locations), ``labels`` (USPVDB), ``population``
(parcels, de-duplication, case labels), ``sampling`` (case-control draw and fetch plan),
``features`` (vector and raster overlays), ``modeling`` (models, metrics, splits).
"""
