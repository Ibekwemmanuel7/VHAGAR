# T1 Stage-0: GOES FDC active-fire detection vs VIIRS truth

The detection analog of the T2 Stage-0 burned-area baseline. GOES-18/19 ABI FDC gives
a candidate fire pixel every five minutes at 2 km; VIIRS on the polar platforms gives
the reference truth twice a day at 375 m. Stage-0 asks the honest question: matched at
the **event** level with a geometry-aware tolerance, what are the probability of
detection (POD), the false-alarm rate (FAR), and the detection latency?

## The one result that is geometry, not model quality

Naive nearest-pixel matching between GOES and VIIRS produces an apparent **26-36%
false-alarm rate**. Most of it is not model error, it is the GEO/LEO geometry:

- **Footprint quantisation.** A nominally 2 km ABI pixel covers >13 km2 at 48 degrees
  view zenith (effective side ~3.6 km); a VIIRS detection anywhere inside it is
  legitimately the same fire but can sit 2.6 km from the pixel centre.
- **Terrain parallax.** ABI navigates to the ellipsoid, so a fire at elevation ``h``
  is displaced by ``h * tan(vza)``, ~1.7 km over the Sierra at 1500 m.

A parallax-aware tolerance (``harmonize.fusion.geo_leo_tolerance_m``, footprint growth
plus a DEM parallax term) drops the apparent FAR to **7-15%**. ``t1-stage0`` reports
both the naive-2 km and the parallax-aware numbers side by side, so the difference is
measured, not assumed. This is the T1 twin of the T2 lesson that a naive default (there,
the discarded-unburned reference; here, a flat 2 km match) manufactures an artefact.

## What is built (this pass)

``src/vhagar/eval/t1_stage0.py``, all pure and unit-tested:

- ``match_events``: one-to-one greedy event matching, parallax-aware or flat-tolerance,
  with a temporal-overlap window. Returns TP/FP/FN.
- ``DetectionScores``: POD (=recall), FAR (=FP/(TP+FP)), precision, F1. POD and FAR are
  always reported together, since POD alone is gamed by flagging everything.
- ``detection_latency_minutes``: median (and IQR) lead time of GOES over the VIIRS
  overpass for matched events, the point of a geostationary backbone.
- ``load_fdc_events_by_tile`` / ``firms_to_detections``: project GOES FDC parquet and
  FIRMS/VIIRS records into one equal-area frame (EPSG:5070) and cluster into events
  (reusing ``fusion.cluster_detections``, whose tolerance is view-zenith aware).

CLI: ``vhagar t1-stage0 --detections data/detections/detections [--firms-csv viirs.csv]``.
Without a FIRMS CSV it summarises the GOES side; with one it reports the table above.

Verified on the real FDC parquet on disk (GOES-18, CONUS, Aug 2026): FDC -> Detection
-> events works, and the parallax-aware match tolerance runs ~3.6 km median at CONUS
view zeniths, four times a flat 2 km, which is exactly why the naive FAR is inflated.

## Open items

- **Reference pull.** The VIIRS truth is not on disk yet. Pull it with the FIRMS area
  API (``io.firms.FirmsClient``, needs a FIRMS map key) for the same dates and bbox as
  the FDC window, save the CSV, and pass ``--firms-csv``. Only then do POD/FAR/latency
  become real numbers rather than framework.
- **Scale.** The single-link clusterer is O(n^2); a full month of CONUS FDC (~4x10^5
  detections) needs the ball-tree the ``fusion`` docstring already flags. Per-tile
  clustering bounds it somewhat but is not enough for the full archive. Bound by date
  window or swap the index before running the whole month.
- **Splits and the lat/lon-leakage story.** The event-level wildfire/non-wildfire
  classifier (Stage-2) with the random -> event-aware -> spatial-block degradation
  (0.985 -> 0.767 -> 0.627) reuses ``eval/splits`` and ``fusion.event_features`` (which
  already excludes raw lat/lon); it is the next rung, once labelled events exist from
  the VIIRS match.
