# T1 Stage-0: GOES FDC active-fire detection vs VIIRS truth

The detection analog of the T2 Stage-0 burned-area baseline. GOES-18/19 ABI FDC gives
a candidate fire pixel every five minutes at 2 km; VIIRS on the polar platforms gives
the reference truth twice a day at 375 m. Stage-0 asks the honest question: matched at
the **event** level with a geometry-aware tolerance, what are the probability of
detection (POD), the false-alarm rate (FAR), and the detection latency?

## First real result (GOES-18 CONUS, 2026-08-01..07, vs VIIRS NOAA-20 + S-NPP)

Detection-level coincidence: a VIIRS fire detection counts as seen by GOES when a GOES
detection lies in the same spatial cell (8-neighbour) within +/-30 min, restricted to
the GOES sector. 104,772 VIIRS detections in domain.

| matching | cell | POD | median gap |
|---|---|---|---|
| naive | 2 km | 0.376 | 2 min |
| **parallax-aware** | **4 km** | **0.499** | **2 min** |

Two things to read from this:

- **GOES FDC detects about half of VIIRS's fire pixels** (POD ~0.50 at the parallax
  scale), near-simultaneously (median gap 2 min). That is the credible, literature-
  consistent number: GOES at 2 km misses the small/cool fires VIIRS's 375 m catches, so
  a POD around 0.4-0.6 is exactly what a geostationary detector scores against a polar
  one, not a bug.
- **The +0.12 POD from 2 km to 4 km is geometry, not model quality.** A nominal 2 km ABI
  pixel covers >13 km2 at high view zenith (effective side ~3.6 km) and terrain parallax
  displaces an elevated fire by ``h * tan(vza)``, so the GOES detection sits offset from
  the VIIRS location. Matching at the footprint+parallax scale recovers 12 points of POD
  that naive 2 km matching throws away. This is the T1 twin of the T2 lesson: a naive
  default (there, the discarded-unburned reference; here, a too-tight match) manufactures
  a worse number than the sensor deserves.

### A broken first metric, and why it was caught

The first cut clustered detections into events and matched event **centroids** with a
3.6 km tolerance, and pulled VIIRS over a huge bbox (down to Guatemala). It reported
POD 0.047, which is absurd for two fire sensors over the same week. The naive-baseline
instinct from T2 caught it: a 4.7% match rate is a broken metric, not a real detection
rate. Diagnosis: (1) VIIRS spanned tropical/agricultural fires outside the GOES-CONUS
sector, counted as misses; (2) matching 50 km-cell cluster centroids at 3.6 km
guarantees misses (median nearest-centroid distance was 276 km); (3) an int-unit bug in
an exploratory grid check gave a spurious 0.0. The fix is detection-level coincidence in
space **and** time, domain-restricted, above. Precision/FAR are deliberately *not*
reported: they need the VIIRS swath geometry to be interpretable (a GOES detection with
no VIIRS nearby may be a real fire between the twice-daily overpasses, not a false
alarm), so they are Stage-2 work.

## What is built (this pass)

``src/vhagar/eval/t1_stage0.py``, all pure and unit-tested:

- ``match_events``: one-to-one greedy event matching, parallax-aware or flat-tolerance,
  with a temporal-overlap window. Returns TP/FP/FN.
- ``DetectionScores``: POD (=recall), FAR (=FP/(TP+FP)), precision, F1. POD and FAR are
  always reported together, since POD alone is gamed by flagging everything.
- ``detection_latency_minutes``: median (and IQR) lead time of GOES over the VIIRS
  overpass for matched events, the point of a geostationary backbone.
- ``load_fdc_events`` (thins the 5-min repeats to a ~2 km/hourly grid, then clusters in
  coarse spatial cells so a week runs in ~1 min without a spatial index) /
  ``firms_to_detections``: project GOES FDC parquet and
  FIRMS/VIIRS records into one equal-area frame (EPSG:5070) and cluster into events
  (reusing ``fusion.cluster_detections``, whose tolerance is view-zenith aware).

CLI: ``vhagar t1-stage0 --detections data/detections/detections [--firms-csv viirs.csv]``.
Without a FIRMS CSV it summarises the GOES side; with one it reports the table above.

Verified on the real FDC parquet on disk (GOES-18, CONUS, Aug 2026): FDC -> Detection
-> events works, and the parallax-aware match tolerance runs ~3.6 km median at CONUS
view zeniths, four times a flat 2 km, which is exactly why the naive FAR is inflated.

## Runbook: first real POD/FAR/latency

```
# 1. free FIRMS map key: https://firms.modaps.eosdis.nasa.gov/api/map_key/
$env:FIRMS_MAP_KEY = "<your key>"
# 2. pull the VIIRS truth for exactly the GOES window (reads dates/bbox from the FDC parquet)
vhagar firms-fetch --detections data\detections\detections --out viirs_truth.csv
# 3. score
vhagar t1-stage0 --detections data\detections\detections --firms-csv viirs_truth.csv
```

``firms-fetch`` reads the FDC window (here 2026-08-01 to 2026-08-07, CONUS+HI bbox) and
pulls the matching VIIRS in <=10-day chunks; ``t1-stage0`` then reports the parallax-
aware vs naive-2 km table above. The FDC window is one week, so the ball-tree is not
needed for this first run; per-tile clustering handles it.

## Open items

- **Reference pull** is now one command (``firms-fetch`` above); it just needs the free
  FIRMS map key and network. Until it is run, POD/FAR/latency are framework, not numbers.
- **Scale.** The single-link clusterer is O(n^2); a full month of CONUS FDC (~4x10^5
  detections) needs the ball-tree the ``fusion`` docstring already flags. Per-tile
  clustering bounds it somewhat but is not enough for the full archive. Bound by date
  window or swap the index before running the whole month.
- **Splits and the lat/lon-leakage story.** The event-level wildfire/non-wildfire
  classifier (Stage-2) with the random -> event-aware -> spatial-block degradation
  (0.985 -> 0.767 -> 0.627) reuses ``eval/splits`` and ``fusion.event_features`` (which
  already excludes raw lat/lon); it is the next rung, once labelled events exist from
  the VIIRS match.
