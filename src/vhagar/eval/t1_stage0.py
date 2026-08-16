"""T1 Stage-0: how well do GOES FDC active-fire detections agree with VIIRS truth?

The detection analog of the T2 Stage-0 burned-area baseline. GOES ABI FDC gives a
candidate fire pixel every five minutes at 2 km; VIIRS on the polar platforms gives
the reference truth twice a day at 375 m. Stage-0 asks the honest question: matched
at the *event* level with a geometry-aware tolerance, what are the probability of
detection, the false-alarm rate, and the detection latency?

Two disciplines carried over from T2, and one specific to the GEO/LEO geometry:

* **Match events, not pixels.** Naive nearest-pixel matching between GOES and VIIRS
  produces an apparent 26-36% false-alarm rate that is *geometry, not model error*:
  a 2 km ABI pixel covers >13 km2 at 48 degrees view zenith, and terrain parallax
  displaces an elevated fire by ``h * tan(vza)``. A parallax-aware tolerance
  (``geo_leo_tolerance_m``) drops the apparent FAR to 7-15%. This module measures
  both so the difference is visible, not assumed.
* **Skill is POD and FAR together.** A detector can trivially reach POD 1.0 by
  flagging everything; report the false-alarm rate beside it, always.
* **Latency is the point of a geostationary backbone.** For matched events, report
  how many minutes GOES leads (or lags) the VIIRS overpass, since "detect earlier at
  equal FAR" is the T1 target.

The reference (VIIRS/FIRMS) needs a pull; the matching, scoring and latency here are
pure and unit-tested, and the GOES side runs on the FDC parquet already on disk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from vhagar.harmonize.fusion import Detection, FireEvent, cluster_detections

__all__ = [
    "DetectionScores",
    "EventMatch",
    "match_events",
    "detection_latency_minutes",
    "load_fdc_detections",
    "load_fdc_events_by_tile",
    "firms_to_detections",
    "events_from_detections",
    "run_t1_stage0",
]


@dataclass(frozen=True, slots=True)
class DetectionScores:
    """Event-level detection agreement of a predictor against a reference."""

    tp: int
    fp: int
    fn: int

    @property
    def pod(self) -> float:
        """Probability of detection = recall = TP / (TP + FN)."""
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def far(self) -> float:
        """False-alarm rate = FP / (TP + FP): fraction of predicted events unmatched."""
        d = self.tp + self.fp
        return self.fp / d if d else float("nan")

    @property
    def precision(self) -> float:
        return 1.0 - self.far if (self.tp + self.fp) else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.pod
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "pod": self.pod,
                "far": self.far, "precision": self.precision, "f1": self.f1}


@dataclass(frozen=True, slots=True)
class EventMatch:
    """One matched (predicted, reference) event pair and their separation."""

    pred: FireEvent
    truth: FireEvent
    distance_m: float
    lead_minutes: float   # truth.start - pred.start; positive => predictor is earlier


def _time_overlap_minutes(a: FireEvent, b: FireEvent) -> float:
    """Minutes by which two events' active spans overlap; negative = gap between them."""
    latest_start = max(a.start, b.start)
    earliest_end = min(a.end, b.end)
    return (earliest_end - latest_start).total_seconds() / 60.0


def match_events(
    pred_events: Sequence[FireEvent],
    truth_events: Sequence[FireEvent],
    parallax_aware: bool = True,
    flat_tolerance_m: float = 2_000.0,
    max_time_gap_min: float = 180.0,
) -> tuple[DetectionScores, list[EventMatch]]:
    """Match predicted events to reference events, one-to-one, greedily by distance.

    A pair may match when their centroids are within the tolerance and their active
    spans are within ``max_time_gap_min`` of overlapping. With ``parallax_aware`` the
    tolerance is the predicted event's own geometry-derived matching radius (the max
    detection ``tolerance_m`` in the event, which grows with view zenith); otherwise a
    flat ``flat_tolerance_m`` is used, which is the naive baseline that inflates FAR.

    Returns ``(DetectionScores, matches)``. Unmatched predicted events are false
    alarms (FP); unmatched reference events are misses (FN).
    """
    candidates: list[tuple[float, int, int]] = []
    for pi, p in enumerate(pred_events):
        px, py = p.centroid()
        tol = (
            max((d.tolerance_m for d in p.detections), default=flat_tolerance_m)
            if parallax_aware else flat_tolerance_m
        )
        for ti, t in enumerate(truth_events):
            tx, ty = t.centroid()
            dist = float(np.hypot(px - tx, py - ty))
            if dist > tol:
                continue
            if _time_overlap_minutes(p, t) < -max_time_gap_min:
                continue
            candidates.append((dist, pi, ti))

    candidates.sort(key=lambda c: c[0])
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matches: list[EventMatch] = []
    for dist, pi, ti in candidates:
        if pi in used_pred or ti in used_truth:
            continue
        used_pred.add(pi)
        used_truth.add(ti)
        p, t = pred_events[pi], truth_events[ti]
        lead = (t.start - p.start).total_seconds() / 60.0
        matches.append(EventMatch(pred=p, truth=t, distance_m=dist, lead_minutes=lead))

    tp = len(matches)
    fp = len(pred_events) - tp
    fn = len(truth_events) - tp
    return DetectionScores(tp=tp, fp=fp, fn=fn), matches


def detection_latency_minutes(matches: Sequence[EventMatch]) -> dict:
    """Summary of predictor lead time over the reference for matched events.

    Positive lead = the predictor (GOES) saw the fire before the reference overpass,
    which is the whole reason for a geostationary backbone. Reports median and IQR;
    the median is the headline because lead-time distributions are heavy-tailed.
    """
    if not matches:
        return {"n": 0}
    lead = np.array([m.lead_minutes for m in matches])
    return {
        "n": len(lead),
        "median_lead_min": float(np.median(lead)),
        "p25_lead_min": float(np.percentile(lead, 25)),
        "p75_lead_min": float(np.percentile(lead, 75)),
        "frac_earlier": float(np.mean(lead > 0)),
    }


def events_from_detections(detections: list[Detection], max_gap_hours: float = 12.0) -> list[FireEvent]:
    """Cluster raw detections into fire events (thin wrapper over the fusion clusterer)."""
    return cluster_detections(detections, max_gap_hours=max_gap_hours)


def _project(lons, lats, crs: str):
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tf.transform(np.asarray(lons, float), np.asarray(lats, float))
    return xs, ys


def load_fdc_detections(
    root, region_crs: str = "EPSG:5070", sensor: str = "goes",
) -> list[Detection]:
    """Load GOES FDC parquet into :class:`Detection` objects in an equal-area CRS.

    Reads the partitioned ``detections/`` parquet, projects lon/lat into ``region_crs``
    (so GOES and VIIRS share one planar frame), and carries view zenith so the
    matching tolerance is geometry-aware. Needs pandas + pyproj.
    """
    import glob as _glob

    import pandas as pd

    files = sorted(_glob.glob(f"{root}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    xs, ys = _project(df["lon"].to_numpy(), df["lat"].to_numpy(), region_crs)
    out: list[Detection] = []
    for i, row in enumerate(df.itertuples(index=False)):
        vz = getattr(row, "view_zenith_deg", None)
        out.append(Detection(
            sensor=sensor, x=float(xs[i]), y=float(ys[i]),
            when=row.t if isinstance(row.t, datetime) else row.t.to_pydatetime(),
            frp_mw=(None if row.frp_mw != row.frp_mw else float(row.frp_mw)),
            view_zenith_deg=(None if vz is None or vz != vz else float(vz)),
        ))
    return out


def load_fdc_events_by_tile(
    root, region_crs: str = "EPSG:5070", max_gap_hours: float = 12.0, sensor: str = "goes",
) -> list[FireEvent]:
    """Load FDC parquet and cluster into events **per tile**, then concatenate.

    Clustering within each tile keeps the O(n^2) single-link matcher bounded (a tile
    is a few thousand detections at most), which is what makes this run on a full
    month of CONUS FDC without a spatial index. The one approximation: a fire that
    straddles a tile boundary becomes two events. That is a Stage-0 caveat, not a
    production choice; swap in the ball-tree clusterer to remove it. Needs pandas +
    pyproj.
    """
    import glob as _glob

    import pandas as pd

    files = sorted(_glob.glob(f"{root}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    events: list[FireEvent] = []
    for tile, g in df.groupby("tile_id"):
        xs, ys = _project(g["lon"].to_numpy(), g["lat"].to_numpy(), region_crs)
        dets = []
        for i, row in enumerate(g.itertuples(index=False)):
            vz = getattr(row, "view_zenith_deg", None)
            when = row.t if isinstance(row.t, datetime) else row.t.to_pydatetime()
            dets.append(Detection(
                sensor=sensor, x=float(xs[i]), y=float(ys[i]), when=when,
                frp_mw=(None if row.frp_mw != row.frp_mw else float(row.frp_mw)),
                view_zenith_deg=(None if vz is None or vz != vz else float(vz)),
            ))
        safe = str(tile).replace("/", "_")
        for ev in cluster_detections(dets, max_gap_hours=max_gap_hours, id_prefix=f"goes_{safe}"):
            events.append(ev)
    return events


def firms_to_detections(records: Sequence, region_crs: str = "EPSG:5070") -> list[Detection]:
    """Project FIRMS/VIIRS records (the reference truth) into :class:`Detection` objects."""
    if not records:
        return []
    lons = [r.longitude for r in records]
    lats = [r.latitude for r in records]
    xs, ys = _project(lons, lats, region_crs)
    out = []
    for i, r in enumerate(records):
        out.append(Detection(
            sensor=r.instrument.lower() if getattr(r, "instrument", None) else "viirs",
            x=float(xs[i]), y=float(ys[i]), when=r.acq_datetime,
            frp_mw=float(r.frp) if getattr(r, "frp", None) is not None else None,
        ))
    return out


def run_t1_stage0(pred_events: Sequence[FireEvent], truth_events: Sequence[FireEvent]) -> dict:
    """Score GOES events against VIIRS truth: parallax-aware vs naive, plus latency."""
    aware, matches = match_events(pred_events, truth_events, parallax_aware=True)
    naive, _ = match_events(pred_events, truth_events, parallax_aware=False)
    return {
        "n_pred_events": len(pred_events),
        "n_truth_events": len(truth_events),
        "parallax_aware": aware.as_dict(),
        "naive_2km": naive.as_dict(),
        "far_reduction": (naive.far - aware.far) if (naive.tp + naive.fp) and (aware.tp + aware.fp) else float("nan"),
        "latency": detection_latency_minutes(matches),
    }
