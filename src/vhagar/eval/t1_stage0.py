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
    "coincidence_scores",
    "detection_latency_minutes",
    "load_fdc_detections",
    "load_fdc_events",
    "fdc_window",
    "firms_to_detections",
    "events_from_detections",
    "cluster_events_gridded",
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


def _posix(dt: datetime) -> float:
    from datetime import UTC

    return (dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt).timestamp()


def coincidence_scores(
    goes: list[Detection],
    viirs: list[Detection],
    cell_m: float = 4_000.0,
    window_min: float = 30.0,
    restrict_domain: bool = True,
) -> dict:
    """Detection-level GEO/LEO agreement, matched in space **and** time.

    The right T1 detection metric, and the one the event-centroid matcher got wrong.
    A VIIRS detection counts as detected by GOES when a GOES detection lies in the
    same ``cell_m`` cell (or an 8-neighbour) within ``window_min`` of the VIIRS time,
    conditioning on the temporal coincidence that a geostationary/polar comparison
    demands (GOES scans continuously, VIIRS passes twice a day, so a plain cell-day
    overlap is uninterpretable). ``restrict_domain`` keeps only VIIRS inside the GOES
    detection bbox, so tropical/agricultural fires outside the GOES sector do not
    count as misses.

    Returns ``pod`` (probability of detection), ``median_gap_min`` for matched pairs,
    and the counts. Precision/FAR need the VIIRS swath geometry to be interpretable
    (a GOES detection with no VIIRS nearby may be a real fire between overpasses), so
    they are deliberately not reported here; see docs/12.
    """
    import numpy as np

    if not goes or not viirs:
        return {"pod": float("nan"), "n_viirs": 0, "n_goes": len(goes), "tp": 0,
                "median_gap_min": float("nan"), "cell_m": cell_m, "window_min": window_min}
    gx = np.array([d.x for d in goes])
    gy = np.array([d.y for d in goes])
    gt = np.array([_posix(d.when) for d in goes])
    vx = np.array([d.x for d in viirs])
    vy = np.array([d.y for d in viirs])
    vt = np.array([_posix(d.when) for d in viirs])
    if restrict_domain:
        m = (vx >= gx.min()) & (vx <= gx.max()) & (vy >= gy.min()) & (vy <= gy.max())
        vx, vy, vt = vx[m], vy[m], vt[m]

    grid: dict[tuple[int, int], np.ndarray] = {}
    from collections import defaultdict
    tmp = defaultdict(list)
    for i in range(len(gx)):
        tmp[(int(gx[i] // cell_m), int(gy[i] // cell_m))].append(gt[i])
    for k, ts in tmp.items():
        grid[k] = np.sort(np.asarray(ts))

    w = window_min * 60.0
    tp, gaps = 0, []
    for i in range(len(vx)):
        cx, cy = int(vx[i] // cell_m), int(vy[i] // cell_m)
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                arr = grid.get((cx + dx, cy + dy))
                if arr is None:
                    continue
                j = int(np.searchsorted(arr, vt[i]))
                for jj in (j - 1, j):
                    if 0 <= jj < len(arr):
                        gap = abs(float(arr[jj]) - vt[i])
                        if best is None or gap < best:
                            best = gap
        if best is not None and best <= w:
            tp += 1
            gaps.append(best / 60.0)
    return {
        "pod": tp / len(vx) if len(vx) else float("nan"),
        "n_viirs": int(len(vx)), "n_goes": int(len(gx)), "tp": tp,
        "median_gap_min": float(np.median(gaps)) if gaps else float("nan"),
        "cell_m": cell_m, "window_min": window_min,
    }


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


def cluster_events_gridded(
    detections: list[Detection], cell_m: float = 50_000.0, max_gap_hours: float = 12.0,
) -> list[FireEvent]:
    """Cluster into events within coarse spatial cells, then concatenate.

    The exact single-link clusterer is O(n^2); on 10^5 detections that is too slow.
    Binning detections into ``cell_m`` cells first keeps each clustering call small,
    at the Stage-0 cost that a fire straddling a cell boundary splits into two events
    (cells are ~50 km, far larger than a fire, so this is rare). A production build
    uses the ball-tree the fusion docstring flags; this is the pragmatic offline
    substitute so a full week of VIIRS clusters in seconds, not minutes.
    """
    buckets: dict[tuple[int, int], list[Detection]] = {}
    for d in detections:
        buckets.setdefault((int(d.x // cell_m), int(d.y // cell_m)), []).append(d)
    events: list[FireEvent] = []
    for (cx, cy), members in buckets.items():
        for ev in cluster_detections(members, max_gap_hours=max_gap_hours, id_prefix=f"c{cx}_{cy}"):
            events.append(ev)
    return events


def _naive_utc(dt: datetime) -> datetime:
    """Drop timezone to naive UTC so GOES (parquet, naive) and VIIRS (FIRMS, tz-aware)
    times compare. Both feeds are UTC, so this only removes the tzinfo tag."""
    if dt.tzinfo is not None:
        from datetime import UTC

        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


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


def load_fdc_events(
    root, region_crs: str = "EPSG:5070", max_gap_hours: float = 12.0, sensor: str = "goes",
    thin_deg: float = 0.02, thin_minutes: int = 60, cell_m: float = 50_000.0,
) -> list[FireEvent]:
    """Load FDC parquet, thin the 5-minute repeats, and cluster into events (bounded).

    GOES FDC re-detects the same pixel every five minutes, so a week of one fire is
    thousands of near-identical rows. Two performance moves make this tractable
    without a spatial index, both vectorised in pandas before any Python object is
    built: (1) **thin** to one detection per ``thin_deg`` cell per ``thin_minutes``
    bin (a ~2 km, hourly grid, so a persistent fire pixel contributes ~24 rows/day,
    not 288); (2) cluster within coarse ``cell_m`` spatial cells (see
    :func:`cluster_events_gridded`). Both are Stage-0 approximations; the production
    path is the ball-tree the fusion docstring flags. Needs pandas + pyproj.
    """
    import glob as _glob

    import pandas as pd

    files = sorted(_glob.glob(f"{root}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["t"] = pd.to_datetime(df["t"])
    # Vectorised thinning: keep the first row in each (lon-cell, lat-cell, time-bin).
    key_lon = (df["lon"] / thin_deg).round().astype("int64")
    key_lat = (df["lat"] / thin_deg).round().astype("int64")
    key_t = (df["t"].astype("int64") // (thin_minutes * 60 * 1_000_000_000))
    df = df.loc[~pd.DataFrame({"a": key_lon, "b": key_lat, "c": key_t}).duplicated()]
    xs, ys = _project(df["lon"].to_numpy(), df["lat"].to_numpy(), region_crs)
    vz_col = df["view_zenith_deg"].to_numpy() if "view_zenith_deg" in df else np.full(len(df), np.nan)
    frp_col = df["frp_mw"].to_numpy() if "frp_mw" in df else np.full(len(df), np.nan)
    times = [_naive_utc(ts.to_pydatetime()) for ts in df["t"]]
    dets = [
        Detection(
            sensor=sensor, x=float(xs[i]), y=float(ys[i]), when=times[i],
            frp_mw=(None if frp_col[i] != frp_col[i] else float(frp_col[i])),
            view_zenith_deg=(None if vz_col[i] != vz_col[i] else float(vz_col[i])),
        )
        for i in range(len(df))
    ]
    return cluster_events_gridded(dets, cell_m=cell_m, max_gap_hours=max_gap_hours)


def fdc_window(root, pad_deg: float = 0.25) -> dict:
    """The date range and padded lon/lat bbox spanned by the FDC parquet.

    Used to pull exactly the VIIRS reference that overlaps the GOES window (same
    dates, same area, padded so an edge fire is not clipped). Pure read; needs pandas.
    """
    import glob as _glob

    import pandas as pd

    files = sorted(_glob.glob(f"{root}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}")
    df = pd.concat((pd.read_parquet(f, columns=["t", "lon", "lat"]) for f in files), ignore_index=True)
    t = pd.to_datetime(df["t"])
    return {
        "start_date": t.min().date().isoformat(),
        "end_date": t.max().date().isoformat(),
        "n_days": int((t.max().date() - t.min().date()).days) + 1,
        "bbox": (
            float(df["lon"].min()) - pad_deg, float(df["lat"].min()) - pad_deg,
            float(df["lon"].max()) + pad_deg, float(df["lat"].max()) + pad_deg,
        ),
        "n_detections": int(len(df)),
    }


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
            x=float(xs[i]), y=float(ys[i]), when=_naive_utc(r.acq_datetime),
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
