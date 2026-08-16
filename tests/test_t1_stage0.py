"""T1 Stage-0: event matching, POD/FAR, parallax-vs-naive, latency. All pure."""

from __future__ import annotations

from datetime import datetime, timedelta

from vhagar.eval.t1_stage0 import (
    DetectionScores,
    detection_latency_minutes,
    match_events,
    run_t1_stage0,
)
from vhagar.harmonize.fusion import Detection, FireEvent

T0 = datetime(2026, 8, 1, 21, 0)


def _event(eid, x, y, when=T0, sensor="goes", vza=None):
    return FireEvent(event_id=eid, detections=[
        Detection(sensor=sensor, x=x, y=y, when=when, view_zenith_deg=vza)
    ])


def test_detection_scores_pod_far_f1():
    s = DetectionScores(tp=8, fp=2, fn=2)
    assert s.pod == 0.8                       # 8 / (8+2)
    assert s.far == 0.2                       # 2 / (8+2)
    assert s.precision == 0.8
    assert round(s.f1, 3) == 0.8


def test_match_pairs_close_events_and_counts_misses_and_false_alarms():
    truth = [_event("t1", 0, 0, sensor="viirs"), _event("t2", 100_000, 0, sensor="viirs")]
    # one GOES event on top of t1, one spurious far from any truth
    pred = [_event("p1", 500, 0), _event("p_fp", 400_000, 0)]
    scores, matches = match_events(pred, truth)
    assert scores.tp == 1 and scores.fp == 1 and scores.fn == 1
    assert len(matches) == 1 and matches[0].truth.event_id == "t1"


def test_parallax_aware_tolerance_matches_where_naive_2km_does_not():
    # A GOES event at 48 deg view zenith has a ~4.25 km tolerance; a truth fire 3 km
    # away is the same fire, but a flat 2 km match would call it a false alarm.
    truth = [_event("t", 0, 0, sensor="viirs")]
    pred = [_event("p", 3_000, 0, vza=48.0)]
    aware, _ = match_events(pred, truth, parallax_aware=True)
    naive, _ = match_events(pred, truth, parallax_aware=False, flat_tolerance_m=2_000.0)
    assert aware.tp == 1 and aware.far == 0.0        # parallax-aware: matched
    assert naive.tp == 0 and naive.fp == 1           # naive 2 km: spurious false alarm


def test_one_to_one_matching_does_not_double_count():
    # two predicted events near a single truth event: only one may match it
    truth = [_event("t", 0, 0, sensor="viirs")]
    pred = [_event("p1", 300, 0, vza=10.0), _event("p2", 600, 0, vza=10.0)]
    scores, matches = match_events(pred, truth)
    assert scores.tp == 1 and scores.fp == 1 and len(matches) == 1


def test_latency_reports_positive_lead_when_goes_is_earlier():
    truth = [_event("t", 0, 0, when=T0, sensor="viirs")]
    pred = [_event("p", 200, 0, when=T0 - timedelta(minutes=40), vza=10.0)]
    _, matches = match_events(pred, truth)
    lat = detection_latency_minutes(matches)
    assert lat["n"] == 1
    assert lat["median_lead_min"] == 40.0            # GOES 40 min before the overpass
    assert lat["frac_earlier"] == 1.0


def test_run_reports_far_reduction_from_parallax():
    truth = [_event("t", 0, 0, sensor="viirs")]
    pred = [_event("p", 3_000, 0, vza=48.0)]
    out = run_t1_stage0(pred, truth)
    assert out["parallax_aware"]["far"] == 0.0
    assert out["naive_2km"]["far"] == 1.0
    assert out["far_reduction"] == 1.0               # naive 1.0 -> aware 0.0
