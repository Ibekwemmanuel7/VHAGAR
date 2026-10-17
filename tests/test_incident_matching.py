"""Synthetic incident fixtures test accounting, not Southern performance."""

from datetime import UTC, datetime, timedelta
from itertools import permutations

import pytest

from vhagar.eval.incident_matching import (
    IncidentObservation,
    ReferenceIncident,
    _assignment,
    score_incidents,
)

T = datetime(2026, 2, 1, 12, tzinfo=UTC)


def ref(rid="r", lon=-100, when=T):
    return ReferenceIncident(rid, lon, 32, when, "independent-agency:snapshot-1")


def obs(oid="o", lid="e", lon=-100, when=T, known=T, masked=False):
    return IncidentObservation(oid, lid, lon, 32, when, known, "sensor:revision-1", masked)


def score(refs, observations, **kwargs):
    return score_incidents(refs, observations, as_of=T + timedelta(days=1),
                           prediction_run="code/config/input:sha256", **kwargs)


def test_one_lineage_cannot_detect_two_incidents_or_split_into_two_hits():
    report = score([ref("a"), ref("b")], [obs(), obs("p", "e")])
    assert report["n_matched"] == 1
    assert report["n_missed"] == 1
    assert report["ledger"][1]["miss_reason"] == "lineage_conflict"


def test_maximum_cardinality_reassigns_greedy_match():
    # a can use either event; b can only use e1. Greedy a->e1 loses a hit.
    report = score([ref("a", -100), ref("b", -100.04)],
                   [obs("o1", "e1", -100.01), obs("o2", "e2", -99.96)])
    assert [(r["incident_id"], r["lineage_id"]) for r in report["ledger"]] == [
        ("a", "e2"), ("b", "e1")]


def test_time_precedes_distance_and_any_observation_can_match():
    report = score([ref()], [obs("far", "e1", -99.99),
                            obs("close", "e2", when=T - timedelta(seconds=1)),
                            obs("old", "e1", -90)])
    assert report["ledger"][0]["observation_id"] == "far"


def test_missed_fire_ledger_keeps_masks_unavailable_and_quality_flags():
    refs = [ref("mask", -100), ref("late", -101), ref("unknown", -102),
            ref("absent", -103), ReferenceIncident("uncertain", -104, 32, T, "agency:v1",
                                                  ("uncertain_origin",))]
    observations = [obs("m", "m", -100, masked=True),
                    obs("l", "l", -101, known=T + timedelta(days=2)),
                    obs("u", "u", -102, known=None)]
    report = score(refs, observations)
    assert report["n_reference"] == report["n_missed"] == 5
    reasons = {r["incident_id"]: r["miss_reason"] for r in report["ledger"]}
    assert reasons["mask"] == "masked_only"
    assert reasons["late"] == reasons["unknown"] == "availability_unknown_or_after_cutoff"
    assert reasons["absent"] == "no_observation_within_tolerances"
    assert next(r for r in report["ledger"] if r["incident_id"] == "uncertain")[
        "quality_flags"] == ["uncertain_origin"]


def test_exact_boundaries_and_future_observation_cannot_leak():
    report = score([ref()], [obs(when=T + timedelta(hours=6),
                                known=T + timedelta(hours=6))], radius_m=0)
    assert report["n_matched"] == 1
    assert score([ref()], [obs(when=T + timedelta(hours=6, microseconds=1),
                              known=T + timedelta(hours=7))])["n_matched"] == 0
    assert score([ref()], [obs(lon=-99)])["n_matched"] == 0


def test_availability_cutoff_is_inclusive_and_mask_does_not_hide_visible_match():
    cutoff = T + timedelta(days=1)
    assert score([ref()], [obs(known=cutoff)])["n_matched"] == 1
    assert score([ref()], [obs(known=cutoff + timedelta(microseconds=1))])["n_matched"] == 0
    assert score([ref()], [obs("masked", masked=True), obs("visible")])["n_matched"] == 1


def test_reproducible_permutations_and_manifest_change():
    refs = [ref("a"), ref("b")]
    observations = [obs("x", "x"), obs("y", "y")]
    original = score(refs, observations)
    for rs in permutations(refs):
        for os in permutations(observations):
            assert score(list(rs), list(os)) == original
    assert score(refs, observations, radius_m=1)["manifest_sha256"] != original["manifest_sha256"]
    assert score([], [obs()])["detection_fraction"] is None
    assert score([ref()], [])["n_missed"] == 1


def test_assignment_matches_exhaustive_optimum():
    import random

    rng = random.Random(72)
    for _ in range(50):
        pairs = {(r, e): (rng.randrange(4), rng.randrange(10), "o")
                 for r in range(4) for e in range(4) if rng.random() < 0.6}
        candidates = []

        def visit(r, used, count, time, distance, candidates=candidates, pairs=pairs):
            if r == 4:
                candidates.append((-count, time, distance))
                return
            visit(r + 1, used, count, time, distance)
            for e in range(4):
                if e not in used and (r, e) in pairs:
                    cost = pairs[r, e]
                    visit(r + 1, used | {e}, count + 1, time + cost[0], distance + cost[1])

        visit(0, set(), 0, 0, 0)
        actual = _assignment(pairs)
        assert (-len(actual), sum(pairs[r, e][0] for r, e in actual.items()),
                sum(pairs[r, e][1] for r, e in actual.items())) == min(candidates)


def test_reject_ambiguous_identity_and_invalid_inputs():
    with pytest.raises(ValueError, match="deduplicate reference"):
        score([ref(), ref()], [])
    with pytest.raises(ValueError, match="deduplicate observation"):
        score([], [obs(), obs()])
    with pytest.raises(ValueError, match="timezone"):
        ref(when=T.replace(tzinfo=None))
    with pytest.raises(ValueError, match="precedes"):
        obs(known=T - timedelta(seconds=1))
    with pytest.raises(ValueError, match="longitude"):
        ref(lon=float("nan"))
    with pytest.raises(ValueError, match="tolerances"):
        score([], [], radius_m=-1)
    with pytest.raises(ValueError, match="tolerances"):
        score([], [], window=timedelta(seconds=-1))


def test_cli_round_trip_frozen_manifest(tmp_path, monkeypatch):
    import json
    import runpy
    from pathlib import Path

    original = score([ref()], [obs()])
    source = tmp_path / "manifest.json"
    target = tmp_path / "report.json"
    source.write_text(json.dumps(original["manifest"]), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "score_incidents.py"
    monkeypatch.setattr("sys.argv", [str(script), str(source), "--output", str(target)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name="__main__")
    assert result.value.code == 0
    assert json.loads(target.read_text(encoding="utf-8")) == original
