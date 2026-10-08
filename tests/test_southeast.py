"""VHAGAR Southeast release 1: association, control, threat, shadow routing.

Each test pins one rule from docs/25 sections 6 and 7. Synthetic geometry near
Waycross, Georgia; no network, no geospatial stack."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vhagar.io.firms import FirmsRecord
from vhagar.southeast import (
    AssessConfig,
    Asset,
    BurnRecord,
    BurnRegistry,
    Obs,
    ObservedEvent,
    assess_event,
    associate,
    find_candidates,
)
from vhagar.southeast.events import cluster_obs, cluster_obs_grid, obs_from_firms
from vhagar.southeast.metrics import (
    clopper_pearson_upper,
    dangerous_downgrade,
    detection_coverage,
    nuisance_reduction,
    season_summary,
)
from vhagar.southeast.records import records_from_csv, records_from_geojson

T0 = datetime(2027, 2, 10, 13, 0, tzinfo=UTC)          # burn day, 08:00 EST
DAY_START = T0.replace(hour=12)
DAY_END = T0.replace(hour=23)
LON, LAT = -82.35, 31.20
# ~1.9 km x 2.4 km unit
UNIT = ((LON - 0.01, LAT - 0.01), (LON + 0.01, LAT - 0.01),
        (LON + 0.01, LAT + 0.01), (LON - 0.01, LAT + 0.01))


def unit_record(**kw) -> BurnRecord:
    base = dict(record_id="U44", source="CUSTOMER", burn_type="broadcast", status="active",
                geometry_kind="unit_polygon", valid_from=DAY_START, valid_to=DAY_END,
                ingested_at=T0 - timedelta(hours=6), ring=UNIT, acres=1100.0,
                holder_id="OWNER-A", customer_id="CUST1")
    base.update(kw)
    return BurnRecord(**base)


def permit(rid="P1", lon=LON, lat=LAT, ingested=T0 - timedelta(hours=3), **kw) -> BurnRecord:
    base = dict(record_id=rid, source="GA_GFC", burn_type="silviculture", status="active",
                geometry_kind="address_point", valid_from=DAY_START, valid_to=DAY_END,
                ingested_at=ingested, point=(lon, lat), acres=150.0, holder_id="OWNER-B")
    base.update(kw)
    return BurnRecord(**base)


def event(points, when=T0 + timedelta(hours=5), eid="E1", frp=20.0) -> ObservedEvent:
    return ObservedEvent(eid, [Obs(lon, lat, when, "viirs", frp, 400.0, 375.0 ** 2)
                               for lon, lat in points])


def registry(*records, polled_at=None) -> BurnRegistry:
    reg = BurnRegistry()
    for r in records:
        reg.upsert(r)
    if polled_at is not None:
        reg.heartbeat("GA_GFC", polled_at)
    return reg


NOW = T0 + timedelta(hours=5, minutes=10)


# --- records -----------------------------------------------------------------------

def test_record_validation():
    with pytest.raises(ValueError):
        unit_record(valid_to=DAY_START)                    # empty window
    with pytest.raises(ValueError):
        unit_record(ring=None)                             # polygon without ring
    with pytest.raises(ValueError):
        unit_record(valid_from=datetime(2027, 2, 10))      # naive datetime
    with pytest.raises(ValueError):
        unit_record(status="approved")


def test_registry_is_bitemporal():
    late = permit(ingested=T0 + timedelta(hours=8))
    reg = registry(late)
    # The permit arrived after the decision time, so it must not explain the event.
    assert reg.known_at(NOW) == []
    assert len(reg.known_at(T0 + timedelta(hours=9))) == 1
    # A new version supersedes the old one only from its own ingest time.
    reg.upsert(permit(ingested=T0 + timedelta(hours=10), status="cancelled"))
    assert reg.known_at(T0 + timedelta(hours=9))[0].status == "active"
    assert reg.known_at(T0 + timedelta(hours=11))[0].status == "cancelled"
    assert len(reg.versions("P1")) == 2


def test_loaders():
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in UNIT] + [list(UNIT[0])]]},
        "properties": {"record_id": "U1", "valid_from": DAY_START.isoformat(),
                       "valid_to": DAY_END.isoformat(), "acres": 500}}]}
    recs = records_from_geojson(fc, ingested_at=T0)
    assert recs[0].is_polygon and len(recs[0].ring) == 4 and recs[0].is_customer
    csv_text = ("record_id,lon,lat,valid_from,valid_to,acres\n"
                f"P9,{LON},{LAT},{DAY_START.isoformat()},{DAY_END.isoformat()},80\n")
    recs = records_from_csv(csv_text, source="GA_GFC", ingested_at=T0)
    assert recs[0].geometry_kind == "address_point" and recs[0].acres == 80
    bad = {"features": [{"geometry": {"type": "MultiPolygon", "coordinates": []},
                         "properties": {}}]}
    with pytest.raises(ValueError):
        records_from_geojson(bad, ingested_at=T0)


# --- association ---------------------------------------------------------------------

def test_declared_match_inside_unit_downgrades_in_shadow_only():
    reg = registry(unit_record())
    a = assess_event(event([(LON, LAT), (LON + 0.003, LAT)]), reg, NOW)
    assert a.association.level == "DECLARED_MATCH"
    assert a.control.status == "IN_BOUNDS"
    assert a.routing.shadow_action == "DOWNGRADE"
    # Release 1 never changes the operational alert.
    assert a.routing.operational_action == "PASS_THROUGH"
    live = assess_event(event([(LON, LAT)]), reg, NOW, cfg=AssessConfig(shadow_mode=False))
    assert live.routing.operational_action == "DOWNGRADE"


def test_no_record_alerts():
    a = assess_event(event([(LON, LAT)]), registry(), NOW)
    assert a.association.level == "NONE"
    assert a.control.status == "NOT_APPLICABLE"
    assert a.routing.shadow_action == "ALERT"


def test_owner_only_never_reaches_strong_association():
    far = permit(lon=LON + 0.2, lat=LAT + 0.2)              # address ~29 km away
    reg = registry(far, polled_at=NOW)
    a = assess_event(event([(LON, LAT)]), reg, NOW, owner_of=lambda lon, lat: "OWNER-B")
    assert a.association.level == "OWNER_ONLY"
    assert a.association.score <= AssessConfig().owner_only_cap < AssessConfig().strong_score
    # The address point is far from the fire, but that is not escape evidence.
    assert a.control.status == "UNKNOWN"
    assert a.routing.shadow_action == "ANALYST"


def test_many_nearby_permits_do_not_inflate_score():
    """Noisy-OR would turn five weak candidates into a confident match."""
    ev = event([(LON, LAT)])
    off = 0.016                                             # ~1.5 km away: weak each
    weak = [permit(rid=f"W{i}", lon=LON + off, lat=LAT, holder_id=f"H{i}") for i in range(5)]
    cfg = AssessConfig()
    reg = registry(*weak, polled_at=NOW)
    cands = find_candidates(ev, reg, NOW, cfg)
    single = associate(cands[:1], cfg)
    many = associate(cands, cfg)
    assert many.n_groups == 5
    assert many.score == pytest.approx(single.score)
    assert many.level == "WEAK_MATCH"


def test_correlated_records_are_one_group():
    ev = event([(LON, LAT)])
    reg = registry(permit(rid="A"), permit(rid="B"), polled_at=NOW)   # same holder, same day
    a = associate(find_candidates(ev, reg, NOW, AssessConfig()), AssessConfig())
    assert a.n_candidates == 2 and a.n_groups == 1


def test_withdrawn_record_never_explains_a_fire():
    reg = registry(unit_record(status="revoked_burn_ban"))
    a = assess_event(event([(LON, LAT)]), reg, NOW)
    assert a.association.level == "NONE"
    assert a.routing.shadow_action == "ALERT"


def test_outside_window_is_not_a_candidate():
    ev = event([(LON, LAT)], when=DAY_END + timedelta(hours=80))
    assert find_candidates(ev, registry(unit_record()), ev.end + timedelta(minutes=5),
                           AssessConfig()) == []


def test_stale_state_source_blocks_downgrade():
    reg = registry(permit(), polled_at=None)                # only the ingest-time poll
    stale_now = T0 + timedelta(hours=5)                     # ingest was 8 h earlier
    a = assess_event(event([(LON, LAT)], when=stale_now - timedelta(minutes=5)), reg, stale_now)
    assert a.association.level == "PERMIT_MATCH"
    assert a.association.best.source_fresh is False
    assert a.routing.shadow_action == "ANALYST"
    fresh = registry(permit(), polled_at=stale_now - timedelta(minutes=10))
    b = assess_event(event([(LON, LAT)], when=stale_now - timedelta(minutes=5)), fresh, stale_now)
    assert b.routing.shadow_action == "DOWNGRADE"


# --- control ----------------------------------------------------------------------------

def test_detection_outside_unit_is_escape_even_when_declared():
    reg = registry(unit_record())
    pts = [(LON, LAT), (LON + 0.03, LAT)]                   # second point ~2 km past the edge
    a = assess_event(event(pts), reg, NOW)
    assert a.control.status == "ESCAPE_SUSPECTED"
    assert a.routing.shadow_action == "ALERT_ESCAPE"


def test_area_larger_than_declared_is_escape():
    reg = registry(unit_record(acres=20.0))
    pts = [(LON + i * 0.004, LAT + j * 0.004) for i in range(-2, 3) for j in range(-2, 3)]
    a = assess_event(event(pts), reg, NOW)                  # 25 pixels x 35 ac >> 20 ac
    assert a.control.status == "ESCAPE_SUSPECTED"
    assert any("declared" in r for r in a.control.reasons)


def test_unknown_control_never_downgrades():
    reg = registry(unit_record())
    later = T0 + timedelta(hours=10)                        # last detection 5 h...
    ev = event([(LON, LAT)], when=T0 + timedelta(hours=1))  # ... make it 9 h old
    a = assess_event(ev, reg, later)
    assert a.control.status == "UNKNOWN"
    assert a.routing.shadow_action == "HOLD"


def test_reported_completion_allows_in_bounds_after_gap():
    reg = registry(unit_record(completion_reported_at=T0 + timedelta(hours=4)))
    a = assess_event(event([(LON, LAT)], when=T0 + timedelta(hours=1)), reg,
                     T0 + timedelta(hours=10))
    assert a.control.status == "IN_BOUNDS"


def test_weather_outside_prescription_is_escape_evidence():
    a = assess_event(event([(LON, LAT)]), registry(unit_record()), NOW,
                     weather_outside_prescription=True)
    assert a.routing.shadow_action == "ALERT_ESCAPE"


# --- threat ------------------------------------------------------------------------------

def test_high_threat_alerts_even_for_a_declared_burn():
    reg = registry(unit_record())
    sub = Asset("SUBSTATION-7", LON + 0.012, LAT)           # just outside the unit edge
    a = assess_event(event([(LON + 0.008, LAT)]), reg, NOW, assets=[sub])
    assert a.association.level == "DECLARED_MATCH"
    assert a.threat.level == "HIGH"
    assert a.routing.shadow_action == "ALERT"


def test_threat_bands():
    a = assess_event(event([(LON, LAT)]), registry(), NOW,
                     assets=[Asset("FAR", LON + 0.5, LAT)])
    assert a.threat.level == "LOW"
    b = assess_event(event([(LON, LAT)]), registry(), NOW,
                     assets=[Asset("MID", LON + 0.05, LAT)])
    assert b.threat.level == "ELEVATED"


def test_static_source_is_logged_not_associated():
    ev = ObservedEvent("S", [Obs(LON, LAT, T0, "viirs", 5.0, static_anomaly=True)])
    a = assess_event(ev, registry(unit_record()), NOW)
    assert a.routing.shadow_action == "LOG_STATIC"
    assert a.association.level == "NONE"


def test_audit_record_is_json_ready():
    import json
    a = assess_event(event([(LON, LAT)]), registry(unit_record()), NOW)
    rec = a.to_record()
    json.dumps(rec)
    assert rec["association"]["best_record_id"] == "U44"
    assert rec["routing"]["operational_action"] == "PASS_THROUGH"
    assert len(rec["config"]) == 12


def test_config_guards_owner_cap():
    with pytest.raises(ValueError):
        AssessConfig(owner_only_cap=0.9, strong_score=0.8)


# --- events -----------------------------------------------------------------------------

def test_firms_to_events():
    def rec(lon, lat, h):
        return FirmsRecord(lat, lon, 330.0, 0.4, 0.4, T0 + timedelta(hours=h), "N21", "VIIRS",
                           "n", "2.0NRT", 290.0, 6.0, "D")
    obs = obs_from_firms([rec(LON, LAT, 0), rec(LON + 0.004, LAT, 1), rec(LON + 1.0, LAT, 0)])
    assert obs[0].location_error_m == pytest.approx(0.5 * (0.4 ** 2 * 2) ** 0.5 * 1000 + 100)
    events = cluster_obs(obs)
    assert sorted(len(e.detections) for e in events) == [1, 2]


def test_grid_clustering_matches_fusion():
    import random
    rng = random.Random(7)
    obs = []
    for _ in range(12):                                     # 12 burns, some close together
        clon, clat = LON + rng.uniform(-0.3, 0.3), LAT + rng.uniform(-0.3, 0.3)
        for _k in range(rng.randint(1, 15)):
            obs.append(Obs(clon + rng.gauss(0, 0.004), clat + rng.gauss(0, 0.004),
                           T0 + timedelta(hours=rng.choice([0, 0, 12, 13, 30, 60])), "viirs"))
    a = cluster_obs(obs, max_gap_hours=24)
    b = cluster_obs_grid(obs, tolerance_m=1_125.0, max_gap_hours=24)

    def parts(events):
        return sorted(sorted((o.lon, o.lat, o.when) for o in e.detections) for e in events)

    assert parts(a) == parts(b)


# --- metrics ------------------------------------------------------------------------------

def test_metrics():
    assert clopper_pearson_upper(0, 30) == pytest.approx(0.0950, abs=1e-3)
    actions = {"a": "DOWNGRADE", "b": "ALERT", "c": "DOWNGRADE", "d": "ANALYST", "e": "HOLD"}
    labels = {"a": "DECLARED_BURN", "b": "WILDFIRE", "c": "ESCAPE", "d": "DECLARED_BURN"}
    dd = dangerous_downgrade(actions, labels)
    assert dd["n_dangerous"] == 2 and dd["n_downgraded"] == 1 and dd["downgraded_event_ids"] == ["c"]
    assert dd["note"]
    assert nuisance_reduction(actions, labels)["share_removed"] == 0.5
    s = season_summary(actions, labels, {"d": "2027-02-10"})
    assert s["recall_among_detected"] == 0.5
    assert s["analyst_load"]["peak_per_day"] == 1


def test_detection_coverage():
    docs = [{"id": 1, "lon": LON, "lat": LAT, "date": "2027-02-10", "size_class": "B"},
            {"id": 2, "lon": LON + 1, "lat": LAT, "date": "2027-02-10", "size_class": "A"}]
    pts = [{"lon": LON + 0.01, "lat": LAT, "date": "2027-02-11"}]
    c = detection_coverage(docs, pts)
    assert c["coverage"] == 0.5
    assert c["by_size_class"]["A"]["coverage"] == 0.0
