"""T4 batch event-catalog runner: determinism, worker-invariance, fuel physics, EP aggregate."""
from __future__ import annotations

from vhagar.eval.catalog import (
    load_dins_fire,
    make_catalog,
    run_catalog,
    run_event,
    summarize,
)

_DINS = (
    "* Damage,* Incident Name,Assessed Improved Value (parcel),Latitude,Longitude\n"
    "Destroyed (>50%),Test,500000,34.10,-118.20\n"
    "No Damage,Test,300000,34.11,-118.21\n"
    "Destroyed (>50%),Test,9999999999,34.12,-118.22\n"     # outlier, winsorized
    "Destroyed (>50%),Other,400000,35.0,-119.0\n"          # different incident
    "Destroyed (>50%),Test,,34.13,-118.23\n"               # missing value -> 0
)


def test_make_catalog_spans_fuels_and_carries_rates():
    specs = make_catalog(n_events=12, seed=0, base_lambda=6.0)
    assert len(specs) == 12
    assert {s.fuel for s in specs} == {"grass", "shrub", "timber", "mixed"}
    assert abs(sum(s.rate for s in specs) - 6.0) < 1e-9      # rates sum to base lambda
    assert all(1.5 <= s.wind_speed <= 11.0 for s in specs)


def test_run_event_shape_and_bounds():
    spec = make_catalog(n_events=4, seed=1, grid=60)[0]
    r = run_event(spec)
    assert set(r) >= {"event_id", "fuel", "rate", "burned_ha", "loss", "runtime_ms", "n_reached"}
    assert r["burned_ha"] > 0.0            # a fire always burns some area
    assert r["loss"] >= 0.0


def test_run_event_is_seed_pure():
    spec = make_catalog(n_events=4, seed=2, grid=60)[1]
    a, b = run_event(spec), run_event(spec)
    assert a["burned_ha"] == b["burned_ha"] and a["loss"] == b["loss"]


def test_catalog_results_invariant_to_worker_count():
    specs = make_catalog(n_events=8, seed=3, grid=60)
    r1 = run_catalog(specs, workers=1)
    r2 = run_catalog(specs, workers=2)
    assert [x["loss"] for x in r1] == [x["loss"] for x in r2]
    assert [x["burned_ha"] for x in r1] == [x["burned_ha"] for x in r2]


def test_fuel_drives_spread_grass_over_timber():
    # same run, compare mean burned area by fuel: flashy grass must out-spread slow timber
    specs = make_catalog(n_events=40, seed=4, grid=60)
    summ = summarize(run_catalog(specs, workers=1))
    assert summ["by_fuel"]["grass"]["mean_burned_ha"] > summ["by_fuel"]["timber"]["mean_burned_ha"]


def test_summarize_ep_aggregate():
    specs = make_catalog(n_events=20, seed=5, grid=60)
    summ = summarize(run_catalog(specs, workers=1))
    assert summ["n_events"] == 20 and summ["synthetic"] is True
    assert summ["aal"] > 0.0
    rpl = summ["return_period_loss"]
    assert set(rpl) == {10, 25, 50, 100, 250}
    # return-period loss is non-decreasing with return period (rarer => at least as large)
    vals = [rpl[k] for k in (10, 25, 50, 100, 250)]
    assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:], strict=False))


def test_load_dins_fire_filters_incident_and_winsorizes(tmp_path):
    p = tmp_path / "dins.csv"
    p.write_text(_DINS, encoding="utf-8")
    lat, lon, value, destroyed = load_dins_fire(str(p), "Test")
    assert lat.size == 4                       # only the four "Test" rows
    assert destroyed.tolist() == [True, False, True, True]
    assert value.max() < 9999999999           # the outlier was winsorized down
    assert value.min() == 0.0                  # the missing value became 0


def test_summary_deterministic_across_runs():
    specs = make_catalog(n_events=16, seed=6, grid=60)
    s1 = summarize(run_catalog(specs, workers=1))
    s2 = summarize(run_catalog(specs, workers=1))
    assert s1["aal"] == s2["aal"]
    assert s1["burned_ha"] == s2["burned_ha"]
