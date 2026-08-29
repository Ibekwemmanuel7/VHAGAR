"""Serving regression tests for Phase 1 fixes:

* the FileNotFoundError -> 503 handler, so data-less endpoints degrade gracefully
  instead of returning a bare 500;
* the /api/candidates event-suppression radius, which must be the event radius
  (perimeter / 2*pi), not half the perimeter (which over-suppressed by ~pi).
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest


def _load_app():
    pytest.importorskip("sklearn")
    pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    pytest.importorskip("pandas")
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("vhagar_api", root / "serve" / "vhagar_api.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.slow
@pytest.mark.parametrize("path", [
    "/api/detections?region=conus&days=1",
    "/api/events?region=conus&days=1",
    "/api/candidates?region=conus&days=1",
])
def test_endpoints_return_503_when_no_state(monkeypatch, path):
    m = _load_app()
    from starlette.testclient import TestClient

    def _no_state():
        raise FileNotFoundError("snapshot not available")

    monkeypatch.setattr(m, "get_state", _no_state)
    c = TestClient(m.app, raise_server_exceptions=False)
    r = c.get(path)
    assert r.status_code == 503
    assert r.json().get("status") == "no_data"


def _crafted_state(m):
    """One confirmed event (perimeter 100 km at -98, 33) plus two GOES detections:
    one just outside the true radius and one inside it. No polar detections."""
    import pandas as pd

    now = pd.Timestamp.utcnow().tz_localize(None)
    df = pd.DataFrame({
        "t": [now, now],
        "lon": [-97.75, -98.05],   # 0.25 deg away (outside), 0.05 deg away (inside)
        "lat": [33.0, 33.0],
        "sensor": ["GOES-19", "GOES-19"],
        "frp_mw": [25.0, 40.0],
        "temp_k": [330.0, 340.0],
    })
    evs = [{"centroid_lon": -98.0, "centroid_lat": 33.0,
            "perimeter_km": 100.0, "_t1": now}]
    return df, evs


@pytest.mark.slow
def test_candidates_suppression_radius_is_perimeter_over_2pi(monkeypatch):
    m = _load_app()
    from starlette.testclient import TestClient

    df, evs = _crafted_state(m)
    monkeypatch.setattr(m, "get_state", lambda: (df, evs))
    c = TestClient(m.app)
    feats = c.get("/api/candidates?region=conus&days=1").json()["features"]
    lons = {round(f["geometry"]["coordinates"][0], 2) for f in feats}

    # true event radius = 100 / (2*pi) / 111 deg + pad ~ 0.163 deg.
    # -97.75 is 0.25 deg out (outside) -> kept; the old perimeter/2 (~0.48 deg)
    # would have wrongly suppressed it.
    assert -97.75 in lons, "detection beyond the event radius must not be suppressed"
    # -98.05 is 0.05 deg in (inside the radius) -> suppressed.
    assert -98.05 not in lons, "detection inside the event radius must be suppressed"
    # every returned candidate is explicitly unconfirmed
    assert all(f["properties"]["status"] == "unconfirmed" for f in feats)


@pytest.mark.slow
def test_refresh_keeps_live_state_when_only_demo_reachable(monkeypatch):
    """The downgrade guard: a refresh that can only reach the frozen demo must not
    clobber a good live feed, and it must leave state and source consistent."""
    m = _load_app()
    monkeypatch.setattr(m, "_STATE", ("LIVE_DF", "LIVE_EV"))
    monkeypatch.setattr(m, "_STATE_SOURCE", "snapshot")

    def _build_demo():
        m._build_ctx.source = "frozen"
        return ("DEMO_DF", "DEMO_EV")

    monkeypatch.setattr(m, "_build_state", _build_demo)
    m.refresh_state()
    assert m._STATE == ("LIVE_DF", "LIVE_EV")   # live state preserved
    assert m._STATE_SOURCE == "snapshot"        # provenance preserved, not "frozen"


@pytest.mark.slow
def test_refresh_swaps_state_and_source_together(monkeypatch):
    """A successful refresh swaps state and its provenance atomically (both new)."""
    m = _load_app()
    monkeypatch.setattr(m, "_STATE", ("OLD_DF", "OLD_EV"))
    monkeypatch.setattr(m, "_STATE_SOURCE", "snapshot")

    def _build_new():
        m._build_ctx.source = "snapshot"
        return ("NEW_DF", "NEW_EV")

    monkeypatch.setattr(m, "_build_state", _build_new)
    m.refresh_state()
    assert m._STATE == ("NEW_DF", "NEW_EV")
    assert m._STATE_SOURCE == "snapshot"
