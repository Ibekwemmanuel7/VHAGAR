"""The self-hosted API's /v1/danger endpoint: three separate T3 quantities."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest


def _load_app():
    pytest.importorskip("sklearn")
    pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("vhagar_api", root / "serve" / "vhagar_api.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.slow
def test_v1_danger_returns_three_separate_quantities():
    m = _load_app()
    from starlette.testclient import TestClient
    c = TestClient(m.app)
    hot = c.get("/v1/danger?dryness=0.9&fuel=0.8&wind=0.7&temp=34&rh=12").json()
    wet = c.get("/v1/danger?dryness=0.1&fuel=0.1&wind=0.1&temp=15&rh=70&rainfall=8").json()
    # three separate quantities, never collapsed into one number
    assert "fire_danger" in hot and "fwi" in hot["fire_danger"]
    assert "ignition_probability" in hot and "expected_burned_area_ha" in hot
    # FWI is live-weather driven in either data source: hot/dry outranks cool/wet
    assert hot["fire_danger"]["fwi"] > wet["fire_danger"]["fwi"]

    if hot.get("data_source") == "synthetic-demo":
        # the demo ignition/E[BA] models respond to dryness/fuel
        assert hot["ignition_probability"] > wet["ignition_probability"]
        assert hot["expected_burned_area_ha"] >= wet["expected_burned_area_ha"]
    else:
        # real FPA-FOD path: ignition/E[BA] are month climatology, not weather.
        # They must be finite, positive, and vary seasonally (peak summer > winter).
        assert hot["ignition_probability"] > 0 and hot["expected_burned_area_ha"] > 0
        summer = c.get("/v1/danger?month=7").json()["ignition_probability"]
        winter = c.get("/v1/danger?month=1").json()["ignition_probability"]
        assert summer > winter   # real occurrence climatology peaks in fire season
