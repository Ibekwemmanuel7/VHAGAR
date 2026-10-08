"""Review regressions (Sept 2026) for the serving freshness and window contracts.

9: a timezone-aware snapshot must not be labelled 'live'; unknown age is 'unknown'.
10: /api/events selects events by window but reports lifetime statistics; the
    response must say so.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest


def _load():
    pytest.importorskip("fastapi")
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("vhagar_api_fresh", root / "serve" / "vhagar_api.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _state(m, times):
    import pandas as pd
    df = pd.DataFrame({"t": times, "lon": -120.0, "lat": 38.0, "sensor": "GOES-18"})
    m._STATE = (df, [])
    m._STATE_SOURCE = "snapshot"


def test_old_aware_snapshot_is_stale_not_live():
    import pandas as pd
    m = _load()
    _state(m, pd.to_datetime(["2020-01-01T00:00:00Z"]))
    assert m._data_age_hours() > 24 * 365
    assert m._state_mode() == "stale"


def test_recent_naive_and_aware_agree():
    import pandas as pd
    m = _load()
    now = pd.Timestamp.now(tz="UTC")
    _state(m, pd.Series([now.tz_localize(None)]))
    a = m._data_age_hours()
    _state(m, pd.Series([now]))
    b = m._data_age_hours()
    assert a == pytest.approx(b, abs=0.01) and m._state_mode() == "live"


def test_unknown_age_is_not_live():
    import pandas as pd
    m = _load()
    _state(m, pd.Series([pd.NaT]))
    assert m._data_age_hours() is None
    assert m._state_mode() == "unknown"


def test_to_naive_utc_converts_offsets():
    import pandas as pd
    m = _load()
    s = m._to_naive_utc(pd.Series(["2026-09-01T12:00:00-05:00", "2026-09-01T17:00:00Z"]))
    assert s.dt.tz is None and s.iloc[0] == s.iloc[1] == pd.Timestamp("2026-09-01T17:00:00")


def test_events_declare_lifetime_stats_and_window(monkeypatch):
    import pandas as pd
    m = _load()
    t_old, t_new = pd.Timestamp("2026-09-01T00:00:00"), pd.Timestamp("2026-09-10T00:00:00")
    df = pd.DataFrame({"t": [t_old, t_new], "lon": [-120.0, -120.0], "lat": [38.0, 38.0],
                       "sensor": ["GOES-18", "GOES-18"]})
    ev = {"event_id": 1, "label": "Cluster 1", "centroid_lat": 38.0, "centroid_lon": -120.0,
          "n_detections": 2, "total_frp_mw": 20.0, "max_frp_mw": 12.0, "mean_frp_mw": 10.0,
          "perimeter_km": 1.0, "footprint_ha": 5.0, "first_seen": t_old.isoformat(),
          "last_seen": t_new.isoformat(), "sensors": ["GOES-18"],
          "geometry": [[[-120, 38], [-119.99, 38], [-119.99, 38.01], [-120, 38]]],
          "_t0": t_old, "_t1": t_new}
    monkeypatch.setattr(m, "get_state", lambda: (df, [ev]))
    monkeypatch.setattr(m, "_enrich_weather", lambda feats: "disabled-in-test")
    fc = m._events_fc("california", 1)
    p = fc["features"][0]["properties"]
    assert p["stats_scope"] == "event_lifetime" and p["starts_before_window"] is True
    md = fc["metadata"]
    assert md["window_start_utc"].startswith("2026-09-09") and "event_lifetime" in md["event_stats_scope"]
    assert md["detection_count"] == 1          # window-only count
