from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from vhagar.features import physics_features as PF
from vhagar.io import sensors as S

# --------------------------------------------------------- sensors ------


def test_all_jpss_platforms_carry_the_same_instrument():
    """VIIRS is an instrument. NOAA-20/21/22 and S-NPP are the spacecraft."""
    for key in ("snpp", "noaa20", "noaa21", "noaa22"):
        assert S.PLATFORMS[key].sensor == "viirs"
        assert S.get_sensor(key).name.startswith("VIIRS")


def test_snpp_drops_out_on_the_cessation_date():
    before = {p.name for p in S.active_platforms(date(2026, 10, 31))}
    after = {p.name for p in S.active_platforms(date(2026, 11, 1))}
    assert "Suomi-NPP" in before
    assert "Suomi-NPP" not in after
    assert "NOAA-21 (JPSS-2)" in after


def test_viirs_platform_count_halves_and_the_report_says_so():
    report = S.coverage_report(date(2026, 12, 1))
    assert "~50 min" in report
    leo = S.active_platforms(date(2026, 12, 1), kind="leo")
    viirs = [p for p in leo if p.sensor == "viirs"]
    assert len(viirs) == 2


def test_noaa21_frp_bias_is_applied_not_ignored():
    """Splicing platforms without this correction creates a fake trend step."""
    raw = 110.0
    corrected = float(S.frp_to_reference_scale(raw, "noaa21"))
    assert corrected == pytest.approx(100.0, rel=1e-9)
    assert float(S.frp_to_reference_scale(raw, "noaa20")) == pytest.approx(raw)


def test_noaa21_has_no_earth_engine_collection():
    """A live single point of failure that must not be silently assumed away."""
    assert S.PLATFORMS["noaa21"].access.get("gee") == ""
    assert S.PLATFORMS["noaa20"].access["gee"].startswith("NASA/LANCE/")


def test_slstr_saturation_is_the_documented_daytime_problem():
    slstr = S.SENSORS["slstr"]
    assert slstr.mir_saturation_k == pytest.approx(311.0)
    assert slstr.mir_saturation_k < S.SENSORS["viirs"].mir_saturation_k
    assert slstr.has_swir, "S5/S6 are what make the gas-flare discriminant possible"


def test_sensors_with_swir_can_do_the_flare_discriminant():
    with_swir = {k for k, v in S.SENSORS.items() if v.has_swir}
    assert {"slstr", "abi", "fci"} <= with_swir
    assert "modis" not in with_swir


def test_modis_is_marked_as_ending():
    assert S.PLATFORMS["terra"].status == "ending"
    assert S.PLATFORMS["aqua"].status == "ending"
    assert not any(
        p.sensor == "modis" for p in S.active_platforms(date(2027, 6, 1))
    )


def test_unknown_platform_raises():
    with pytest.raises(KeyError, match="unknown platform"):
        S.get_sensor("hubble")


# -------------------------------------------------- physics features ------


def _inputs(n: int = 5) -> PF.PhysicsInputs:
    rng = np.random.default_rng(0)
    return PF.PhysicsInputs(
        bt_mir_k=330.0 + rng.random(n) * 40,
        bt_tir_k=298.0 + rng.random(n) * 6,
        bt_mir_background_k=np.full(n, 304.0),
        bt_tir_background_k=np.full(n, 297.0),
        latitude_deg=np.linspace(34.0, 48.0, n),
        longitude_deg=np.linspace(-122.0, -100.0, n),
        day_of_year=np.full(n, 200),
        utc_hour=np.full(n, 21.0),
        view_zenith_deg=np.linspace(0.0, 55.0, n),
        view_azimuth_deg=np.full(n, 100.0),
        solar_zenith_deg=np.linspace(20.0, 60.0, n),
        solar_azimuth_deg=np.full(n, 200.0),
        tcwv_kg_m2=np.full(n, 18.0),
    )


def test_feature_matrix_shape_and_names_agree():
    x, names = PF.build_features(_inputs(7))
    assert x.shape == (7, len(names))
    assert len(set(names)) == len(names), "duplicate feature names"


def test_no_coordinate_feature_can_reach_the_model():
    _, names = PF.build_features(_inputs())
    assert not (set(n.lower() for n in names) & PF.FORBIDDEN_FEATURES)


@pytest.mark.parametrize("bad", ["lat", "longitude", "x", "h3", "tile_x", "geohash"])
def test_forbidden_feature_guard_actually_fires(bad):
    with pytest.raises(ValueError, match="forbidden coordinate"):
        PF.assert_no_forbidden_features(["bt_mir_k", bad])


def test_geometry_features_are_present_and_finite():
    x, names = PF.build_features(_inputs())
    for key in ("glint_angle_deg", "air_mass_factor", "transmittance_mir", "local_solar_time_h"):
        col = x[:, names.index(key)]
        assert np.all(np.isfinite(col)), f"{key} produced non-finite values"


def test_transmittance_feature_falls_with_view_angle():
    x, names = PF.build_features(_inputs())
    tau = x[:, names.index("transmittance_mir")]
    assert tau[0] > tau[-1], "view zenith increases across the fixture"


def test_missingness_is_indicated_not_imputed():
    """SWIR is missing-not-at-random: smoke degrades it more than MIR."""
    x, names = PF.build_features(_inputs())
    assert np.isnan(x[:, names.index("swir16_mir_ratio")]).all()
    x2, names2 = PF.missingness_indicators(x, names)
    assert "missing__swir16_mir_ratio" in names2
    assert x2.shape[1] > x.shape[1]
    assert x2[:, names2.index("missing__swir16_mir_ratio")].all()


def test_excess_ratio_separates_a_subpixel_fire_from_a_warm_surface():
    """A uniform warm surface raises MIR and TIR together; a fire does not."""
    n = 2
    inp = PF.PhysicsInputs(
        # sample 0: sub-pixel fire (MIR way up, TIR barely).
        # sample 1: uniformly warm ground (both up by similar amounts).
        bt_mir_k=np.array([400.0, 312.0]),
        bt_tir_k=np.array([302.0, 310.0]),
        bt_mir_background_k=np.full(n, 300.0),
        bt_tir_background_k=np.full(n, 300.0),
        latitude_deg=np.full(n, 35.0),
        longitude_deg=np.full(n, -110.0),
        day_of_year=np.full(n, 200),
        utc_hour=np.full(n, 21.0),
        view_zenith_deg=np.zeros(n),
        view_azimuth_deg=np.zeros(n),
        solar_zenith_deg=np.full(n, 25.0),
        solar_azimuth_deg=np.full(n, 180.0),
        tcwv_kg_m2=np.full(n, 15.0),
    )
    x, names = PF.build_features(inp)
    ratio = x[:, names.index("mir_tir_excess_ratio")]
    assert ratio[0] > 20, "sub-pixel fire: MIR excess >> TIR excess"
    assert ratio[1] < 3, "warm ground: MIR and TIR move together"
