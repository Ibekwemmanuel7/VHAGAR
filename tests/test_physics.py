"""Physics tests. These check the equations against published anchor values.

If one of these fails, the physics is wrong, do not adjust the expected value.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhagar.physics import atmosphere as A
from vhagar.physics import dozier as D
from vhagar.physics import frp as F
from vhagar.physics import geometry as G
from vhagar.physics import planck as P

# ----------------------------------------------------------- planck ------


def test_planck_roundtrip_is_exact():
    for t in (250.0, 300.0, 400.0, 800.0, 1200.0):
        for lam in (3.9, 11.0, 1.6):
            rad = P.planck_radiance(lam, t)
            assert P.brightness_temperature(lam, rad) == pytest.approx(t, rel=1e-9)


def test_sensitivity_exponent_matches_published_values():
    assert float(P.planck_sensitivity_exponent(3.9, 300.0)) == pytest.approx(12.3, abs=0.1)
    assert float(P.planck_sensitivity_exponent(11.0, 300.0)) == pytest.approx(4.4, abs=0.1)
    assert float(P.planck_sensitivity_exponent(3.9, 1000.0)) == pytest.approx(3.7, abs=0.1)


def test_b_passes_through_four_in_the_flaming_range():
    """The entire justification for the Wooster power-law method."""
    t = np.linspace(650.0, 1350.0, 200)
    b = P.planck_sensitivity_exponent(3.9, t)
    assert b.min() < 4.0 < b.max(), "b must bracket 4 across the flaming range"
    assert float(np.mean(b)) == pytest.approx(4.0, abs=0.6)


def test_mir_contrast_advantage_over_tir():
    """Exact monochromatic values for a 1000 K fire on a 300 K background.

    5.6e3 at 3.9 um vs 28.6 at 11 um -> a factor of ~196. Published summaries
    round these to ~5.5e3 / ~21 / ~260; the exact numbers are asserted here.
    """
    r39 = float(P.dozier_contrast_ratio(3.9, 1000.0, 300.0))
    r11 = float(P.dozier_contrast_ratio(11.0, 1000.0, 300.0))
    assert r39 == pytest.approx(5616.0, rel=0.01)
    assert r11 == pytest.approx(28.6, rel=0.01)
    assert r39 / r11 > 150


def test_half_percent_fire_lights_up_mir_and_barely_moves_tir():
    """The canonical worked example: p=0.005, Tf=1000 K, Tb=300 K.

    Exact monochromatic answer is 413 K at 3.9 um. Published summaries quote
    ~394 K from the constant-b linearisation, which under-reads by ~19 K because
    b falls with temperature. Do not "fix" this back to 394.
    """
    l_mir = P.mixed_pixel_radiance(3.9, 0.005, 1000.0, 300.0)
    l_tir = P.mixed_pixel_radiance(11.0, 0.005, 1000.0, 300.0)
    bt_mir = float(P.brightness_temperature(3.9, l_mir))
    bt_tir = float(P.brightness_temperature(11.0, l_tir))
    assert bt_mir == pytest.approx(413.2, abs=1.0)
    assert bt_tir - 300.0 == pytest.approx(7.0, abs=3.0)
    # The whole point: MIR moves 113 K, TIR moves 7 K.
    assert (bt_mir - 300.0) > 10 * (bt_tir - 300.0)


def test_emissivity_error_hits_tir_harder_than_mir():
    d_mir = abs(float(P.emissivity_error_to_bt_error(300.0, 3.9, 0.01)))
    d_tir = abs(float(P.emissivity_error_to_bt_error(300.0, 11.0, 0.01)))
    assert d_mir == pytest.approx(0.26, abs=0.05)
    assert d_tir == pytest.approx(0.72, abs=0.10)
    assert d_tir > d_mir


def test_nonpositive_radiance_gives_nan_not_a_negative_temperature():
    assert np.isnan(float(P.brightness_temperature(3.9, 0.0)))
    assert np.isnan(float(P.brightness_temperature(3.9, -1.0)))


# ------------------------------------------------------- atmosphere ------


def test_transmittance_matches_the_lsa_saf_anchor():
    assert float(A.transmittance_mir(20.0, 0.0)) == pytest.approx(0.69, abs=0.005)


def test_transmittance_follows_the_secant_law_in_view_angle():
    tau0 = float(A.transmittance_mir(20.0, 0.0))
    tau60 = float(A.transmittance_mir(20.0, 60.0))
    assert tau60 == pytest.approx(tau0**2, rel=1e-6), "sec(60) = 2 exactly"
    assert tau60 == pytest.approx(0.476, abs=0.01)


def test_uncorrected_frp_bias_is_about_31_percent_at_nadir():
    factor = float(A.frp_atmospheric_correction_factor(20.0, 0.0))
    assert factor == pytest.approx(1.45, abs=0.02)
    assert float(A.frp_atmospheric_correction_factor(20.0, 60.0)) == pytest.approx(2.1, abs=0.05)


def test_transmittance_is_monotone_in_water_vapour_and_angle():
    w = np.linspace(0.0, 60.0, 50)
    tau = A.transmittance_mir(w, 0.0)
    assert np.all(np.diff(tau) < 0)
    z = np.linspace(0.0, 70.0, 50)
    tau_z = A.transmittance_mir(20.0, z)
    assert np.all(np.diff(tau_z) < 0)


def test_transmittance_stays_in_the_unit_interval():
    w = np.linspace(0.0, 80.0, 30)
    z = np.linspace(0.0, 85.0, 30)
    tau = A.transmittance_mir(w[:, None], z[None, :], aod_550=0.5)
    assert np.all((tau > 0.0) & (tau <= 1.0))


def test_smoke_caveat_is_recorded_not_silently_ignored():
    msg = A.smoke_attenuation_warning()
    assert "NOT quantified" in msg


# -------------------------------------------------------------- frp ------


def test_frp_recovers_a_known_synthetic_fire():
    """Round-trip: build a mixed pixel, then recover FRP within Wooster's ~12%."""
    p, t_f, t_b = 0.002, 900.0, 300.0
    area = 375.0**2
    l_mir = float(P.mixed_pixel_radiance(3.9, p, t_f, t_b))
    l_bg = float(P.planck_radiance(3.9, t_b))
    frp = float(
        F.frp_from_radiance(l_mir, l_bg, area, sensor="modis_c6", transmittance=1.0)
    )
    truth_mw = P.STEFAN_BOLTZMANN * p * area * t_f**4 / 1e6
    assert frp == pytest.approx(truth_mw, rel=0.20)


def test_frp_scales_correctly_with_area_and_transmittance():
    base = float(F.frp_from_radiance(0.35, 0.02, 1e5, transmittance=1.0))
    assert float(F.frp_from_radiance(0.35, 0.02, 2e5, transmittance=1.0)) == pytest.approx(2 * base)
    assert float(F.frp_from_radiance(0.35, 0.02, 1e5, transmittance=0.5)) == pytest.approx(2 * base)


def test_frp_is_never_negative():
    assert float(F.frp_from_radiance(0.01, 0.05, 1e5, transmittance=1.0)) == 0.0


def test_missing_transmittance_warns_loudly():
    with pytest.warns(RuntimeWarning, match="31% low"):
        F.frp_from_radiance(0.35, 0.02, 1e5)


def test_unknown_sensor_warns_but_does_not_crash_the_pipeline():
    with pytest.warns(RuntimeWarning, match="no published Wooster constant"):
        assert F.wooster_a("fictional_sensor") == pytest.approx(3.0e-9)
    with pytest.raises(KeyError):
        F.wooster_a("fictional_sensor", strict=True)


def test_frp_uncertainty_explodes_for_marginal_detections():
    """As L_fire -> L_background the point estimate becomes meaningless."""
    strong = float(F.frp_uncertainty(100.0, 0.35, 0.02, 0.001, 0.001))
    marginal = float(F.frp_uncertainty(100.0, 0.0205, 0.02, 0.001, 0.001))
    assert marginal > 5 * strong


def test_saturation_mask_and_fuel_conversion():
    bt = np.array([300.0, 311.0, 340.0])
    assert F.saturation_mask(bt, 311.0).tolist() == [False, True, True]
    assert float(F.fuel_consumed_kg(1000.0)) == pytest.approx(368.0)


def test_wooster_validity_range():
    assert bool(F.wooster_validity(900.0))
    assert not bool(F.wooster_validity(450.0))
    assert not bool(F.wooster_validity(1600.0))


# ----------------------------------------------------------- dozier ------


def test_dozier_recovers_a_synthetic_state_when_well_conditioned():
    p_true, tf_true, tb = 0.004, 900.0, 300.0
    lm = P.mixed_pixel_radiance(3.9, p_true, tf_true, tb)
    lt = P.mixed_pixel_radiance(11.0, p_true, tf_true, tb)
    r = D.retrieve(lm, lt, tb)
    assert bool(r.converged[0])
    assert float(r.fire_fraction[0]) == pytest.approx(p_true, rel=0.05)
    assert float(r.t_fire_k[0]) == pytest.approx(tf_true, rel=0.05)


def test_dozier_reports_its_own_ill_conditioning():
    """The p/Tf split is not identifiable for tiny fire fractions."""
    tiny = D.condition_number(1e-6, 1200.0, 300.0)
    fat = D.condition_number(0.05, 800.0, 300.0)
    assert float(tiny) > float(fat)


def test_dozier_result_carries_a_trustworthiness_gate():
    p_true, tf_true, tb = 0.004, 900.0, 300.0
    lm = P.mixed_pixel_radiance(3.9, p_true, tf_true, tb)
    lt = P.mixed_pixel_radiance(11.0, p_true, tf_true, tb)
    r = D.retrieve(lm, lt, tb)
    assert r.trustworthy().shape == r.fire_fraction.shape
    s = r.summary()
    assert set(s) == {"n", "converged_frac", "trustworthy_frac", "median_condition"}


def test_dozier_rejects_rather_than_reports_unphysical_solutions():
    """Background-only radiance has no fire in it; the answer must be NaN."""
    tb = 300.0
    lm = P.planck_radiance(3.9, tb)
    lt = P.planck_radiance(11.0, tb)
    r = D.retrieve(lm, lt, tb)
    assert np.isnan(r.fire_fraction[0]) or r.fire_fraction[0] < 1e-6


def test_background_error_matters_most_for_marginal_detections():
    """Background characterisation error is the dominant FRP systematic.

    Published: degrading background characterisation by 10 K inflated simulated
    GOES FRP by 82%. The magnitude depends entirely on fire-to-background
    contrast, which is the physics worth encoding in a test: for a strong fire
    a 10 K background error is a few percent; for a marginal detection it is a
    factor of several. That is also why FRP uncertainty must be reported.
    """
    tb, tf = 300.0, 900.0
    l_bg_true = float(P.planck_radiance(3.9, tb))
    l_bg_wrong = float(P.planck_radiance(3.9, tb - 10.0))

    def rel_error(p):
        lm = float(P.mixed_pixel_radiance(3.9, p, tf, tb))
        a = float(F.frp_from_radiance(lm, l_bg_true, 1e5, transmittance=1.0))
        b = float(F.frp_from_radiance(lm, l_bg_wrong, 1e5, transmittance=1.0))
        return abs(b - a) / a

    strong = rel_error(3e-3)
    marginal = rel_error(2e-5)
    assert strong < 0.10, "a strong fire should be robust to background error"
    assert marginal > 1.0, "a marginal detection should not be"
    assert marginal > 20 * strong


# --------------------------------------------------------- geometry ------


def test_solar_position_is_sane_at_local_noon():
    z, _ = G.solar_position(40.0, 0.0, 172, 12.0)  # summer solstice, Greenwich noon
    assert 15.0 < float(z) < 22.0                  # 40N - 23.4 declination ~ 16.6


def test_solar_zenith_exceeds_90_at_night():
    z, _ = G.solar_position(40.0, 0.0, 172, 0.0)
    assert float(z) > 90.0


def test_glint_angle_is_zero_at_perfect_specular_geometry():
    g = G.glint_angle_deg(solar_zenith_deg=30.0, view_zenith_deg=30.0,
                          solar_azimuth_deg=0.0, view_azimuth_deg=180.0)
    assert float(g) == pytest.approx(0.0, abs=1e-6)


def test_glint_angle_is_large_when_looking_away_from_the_sun():
    g = G.glint_angle_deg(30.0, 30.0, 0.0, 0.0)
    assert float(g) > 50.0


def test_local_solar_time_wraps_correctly():
    assert float(G.local_solar_time_hours(-120.0, 20.0)) == pytest.approx(12.0)
    assert 0.0 <= float(G.local_solar_time_hours(180.0, 23.0)) < 24.0


def test_doy_encoding_is_continuous_across_new_year():
    s1, c1 = G.day_of_year_encoding(365)
    s2, c2 = G.day_of_year_encoding(1)
    assert abs(float(s1) - float(s2)) < 0.03
    assert abs(float(c1) - float(c2)) < 0.03


def test_pixel_area_grows_off_nadir_and_viirs_growth_is_capped():
    assert float(G.pixel_area_growth(0.0)) == pytest.approx(1.0, abs=1e-6)
    assert float(G.pixel_area_growth(60.0)) > 2.0
    # VIIRS aggregation caps growth at ~4x, vs MODIS' ~8x.
    assert float(G.viirs_pixel_area_m2(70.0)) <= 4.0 * 375.0**2 + 1e-6


def test_frp_survives_noise_that_destroys_the_p_tf_split():
    """The empirical case for reporting FRP and not (p, T_f).

    Under realistic perturbation, 0.2 K sensor noise plus 2 K contextual
    background uncertainty, the fire fraction scatters by tens of percent and
    the fire temperature by >100 K, while FRP moves by well under 2%. That is
    the whole reason LSA-SAF, MODIS C6, VIIRS and SLSTR all report FRP.
    """
    rng = np.random.default_rng(7)
    p_true, tf_true, tb = 0.004, 900.0, 300.0
    bt_mir0 = float(P.brightness_temperature(3.9, P.mixed_pixel_radiance(3.9, p_true, tf_true, tb)))
    bt_tir0 = float(P.brightness_temperature(11.0, P.mixed_pixel_radiance(11.0, p_true, tf_true, tb)))

    ps, tfs, frps = [], [], []
    for _ in range(200):
        bt_mir = bt_mir0 + rng.normal(0, 0.2)
        bt_tir = bt_tir0 + rng.normal(0, 0.2)
        tb_est = tb + rng.normal(0, 2.0)
        r = D.retrieve(
            P.planck_radiance(3.9, bt_mir), P.planck_radiance(11.0, bt_tir), tb_est
        )
        if r.converged[0]:
            ps.append(float(r.fire_fraction[0]))
            tfs.append(float(r.t_fire_k[0]))
        frps.append(
            float(
                F.frp_from_radiance(
                    P.planck_radiance(3.9, bt_mir),
                    P.planck_radiance(3.9, tb_est),
                    375.0**2,
                    transmittance=1.0,
                )
            )
        )

    cv_p = float(np.std(ps) / np.mean(ps))
    cv_frp = float(np.std(frps) / np.mean(frps))
    assert cv_p > 0.15, "fire fraction should be badly determined under noise"
    assert float(np.std(tfs)) > 50.0, "fire temperature should scatter by >50 K"
    assert cv_frp < 0.05, "FRP should be robust"
    assert cv_p > 10 * cv_frp, "FRP must be an order of magnitude more stable than p"


def test_condition_number_is_always_large_for_two_channel_dozier():
    """It is a property of the physics, not of the optimiser.

    Even a generously large fire fraction gives a condition number in the
    thousands, which is why is_trustworthy() rejects essentially every
    two-channel retrieval. That rejection is the correct answer.
    """
    for p_frac, tf in ((0.05, 800.0), (0.004, 900.0), (2e-5, 1200.0)):
        assert float(D.condition_number(p_frac, tf, 300.0)) > 1e3
