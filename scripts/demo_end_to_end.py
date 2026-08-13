"""A runnable tour of what VHAGAR already enforces.

    python scripts/demo_end_to_end.py

No network, no GDAL, no GPU. Everything here runs in the core install and is
covered by the test suite.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np

from vhagar.eval import splits as S
from vhagar.eval.area_estimation import estimate_areas
from vhagar.eval.baselines import persistence, persistence_with_buffer
from vhagar.eval.metrics import average_precision, burned_area_ratio, iou
from vhagar.features.fwi import FWIState, fwi_system
from vhagar.features.indices import classify_severity, dnbr, nbr, rbr
from vhagar.grid import AnalysisGrid
from vhagar.harmonize.fusion import Detection, cluster_detections, event_features


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# 1 -------------------------------------------------------------------------
rule("1. Analysis grid, one equal-area 375 m grid per region")
for region in ("conus", "canada", "europe"):
    g = AnalysisGrid(region)
    print(f"  {region:<8} {g.crs}  {g.n_x}x{g.n_y} = {g.n_tiles:>6,} tiles of 96 km")
t = AnalysisGrid("conus").tile(20, 15)
print(f"  example tile {t.tile_id}  core {t.bounds}  array {t.shape} (incl. 32-cell halo)")


# 2 -------------------------------------------------------------------------
rule("2. Splits, random splitting is structurally unavailable")
try:
    S.random_split([])
except NotImplementedError as exc:
    print(f"  random_split() -> NotImplementedError:\n    {str(exc)[:120]}...")

units = [
    S.SplitUnit(
        uid=f"fire{i:03d}",
        lon=-122.0 + (i % 8) * 3.0,
        lat=34.0 + (i // 8) * 2.5,
        when=date(2018 + i % 5, 8, 1 + i % 25),
        group=f"fire{i:03d}",
        ecoregion=f"eco{i % 3}",
        continent="NA",
    )
    for i in range(48)
]
for m in (S.leave_year_out(units), S.spatial_block_split(units, n_folds=4, block_degrees=5.0)):
    S.verify_no_overlap(m)
    print("\n" + S.summarise(m))


# 3 -------------------------------------------------------------------------
rule("3. Spectral severity. RBR is primary, dNBR kept for comparability")
pre_nir, pre_swir = 0.34, 0.09
post_nir, post_swir = 0.11, 0.31
n_pre, n_post = nbr(pre_nir, pre_swir), nbr(post_nir, post_swir)
print(f"  NBR pre  {float(n_pre):+.3f}   NBR post {float(n_post):+.3f}")
print(f"  dNBR     {float(dnbr(n_pre, n_post)):7.1f}")
print(f"  RBR      {float(rbr(n_pre, n_post)):7.1f}   -> class {int(classify_severity(rbr(n_pre, n_post)))}")
print("  RdNBR diverges as NBR_pre -> 0; RBR does not. That is why RBR is primary.")


# 4 -------------------------------------------------------------------------
rule("4. FWI System, 30 days of drought from season-start values")
state = FWIState.season_start()
for day in range(1, 31):
    out, state = fwi_system(29.0, 22.0, 24.0, 0.0, state, month=7)
    if day in (1, 5, 10, 20, 30):
        print(
            f"  day {day:>2}: FFMC {float(out['ffmc']):5.1f}  DMC {float(out['dmc']):6.1f}  "
            f"DC {float(out['dc']):6.1f}  ISI {float(out['isi']):5.1f}  "
            f"BUI {float(out['bui']):6.1f}  FWI {float(out['fwi']):6.1f}  DSR {float(out['dsr']):5.2f}"
        )
print("  Average DSR, never raw FWI.")


# 5 -------------------------------------------------------------------------
rule("5. Multi-sensor fusion, parallax tolerance is geometry, not tuning")
t0 = datetime(2026, 7, 15, 13, 0, tzinfo=UTC)
dets = [
    Detection("viirs", 0.0, 0.0, t0, frp_mw=14.0, bt_mir_k=338.0, bt_tir_k=299.0, landcover="forest"),
    Detection("goes", 1_600.0, 400.0, t0 + timedelta(minutes=25), frp_mw=52.0, landcover="forest"),
    Detection("goes", 1_800.0, 300.0, t0 + timedelta(minutes=55), frp_mw=91.0, landcover="forest"),
    Detection("viirs", 240_000.0, 0.0, t0, frp_mw=6.0, static_anomaly=True, landcover="other"),
]
events = cluster_detections(dets)
print(f"  {len(dets)} detections -> {len(events)} events")
for ev in events:
    f = event_features(ev)
    print(
        f"    {ev.event_id}: n={int(f['n_detections'])} sensors={int(f['n_sensors'])} "
        f"peak_frp={f['peak_frp_mw']:.0f} MW growth={f['frp_growth_mw_per_h']:+.1f} MW/h "
        f"static_anomaly={f['static_anomaly_fraction']:.0%}"
    )
print(f"  feature names: {sorted(event_features(events[0]))[:6]} ...")
print("  no lat/lon/x/y among them, spatial memorisation is excluded by construction")


# 6 -------------------------------------------------------------------------
rule("6. Baselines and metrics, persistence is the bar to clear")
rng = np.random.default_rng(0)
yesterday = np.zeros((64, 64), dtype=int)
yesterday[26:38, 26:38] = 1
today = np.zeros_like(yesterday)
today[24:40, 24:42] = 1                       # grew, mostly downwind

model_score = rng.random((64, 64)) * 0.2
model_score[23:41, 23:43] += 0.7

print(f"  persistence            IoU {iou(today, persistence(yesterday)):.3f}   "
      f"area ratio {burned_area_ratio(today, persistence(yesterday)):.2f}")
p2 = persistence_with_buffer(yesterday, 2)
print(f"  persistence + buffer   IoU {iou(today, p2):.3f}   "
      f"area ratio {burned_area_ratio(today, p2):.2f}   <- the baseline nobody reports")
print(f"  model (synthetic)       AP {average_precision(today, model_score):.3f}")
print("  Published SOTA on real next-day spread is AP 0.35-0.45. Persistence is 0.19.")


# 7 -------------------------------------------------------------------------
rule("7. Burned area, always an error-adjusted estimate with a CI")
conf = np.array([[96.0, 4.0], [12.0, 88.0]])
mapped = np.array([2_400_000.0, 143_200.0])   # hectares
for e in estimate_areas(conf, mapped, ["unburned", "burned"]):
    print(f"  {e}")
print("\n  The mapped 143,200 ha is a biased estimate. Never publish a pixel count.")

# 8 -------------------------------------------------------------------------
rule("8. Physics, why the mid-infrared channel exists")
from vhagar.physics.planck import (  # noqa: E402
    brightness_temperature,
    dozier_contrast_ratio,
    mixed_pixel_radiance,
    planck_sensitivity_exponent,
)

for lam, t in ((3.9, 300.0), (11.0, 300.0), (3.9, 1000.0)):
    print(f"  b({lam:>4} um, {t:6.0f} K) = {float(planck_sensitivity_exponent(lam, t)):5.2f}")
r39 = float(dozier_contrast_ratio(3.9, 1000.0, 300.0))
r11 = float(dozier_contrast_ratio(11.0, 1000.0, 300.0))
print(f"  1000 K fire / 300 K background:  3.9 um x{r39:,.0f}   11 um x{r11:.1f}   ratio {r39 / r11:.0f}")
for p_frac in (0.0005, 0.005, 0.05):
    bm = float(brightness_temperature(3.9, mixed_pixel_radiance(3.9, p_frac, 1000.0, 300.0)))
    bt = float(brightness_temperature(11.0, mixed_pixel_radiance(11.0, p_frac, 1000.0, 300.0)))
    print(f"  p={p_frac:6.4f} of a 1000 K fire ->  BT(3.9) {bm:6.1f} K   BT(11) {bt:6.1f} K")
print("  A sub-pixel fire moves MIR by ~100 K and TIR by ~7 K. Warm ground moves both.")


# 9 -------------------------------------------------------------------------
rule("9. Atmosphere, the largest correctable systematic in FRP")
from vhagar.physics.atmosphere import (  # noqa: E402
    frp_atmospheric_correction_factor,
    transmittance_mir,
)

print("  TCWV  view_zenith    tau    correction")
for tcwv in (5.0, 20.0, 45.0):
    for vz in (0.0, 40.0, 60.0):
        tau = float(transmittance_mir(tcwv, vz))
        print(f"  {tcwv:4.0f}  {vz:11.0f}  {tau:5.3f}    x{float(frp_atmospheric_correction_factor(tcwv, vz)):.2f}")
print("  Leaving tau=1 costs ~31% at nadir and >50% at 60 deg. It is deterministic.")


# 10 ------------------------------------------------------------------------
rule("10. Dozier sub-pixel retrieval, and its condition number")
from vhagar.physics.dozier import retrieve  # noqa: E402

for p_true, tf_true in ((0.05, 800.0), (0.004, 900.0), (2e-5, 1200.0)):
    lm = mixed_pixel_radiance(3.9, p_true, tf_true, 300.0)
    lt = mixed_pixel_radiance(11.0, p_true, tf_true, 300.0)
    r = retrieve(lm, lt, 300.0)
    ok = bool(r.trustworthy()[0])
    print(
        f"  true p={p_true:8.5f} Tf={tf_true:6.0f} -> "
        f"p={float(r.fire_fraction[0]):8.5f} Tf={float(r.t_fire_k[0]):6.0f} "
        f"cond={float(r.condition[0]):8.1f}  trustworthy={ok}"
    )
print()
print("  Condition numbers of 1e4-1e8 mean the MIR and TIR Jacobian rows are")
print("  nearly parallel. Noise-free synthetic data inverts exactly; real data")
print("  does not. Watch 0.2 K sensor noise plus 2 K background uncertainty. ")
print("  the background term is the documented dominant error:")

rng2 = np.random.default_rng(7)
p_true, tf_true, tb = 0.004, 900.0, 300.0
lm0 = float(mixed_pixel_radiance(3.9, p_true, tf_true, tb))
lt0 = float(mixed_pixel_radiance(11.0, p_true, tf_true, tb))
from vhagar.physics.frp import frp_from_radiance  # noqa: E402
from vhagar.physics.planck import planck_radiance  # noqa: E402

ps, tfs, frps = [], [], []
for _ in range(200):
    lm = float(brightness_temperature(3.9, lm0)) + rng2.normal(0, 0.2)
    lt = float(brightness_temperature(11.0, lt0)) + rng2.normal(0, 0.2)
    tb_est = tb + rng2.normal(0, 2.0)      # contextual-window uncertainty
    r = retrieve(planck_radiance(3.9, lm), planck_radiance(11.0, lt), tb_est)
    if r.converged[0]:
        ps.append(float(r.fire_fraction[0]))
        tfs.append(float(r.t_fire_k[0]))
    frps.append(float(frp_from_radiance(planck_radiance(3.9, lm), planck_radiance(3.9, tb_est),
                                        375.0**2, transmittance=1.0)))
print(f"    converged        : {len(ps)}/200 runs")
if ps:
    print(f"    fire fraction p : true {p_true:.5f}  retrieved {np.mean(ps):.5f} "
          f"+/- {np.std(ps):.5f}  ({100 * np.std(ps) / np.mean(ps):.0f}% spread)")
    print(f"    fire temp    Tf : true {tf_true:.0f} K  retrieved {np.mean(tfs):.0f} "
          f"+/- {np.std(tfs):.0f} K")
print(f"    FRP             : {np.mean(frps):.1f} +/- {np.std(frps):.1f} MW "
      f"({100 * np.std(frps) / np.mean(frps):.1f}% spread)  <- stable")
print()
print("  Published: GOES Dozier fire AREA correlated with reference at r = -0.22.")
print("  Report FRP. Gate (p, Tf) on the condition number, or do not report it.")


# 11 ------------------------------------------------------------------------
rule("11. Constellation, the 2026-11-01 cliff is in code, not a comment")
from datetime import date as _date  # noqa: E402

from vhagar.io.sensors import coverage_report, frp_to_reference_scale  # noqa: E402

print(coverage_report(_date(2026, 10, 1)))
print()
print(coverage_report(_date(2026, 12, 1)))
print(f"\n  NOAA-21 raw FRP 110 MW -> {float(frp_to_reference_scale(110.0, 'noaa21')):.1f} MW "
      "on the NOAA-20 scale (+10% instrument bias)")


# 12 ------------------------------------------------------------------------
rule("12. Physics features, coordinates cannot reach the model")
from vhagar.features.physics_features import (  # noqa: E402
    FORBIDDEN_FEATURES,
    PhysicsInputs,
    build_features,
)

inp = PhysicsInputs(
    bt_mir_k=np.array([400.0, 312.0]),
    bt_tir_k=np.array([302.0, 310.0]),
    bt_mir_background_k=np.full(2, 300.0),
    bt_tir_background_k=np.full(2, 300.0),
    latitude_deg=np.full(2, 35.0),
    longitude_deg=np.full(2, -110.0),
    day_of_year=np.full(2, 200),
    utc_hour=np.full(2, 21.0),
    view_zenith_deg=np.zeros(2),
    view_azimuth_deg=np.zeros(2),
    solar_zenith_deg=np.full(2, 25.0),
    solar_azimuth_deg=np.full(2, 180.0),
    tcwv_kg_m2=np.full(2, 15.0),
)
X, fnames = build_features(inp)
print(f"  {X.shape[1]} features, {X.shape[0]} samples")
print(f"  forbidden names blocked: {sorted(FORBIDDEN_FEATURES)[:8]} ...")
ratio = X[:, fnames.index("mir_tir_excess_ratio")]
print(f"  sub-pixel fire  mir/tir excess ratio = {ratio[0]:6.1f}")
print(f"  warm bare ground mir/tir excess ratio = {ratio[1]:6.1f}")
print("  Same discriminant a contextual algorithm uses, handed to the model directly.")

print("\nDone. See docs/02_VALIDATION.md and docs/03_PHYSICS.md.\n")
