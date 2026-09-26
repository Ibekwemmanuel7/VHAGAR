# Post-fire structure-damage screen, validated against CAL FIRE DINS

The rapid-damage-assessment tier from `docs/21_PRODUCT_POSITIONING.md` and the product
review, built and validated on real data. The industry-standard post-fire task is to
classify each structure's damage from post-fire evidence and check it against ground
truth. The accepted US ground truth is CAL FIRE DINS; the ML community's taxonomy is
the xView2/xBD four-level scale (No Damage, Minor, Major, Destroyed).

Here the post-fire evidence is the MTBS burn-severity raster, so this is a mapped-burn
SCREEN, not a per-structure adjudication or an imagery-ML classifier. It is the honest
baseline every damage-assessment product is measured against, and it is a real,
skillful product in its own right.

## What it does

`scripts/postfire_damage_screen.py`, per named fire:

1. Loads the DINS inspections for that fire (the real 132k-row CAL FIRE master export),
   reduces `* Damage` to the four ordinal classes, and drops Inaccessible.
2. Samples the MTBS 2021 severity code at each structure (windowed read, reprojected
   to the raster CRS).
3. Predicts two binary destroyed screens and a four-class grade:
   - inside-burn: severity low, moderate, or high;
   - high-severity-only: severity high;
   - four-class grade: unburned to No Damage, low to Minor, moderate to Major, high to
     Destroyed.
4. Scores against DINS with the mandatory baselines.

The scoring core is `src/vhagar/eval/damage_screen.py`, pure numpy and CI-safe:
`binary_screen` (POD, FAR, precision, F1, accuracy, the predict-all-destroyed baseline,
and the F1 skill over it), `confusion`, and `ordinal_metrics` (accuracy, per-class
recall/precision, and the quadratic-weighted kappa, the standard ordered-category
agreement metric). Tests in `tests/test_damage_screen.py`.

## Result (real DINS + MTBS, 2021)

| fire | structures | inside-burn F1 | predict-all baseline | skill | 4-class accuracy | QWK |
|---|---|---|---|---|---|---|
| Caldor | 4,442 | 0.78 | 0.37 | **+0.41** | 0.74 | 0.80 |
| Dixie | 3,831 | 0.73 | 0.51 | **+0.22** | 0.53 | 0.50 |

The honest reading:

- The inside-burn screen **beats the predict-all-destroyed baseline on both fires**
  (skill +0.41 and +0.22). This is the opposite of the T5 WUI *predictive* loss model,
  which does not beat its baseline. A mapped-burn screen genuinely discriminates
  destroyed from surviving structures, because DINS carries many No-Damage structures
  outside the burn that the screen correctly clears. That is why post-fire evidence is
  product-ready while predictive property loss is not.
- The high-severity-only screen is worse (Caldor skill +0.18, Dixie -0.32): MTBS
  "high severity" is a vegetation-dNBR judgment, and many structures are destroyed in
  low or moderate vegetation severity, so restricting to high severity misses them.
  The inside-burn screen is the right operating point.
- The four-class grade is strong on Caldor (accuracy 0.74, QWK 0.80) and moderate on
  Dixie (0.53, 0.50). MTBS severity systematically over-grades structure damage
  (vegetation severity is not structure severity), so the grade is a screening prior,
  not a calibrated damage class.

## Honest scope

MTBS severity is a 30 m, dNBR-derived vegetation-severity product, not a structure
inspection, so this is a screen for inspection targeting, not a damage adjudication.
DINS is destroyed-heavy because inspectors concentrate near destruction, so the raw F1
is inflated by base rate and the **skill over the predict-all-destroyed baseline is the
honest number**. MTBS 2021 is an end-of-year mosaic, appropriate for these 2021 fires;
for a live event the same screen runs on VHAGAR's own T2 burned-area output or a rapid
satellite burn map instead of MTBS.

## Where this goes next (toward the SOTA)

The frontier product is per-structure damage from pre/post imagery on the xView2/xBD
4-class scale, which resolves what a vegetation-severity screen cannot. This module is
the mandatory baseline that any such model must beat, on the same DINS folds, with the
same skill-over-baseline discipline. Build order and status: (1) **done** the Evidence
Pack now screens on VHAGAR's own T2 scaled-RBR severity via `rbr_to_class_index`
(`--severity-scheme rbr`), with MTBS kept only as a validation reference (see
`docs/22_EVIDENCE_PACK.md`); (2) **done** building-footprint sampling instead of points
(footprint-to-footprint in the Evidence Pack); (3) train an imagery damage classifier
and score it against this screen and the predict-all baseline.

Figures `outputs/postfire_Caldor.png`, `outputs/postfire_Dixie.png`; summaries
`outputs/postfire_Caldor.json`, `outputs/postfire_Dixie.json`.
