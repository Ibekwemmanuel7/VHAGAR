# T5 — catastrophe loss: hazard -> exposure -> vulnerability -> loss -> EP curve

Status: **working generic CAT-loss pipeline on VHAGAR's own hazard output.** The
vulnerability curve is a documented logistic surrogate, swap in a curve fitted to
DINS damage classes (docs/17) for a calibrated model. This tier is deliberately
platform-neutral: it demonstrates the standard catastrophe-modelling chain in the
(re)insurance vocabulary, not any vendor system.

## Why this exists

The upstream tiers produce the **hazard**: where fire burns and how severely (T2/T4)
and, in the WUI, which structures are destroyed (`models/wui.py`). Catastrophe (CAT)
modelling turns hazard into the numbers an insurer prices on. The chain:

    hazard -> exposure -> vulnerability -> per-event loss -> loss catalogue -> EP curve + AAL

`models/loss.py` implements the loss half.

## The logic

**Exposure.** What is at risk and its value: building footprints with a replacement
value per structure (footprint area x regional cost, or a supplied value).

**Vulnerability** (`vulnerability_damage_ratio`). A damage ratio in [0, 1] as a
function of normalised hazard intensity at each structure, a logistic curve
(`midpoint` = intensity at 50% damage, `steepness` = sharpness). This is the link
that a DINS fit calibrates: DINS damage classes (None -> Affected -> Minor -> Major
-> Destroyed) map to damage ratios, and a curve fit to them replaces the default.

**Per-event loss.** `ground_up_loss(damage_ratio, exposure_value)` sums
`damage_ratio x value`. For the binary WUI output, `structure_event_loss(destroyed,
value)` is the value of destroyed structures (damage ratio 1).

**Catalogue + EP curves.** Given per-event losses and annual occurrence rates:

- **AAL** (`average_annual_loss`) = `sum_i rate_i x loss_i`, the expected annual loss,
  the headline pure-premium number.
- **OEP** (`oep_curve`), occurrence exceedance probability, the annual probability
  the *largest single event* exceeds a loss threshold. Closed form under a Poisson
  event process: `OEP(x) = 1 - exp(-sum_i rate_i [loss_i > x])`.
- **AEP** (`aep_curve`), aggregate exceedance probability, the annual probability the
  *year's total* loss exceeds a threshold. Estimated by Monte-Carlo simulation of
  Poisson event years, with the Monte-Carlo AAL returned alongside the analytic one
  as a built-in sanity check (they should match up to sampling error).

Both curves also return **return periods** (`1 / exceedance probability`), the
"1-in-N-year loss" framing insurers use.

## End-to-end

From a catalogue of simulated events (each a `wui_spread` run seeded from a
sampled ignition, with an annual rate), compute each event's `structure_event_loss`
against the exposure values, then feed the loss vector + rates to `oep_curve` /
`aep_curve`. That yields the AAL and the OEP/AEP curves for the portfolio, the CAT
deliverable, built directly on VHAGAR's physics-based hazard rather than a
black-box.

## What this is and isn't

It **is**: the complete, tested hazard->exposure->vulnerability->loss->EP pipeline,
in insurer vocabulary (AAL, OEP, AEP, return period), on VHAGAR's own hazard.

It **isn't**: calibrated (the vulnerability curve is a default until fit to DINS),
nor a replacement for a production CAT platform's exposure/financial engine
(insurance terms, reinsurance structures, secondary uncertainty). Those are the
financial-module next steps; the science-to-loss spine is here.
