# VHAGAR product positioning — where it can be sold honestly

**Positioning principle:** VHAGAR is currently a research-grade, physics-grounded
wildfire intelligence platform with several commercial *decision-support* products.
It is not yet a primary insurance catastrophe model, an approved rating model, or
an operational fire-perimeter forecast. Product claims must follow the validation
state of each tier.

The strongest near-term offer is therefore not “we predict every wildfire loss.” It
is: **we provide independently auditable evidence of wildfire occurrence, affected
area, and portfolio exposure, with clear provenance and uncertainty.**

## What is product-ready now

### 1. Incident intelligence and portfolio exposure monitoring

**Buyer:** insurers, MGAs, reinsurers, brokers, corporate risk teams, utilities,
and public-sector emergency teams.

**Buyer question:** “A wildfire is reported. Which assets may be affected, what is
the evidence, and how certain is it?”

**VHAGAR product:** an API/dashboard that fuses GOES, VIIRS, and MODIS thermal
detections into candidate and confirmed fire events; attaches observation times,
sensor provenance, and event conditions; intersects an event footprint or buffer
with a customer-provided location portfolio.

**Why it is sellable:** T1 uses real observations and the console already exposes
the operational ingredients: multi-sensor provenance, candidate-versus-confirmed
status, and freshness. It is useful before a final agency perimeter exists.

**Do not claim:** that the detection hull is a legal/agency fire perimeter, that
every candidate is a confirmed incident, or that the service proves cause of loss.

**Commercial outputs:** alert feed; CSV/API of potentially affected locations;
sensor-evidence timeline; confidence and data-freshness flags; event audit trail.

### 2. Post-fire burn-footprint and impact evidence

**Buyer:** claims teams, MGAs, parametric insurers, brokers, restoration firms,
public agencies, and risk managers.

**Buyer question:** “Which locations intersect the burned area, and where should
we triage first?”

**VHAGAR product:** a post-fire mapping service that produces a documented burned
area from optical imagery, compares it with authoritative references when
available, and intersects it with insured or customer locations. T2's
Sentinel-2/MTBS workflow, spectral baselines, and leakage-safe evaluation make this
the strongest model-development evidence in the platform.

**Why it is sellable:** the decision is bounded and verifiable: prioritize site
inspection, outreach, or claims review for locations inside or near a documented
burn footprint. It does not require VHAGAR to price a portfolio or predict a future
loss.

**Do not claim:** a building-damage determination from a burn-footprint overlap.
Intersection is triage evidence, not a coverage decision or a final DINS-style
damage assessment.

**Commercial outputs:** dated burned-area polygon/raster; location-intersection
table; commission/omission and source metadata; before/after imagery links; a
case-level evidence report.

### 3. Parametric-trigger verification and basis-risk review

**Buyer:** parametric insurers, MGAs, brokers, reinsurers, and corporate buyers of
parametric cover.

**Buyer question:** “Did a contract trigger occur, and can we independently
document the spatial and temporal evidence?”

**VHAGAR product:** a neutral verification layer that evaluates pre-agreed trigger
rules against independently sourced detections and burned-area products. Examples:
fire detected within a specified radius, event persistence for a stated duration,
or a final burned footprint intersecting a covered location.

**Why it is sellable:** it reduces operational ambiguity after an event. The value
is provenance, reproducibility, and a pre-agreed calculation—not a promise that
VHAGAR eliminates basis risk.

**Do not claim:** that a detection cluster alone settles a payment dispute. Contract
language must define data sources, timing, resolution, missing-data treatment,
appeal process, and the authoritative trigger calculation.

**Commercial outputs:** signed event-evidence packet; reproducible trigger run;
source/processing ledger; covered-location intersection; exception report.

### 4. Spread scenario and event-reconstruction studies

**Buyer:** specialty insurers, reinsurers, brokers, utilities, law firms,
engineering consultants, land managers, and resilience planners.

**Buyer question:** “Under stated wind, fuel, and ignition assumptions, what spread
path is physically plausible, and which inputs drive the result?”

**VHAGAR product:** a scoped professional-services study using Rothermel surface
fire rate of spread and an anisotropic Eikonal arrival-time solver. It can compare
scenarios, reconstruct observed progression, and make sensitivity to fuels, wind,
and assumptions explicit.

**Why it is sellable:** the physics core, replay harness, and validation framework
are real. A client may value a reproducible scenario analysis even where the result
is not an operational forecast.

**Do not claim:** a deterministic next-day perimeter, live suppression-aware
forecast, spotting-complete WUI forecast, or forensic proof of origin/cause. The
current operational replay has high overprediction against a sparse VIIRS proxy,
and the spatial fuel signal has not yet beaten simple baselines consistently.

**Commercial outputs:** assumptions register; scenario maps and arrival-time bands;
sensitivity table; observations-versus-model comparison; an explicit limitations
statement suitable for review.

### 5. Portfolio wildfire-data quality and model-risk review

**Buyer:** insurers, reinsurers, brokers, MGAs, lenders, and large property owners.

**Buyer question:** “Can we trust this wildfire model, vendor score, or exposure
dataset for the decision we want to make?”

**VHAGAR product:** independent model-risk and data-quality review. VHAGAR's
leakage-safe split machinery, baseline comparisons, calibration metrics,
provenance, and error-adjusted area methods are a tangible differentiator.

**Why it is sellable:** customers often own several vendor outputs but lack a
transparent way to test spatial leakage, stale sensor assumptions, uncalibrated
probabilities, or whether a complex model improves on a simple baseline.

**Do not claim:** regulatory approval, actuarial sign-off, or replacement of an
insurer's formal model-validation governance.

**Commercial outputs:** validation protocol; reproducible scorecard; bias/leakage
audit; data lineage review; benchmark report; prioritized remediation roadmap.

## Products to sell only after defined upgrades

### Property-level underwriting and mitigation score

This can become a strong insurance product after VHAGAR has a representative
structure inventory, building and defensible-space features, a calibration set that
includes undamaged structures, spatially blocked validation, and an explainable
mitigation counterfactual. The product should separate:

- **Exposure:** probability that a property encounters wildfire hazard in a stated
  period and scenario.
- **Vulnerability:** conditional probability or damage ratio given a defined ember,
  flame, and heat exposure.

Today, T5 is a research prototype. Its leave-one-fire-out F1 does not beat the
predict-all-destroyed baseline on the current DINS sample, so it must not be sold
as a predictive property-loss score.

### Portfolio catastrophe analytics

VHAGAR already has a tested hazard-to-exposure-to-vulnerability-to-loss pipeline
and can calculate illustrative AAL, OEP, and AEP curves. It becomes a commercial
portfolio-cat product only after it has: a stochastic event catalog with fitted
frequency and spatial dependence; validated exposure and vulnerability modules;
financial terms and secondary uncertainty; OED/ORD interoperability or an
equivalent governed financial-engine integration; and historical loss validation.

Until then, sell the catalog runner as an **R&D scenario and workflow capability**,
not as pricing, PML, capital, or reinsurance-placement advice.

### Real-time operational spread forecasting

This requires spatial and time-varying weather, pre-fire fuels, DEM slope and
aspect, fuel moisture, spotting, suppression treatment, uncertainty ensembles,
and validation against time-resolved agency perimeters. It is an excellent research
and partner-development roadmap, not a current product promise.

## Recommended market entry

Start with a paid pilot called **Wildfire Event Evidence Pack**:

1. A customer supplies a portfolio of coordinates or building footprints.
2. VHAGAR monitors and detects candidate/confirmed events.
3. After an event, VHAGAR produces a timestamped detection record, burn-footprint
   intersection, affected-location list, confidence/source metadata, and an
   optional spread-scenario appendix.
4. The customer measures operational value: time to triage, inspection targeting,
   claims-contact timing, or parametric-trigger reconciliation.

This is narrow enough to deliver now and creates the exposure, claims, and
validation data needed for later underwriting and catastrophe-model products.

## Product language to use

Use:

- “independent wildfire event evidence”
- “portfolio exposure monitoring and triage”
- “post-fire burn-footprint intersection”
- “reproducible parametric-trigger verification”
- “physics-grounded scenario analysis”
- “validation and model-risk review”

Avoid:

- “operational perimeter forecast”
- “claims adjudication”
- “property-loss prediction”
- “approved underwriting score”
- “replacement CAT model”
- “forensic cause determination”

## The one-sentence positioning

**VHAGAR helps insurers and risk owners turn wildfire observations into auditable
event evidence, burn-footprint exposure intelligence, and physics-grounded
scenario analysis—while keeping the uncertainty and decision limits explicit.**
