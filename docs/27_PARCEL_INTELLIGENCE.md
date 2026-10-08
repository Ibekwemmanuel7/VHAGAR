# 27. Parcel Evidence & Suitability (demo slice)

A small, self-contained vertical that scores a parcel for a proposed land use
(solar, wind, conservation, residential, data center) and returns a transparent,
auditable suitability result. It exists to demonstrate that VHAGAR's geospatial
discipline, imperfect-evidence handling, leakage-safe evaluation, provenance, and
honest refusal, transfers beyond wildfire to land-intelligence decisions.

This is a **demo slice on synthetic fixtures**. It is **not** a validated land
valuation (AVM), a bankable number, or a measured community-sentiment product, and
nothing here should be quoted as a performance claim for VHAGAR parcel suitability.
The wildfire platform (T1-T5) is unchanged; wildfire enters this slice only as one
hazard input (`wildfire_exposure_0_100`, sourced from VHAGAR T3/T4).

## What it does

Given a parcel geometry and a proposed use, it returns:

1. a transparent overall suitability score (0-100), or an explicit "cannot score"
   outcome when evidence is too thin;
2. **separate** component scores, physical/environmental, access/infrastructure
   proxy, hazard/ecological constraint, and market/community evidence, never
   collapsed into a single black-box aggregate;
3. provenance for every input (source, vintage, resolution, method, coverage,
   licence, evidence status);
4. a confidence/completeness statement (how many inputs are observed, derived,
   assumed, synthetic, or missing);
5. a plain-language explanation naming the strengths, constraints, and drivers.

## Module map (`src/vhagar/parcel/`)

| Module | Responsibility |
|---|---|
| `schemas.py` | Pure-stdlib dataclasses and enums: `ProposedUse`, `EvidenceStatus`, `Provenance`, `FeatureValue`, `Parcel` (computes centroid + equal-area hectares), `ComponentScore`, `Completeness`, `SuitabilityResult`. |
| `manifest.py` | The versioned feature manifest (data contract). Each feature declares the real dataset it would come from (USGS 3DEP, NREL NSRDB / WIND Toolkit, FEMA NFHL, HIFLD, TIGER, PAD-US, GAP, NLCD, VHAGAR T3/T4). In this build every value is `SYNTHETIC`. |
| `config.py` | The versioned, explainable scoring config: per-use component weights, feature weights, min-area gates, and transparent scoring curves (`ramp_up`, `ramp_down`, `band`, `frac_up`, `frac_down`, `category`). No magic constants in code. |
| `engine.py` | The scoring engine. Weighted-mean components over present inputs, renormalised, gated by `min_inputs`; overall withheld unless the area gate passes, the physical component scored, and scored components cover at least half the use's weight. Emits reason codes and a plain-language explanation. |
| `llm.py` | Provider-agnostic community/market-evidence extractor. `EvidenceExtractor` protocol + deterministic `MockEvidenceExtractor` (no API key). Structured cited output; guardrails for malformed output, missing citations, unsupported certainty, and prompt injection. Output is **evidence for human review, never ground truth**. |
| `evaluation.py` | Leakage-safe evaluation scaffold. Reuses `vhagar.eval.splits.spatial_block_split`; mandatory rules and null/majority baselines; `EvaluationReport` leads with the deployment question and label limitations. Synthetic fixtures only, no invented numbers. |
| `store.py` | `ParcelStore` protocol, `InMemoryParcelStore`, and an optional, documented `PostGISParcelStore` (psycopg imported lazily, not a hard dependency). |
| `service.py` | Ties store + engine + JSON serialisation for the API and console panel. |
| `fixtures.py` | Clearly-labelled synthetic demo parcels and feature sets. |

## The honesty mechanics

- **Evidence gating, not gap-filling.** A missing input is recorded as a reason
  code (`missing:<feature>`), never silently imputed. A component below its
  `min_inputs` returns `insufficient_evidence`, and the overall score is withheld
  rather than computed from a biased subset.
- **Provenance per input.** Every `FeatureValue` carries a `Provenance` record and
  an `EvidenceStatus` (observed / derived / assumed / synthetic / missing). The
  result's completeness statement counts them, so a reader always knows how real
  the inputs were (here: 0% real, 100% synthetic).
- **Components stay separate.** There is no single opaque aggregate; the four
  components are always returned individually with their weights, so a user can see
  *why*, and disagree with the weighting, rather than trust a number.
- **Hard area gate.** Each use declares a minimum parcel size; below it the screen
  refuses to score rather than extrapolating.
- **LLM as evidence, not oracle.** The market/community signal only becomes a
  scorable `market_support_status` after a human reviewer accepts the extraction
  (`reviewed_market_status`); otherwise it is `unknown` and the market component
  abstains. Instruction-like text in a source document is flagged as possible
  prompt injection and never followed; confidence is bounded and can only be
  lowered, never raised, by injected text.

## Evaluation design

Suitability labels are sparse, class-imbalanced, spatially autocorrelated, and
partly contested. `evaluate_spatial_block` therefore holds out **whole lat/lon
blocks** (not individual parcels) via the same split logic VHAGAR uses for fire
work, and compares the engine against two mandatory baselines:

- a **null/majority** baseline (predict the training-fold majority class), and
- a transparent **rules** baseline (suitable unless an obvious disqualifier is
  present).

`EvaluationReport` leads with the prediction unit, the deployment question, the
split method, and the standing label limitations, then reports accuracy /
precision / recall / F1 with **skill over the majority baseline**, plus known
failure modes (small-fold instability, engine abstentions excluded from metrics,
threshold as a product decision). The only data shipped is synthetic; real metrics
appear only when real, documented labels are supplied.

## API and console panel

- `GET /api/parcel/demo`, list the synthetic demo parcels and their bundled use.
- `POST /api/parcel/suitability`, body `{parcel_id[, use]}`, returns the full
  auditable result (separate components, provenance, completeness, reason codes,
  explanation, disclaimer).
- `GET /parcel`, a standalone panel (`vhagar_parcel.html`) kept **separate from the
  live wildfire console**, badged "DEMO · SYNTHETIC DATA".

## How to move this toward production (not done here)

1. **Ingest real features.** Replace `fixtures.synthetic_feature` calls with a real
   ingest per the manifest contract, flipping each `EvidenceStatus` to `OBSERVED`
   or `DERIVED` only when the source is wired and validated: slope from 3DEP,
   irradiance from NSRDB, wind from the WIND Toolkit, floodplain from NFHL,
   transmission/road distances from HIFLD/TIGER, protected overlap from PAD-US,
   habitat from GAP, developed cover from NLCD.
2. **Persist geometry in PostGIS.** Implement `PostGISParcelStore` against the
   documented schema (`geography(Polygon, 4326)`, `ST_*` queries).
3. **Obtain real labels and calibrate.** Only then run `evaluate_spatial_block` for
   real metrics, sweep the decision threshold, and add a calibration section for
   any probabilistic output.
4. **Wire a real LLM provider** behind the `EvidenceExtractor` protocol, keeping
   the guardrails and the human-review gate.

Until those steps are done, this slice demonstrates the *method*, transparent,
provenance-tracked, leakage-aware land screening that refuses to overclaim, not a
validated land-intelligence product.
