# 27. Parcel Evidence & Suitability (demo slice)

A small, self-contained vertical that screens a parcel for a proposed land use (solar,
wind, conservation, residential, data center) and returns a transparent, auditable
result. It exists to show that VHAGAR's discipline (provenance on every input,
leakage-safe evaluation, and refusal to claim more than the evidence supports)
transfers from wildfire to land-intelligence decisions.

This is a **demo slice on synthetic fixtures**. It is **not** a validated land valuation
(AVM), a bankable number, or a measured community-sentiment product, and nothing here is
a performance claim. The wildfire platform (T1-T5) is unchanged; wildfire enters this
slice only as one hazard input (`wildfire_exposure_0_100`, sourced from VHAGAR T3/T4).

## What it returns

For a parcel geometry and a proposed use:

1. an **overall status**: `scored`, `ineligible` (a hard exclusion or the minimum-area
   gate fired; no score), or `insufficient_evidence` (a required input is missing; no
   score, and the missing inputs are named);
2. an **evidence grade**: `evidence_based` (every required input observed or derived),
   `provisional` (some assumed), `illustrative` (some synthetic), or `no_evidence`;
3. **separate component scores** (physical/environmental, access/infrastructure,
   hazard/ecological constraint) with fixed, versioned weights;
4. **community/market evidence** reported on its own, never blended into the score;
5. provenance for every input, a completeness count by evidence status, reason codes,
   and a plain-language explanation that leads with the evidence grade.

## Module map (`src/vhagar/parcel/`)

| Module | Responsibility |
|---|---|
| `schemas.py` | Dataclasses and enums (`ProposedUse`, `EvidenceStatus`, `Provenance`, `FeatureValue`, `Parcel`, `ComponentScore`, `Completeness`, `SuitabilityResult`). Ring normalisation and validation (range, self-intersection, zero area), area-weighted centroid, geodesic area via `pyproj.Geod` when installed, local equal-area fallback otherwise; the method is recorded. |
| `manifest.py` | Versioned feature manifest (data contract): source, vintage, resolution, method, coverage, licence, and status per feature (USGS 3DEP, NREL NSRDB / WIND Toolkit, FEMA NFHL, HIFLD, TIGER, PAD-US, GAP, NLCD, VHAGAR T3/T4). |
| `config.py` | Versioned scoring config (`parcel-scoring-0.2.0`): per use, the required components, fixed feature weights, scoring curves, hard exclusions, and minimum area. |
| `engine.py` | Decision order: area gate, hard exclusions, required evidence, then a fixed-weight score. No renormalisation around missing inputs. |
| `llm.py` | Provider-agnostic evidence extraction: `EvidenceExtractor` protocol, `CallableLLMExtractor` (any `prompt -> text` function), deterministic `MockEvidenceExtractor`, `parse_extraction` for raw model output, and `validate_extraction` / `finalize` guardrails checked against the source document. |
| `evaluation.py` | Leakage-safe evaluation: spatial-block, leave-one-group-out, or leave-year-out splits from `vhagar.eval.splits`; coverage and abstention reporting; engine vs majority and rules baselines on identical parcels; F1, balanced accuracy, and PR-AUC. |
| `store.py` | `ParcelStore` protocol, `InMemoryParcelStore`, and an optional `PostGISParcelStore` (psycopg imported lazily; not a hard dependency). |
| `service.py` | Store + engine + JSON serialisation; parses caller-supplied polygons and features. |
| `fixtures.py` | Clearly labelled synthetic demo parcels, including a hard-exclusion case and a thin-evidence case. |

## The evidence contract

- **Every listed input is required.** A component is scored only when all of its
  features are present, and its feature weights are fixed. An overall score requires the
  physical, access, and hazard components and every exclusion input. Missing evidence
  produces `insufficient_evidence` with the missing inputs named; it is never estimated,
  skipped, or renormalised away, so it can never raise a score.
- **Hard exclusions are configuration.** For example, solar is `ineligible` when
  protected-area overlap exceeds 10% or floodplain overlap exceeds 25%; data center is
  `ineligible` above 10% floodplain overlap. Thresholds are screening choices recorded in
  the versioned config for audit, not legal or regulatory determinations. An exclusion
  whose input is missing cannot be cleared.
- **The score says how real it is.** A number computed from synthetic inputs is labelled
  "Illustrative demo result" in the explanation, the API (`evidence_grade`), and the page.
- **Market evidence stays separate.** Community and market evidence is contested,
  LLM-assisted, and human-reviewed. It is reported next to the score and never changes it.
- **LLM output is evidence for review, not an oracle.** A cited `text_span` must appear
  verbatim (whitespace-normalised) in the source document; `doc_id` and date must match;
  a span that is itself instruction text is rejected; malformed JSON, missing fields,
  wrong types, and provider errors become `abstain`. Any failure forces `abstain` and caps
  confidence at 0.3. Only `reviewed_market_status` with explicit reviewer acceptance
  produces a `market_support_status` value.

## Adversarial review (2026-10-08)

The first version (commit `9fe64dc`) passed its own tests but failed ordinary,
off-fixture product use. Each counterexample below is now a regression test in
`tests/test_parcel.py`.

| Counterexample | Before | After |
|---|---|---|
| Solar parcel scored for **wind** with no wind-resource input | 91.5, "strong" (physical scored from slope alone) | `insufficient_evidence`, missing `mean_wind_ms_100m` |
| All hazard inputs removed | score rose from 87.2 to 87.9; explanation silent | `insufficient_evidence`, missing inputs named |
| Transmission distance removed | access rose to 100 | `insufficient_evidence` |
| Parcel 100% in a protected area and 100% in floodplain (solar) | 79.2, "strong" | `ineligible` (protected_area_overlap, floodplain_overlap) |
| Every input an assumed default | full score, no warning | score labelled `provisional` |
| LLM extraction quoting text that is not in the document | passed validation | `citation_not_in_source`, forced `abstain` |
| Evaluation with engine abstentions | engine and baselines on different parcel sets; accuracy-based skill | identical parcels, coverage and abstention rate reported, F1 / balanced-accuracy / PR-AUC |

The fix in each case changed the product contract (what the engine is allowed to claim),
not the score.

## Evaluation design

Suitability labels are sparse, imbalanced, spatially autocorrelated, and partly
contested. `evaluate_suitability` holds out whole spatial blocks (or groups, or label
years) and reports, in order: the deployment question, the split, **coverage** (how many
held-out parcels the engine decided) and the abstention rate, the standing label
limitations, then metrics on two sets:

- **matched**: the identical parcels the engine decided, for the engine, the
  null/majority baseline, and the transparent rules baseline;
- **full**: every held-out parcel, with an engine abstention counted as "not suitable".

Metrics are precision, recall, F1, balanced accuracy, and the engine's average precision
(PR-AUC), with skill expressed as differences against both baselines. The engine output
is a 0-100 score, not a probability, so calibration is not reported until scores are
mapped to probabilities on real labels. The only labels shipped are synthetic; real
metrics appear only when real, documented labels are supplied.

## API and page

- `GET /api/parcel/demo`: list the synthetic demo parcels and their bundled use.
- `POST /api/parcel/suitability`:
  - demo mode, `{parcel_id[, use]}`;
  - caller-polygon mode, `{geometry, use, features?, parcel_id?}` where `geometry` is a
    GeoJSON Polygon (outer ring) or a `[[lon, lat], ...]` ring and `features` maps feature
    names to `{value, status, source?, vintage?}`. No feature pipeline is connected, so
    nothing is looked up or filled; without supplied features the result is
    `insufficient_evidence`. Caller-declared statuses are recorded as declared and are not
    verified. Invalid geometry, unknown features, and invalid statuses return 400.
- `GET /parcel`: a standalone page (`vhagar_parcel.html`), kept separate from the live
  wildfire console, badged "DEMO · SYNTHETIC DATA".

## Toward production (not done here)

1. **Ingest real features** per the manifest contract (3DEP slope, NSRDB irradiance,
   WIND Toolkit, NFHL, HIFLD/TIGER distances, PAD-US, GAP, NLCD), flipping each status to
   `OBSERVED` or `DERIVED` only when the source is wired and validated. Do the overlays in
   PostGIS or GDAL/OGR, not in the lightweight geometry helpers here.
2. **Persist parcels in PostGIS** via `PostGISParcelStore` (`geography(Polygon, 4326)`).
3. **Collect real labels**, run `evaluate_suitability`, sweep the decision threshold, and
   add calibration if scores are mapped to probabilities.
4. **Wire a real LLM provider** through `CallableLLMExtractor`, hand-label a sample of
   real planning documents, and report citation accuracy, precision, and abstention rate.
5. **Review exclusion thresholds** with domain experts for each use and jurisdiction.
