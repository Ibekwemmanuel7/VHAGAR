# Southern pilot evaluation readiness

Assessment date: 17 October 2026. Part of [Lillie US](https://github.com/Ibekwemmanuel7/Lillie_US).

## Decision

VHAGAR is ready for bounded retrospective evaluation development, not a validated
Southern operational service. Neither Southern scientific acceptance nor shadow
service readiness has been demonstrated. Detection, prescribed-burn context,
burned-area mapping and fire evidence belong here; forestry management decisions
remain with Vepris.

This review considered the repository validation contract, portfolio scope and
interfaces, current tests, T1/T2 result caveats, Southeast replay and association
code, and the locally accessible Southern pilot specification and architecture
(content version 0.2, original v0.1 filenames). Those planning documents define
future acceptance, not measured performance. The older state-of-the-art assessment
is a historical bug ledger; individual findings require verification against
current code before treating them as still present.

Existing T1 evaluations compare sensor coincidence rather than independent
Southern incident coverage. T2 documentation records reference/window artifacts
and small transfer cohorts. Synthetic spread results are not operational
validation. Existing replay proximity matching can reuse a single event for
multiple reference fires; unmatched candidates do not establish false alarms or
executed prescribed burns. These limitations motivate an isolated scoring module
before any headline pilot accuracy is calculated.

## Bounded improvement

`vhagar.eval.incident_matching.score_incidents` scores a frozen, independently
assembled eligible incident frame against observations from a completed prediction
run. It implements maximum-cardinality one-to-one incident/lineage matching, then
minimises total absolute discovery/observation time difference, then total great
circle distance. Stable sorted IDs resolve remaining ties. The default inclusive
tolerances are six hours and 5 km. Each pair uses an actual qualifying observation,
not a multi-day event centroid. The flow solver can reverse earlier assignments;
a greedy nearest-neighbour match is insufficient.

Supply canonical lineage IDs across event merges/splits, globally unique source
observation IDs, deduplicated reference IDs, source snapshot/revision identifiers,
and timezone-aware times. The caller owns deduplication, incident independence,
eligibility, lineage construction and adjudication. Repeated observations cannot
create extra true positives. Quality flags stay in the headline denominator.

An observation qualifies only when its `known_at` is known and no later than the
explicit replay cutoff. This should represent availability to the prediction
system, not merely acquisition time. Unknown historical availability remains
unknown; do not substitute acquisition time. This scorer cannot establish that
the upstream predictor itself used only as-of context: that requires an audited
prediction manifest and separate feature/reference access controls.

All eligible references receive a ledger row, including misses due to masking,
unavailable/unknown input timing, lineage conflicts, or no observation within the
tolerances. Keep raw masked observations in the input. An absent observation does
not identify whether a source outage, cloud, sensitivity, or processing caused
the miss; investigate separately without removing the incident. Unmatched
lineages remain unresolved, not false alarms. Match timestamps do not represent
analyst availability or observed ignition lead time.

## Reproduce a score

The Python API returns a JSON-ready report containing a complete canonical input
manifest and its SHA-256 fingerprint. `prediction_run` must identify the frozen
code, configuration and source-input manifest, including sensor epochs and
development-only masks. Archive actual sources and environment versions alongside
it; a hash alone does not prove source authenticity or independence. The report
manifest is the CLI input format:

```python
import json
from pathlib import Path
from vhagar.eval.incident_matching import score_incidents

# references and observations are typed records constructed by your adapters.
report = score_incidents(references, observations, as_of=cutoff,
                         prediction_run="code/config/source-manifest identifiers")
Path("manifest.json").write_text(json.dumps(report["manifest"]), encoding="utf-8")
```

```shell
python scripts/score_incidents.py manifest.json --output report.json
python -m pytest tests/test_incident_matching.py tests/test_southeast.py -q
```

The manifest has `schema_version: 1`, `prediction_run`, ISO-offset `as_of`,
`radius_m`, integer `window_microseconds`, `references` and `observations`.
Reference fields are `incident_id`, `lon`, `lat`, `discovery_at`,
`source_revision`, optional `quality_flags`. Observation fields are
`observation_id`, `lineage_id`, `lon`, `lat`, `observed_at`, nullable `known_at`,
`source_revision`, optional `masked` (default false). Persist source revision
identifiers with their namespace, not just a locally ambiguous revision number.
Keep manifests with sensitive reference locations in controlled data storage,
outside source commits. Reversing input order must reproduce the same report.

Run all methods and baselines on the same frozen reference frame and availability
policy. Re-score at 15 minutes, one hour, three hours and six hours as predefined
sensitivity analyses; do not select the most flattering window. This module does
not replace the historical Southeast nuisance-volume replay or silently change
its published semantics. The solver is intended for pilot-sized batch frames;
pair construction is quadratic and residual flow uses Bellman-Ford. Benchmark
before large archive runs.

## Remaining gates

- Freeze AOIs, independent reference snapshots, eligibility, discovery quality,
  correlated-incident adjudication and development/held-out exposure records.
- Verify source availability and identifiers for NOAA-20/21 and historical GOES
  epochs; the old replay still defaults to S-NPP and is not a pilot input profile.
- Implement predictor-side as-of replay and development-only masks, with frozen
  source/configuration/environment manifests and baseline fairness tests.
- Add independent-incident uncertainty and stratified reports, minimum-evidence
  checks and pass/fail/insufficient-evidence gates. Current fractions are descriptive.
- Independently sample and adjudicate burned pixels; incident perimeter agreement
  cannot validate burned-pixel accuracy. Compare dNBR on identical samples.
- Obtain independent executed-burn and wildfire/escape outcomes for association
  safety; permit intent alone is inadequate. Retain human review for candidates.
- Demonstrate idempotent/revision-aware ingestion, event history, prospective
  latency/outage/freshness accounting, tenant isolation, triage and restore tests.

No new real-data performance score, independent reference dataset, operational
latency result, deployment, or automatic suppression authority is delivered by
this change. Coordinate integration by the exact product commit after tests;
do not silently advance the Lillie US runtime dependency pin.

## Verification for this change

On Python 3.12.10, NumPy 2.2.6 and SciPy 1.14.1:

- Focused incident and Southeast suite: **35 passed**, including CLI manifest
  round-trip, time/space/availability boundaries, masking, lineage competition,
  input-order invariance and exhaustive optimum comparisons on 50 small graphs.
- `ruff check src tests scripts/score_incidents.py`: passed.
- Physics module doctests: **9 passed**. Git whitespace check: passed.
- Full suite completed with two failures and seven skips. Both failures were
  existing parcel-study raster tests using PostgreSQL's incompatible PROJ
  database (layout minor 2, rasterio expected at least 6). Re-running those two
  tests with process-local `PROJ_DATA` and `PROJ_LIB` pointing to rasterio's
  `proj_data` directory: **2 passed**. No machine-wide setting was changed.
  The full suite was not repeated after that environment correction. Skips
  covered two absent demo SQLite datasets and five PostGIS tests without
  `VHAGAR_PG_DSN`. Optional dependencies available on this machine differ from
  the minimal CI environment; this is not a validation of all deployments.

The initial sandbox run could not create pytest temporary directories. A full
run outside the sandbox and focused runs with workspace-local `--basetemp`
resolved that test-environment restriction. No source fix was made for it.
