# VHAGAR — State of the Art Verdict & Assessment

**Full-repository review · August 20, 2026**
Scope: every source module (60+ files, ~15k lines of Python), all 37 test files, all 16 docs, the serving stack, CI, configs, notebooks, and both HTML frontends. Four independent deep-review passes (physics/ingestion, ML/evaluation, serving/ops, docs/tests) plus a current-literature comparison.

---

## 1. Executive verdict

**VHAGAR is a genuinely impressive research platform whose *methodology* is at or above the current state of the art, whose *scientific core* is verified-correct, and whose *results* are honest but pre-SOTA — limited by data volume, not by rigor.** It is one of the most epistemically honest ML codebases I have reviewed: it retracts its own headline numbers, structurally forbids the leakage patterns that inflate most published wildfire ML results, and refuses to invent fields its sensors cannot measure.

The blunt one-liner: **SOTA-grade discipline, pre-SOTA-scale evidence.** The evaluation machinery would survive peer review at a top venue; the headline claims rest on 3–9 test fires; two of the four model tracks (danger, spread) have never touched real data; and nothing machine-learned is actually served in production yet.

**Scorecard**

| Quadrant | Score | One-line justification |
|---|---|---|
| Physics & data ingestion | 7.5/10 | Reference-grade physics (FWI bit-matches cffdrs; Planck, Dozier, ABI navigation verified exact) marred by one real propagating geometry bug |
| ML & evaluation | 8/10 | Structurally enforced anti-leakage and correct hard statistics; held back by a facade training layer and tiny real-data n |
| Serving, ops, CLI | 6.5/10 | A clever zero-infra live pipeline that really runs, but no model reaches production and the "live" signal can silently go stale |
| Docs, tests, honesty | 8/10 | Top-decile test suite (287 passing verified from cold install) and retraction-first results culture; minor doc drift |
| **Overall** | **7.5/10** | An exceptional one-to-two-week solo/AI-assisted sprint that most teams could not produce in a quarter — with clearly mapped gaps between demo and product |

---

## 2. What VHAGAR actually is

Four tasks, one platform: **T1** active-fire detection (GOES-19/18 ABI FDC, VIIRS, FIRMS), **T2** burned-area + severity mapping (Sentinel-2/Landsat vs MTBS and Copernicus EMSR truth, incl. a Prithvi-EO foundation-model fine-tune), **T3** fire danger (GBDT over weather/fuel covariates), **T4** spread forecasting (level-set/fast-marching physics core with assimilation and a GAN scaffold).

What exists and runs: a real resumable GOES FDC archive backfill with append-only manifests; a Welford/Chan climatology engine; a leakage-hardened evaluation harness with spatial-block and leave-year-out splits enforced in code; a real Prithvi HLS band-matched fine-tune pipeline (chips built locally, trained on Colab, scored back locally); a FastAPI serving layer fed by a GitHub-Actions cron every 30 minutes pulling live NOAA S3 + FIRMS data, fused parallax-aware, published as a rolling snapshot; a professional Mapbox ops console; 44 CLI subcommands, none of which is an empty stub.

What does *not* exist: model inference in the serving path (detections are FDC/FIRMS pass-through, honestly labeled `schema="fdc"`); any real ignition or perimeter data in T3/T4 (both are synthetic-world demos, built in about a day each); a wired Hydra/Lightning training entrypoint (`train/train.py:main` raises); reproducibility from the repo alone (data, caches, checkpoints live on one Windows machine and a Colab session).

---

## 3. Scientific core — verified, with one real bug

The physics was checked formula-by-formula, not taken on faith:

- **Canadian FWI system**: every coefficient of FFMC/DMC/DC/ISI/BUI/FWI/DSR matches Van Wagner & Pickett 1985, and the standard reference case reproduces to the last digit (FFMC 87.69, ISI 10.85, FWI 10.1).
- **Planck/Dozier/Wooster FRP**: constants exact; the Dozier Newton solver recovers p=0.004, T=900 K from clean radiances and — unusually and admirably — ships its Jacobian condition number and a gate that rejects ill-posed retrievals.
- **GOES ABI fixed-grid navigation**: PUG Vol.5 §4.2.8 implemented term-by-term; round-trips at 1e-13 degrees.
- **Spectral indices**: NBR/dNBR/RdNBR/RBR with the Miller & Thode floor and Parks offset, all correct.

**The one real physics defect: the pixel-footprint growth formula is wrong**, in two independent copies (`physics/geometry.py:147-149`, `io/abi_grid.py:184`). At 60° view zenith for geostationary it overestimates area growth ~3.4× (7.86× vs the true ~2.33×). This is not cosmetic: the wrong values are already persisted per-detection into the Tier-A archive (`true_pixel_area_m2`), bias any FRP computed from radiance via pixel area, inflate the GEO/LEO fusion match radius ~40% at high zenith (worsening the documented merge-adjacent-fires failure mode), and ride along as feature 15 of the T1 physics feature vector. The module's own docstrings are internally inconsistent about it ("MODIS grows ~8×" vs the formula's 4.4×). This is the highest-priority fix in the repository.

Everything else in this layer is honest about its limits: the 3.9 µm transmittance module declares itself a parametric surrogate, aerosol absorption is a labeled placeholder, and the southern-hemisphere FWI day-length table is a rolled northern table (fine for CONUS, wrong for the stated Canada/Europe ambition). The archive engineering (append-only JSONL manifests tolerant of truncation, rows-before-manifest crash ordering with the reasoning written down, verify-then-delete compaction, atomic Welford checkpoints) is production-minded and better than most operational pipelines.

---

## 4. ML & evaluation — the crown jewel, with caveats

This is where VHAGAR beats most of the published literature on *practice*:

- `random_split()` **raises by design**, citing the measured inflation (F1 0.985 random → 0.627 spatial-block on a FIRMS task). Splits are manifest-fingerprinted, disjointness-verified, and a **CI grep gate fails the build** on `train_test_split`/`KFold`/`shuffle=True` anywhere in the repo.
- A `FORBIDDEN_FEATURES` guard blocks lat/lon and friends at feature-build time, with the 88.9%-of-split-gain result cited in the error message.
- Every result ships with a naive baseline through the identical metric path (predict-all-burned F1, persistence+buffer, climatology), and headline numbers are *skill over naive*, not raw F1 — the single most common inflation vector in this field.
- Hard statistics are correct: Olofsson error-adjusted area with CIs, Murphy Brier decomposition (verified to ±1e-4), King-Zeng correction, FSS, quantile boosting with GPD tails.
- `ConstrainedFRPHead` does exactly what it advertises: correction hard-bounded to [1/3, 3], zero-init so the model is pure Wooster physics at step 0.
- The Prithvi fine-tune is a **real HLS band-matched transfer, not a hack**: correct six-band order including the B8A narrow-NIR choice, native 30 m, train-fires-only normalization stats, whole-fire splits, per-fire stitching back into the same scorer.

The caveats are equally real. The headline transfer claim (Prithvi +0.488 skill vs NBR-threshold +0.372 on EU fires) rests on **9 fires**; the Prithvi-vs-U-Net comparison on **3 test fires**; by the project's own fold-variance doctrine these margins are not statistically separable, and the project's own declared bar — beat the U-Net, not the threshold, on the transfer set — was never run. T4 spread and assimilation are validated only against synthetic worlds generated by the same fast-marching family the forecaster uses — partially circular, and flagged as "OPTIMISTIC" in the project's own log.

Genuine bugs found in this quadrant: the arrival-GAN's "physics anchor" Eikonal loss is mis-scaled ~30× (hardcoded `tmax=1.0` against a normalized target, `arrival_gan.py:197`), reducing the flagship physics-constrained-GAN claim to L1+LSGAN plus a smoothness prior; the default stratified negative sampler in `datasets/danger.py:206-209` fails to exclude presence ids, so real ignitions can re-enter as negatives; `bss_vs_climatology` (`eval/danger.py:81`) scores against a constant base rate — the exact practice the codebase's own doctrine forbids by name; and `leave_year_out` places validation years *after* the test year, contradicting its docstring.

---

## 5. Serving & operations — clever, honest, not yet a product

The live pipeline is a resourceful zero-infrastructure design that genuinely runs: GitHub-Actions cron (every 30 min) → NOAA public S3 + FIRMS → parallax-aware KD-tree fusion → rolling GitHub Release snapshot → Render-hosted FastAPI that re-pulls every 10 minutes. The API's honesty is exemplary — it refuses to invent burned-area fields FDC cannot measure and labels the danger endpoint `t3-danger-demo`.

But the operational story has structural holes. **Nothing machine-learned is served**: production is FDC/FIRMS pass-through with convex hulls and a hand-rolled 55/30/15 weather heuristic. And the "live" signal can lie three ways at once: `"mode": "live"` is hardcoded even when serving frozen demo data; time windows are relative to the data's own max timestamp rather than wall clock, so stale data always "fits"; GitHub disables cron on 60 days of repo inactivity with no alerting; and a transient snapshot-fetch failure silently swaps months-old demo data over the last good live state (`vhagar_api.py:171-176, 393-400`). Add a real security wart — `pickle.loads` on a network-fetched release asset (remote-code-execution surface if the asset is ever compromised) — a temp-dir leak every refresh cycle, a Docker build that fails at `COPY brand/` (directory absent), the `[serve]` extra that installs an unused dependency (`psycopg`) while missing a needed one (`scikit-learn`), and unhandled 500s on the data endpoints when state is missing.

The two frontends have a split personality. `vhagar_console.html` is a professional live ops console (schema-aware labeling, cold-start backoff, explicit LIVE/SAMPLE/OFFLINE pill, XSS escaping). `dashboard.html` is the opposite: 115 KB of hardcoded data with a pulsing "LIVE" dot and hardcoded headline metrics — pure theater that undercuts the codebase's otherwise scrupulous honesty and should be renamed or deleted.

The 2,644-line `cli.py` is a dumping ground by size but a coherent research runbook in substance — all 44 subcommands delegate to tested library modules; the honest split is real-data commands (backfill, t2-stage0, t1-cohort, emsr-ingest) vs synthetic-by-default demos (t3-*, t4-*).

---

## 6. Tests, docs, and honesty

The test suite is **top-decile for a research codebase**: 287 passed / 21 correctly-guarded skips verified from a cold minimal install. Tests assert *science*, not shapes — FWI against the published worked example, Planck anchors with "do not fix this back" comments, Brier propriety, regrid mass conservation under Hypothesis, and a set of regression tests keyed to real bugs the project actually hit (CF-time overflow, corrupt scan-start recovery, nav-cache concurrency). Leakage is tested directly. Weakest spots: torch model tests are smoke-level, and no golden test runs against a real archived granule.

The documentation is dense, literature-grounded, and unusually candid — docs/11 retracts its own headline F1 0.865 as a window artifact; docs/12 documents five successive artifacts in the temporal detector and lands on an honest negative (−170 min median lead: GOES FDC beats the anomaly detector); the first Prithvi run is reported as a failure before the fix. Blemishes: six dangling references to a `docs/research/` folder that doesn't exist, a stale README status list, an uncorrected wrong-unit table left inline in docs/11, and headline sentences ("genuine cross-continent generalisation", "crowning result") that outrun their sample sizes.

The build log (PROGRESS.md, 1,657 lines) shows the whole system was assembled in roughly **one week** (Aug 13–19, 2026) on top of an earlier base, explicitly AI-assisted, with entire science phases opened and closed in single days. Depth tracks data availability exactly: deep where real data existed (T1, T2), demo-thin where it didn't (T3, T4).

---

## 7. Consolidated bug ledger (priority-ordered)

| # | Severity | Location | Defect |
|---|---|---|---|
| 1 | **High** | `physics/geometry.py:147`, `io/abi_grid.py:184` | Pixel-footprint growth formula wrong (~3.4× area overestimate at 60° zenith); already persisted into archive, biases FRP-from-radiance, fusion tolerance, and a T1 feature |
| 2 | **High** | `serve/vhagar_api.py:171-176, 393-400` | Failed snapshot refresh silently replaces last-good live data with months-old demo data |
| 3 | **High** | `serve/vhagar_api.py:160` | `pickle.loads` on network-fetched release asset — RCE surface; use JSON/parquet |
| 4 | **High** | `models/arrival_gan.py:197` | Eikonal "physics anchor" mis-scaled ~30× (hardcoded `tmax=1.0` vs normalized target) — advertised constraint numerically meaningless |
| 5 | **High** | `datasets/danger.py:206-209` | Default stratified negative sampling doesn't exclude presence ids; positives re-enter as label-0 rows and reporting-intensity weights are bypassed |
| 6 | Med | `io/gee.py:201-207` | `fetch_patches` submits an unpicklable closure to ProcessPoolExecutor — every chip "fails" silently, returns zero results |
| 7 | Med | `serve/vhagar_api.py:502,578` | `"mode": "live"` hardcoded; staleness indistinguishable from liveness (compounded by data-relative windows and silent cron death) |
| 8 | Med | `eval/danger.py:81,90` | `bss_vs_climatology` uses a constant base rate — violates the repo's own stated rule |
| 9 | Med | `eval/splits.py:246-251` | `leave_year_out` validation years are *after* the test year, contra docstring |
| 10 | Med | `goes_reader.py:188`, `cmip_reader.py:165,183` | Naive/aware datetime inconsistencies: naive input crashes; aware non-UTC input silently mislists CMIP hours |
| 11 | Med | `Dockerfile:27`; `serve/demo/` | `COPY brand/` fails (dir absent); frozen-demo dir absent → deploy-check and fresh-clone demo path break |
| 12 | Med | `serve/vhagar_api.py:149` | Temp-dir leak every 10-min refresh; no conditional GET |
| 13 | Low | `eval/metrics.py:162-170, 349-353` | AP mishandles tied scores (±0.01 vs sklearn); CRPS-from-quantiles is a biased quadrature |
| 14 | Low | `io/firms.py:162` | Malformed FIRMS rows silently dropped — schema change indistinguishable from "no fires" |
| 15 | Low | various | Doctest rot under numpy≥2; `filter_fa` dead param; `_conf_pct` renders MODIS confidence 1 as 100%; dead fold-cap logic in `danger_grid.py:117`; per-tile clustering splits fires at tile edges |

No committed secrets were found anywhere in the tree.

---

## 8. How VHAGAR compares to the state of the art

**T2 burned area / foundation models.** The published reference point is IBM/NASA's [Prithvi-EO-2.0-300M-BurnScars](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars), which reports ~87–90 IoU on the curated HLS Burn Scars benchmark; current research pushes efficiency and small-sample adaptation via [LoRA adaptation of geospatial foundation models for wildfire mapping](https://arxiv.org/abs/2605.04989) and [fine-tuning optimization for small-scale burn scar identification](https://www.mdpi.com/2571-6255/9/4/161). VHAGAR's burn IoU of 0.21 looks far below that — but the comparison is not apples-to-apples: the benchmark is a large curated dataset scored with random-ish chip splits, while VHAGAR scores whole held-out fires under grouped splits with skill-over-naive on 20 fires total. VHAGAR's *evaluation setting is harder and more honest than the benchmark's*; its *evidence base is far smaller*. Verdict: methodology ahead of published practice, results behind published numbers, and the gap is data volume (the project's own next step — scale to ~60 CONUS fires — is exactly right). Multi-source rapid burn-scar mapping ([arXiv 2601.09262](https://arxiv.org/html/2601.09262)) and multi-task time-series datasets like [TS-SatFire](https://www.nature.com/articles/s41597-025-06271-3) are the community direction VHAGAR should plug into for scale.

**T1 detection.** Serving NOAA's operational FDC plus FIRMS with parallax-aware fusion is solid operational practice, not novel detection. The parallax study (POD 0.376→0.499, FAR 15.7%→5.6% on 188k detections) is real, careful measurement science. The temporal-anomaly detector was an honest negative — correctly banked rather than spun. Meanwhile the detection frontier is moving to purpose-built constellations: [FireSat](https://sites.research.google/gr/wildfires/firesat/) ([first protoflight imagery, Earth Fire Alliance](https://www.earthfirealliance.org/press-release/firesat-first-wildfire-images); [three more satellites launched](https://blog.google/innovation-and-ai/models-and-research/google-research/firesat-satellites/)) detects ~5×5 m fires with ~20-minute revisit at full constellation — a capability class no GOES-based system can match. VHAGAR's own constellation doc already anticipates this landscape, which is to its credit.

**T3 danger / T4 spread.** Here VHAGAR is furthest from SOTA. The community has real-data deep-learning danger forecasting ([Deep Learning Methods for Daily Wildfire Danger Forecasting](https://ar5iv.labs.arxiv.org/html/2111.02736); [ML + FWI operational practice](https://www.haskoning.com/en/twinn/blogs/2025/machine-learning-and-fire-weather-index-in-wildfire-danger-prediction)) and public next-day-spread benchmarks with attention/time-series models ([Predicting Next-Day Wildfire Spread](https://arxiv.org/html/2502.12003v1); [survey](https://www.mdpi.com/2571-6255/7/12/482)). VHAGAR's T3/T4 are methodologically well-constructed harnesses that have never seen a real ignition record or a real perimeter. The machinery (target-group sampling, King-Zeng, CRPS, assimilation scaffolding) is the right machinery; the evidence is zero real-world bits so far.

**Overall SOTA verdict.** On *evaluation discipline* — enforced anti-leakage splits, baseline culture, error-adjusted area estimation, retraction practice — VHAGAR is ahead of the median published wildfire-ML paper, several of which the recent [lab-metrics-vs-real-world-validation review](https://www.mdpi.com/2673-2688/6/10/253) criticizes for exactly the practices VHAGAR bans. On *demonstrated results*, it is below SOTA everywhere, honestly so. On *operational capability*, it is a well-built demo tier below systems like FireSat or national operational services — as a one-person, one-week, zero-budget build, that is not a criticism so much as a coordinate.

---

## 9. Recommendations, in order

1. **Fix the pixel-area growth formula** (bug 1) and re-derive from slant range; add a golden test against published MODIS/ABI footprint tables; re-run the Tier-A archive or version the schema, since wrong values are persisted.
2. **Make staleness visible** (bugs 2, 7): keep last-good state on refresh failure, compute `mode` from data age vs wall clock, alert on cron death. For a fire-information product, a stale feed that looks live is the worst failure mode the system can have.
3. **Replace pickle with parquet/JSON in the snapshot path** (bug 3).
4. **Scale T2 before adding anything new**: ~60 CONUS + a larger EU set, then run the project's own declared decisive test — Prithvi vs U-Net head-to-head on the same transfer fires with bootstrap CIs. This single experiment settles the repo's central scientific claim.
5. Fix the GAN Eikonal scaling and the danger negative-sampling contamination (bugs 4, 5) before any real-data T3/T4 run inherits them.
6. **Get real data into T3/T4** (FPA-FOD ignitions; NIROPS/VIIRS perimeters, or the public WildfireSpreadTS-style benchmarks) — until then, label every synthetic number as prominently in the dashboards as the code already does in the API.
7. Delete or honestly relabel `dashboard.html`; the console makes it redundant and its fake LIVE badge is the one place the project's integrity standard slips.
8. Housekeeping: wire or delete the Hydra/Lightning facade, split `cli.py`, fix the Docker build and `[serve]` extras, restore or remove the dangling `docs/research/` references, run doctests in CI on numpy≥2, and repair the datetime discipline in the listing layer.

---

## 10. Closing verdict

VHAGAR reads like what happens when strong engineering judgment, real domain literacy, and AI-assisted velocity are pointed at a hard geophysical problem for a week — with an unusually rare ingredient added: the discipline to report what *didn't* work. The physics is right, the evaluation contract is enforced rather than promised, the tests check science rather than shapes, and the failures are documented with more care than most projects document their successes. What separates it from state of the art is not intelligence or hygiene but evidence: an order of magnitude more labeled fires for T2, any real data at all for T3/T4, and a served model behind the API. The bones are excellent. The burden now is scale, and the bug ledger above is short enough to clear in days.

**Sources:** [Prithvi-EO-2.0-300M-BurnScars model card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars) · [NASA/IBM Prithvi release](https://www.earthdata.nasa.gov/news/nasa-ibm-openly-release-geospatial-ai-foundation-model-nasa-earth-observation-data) · [LoRA adaptation of GFMs for wildfire mapping (arXiv 2605.04989)](https://arxiv.org/abs/2605.04989) · [Fine-tuning EFMs for small-scale burn scars (Fire, MDPI)](https://www.mdpi.com/2571-6255/9/4/161) · [Rapid multi-source burn scar mapping (arXiv 2601.09262)](https://arxiv.org/html/2601.09262) · [TS-SatFire dataset (Nature Sci. Data)](https://www.nature.com/articles/s41597-025-06271-3) · [FireSat (Google Research)](https://sites.research.google/gr/wildfires/firesat/) · [FireSat first imagery (Earth Fire Alliance)](https://www.earthfirealliance.org/press-release/firesat-first-wildfire-images) · [FireSat launches](https://blog.google/innovation-and-ai/models-and-research/google-research/firesat-satellites/) · [Deep learning daily fire danger (arXiv 2111.02736)](https://ar5iv.labs.arxiv.org/html/2111.02736) · [ML + FWI in danger prediction (Haskoning)](https://www.haskoning.com/en/twinn/blogs/2025/machine-learning-and-fire-weather-index-in-wildfire-danger-prediction) · [Next-day spread with attention (arXiv 2502.12003)](https://arxiv.org/html/2502.12003v1) · [ML/DL spread prediction review (Fire, MDPI)](https://www.mdpi.com/2571-6255/7/12/482) · [AI for wildfire management: lab metrics vs real-world validation (MDPI)](https://www.mdpi.com/2673-2688/6/10/253)
