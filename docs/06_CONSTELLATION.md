# VHAGAR: Sensor Constellation Plan, 2026-2030

Surviving the S-NPP shutdown and the MODIS end of mission.
Encoded in code at `src/vhagar/io/sensors.py`; tested in
`tests/test_sensors_and_features.py`.

---

## 0. The terminology correction, because it changes the plan

**VIIRS is an instrument, not a satellite.** Every JPSS spacecraft carries one:

| Spacecraft | Programme | Carries VIIRS? | Status |
|---|---|---|---|
| Suomi-NPP | NPP demo | **Yes** | data ends **2026-11-01 13:00 UTC** |
| NOAA-20 | JPSS-1 | **Yes** | secondary (past 7-yr design life) |
| NOAA-21 | JPSS-2 | **Yes** | **primary since March 2024** |
| NOAA-22 | **JPSS-4** | **Yes** | launch readiness **2027** |
| JPSS-3 | JPSS-3 | **Yes** | ~2032 (in-orbit name unconfirmed) |

So *"use NOAA-21/22 instead of VIIRS"* resolves to **"switch the VIIRS
active-fire feed from the S-NPP platform to NOAA-21"**, which is exactly what
NOAA directs users to do. Same instrument, same 375 m I-band contextual
algorithm, same product family. In code it is a string change:
`VIIRS_SNPP_NRT` → `VIIRS_NOAA21_NRT`.

Note also: **NOAA-22 is JPSS-4, and JPSS-4 launches *before* JPSS-3.**
Third-party trackers disagree with NOAA on JPSS-3's in-orbit designation; treat
JPSS-3 = NOAA-23 as inferred, not confirmed.

The three consequences that are *not* cosmetic are in §2.

---

## 1. What actually changes on 1 November 2026

**The notice.** S-NPP data delivery ceases **2026-11-01 at 13:00 UTC**. All
instruments. ATMS, CERES, OMPS, **VIIRS**, and **HRD direct broadcast**. Users
are directed to NOAA-21 (primary) and NOAA-20 (secondary).

⚠️ The HRD clause matters: FIRMS **Ultra Real-Time (<60 s)** depends on
direct-readout antennas. S-NPP URT dies with the downlink, not just the archive.

**Revisit.** Over CONUS the sequence today is NOAA-21 → ~25 min → S-NPP →
~25 min → NOAA-20 → 50 min gap. After November the quarter-orbit slot empties:
NOAA-21 → **50 min** → NOAA-20 → 50 min gap.

- VIIRS overpasses at mid-latitudes: **~6/day → ~4/day** (2 day, 2 night).
- Local times unchanged, all JPSS platforms share one plane at **~13:25 LTAN**.
- Canada/Alaska degrades least (orbit convergence, 3000 km swath overlap above
  ~50 °N); southern CONUS and southern Europe degrade most.

**Is there a new diurnal gap? No, not from S-NPP.** S-NPP added redundancy and
a tighter cadence inside the existing 13:30/01:30 band; it never sampled a
distinct time of day. What it removes is one of three independent chances to see
through cloud and smoke, and it doubles the maximum interval between looks.

**The real diurnal hole is opened by MODIS.** Terra held ~10:30 LT before
drifting to ~09:00; Aqua drifted from 13:30 to ~15:50, which means Aqua's final
year has been sampling the **mid-afternoon fire peak**, and that sample
disappears at end of mission with no public-sector polar replacement.

Post-2026 polar diurnal budget:

| Local solar time | Source |
|---|---|
| ~01:30 | NOAA-21, NOAA-20 |
| ~09:30 | Metop-SG-A1 METimage (⚠ no fire L2 confirmed) |
| ~10:00 | Sentinel-3A/B/C SLSTR |
| ~13:30 | NOAA-21, NOAA-20 |
| ~21:30 | Metop-SG-A1 |
| ~22:00 | Sentinel-3 |
| **~15:00-18:00** | **GAP, geostationary only** |

**NASA acknowledges no morning-orbit VIIRS replacement is scheduled.** The
morning orbit now belongs to Europe.

---

## 2. Three non-cosmetic consequences

**1. NOAA-21 carries a ~+10 % FRP bias**, attributed to a shift in the M13
spectral response toward a more transparent part of the atmosphere. Detection
*counts* are consistent across platforms; **FRP magnitude is not.** Splice
without correcting and your fire-intensity time series gains a spurious step at
the handover.
→ `sensors.frp_to_reference_scale(frp, "noaa21")`.

**2. Dedup and growth-rate windows must be re-tuned** for the ~25 → ~50 min
interval change.
→ `sensors.coverage_report(date)` emits this warning automatically when fewer
than three VIIRS platforms are active.

**3. There is no NOAA-21 collection in Google Earth Engine.** Verified: the
expected `NASA/LANCE/NOAA21_VIIRS/C2` path 404s, and the GEE LANCE tag lists
only `FIRMS`, `NASA/LANCE/NOAA20_VIIRS/C2` and `NASA/LANCE/SNPP_VIIRS/C2`. A
GEE-resident pipeline loses S-NPP on 1 Nov and is left with **NOAA-20 alone**.
Options: migrate to NOAA-20 and accept one platform; ingest FIRMS API output
into your own GEE asset; or petition for publication. This is a live single
point of failure, recorded in `PLATFORMS["noaa21"].caveats`.

---

## 3. Sentinel-3 SLSTR as a primary source: yes, with a hard caveat

**Fleet.** S3A (2016) and S3B (2018) are both past their 7-year design life.
**S3C launched 2026-09-14** on Vega-C, roughly concurrent with the S-NPP
shutdown. Allow ~6 months commissioning: **do not plan on operational S3C FRP
before ~Q2 2027.** S3D ~2028.

**Fire-relevant bands** (814.5 km, 10:00 LT descending, 27-day repeat):

| Band | λ | GSD | Role |
|---|---|---|---|
| S5 | 1.613 µm | 500 m | SWIR1 |
| **S6** | **2.25 µm** | 500 m | night AF detection; **gas-flare discrimination** |
| **S7** | 3.742 µm | 1 km | primary MIR. **saturates ~311 K** |
| **F1** | 3.742 µm | 1 km | dedicated high-gain fire channel |
| S8 / F2 | 10.85 µm | 1 km | TIR context, glint rejection |

Swath >1400 km nadir. Nadir-only revisit ~1 d (1 sat) / **0.5 d (2 sats)**. 
roughly 2 day + 2 night looks over Europe and Canada. NRT FRP delivered within
**~3 hours**.

**The night-time result is genuinely excellent.** SLSTR captured 90 % of MODIS
active-fire pixels **plus 44 % additional pixels, predominantly FRP < 5 MW**,
and globally detected ~7× more fire pixels at comparable total FRP.

**The daytime limitation is severe.** S7 saturates at ~311 K, and the algorithm
**cannot process daytime scenes where >1 % of *background* pixels are
saturated**, which describes much of summer daytime Mediterranean, southern
CONUS and the interior West. F1 exists for this, but the fully F1-based daytime
path is listed as future work. S7/F1 mis-registration averages ~1 km and
requires a clustering step. Product maturity is "preliminary operational,
primarily effective at night."

**Dual view is irrelevant to fire.** The oblique view exists for SST aerosol
correction: narrower swath, coarser footprint, doubled atmospheric path. The FRP
algorithm is nadir-only. Budget revisit on nadir numbers.

**Verdict: promote SLSTR to primary for Europe and for the morning/night band.
Do not make it the daytime primary for CONUS or the Mediterranean.** It
complements VIIRS; it does not substitute for it. Its S6 band is also what makes
the flare colour-temperature discriminant possible (`docs/03_PHYSICS.md` §4).

**Access.** CDSE (`SL_2_FRP___`, STAC `sentinel-3-sl-2-frp-nrt`); EUMETSAT Data
Store `EO:EUM:DAT:0417`. ⚠️ **Not available in Google Earth Engine**, only
`COPERNICUS/S3/OLCI` exists. Copernicus free and open.

---

## 4. What replaces MODIS

MODIS uniquely provided a 25-year homogeneous fire record, `MCD64A1` (500 m
burned area) and `MCD14ML`, plus the 10:30 morning sample.

**Active fire →** VIIRS 375 m. *Better* for small fires, but the 1 km→375 m
change means splicing the records without explicit intercalibration produces a
spurious upward step in fire counts. **This harmonisation problem is unsolved**
and is the largest scientific cost of the MODIS loss.

**Burned area →** `VNP64A1` (500 m, monthly, ~5-month latency, record from March
2012), which NASA states provides continuity to MCD64A1.
⚠️ **`VNP64A1` is built on S-NPP.** A NOAA-20/21 equivalent could not be
verified. **If none exists, the NASA standard burned-area record terminates in
November 2026.** This is the highest-priority open query to LP DAAC.

**European fallbacks:** ESA Fire_cci Sentinel-3 SYN burned area (CEDA / CCI Open
Data Portal); Copernicus Land Global Burnt Area, which has already made a
"temporary switch to NOAA VIIRS source."
⚠️ **EFFIS/GWIS still names S-NPP as its VIIRS source** with no published 2026
transition statement. Confirm with JRC before 1 Nov or the European operational
feed silently degrades.

**Metop-SG-A1 / METimage** launched 2025-08-13, 09:30 LTDN, **20 channels,
500 m, 2715 km swath**, with 3.7 and 10.8 µm fire-capable channels; first light
September 2025. This is the best replacement for Terra's morning overpass.
⚠️ **No operational fire/FRP L2 product could be found.** Prospective, not
procurable, in 2026. Watch for an EUMETSAT/LSA-SAF announcement.

---

## 5. Geostationary

- **GOES-19 = GOES-East** (75.2 °W, operational 2025-04-04); **GOES-18 =
  GOES-West** (137 °W); GOES-16 drifting to storage; GOES-17 offline.
  Product: ABI L2 FDC, `FDCC` (CONUS, 5 min), `FDCF` (full disk, 10 min), 2 km.
  Free on AWS and in GEE.
- ⚠️ **GeoXO was descoped in August 2025 and the restructuring approved April
  2026; launches 2032, 2034, 2039, 2043. There is no GOES-20.** GOES-19 must
  carry fire through ~2032, a long-horizon single point of failure.
- **Meteosat-12 (MTG-I1)** became prime for the 0° mission June 2025; MTG-I2
  expected August 2026.
  - **MSG SEVIRI FRP-Pixel `LSA-502`, 3 km / 15 min. OPERATIONAL.**
  - MTG FCI FRP-Pixel `LSA-509`, 1 km / 10 min. **still Demonstration status.**
  → Keep `LSA-502` as the European geostationary feed until `LSA-509` is
  promoted. MSG has finite life; that promotion is a hard watch item.

---

## 6. Migration checklist: before 1 November 2026

1. Swap `VIIRS_SNPP_NRT` → `VIIRS_NOAA21_NRT` (primary) + `VIIRS_NOAA20_NRT`
   (secondary) throughout ingest, dedup and alerting. **Run both in parallel
   through October and compare.**
2. Apply the NOAA-21 FRP correction wherever FRP is used quantitatively.
3. Re-tune dedup windows and growth-rate estimators for ~50 min spacing.
4. If on Earth Engine: migrate to `NASA/LANCE/NOAA20_VIIRS/C2` and stand up a
   FIRMS-API → GEE-asset ingest for NOAA-21.
5. Stand up Sentinel-3 SLSTR FRP ingest via CDSE STAC before winter, so it is in
   production before the 2027 season and before S3C data arrives.
6. Confirm with JRC that EFFIS/GWIS has migrated off S-NPP.
7. Query LP DAAC about a NOAA-20/21 burned-area product.
8. Increase reliance on GOES-19 FDC, the 5-min CONUS cadence is what actually
   absorbs the lost S-NPP sample.

---

## 7. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | VIIRS down to 2 platforms until NOAA-22 (2027+); NOAA-20 past design life | **High** | GOES 5-min FDC for temporal fill; SLSTR for 10:00/22:00; treat 2027 LRD as slippable |
| R2 | No morning-orbit VIIRS ever planned; Terra 10:30 gone | **High** | SLSTR (10:00) + METimage (09:30). Europe now owns the morning orbit |
| R3 | No NOAA-21 collection in Google Earth Engine | Med-High | Use NOAA-20 in GEE; self-ingest NOAA-21 via FIRMS API |
| R4 | `VNP64A1` is S-NPP-based; successor unconfirmed | **High** | Fire_cci S3-SYN BA as European fallback; escalate to LP DAAC |
| R5 | MODIS↔VIIRS record harmonisation (1 km vs 375 m detection floors) | High (science) | Never splice raw counts; report the break explicitly |
| R6 | SLSTR daytime S7 saturation | **High** for Mediterranean / southern CONUS daytime | Never use S3 as daytime primary; watch for the F1-based daytime processor |
| R7 | MTG FCI fire product still Demonstration | Medium | Keep MSG `LSA-502` operational; monitor `LSA-509` |
| R8 | No GOES imager replacement until GeoXO ~2032 | **High**, long-horizon | GOES-16/17 in on-orbit storage as spares; monitor GOES-19 health |
| R9 | URT (<60 s) is direct-broadcast dependent, North America only | Medium | Accept ~3 h NRT for Europe, or fund an EU direct-readout station |
| R10 | FireSat is not open data; ops from 2027 | Medium | Register as Early Adopter; never architect a hard dependency |
| R11 | S3A/S3B both past design life; S3C not operational before ~Q2 2027 | Medium | S3D ~2028 |
| R12 | OroraTech / Hellenic Fire System are commercial or national | Low-Med | Tactical overlay only, never the detection backbone |

---

## 8. Unverified: do not treat as established

1. JPSS-3's in-orbit designation and firm launch date.
2. Existence of a NOAA-20/21 VIIRS burned-area product (`VJ164A1` or similar).
3. Existence of a `VIIRS_NOAA21_SP` standard-processing stream.
4. Whether an operational METimage fire/FRP L2 product exists or is scheduled.
5. Whether the AWS `meeo-s3` mirror carries SLSTR L2 FRP.
6. Whether S3C replaces S3A or S3B, or a 3-satellite constellation will be flown.
7. Whether EFFIS/GWIS has migrated off S-NPP.
8. FireSat and OroraTech pricing and licence terms.
9. Wooster `a` constants for VIIRS, SLSTR, FCI and METimage. **not published**.
   `frp.wooster_a` warns and falls back to the MODIS C6 value; worth ~10 %.
