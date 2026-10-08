# 29. PostGIS mirror of the Texas solar parcel study

Status: run on the real study sample on 2026-10-08 (PostgreSQL 16, PostGIS 3.4.2,
Windows). SQL and GeoPandas agree on all 31,145 parcels and all 13,972 lines. Results:
`outputs/parcel_study/postgis_check.json` and section 7.

## 1. Purpose

The study (doc 28) computes its vector features in GeoPandas and shapely. This step loads
the same inputs into PostgreSQL/PostGIS and recomputes three features in SQL. The two
implementations must agree parcel by parcel. This gives:

1. An independent check of the Python features (two implementations, one answer).
2. A database layer that the parcel API can query (`PostGISParcelStore` in
   `src/vhagar/parcel/store.py`), with the same spatial logic.
3. A customer-style screen written in SQL.

## 2. Schema (`parcel_study`, all geometry in EPSG:5070, GiST-indexed)

| Table | Rows (real run) | Content |
|---|---|---|
| `parcels` | 31,145 | Study sample: pid, county, area_ha, is_case, floodplain_frac (NULL where FEMA has no mapping), slope_mean_pct, geom |
| `tx_lines` | 13,972 | HIFLD in-service lines: kv (NULL if unknown), robust flag, geom |
| `facilities` | USPVDB footprints | Used only by the generator-tie rule |
| `protected` | 10,169 | PAD-US 3.0 fee and easement polygons, made valid on load |
| `features_sql` | 31,145 | The three SQL features per parcel |

Loading uses `COPY` with hex EWKB, so no ORM or extra driver is needed beyond psycopg 3.

## 3. Features computed in SQL

| Feature | SQL method |
|---|---|
| Generator-tie rule | `ST_Length < 15 km` and a merged single LineString whose start or end point is `ST_DWithin` 2 km of a facility. The result is the `robust` flag. |
| `dist_tx_robust_km` | KNN `ORDER BY geom <-> parcel LIMIT 8` through a partial GiST index on robust lines, then exact `min(ST_Distance)` |
| `dist_tx_hv_km` | The same, robust lines of 230 kV or more |
| `protected_overlap_frac` | `ST_Area(ST_Union(ST_Intersection(parcel, protected)))` divided by parcel area. The union prevents double counting where fee and easement polygons overlap. |

Tolerances for agreement: 1 m for distances, 0.001 for the overlap fraction. The tie rule
must agree line by line.

**Parity note found by the integration tests.** For a multi-part line that does not merge
into one LineString, `shapely.get_point` returns `None`, so the Python rule never flags it.
PostGIS 3.4 `ST_StartPoint` returns the first point of the first part instead. The SQL
therefore applies the end-point test only to merged single LineStrings, which matches the
study. Both implementations share this limitation: unmergeable multi-part ties are kept.

## 4. Customer-style screen (`SCREEN_SQL`)

Each sampled parcel gets one outcome, in this order: `too_small` (< 40 ha), `protected`
(>= 10% protected), `far_from_hv` (> 5 km from a 230 kV+ line), `pass_flood_unknown`
(no FEMA mapping), `floodplain` (>= 10% in a Special Flood Hazard Area), `pass`.
Parcels without FEMA mapping are reported separately and never treated as "not in a
floodplain". Counts are case-control sample rows, not population totals.

## 5. How to run (Windows, PowerShell, from the VHAGAR folder)

Option A, Docker Desktop:

```powershell
docker compose -f db/compose.postgis.yml up -d
pip install "psycopg[binary]>=3.1"
python scripts/parcel_postgis.py all
```

Option B, native install: install PostgreSQL 16 (EDB installer), then add PostGIS 3.4
with Stack Builder. Create the role and database, then point the scripts at it:

```powershell
psql -U postgres -c "CREATE ROLE vhagar LOGIN PASSWORD 'vhagar'" -c "CREATE DATABASE vhagar OWNER vhagar"
psql -U postgres -d vhagar -c "CREATE EXTENSION postgis"
$env:VHAGAR_PG_DSN = "postgresql://vhagar:vhagar@localhost:5432/vhagar"
python scripts/parcel_postgis.py all
```

Integration tests (each test uses its own throwaway schema):

```powershell
$env:VHAGAR_PG_DSN = "postgresql://vhagar:vhagar@localhost:5433/vhagar"
pytest tests/test_parcel_postgis.py -v
```

## 6. Verification so far

* `tests/test_parcel_postgis.py`: 4 tests pass on PostgreSQL 16 / PostGIS 3.4.2 (load
  counts; tie rule equal to the study's shapely rule; three features equal to shapely on
  geometry with known answers, including a crossing line, overlapping protected polygons
  and a removed tie; the screen keeps unknown floodplain separate).
* End-to-end CLI run on a synthetic study (400 parcels, 88 lines, 15 facilities, 60
  protected polygons): all three features within tolerance for 100% of parcels, maximum
  difference 0.000000, tie rule 9 removed by both implementations with 0 disagreements.
* Real sample: all three features within tolerance for 100% of 31,145 parcels (max
  |diff| 0.000000); tie rule 128 removed by both, 0 disagreements. Load 9 s, SQL features
  145 s. The first real load failed because some TxGIO parcels carry Z = 0; the loader now
  forces 2D, and `test_load_drops_z` covers it.

## 7. Screen results on the real sample

| Outcome | Sample rows | Case parcels |
|---|---|---|
| too_small (< 40 ha) | 24,007 | 443 |
| far_from_hv (> 5 km to 230 kV+) | 4,819 | 260 |
| pass_flood_unknown | 985 | 179 |
| pass | 746 | 205 |
| floodplain | 349 | 54 |
| protected | 239 | 4 |

Re-weighted to the population (control weight 59.2), the two pass outcomes hold about
80,000 parcels (4.5% of Texas parcels of 2 ha or more) and 34% of the parcels that later
hosted utility-scale solar: about 7.5 times the base rate. Two rules lose the most cases:

1. The per-parcel 40 ha minimum (443 cases). Facilities usually span several parcels, so
   size belongs on an assembly of adjacent parcels, not on one parcel.
2. The 5 km distance to 230 kV lines (260 cases). Many facilities connect to lower-voltage
   lines; the threshold should be relaxed or learned.

The protected-land rule removes only 4 cases, so it costs almost nothing.
