# VHAGAR: Data Catalogue

Every dataset VHAGAR depends on, with identifier, access path, resolution,
cadence and licence. Entries marked ⚠ carry a caveat you must read before use.

---

## 1. Active fire / thermal (T1)

| Dataset | Res | Cadence | Access | Notes |
|---|---|---|---|---|
| VIIRS I-band AF, S-NPP | 375 m | ~2×/day | GEE `NASA/LANCE/SNPP_VIIRS/C2`; FIRMS API | ⚠ **NOAA ceases S-NPP delivery 2026-11-01 13:00 UTC** |
| VIIRS I-band AF, NOAA-20 | 375 m | ~2×/day | GEE `NASA/LANCE/NOAA20_VIIRS/C2` | secondary |
| VIIRS I-band AF, NOAA-21 | 375 m | ~2×/day | FIRMS; GEE ID follows same pattern | **primary after Nov 2026**; verify GEE ID |
| GOES-19 ABI L2 FDC (East) | 2 km | 5 min CONUS / 10 min FD | GEE `NOAA/GOES/19/FDCC`, `/FDCF`; `s3://noaa-goes19` | operational GOES-East since 2025-04 |
| GOES-18 ABI L2 FDC (West) | 2 km | 5 min | GEE `NOAA/GOES/18/FDCC`; `s3://noaa-goes18` | |
| GOES SNS new-object topic |. | event | `arn:aws:sns:us-east-1:123901341784:NewGOES19Object` | drives the hot path |
| MTG-I FCI L2 Fire (FIR) | **1 km** | **10 min** | EUMETSAT Data Store `EO:EUM:DAT:0682` | ⚠ demonstration status; back-processed to 2025-01-01 |
| MSG SEVIRI FRP-PIXEL | 3 km | 15 min | LSA SAF | legacy Europe |
| Sentinel-3 SLSTR FRP | 1 km | ~1×/day | CDSE / EUMETSAT | S6 @2.25 µm discriminates gas flares; NRT <3 h |
| MODIS MOD14/MYD14 | 1 km | ~2×/day | GEE `MODIS/061/MOD14A1`, `MYD14A1`, `FIRMS` | ⚠ Terra/Aqua shutdown begins late 2026 |
| NOAA HMS | vector | daily | ospo.noaa.gov | **analyst-QC'd**, best human-verified CONUS labels |
| FIRMS Static Thermal Anomalies | 400 m | static + dynamic | FIRMS | persistent industrial/volcanic source mask |
| CWFIS Fire M3 hotspots | 375 m/1 km | NRT | cwfis.cfs.nrcan.gc.ca | Canada |
| EFFIS active fire | 375 m | 6×/day, 2-3 h latency | EFFIS API | Europe |

**Latency reference.** FIRMS URT ≈ 25 s (MODIS) / 50 s (VIIRS) end-to-end over
CONUS/PR/HI via direct broadcast; FIRMS RT ≈ 60-90 min faster than global NRT;
LANCE NRT ≈ 3 h; GOES L2 FDC on S3 ≈ 1-2 min after scan end. Do not compete
with URT, consume it.

### Detection benchmarks

| Benchmark | Content | Licence |
|---|---|---|
| ActiveFire (Landsat-8) | >150k 10-band patches, algorithmic + manual masks | github.com/pereira-gha/activefire |
| Land8Fire | >20k **manually annotated** 256² 10-band patches, fixed splits | CC-BY |
| TS-SatFire | 3,552 VIIRS images, 179 CONUS fires 2017-2021, 3 tasks, 71 GB | CC-BY 4.0 |
| Sen2Fire | Sentinel-2 + S5P, extreme imbalance | open |

---

## 2. Burned area & severity (T2)

| Dataset | Region | Res | Range | Access | Licence |
|---|---|---|---|---|---|
| **MTBS** | US | 30 m | 1984-2024 | GEE `USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1`; mtbs.gov | public domain |
| Landsat C2 L3 Burned Area | ⚠ **CONUS only, no Alaska** | 30 m | 1984- | USGS EarthExplorer | public domain |
| Welty & Jeffries polygons | US | vector | 1800s- | ScienceBase | ⚠ recall layer only, not pixel truth |
| NIFC / WFIGS perimeters | US | vector | current + history | data-nifc.opendata.arcgis.com | ⚠ timestamps often reflect *upload*, not observation |
| **NBAC** | Canada | 30 m | 1972- | CWFIS Datamart; GEE community `projects/sat-io/open-datasets/CA_FOREST/NBAC/...` | OGL-Canada |
| CNFDB | Canada | vector | 1917- | CWFIS | coarser; prefer NBAC |
| **EFFIS burnt area** | Europe | 20 m (S2, since 2018) | 2003- | EFFIS WFS | ⚠ semi-automatic + manual QA; ~30 ha MMU |
| **Copernicus EMS (EMSR)** | Europe | VHR | per activation | emergency.copernicus.eu | **hold out entirely as test set** |
| MCD64A1 C6.1 | global | 500 m | 2000-11- | GEE `MODIS/061/MCD64A1` | |
| FireCCI51 | global | 250 m | 2001-2020 | GEE `ESA/CCI/FireCCI/5_1` | |
| GABAM | global | 30 m | 1990-2021 | GEE `projects/sat-io/open-datasets/GABAM` | CC0 |
| GlobFire daily perimeters | global | 500 m | 2003- | GEE `JRC/GWIS/GlobFire/v2/DailyPerimeters` | ⚠ too coarse for tactical labels |

⚠ **MTBS size thresholds:** ≥1,000 acres in the western US, ≥500 acres in the
east. It captures ~95 % of annual burned *area* but is **not** a complete
inventory, a documented small-fire omission bias.

⚠ **Use MTBS continuous dNBR/RBR, not the thematic class.** Thematic
breakpoints are set per fire by analyst review and are not comparable across
fires.

### Imagery

| Source | Res | Access |
|---|---|---|
| Sentinel-2 L2A COG | 10-60 m | `s3://sentinel-cogs` (us-west-2, **not** requester-pays); STAC: earth-search.aws.element84.com/v1 |
| Landsat C2 L2 | 30 m | `s3://usgs-landsat` (us-west-2, **requester-pays**) |
| Sentinel-1 GRD | 10 m | CDSE; ASF |
| Satellite Embedding V1 (AlphaEarth) | 10 m, 64 bands | GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | ⚠ **annual** cadence, auxiliary feature only, cannot separate within-season fires |

---

## 3. Weather & climate (T3, T4)

| Dataset | Res | Cadence | Access |
|---|---|---|---|
| **gridMET** | 4 km CONUS | daily | GEE `IDAHO_EPSCOR/GRIDMET`, bands `bi erc fm100 fm1000 vpd vs th pr tmmn tmmx rmin rmax sph srad` |
| gridMET DROUGHT | 4 km | 5-day | GEE `GRIDMET/DROUGHT`. SPI/SPEI/EDDI/PDSI |
| ERA5 / ERA5-Land | 0.25° / 9 km | hourly | GEE `ECMWF/ERA5_LAND/HOURLY`; ARCO-ERA5 on GCS; `s3://earthmover-icechunk-era5` (CC-BY-4.0) |
| **HRRR** | **3 km CONUS** | hourly init, 18-48 h | `s3://noaa-hrrr-bdp-pds` |
| RTMA / URMA | 2.5 km | hourly / 15 min RU | `s3://noaa-rtma`, use URMA as verification truth |
| ECMWF HRES / ENS | 9 / 18 km | 2×/day | ECMWF Open Data (0.25° free) |
| ECMWF **AIFS v2** | ~28 km | 2×/day | ECMWF Open Data | ⚠ under-disperses extremes; days 3-15 only |
| GFS | 0.25° | 4×/day | NOMADS / AWS |
| Daymet v4 | 1 km | daily | GEE `NASA/ORNL/DAYMET_V4` |
| PRISM | 800 m / 4 km | daily | prism.oregonstate.edu | ⚠ 800 m has licence restrictions |
| CEMS seasonal fire danger | 0.25° | seasonal | `cems-fire-seasonal` on CDS |

⚠ ECMWF stopped running GraphCast, Pangu, FourCastNet and Aurora in real time
after the IFS 50r1 upgrade. AIFS is the surviving operational AI NWP path.

---

## 4. Fuels & vegetation (T3, T4)

| Dataset | Region | Res | Access |
|---|---|---|---|
| LANDFIRE FBFM40 / FBFM13, CBH, CBD, CC, CH | US | 30 m | landfire.gov |
| **CFFDRS FBP Fuel Types 2024** | Canada | **30 m** | open.canada.ca (OGL), from SCANFI + ecozones + NBAC |
| FirEUrisk European fuel map | Europe | 1 km | ESSD 15:1287 (2023) | ⚠ the only harmonised pan-EU map; coarse for spread |
| SMAP L4 root-zone soil moisture | global | 9 km | GEE `NASA/SMAP/SPL4SMGP/007` |
| MODIS NDVI/EVI | global | 500 m | GEE `MODIS/061/MOD13A1` |
| MODIS LST | global | 1 km | GEE `MODIS/061/MOD11A1` |
| MODIS LAI/FPAR | global | 500 m | GEE `MODIS/061/MCD15A3H` |
| GEDI L2A / L4B | 52°N-52°S | 25 m / 1 km | GEE `LARSE/GEDI/...` |
| TerraClimate (PDSI, CWD) | global | 4 km | GEE `IDAHO_EPSCOR/TERRACLIMATE` |
| **Globe-LFMC 2.0** | global | point | figshare `10.6084/m9.figshare.c.6980418`. >280k measurements, 1977-2023 |

---

## 5. Human & ignition drivers (T3)

| Dataset | Access | Notes |
|---|---|---|
| **FPA-FOD** (US ignitions) | USFS RDS-2013-0009.7 | 1992-~2022, ~2.3 M records with cause |
| CNFDB point (Canada) | cwfis.cfs.nrcan.gc.ca | 1930- |
| Europe ignition points | national (Prométhée FR, EGIF ES, ICNF PT) | ⚠ **no pan-European ignition database of FPA-FOD quality** |
| WorldPop | GEE `WorldPop/GP/100m/pop` | |
| GHSL population / built | GEE `JRC/GHSL/P2023A/GHS_POP`, `GHS_BUILT_S` | |
| SILVIS WUI (US) | silvis.forest.wisc.edu | block-level, decadal 1990-2020 |
| Global WUI 10 m (2020) | Zenodo 7941460 | Europe is 15 % WUI by land area, highest globally |
| Roads / rail | OSM (Geofabrik), GRIP4 | |
| Transmission lines (US) | HIFLD | ⚠ transmission only; **distribution** lines cause most utility ignitions |
| GOES GLM lightning | GEE `NOAA/GOES/16/...` | free, continuous, ~8 km; DE ~70-80 % |
| NLDN / CLDN / ENTLN | commercial | higher CG detection efficiency |

---

## 6. Perimeter & progression (T4)

| Source | Res | Cadence | Access |
|---|---|---|---|
| **FEDS** (VIIRS fire event data suite) | 375 m | **12 h**, ~4 h latency | fire.eis.smce.nasa.gov; github.com/Earth-Information-System/fireatlas |
| NIROPS airborne IR | metre-scale | irregular, overnight | nifc.gov/resources/niicd/infrared-branch. **best ground truth** |
| WFIGS perimeters | vector | irregular | data-nifc.opendata.arcgis.com |
| Canadian Fire Spread Dataset | 180 m | daily | OSF `10.17605/OSF.IO/F48RY`, 3,269 fires 2002-2021, ⚠ kriged day-of-burning under-predicts daily area |
| EFFIS Rapid Damage Assessment | 20 m | daily | api2.effis.emergency.copernicus.eu |
| Mesogeos datacube | 1 km | daily | github.com/Orion-AI-Lab/mesogeos. Mediterranean 2006-2022, 25,722 events |

### Spread benchmarks

| Benchmark | Content | Best published AP |
|---|---|---|
| Next Day Wildfire Spread | 1 km CONUS, 18,545 samples, 2012-2020 | ~0.28-0.34 |
| **WildfireSpreadTS** | 375 m, 607 fires, 13,607 images, 23 channels | 0.372 (UTAE), persistence 0.193 |
| WSTS+ | 1,005 fires, 24,462 images, 2016-2023 | 0.404 (SwinUnet, ImageNet-pretrained) |

⚠ **AP is base-rate dependent, never compare NDWS numbers to WSTS numbers.**

⚠ **Label noise floor.** VIIRS-derived perimeters score 0.71-0.93 F1 / IoU
against agency perimeters, and ~9 % of the area inside them is unburned
islands. No model should be credited with skill beyond that ceiling.

---

## 7. Licence summary

| Class | Datasets |
|---|---|
| Public domain (US Gov) | MTBS, Landsat, NIFC/WFIGS, FPA-FOD, LANDFIRE, GOES, HRRR, RTMA, FIRMS |
| Open Government Licence - Canada | NBAC, CNFDB, CFFDRS FBP fuels |
| CC-BY-4.0 | Satellite Embedding V1, TS-SatFire, Land8Fire, Icechunk ERA5 |
| CC0 | GABAM |
| Copernicus free & open | Sentinel-1/2/3, EFFIS, EMS, ERA5 |
| Apache-2.0 (model weights) | Prithvi-EO-2.0, TerraMind, Clay |
| ⚠ Verify before commercial use | Google Earth Engine compute, NLDN/ENTLN lightning, PRISM 800 m, Galileo & DOFA weights, FireSat / OroraTech |
