"""The sensor registry, one place that knows what is flying and when it stops.

Terminology, because it matters and is routinely confused
---------------------------------------------------------
**VIIRS is an instrument, not a satellite.** The Visible Infrared Imaging
Radiometer Suite flies *on* spacecraft:

    Suomi-NPP  carries VIIRS   -> data delivery ends 2026-11-01 13:00 UTC
    NOAA-20    carries VIIRS   -> secondary
    NOAA-21    carries VIIRS   -> PRIMARY since March 2024
    NOAA-22    will carry VIIRS -> JPSS-4, launch readiness 2027

So "switch from VIIRS to NOAA-21" is not a sensor change. It is a **platform
identifier change** in the ingest layer:
``VIIRS_SNPP_NRT -> VIIRS_NOAA21_NRT``. Same instrument, same 375 m I-band
contextual algorithm, same product family.

Two consequences that are *not* cosmetic:

1. **NOAA-21 carries a ~+10% positive FRP bias** relative to NOAA-20/S-NPP,
   attributed to a shift in the M13 spectral response toward a more transparent
   part of the atmosphere. Splice the platforms without correcting and your FRP
   time series gains a spurious step at the handover. See ``frp_bias_factor``.
2. **The CONUS inter-satellite interval doubles**, from ~25 min (three
   platforms) to ~50 min (two). Anything that assumes ~25 min spacing --
   alert dedup windows, fire growth-rate estimators -- must be re-tuned.

What replaces MODIS
-------------------
MODIS (Terra/Aqua) reaches end of mission late 2026 / early 2027. Nothing
replaces its **10:30 morning overpass** in the US programme: all JPSS platforms
share one plane at ~13:25 LTAN. The morning orbit now belongs to Europe --
Sentinel-3 SLSTR at ~10:00 and Metop-SG-A1 METimage at ~09:30.

Sentinel-3 SLSTR, honestly
--------------------------
Promoting SLSTR to primary is right for Europe and for the night/morning band,
but be clear-eyed about the daytime limitation: **S7 (3.74 um) saturates at
~311 K**, and the FRP algorithm cannot process daytime scenes where >1% of
*background* pixels are saturated. That knocks out much of peak-summer daytime
Mediterranean and southern-CONUS coverage. The F1 high-gain channel exists for
this, but the fully F1-based daytime path is still listed as future work.

SLSTR's published <5 MW detection floor is a **night-time** result. Do not
quote it for daytime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

__all__ = ["Platform", "SensorSpec", "SENSORS", "PLATFORMS", "active_platforms", "get_sensor"]


@dataclass(frozen=True, slots=True)
class SensorSpec:
    """Physical and product characteristics of a fire-capable instrument."""

    name: str
    kind: str                       # "leo" | "geo"
    mir_wavelength_um: float
    tir_wavelength_um: float
    #: MIR saturation brightness temperature (K). Saturation is censoring.
    mir_saturation_k: float
    resolution_m: float
    #: Wooster constant key in ``vhagar.physics.frp.WOOSTER_A``, or None.
    wooster_key: str | None
    swir_bands_um: tuple[float, ...] = ()
    notes: str = ""

    @property
    def has_swir(self) -> bool:
        """SWIR bands enable the colour-temperature discriminant for gas flares.

        Flares burn at a median ~1750 K vs ~1000 K for biomass; a 1750 K Wien
        peak sits at 1.66 um, so flares are bright at 1.6/2.2 um where a
        wildfire is orders of magnitude weaker.
        """
        return len(self.swir_bands_um) > 0


@dataclass(frozen=True, slots=True)
class Platform:
    """A spacecraft carrying one or more fire-capable instruments."""

    name: str
    sensor: str
    operator: str
    status: str                     # primary | secondary | planned | ending | retired
    local_time: str                 # equator crossing / GEO longitude
    data_start: date | None = None
    #: Hard date after which no more data. This is the field that should drive
    #: an alert in your pipeline, not a comment in a README.
    data_end: date | None = None
    #: Multiplicative FRP correction to bring this platform onto the
    #: NOAA-20 / S-NPP scale. 1.0 means no known bias.
    frp_bias_factor: float = 1.0
    access: dict[str, str] = field(default_factory=dict)
    caveats: tuple[str, ...] = ()


SENSORS: dict[str, SensorSpec] = {
    "viirs": SensorSpec(
        name="VIIRS I-band / M13",
        kind="leo",
        mir_wavelength_um=3.74,          # I4; M13 at 4.05 um carries FRP
        tir_wavelength_um=11.45,         # I5
        mir_saturation_k=650.0,          # M13, dual gain; I4 saturates ~367 K
        resolution_m=375.0,
        wooster_key=None,                # no published fit; re-derive before use
        notes=(
            "375 m I-band contextual algorithm. Aggregation caps pixel growth at "
            "~800 m at swath edge vs MODIS' ~4.8 km, which is why VIIRS beats "
            "MODIS on small fires off nadir. M13 (4.05 um) overlaps the CO2 band, "
            "so its atmospheric attenuation can be ~2x MODIS' MIR channel."
        ),
    ),
    "slstr": SensorSpec(
        name="Sentinel-3 SLSTR",
        kind="leo",
        mir_wavelength_um=3.742,         # S7, with F1 as the high-gain twin
        tir_wavelength_um=10.85,         # S8 / F2
        mir_saturation_k=311.0,          # S7 -- low, and the core daytime problem
        resolution_m=1000.0,
        wooster_key=None,
        swir_bands_um=(1.613, 2.25),
        notes=(
            "S6 (2.25 um) gives night active-fire detection and gas-flare "
            "discrimination. Night detection floor <5 MW (night only). Daytime "
            "scenes with >1% saturated background S7 pixels are not processed. "
            "S7/F1 mis-registration ~1 km requires a clustering step. Dual-view "
            "(oblique) geometry is for SST aerosol correction and is irrelevant "
            "to fire -- budget revisit on nadir-only numbers."
        ),
    ),
    "abi": SensorSpec(
        name="GOES-R ABI",
        kind="geo",
        mir_wavelength_um=3.9,
        tir_wavelength_um=11.2,
        mir_saturation_k=400.0,
        resolution_m=2000.0,
        wooster_key="goes_imager",
        swir_bands_um=(2.24,),
        notes=(
            "5 min CONUS / 10 min full disk. Minimum detectable fire ~0.004 km2 "
            "at nadir. Processing cut off beyond 80 deg view zenith."
        ),
    ),
    "fci": SensorSpec(
        name="MTG-I FCI",
        kind="geo",
        mir_wavelength_um=3.8,
        tir_wavelength_um=10.5,
        mir_saturation_k=400.0,          # verify against the FCI ATBD
        resolution_m=1000.0,
        wooster_key=None,
        swir_bands_um=(2.2,),
        notes="1 km, 10 min. LSA-SAF FRP-Pixel product LSA-509 -- DEMONSTRATION status.",
    ),
    "seviri": SensorSpec(
        name="MSG SEVIRI",
        kind="geo",
        mir_wavelength_um=3.9,
        tir_wavelength_um=10.8,
        mir_saturation_k=335.0,
        resolution_m=3000.0,
        wooster_key="seviri",
        notes=(
            "3 km, 15 min. LSA-SAF FRP-Pixel LSA-502 is OPERATIONAL -- keep it as "
            "the European geostationary feed until LSA-509 is promoted."
        ),
    ),
    "modis": SensorSpec(
        name="MODIS",
        kind="leo",
        mir_wavelength_um=3.96,
        tir_wavelength_um=11.0,
        mir_saturation_k=331.0,          # band 22; band 21 saturates ~500 K
        resolution_m=1000.0,
        wooster_key="modis_c6",
        notes="End of mission late 2026 / early 2027. Climatology source only.",
    ),
    "metimage": SensorSpec(
        name="Metop-SG METimage",
        kind="leo",
        mir_wavelength_um=3.7,
        tir_wavelength_um=10.8,
        mir_saturation_k=float("nan"),   # not published
        resolution_m=500.0,
        wooster_key=None,
        notes=(
            "20 channels, 500 m, 2715 km swath, 09:30 LTDN -- the best available "
            "replacement for Terra's morning overpass. NO operational fire/FRP L2 "
            "product confirmed as of 2026. Prospective, not procurable."
        ),
    ),
}


PLATFORMS: dict[str, Platform] = {
    # ---- JPSS: all carry VIIRS, all in one plane at ~13:25 LTAN --------
    "snpp": Platform(
        name="Suomi-NPP",
        sensor="viirs",
        operator="NOAA/NASA",
        status="ending",
        local_time="13:25 asc",
        data_start=date(2011, 10, 28),
        data_end=date(2026, 11, 1),
        access={"firms": "VIIRS_SNPP_NRT", "gee": "NASA/LANCE/SNPP_VIIRS/C2"},
        caveats=("Data delivery ceases 2026-11-01 13:00 UTC, including HRD direct broadcast (URT).",),
    ),
    "noaa20": Platform(
        name="NOAA-20 (JPSS-1)",
        sensor="viirs",
        operator="NOAA",
        status="secondary",
        local_time="13:25 asc",
        data_start=date(2017, 11, 18),
        access={"firms": "VIIRS_NOAA20_NRT", "gee": "NASA/LANCE/NOAA20_VIIRS/C2"},
        caveats=("Past its 7-year design life.",),
    ),
    "noaa21": Platform(
        name="NOAA-21 (JPSS-2)",
        sensor="viirs",
        operator="NOAA",
        status="primary",
        local_time="13:25 asc",
        data_start=date(2023, 2, 23),
        frp_bias_factor=1.10,
        access={"firms": "VIIRS_NOAA21_NRT", "gee": ""},
        caveats=(
            "~+10% FRP bias vs NOAA-20/S-NPP (M13 spectral response shift). "
            "Divide by frp_bias_factor to place on the NOAA-20 scale.",
            "NO Google Earth Engine collection exists. If your pipeline runs in "
            "GEE you must ingest FIRMS API output into your own asset.",
            "Data before 2023-02-23 are unreliable.",
            "No VIIRS_NOAA21_SP standard-processing stream confirmed.",
        ),
    ),
    "noaa22": Platform(
        name="NOAA-22 (JPSS-4)",
        sensor="viirs",
        operator="NOAA",
        status="planned",
        local_time="13:25 asc",
        data_start=date(2027, 1, 1),     # launch readiness 2027; treat as slippable
        access={},
        caveats=("Launch readiness date 2027. JPSS-4 launches BEFORE JPSS-3.",),
    ),
    # ---- Sentinel-3 ----------------------------------------------------
    "sentinel3a": Platform(
        name="Sentinel-3A",
        sensor="slstr",
        operator="ESA/EUMETSAT",
        status="primary",
        local_time="10:00 desc",
        data_start=date(2016, 2, 16),
        access={
            "cdse_stac": "sentinel-3-sl-2-frp-nrt",
            "cdse_type": "SL_2_FRP___",
            "eumetsat": "EO:EUM:DAT:0417",
            "gee": "",
        },
        caveats=("Past its 7-year design life.", "NOT available in Google Earth Engine."),
    ),
    "sentinel3b": Platform(
        name="Sentinel-3B",
        sensor="slstr",
        operator="ESA/EUMETSAT",
        status="primary",
        local_time="10:00 desc",
        data_start=date(2018, 4, 25),
        access={"cdse_type": "SL_2_FRP___", "eumetsat": "EO:EUM:DAT:0417"},
        caveats=("Past its 7-year design life.",),
    ),
    "sentinel3c": Platform(
        name="Sentinel-3C",
        sensor="slstr",
        operator="ESA/EUMETSAT",
        status="planned",
        local_time="10:00 desc",
        data_start=date(2027, 4, 1),     # ~6 months commissioning after Sep 2026 launch
        access={"cdse_type": "SL_2_FRP___"},
        caveats=("Launched 2026-09-14. Do not plan on operational FRP before ~Q2 2027.",),
    ),
    # ---- Geostationary --------------------------------------------------
    "goes19": Platform(
        name="GOES-19 (GOES-East)",
        sensor="abi",
        operator="NOAA",
        status="primary",
        local_time="75.2W",
        data_start=date(2025, 4, 4),
        access={"s3": "noaa-goes19", "gee": "NOAA/GOES/19/FDCC"},
        caveats=("No replacement imager until GeoXO ~2032.",),
    ),
    "goes18": Platform(
        name="GOES-18 (GOES-West)",
        sensor="abi",
        operator="NOAA",
        status="primary",
        local_time="137W",
        data_start=date(2023, 1, 10),
        access={"s3": "noaa-goes18", "gee": "NOAA/GOES/18/FDCC"},
    ),
    "mtg_i1": Platform(
        name="Meteosat-12 (MTG-I1)",
        sensor="fci",
        operator="EUMETSAT",
        status="secondary",
        local_time="0 deg",
        data_start=date(2025, 1, 1),
        access={"eumetsat": "EO:EUM:DAT:0682", "lsasaf": "LSA-509"},
        caveats=("FRP product is DEMONSTRATION status, not operational.",),
    ),
    "msg": Platform(
        name="Meteosat Second Generation",
        sensor="seviri",
        operator="EUMETSAT",
        status="primary",
        local_time="0 deg",
        access={"lsasaf": "LSA-502"},
        caveats=("Operational, but finite remaining life. The European GEO fire feed until LSA-509 is promoted.",),
    ),
    # ---- Legacy ---------------------------------------------------------
    "terra": Platform(
        name="Terra",
        sensor="modis",
        operator="NASA",
        status="ending",
        local_time="~09:00 desc (drifted from 10:30)",
        data_end=date(2026, 12, 31),
        access={"gee": "MODIS/061/MOD14A1"},
        caveats=("Orbit drifted; end of mission. NO morning-orbit VIIRS replacement is planned.",),
    ),
    "aqua": Platform(
        name="Aqua",
        sensor="modis",
        operator="NASA",
        status="ending",
        local_time="~15:50 asc (drifted from 13:30)",
        data_end=date(2026, 12, 31),
        access={"gee": "MODIS/061/MYD14A1"},
        caveats=(
            "Drift to ~15:50 has been sampling the mid-afternoon fire peak. "
            "That sample disappears at end of mission and is not replaced by any "
            "public-sector polar mission."
        ,),
    ),
}


def get_sensor(platform: str) -> SensorSpec:
    """Instrument spec for a platform key."""
    p = PLATFORMS.get(platform.lower())
    if p is None:
        raise KeyError(f"unknown platform {platform!r}; known: {sorted(PLATFORMS)}")
    return SENSORS[p.sensor]


def active_platforms(on: date, kind: str | None = None) -> list[Platform]:
    """Platforms delivering data on a given date.

    Use this rather than a hard-coded list, so that 2026-11-01 changes the
    pipeline's behaviour instead of silently producing a coverage hole.

    >>> from datetime import date
    >>> before = {p.name for p in active_platforms(date(2026, 10, 1), kind="leo")}
    >>> after = {p.name for p in active_platforms(date(2026, 12, 1), kind="leo")}
    >>> "Suomi-NPP" in before and "Suomi-NPP" not in after
    True
    """
    out = []
    for p in PLATFORMS.values():
        if p.data_start and on < p.data_start:
            continue
        if p.data_end and on >= p.data_end:
            continue
        if p.status in {"planned", "retired"}:
            continue
        if kind and SENSORS[p.sensor].kind != kind:
            continue
        out.append(p)
    return sorted(out, key=lambda q: q.name)


def frp_to_reference_scale(frp_mw, platform: str):
    """Place a platform's FRP on the common (NOAA-20 / S-NPP) scale.

    NOAA-21 reads ~10% high; without this, a fire-intensity time series shows a
    step change at the S-NPP -> NOAA-21 handover that is an instrument artifact,
    not a geophysical trend.
    """
    import numpy as np

    p = PLATFORMS.get(platform.lower())
    if p is None:
        raise KeyError(f"unknown platform {platform!r}")
    return np.asarray(frp_mw, dtype=np.float64) / p.frp_bias_factor


def coverage_report(on: date) -> str:
    """Human-readable constellation status, for logs and provenance."""
    leo = active_platforms(on, "leo")
    geo = active_platforms(on, "geo")
    lines = [f"VHAGAR constellation on {on.isoformat()}"]
    lines.append(f"  polar ({len(leo)}): " + ", ".join(f"{p.name} [{p.local_time}]" for p in leo))
    lines.append(f"  geo   ({len(geo)}): " + ", ".join(f"{p.name} [{p.local_time}]" for p in geo))
    viirs = [p for p in leo if p.sensor == "viirs"]
    if len(viirs) < 3:
        lines.append(
            f"  NOTE: only {len(viirs)} VIIRS platform(s). CONUS inter-satellite "
            "interval is ~50 min, not ~25 min -- re-tune dedup and growth-rate windows."
        )
    return "\n".join(lines)
