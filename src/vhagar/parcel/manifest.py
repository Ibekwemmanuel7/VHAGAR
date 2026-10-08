"""Versioned feature manifest (data contract) for parcel suitability.

Each feature declares the REAL dataset it would be sourced from in production (source,
vintage, spatial resolution, processing method, coverage, license caveat) so the pipeline
is reproducible and auditable. In this demo build every value is marked ``SYNTHETIC``,
because no proprietary or large datasets are downloaded here; the manifest is the contract
that a real ingest would have to satisfy, feature by feature, flipping each status to
``OBSERVED`` or ``DERIVED`` only when a real source is wired and validated.

Bump ``MANIFEST_VERSION`` whenever a feature's source, method, or resolution changes.
"""
from __future__ import annotations

from vhagar.parcel.schemas import EvidenceStatus, Provenance

__all__ = ["MANIFEST_VERSION", "FEATURE_MANIFEST", "provenance_for"]

MANIFEST_VERSION = "parcel-features-0.1.0"


def _p(feature, source, vintage, resolution, method, coverage, license_, status=EvidenceStatus.SYNTHETIC):
    return Provenance(feature=feature, source=source, vintage=vintage, resolution=resolution,
                      method=method, coverage=coverage, license=license_, status=status)


#: The data contract. Keys are feature names used throughout the engine.
FEATURE_MANIFEST: dict[str, Provenance] = {
    "slope_pct": _p(
        "slope_pct", "USGS 3DEP DEM", "2023", "10 m",
        "slope from DEM (Horn 1981), percent rise", "CONUS", "public domain"),
    "irradiance_kwh_m2_day": _p(
        "irradiance_kwh_m2_day", "NREL NSRDB", "2022", "~4 km",
        "annual mean global horizontal irradiance", "CONUS + territories", "public, attribution"),
    "mean_wind_ms_100m": _p(
        "mean_wind_ms_100m", "NREL WIND Toolkit", "2021", "2 km",
        "100 m hub-height annual mean wind speed", "CONUS offshore+onshore", "public, attribution"),
    "wildfire_exposure_0_100": _p(
        "wildfire_exposure_0_100", "VHAGAR T3 danger + T4 exposure", "live",
        "parcel scale", "VHAGAR fire-danger and exposure, rescaled 0-100 (the wildfire tie-in)",
        "VHAGAR serving regions", "internal, VHAGAR"),
    "floodplain_frac": _p(
        "floodplain_frac", "FEMA NFHL", "2024", "parcel overlay",
        "fraction of parcel in the 1% annual-chance floodplain", "mapped US communities",
        "public, FEMA terms"),
    "transmission_distance_km": _p(
        "transmission_distance_km", "HIFLD transmission lines", "2023", "line vector",
        "distance from parcel centroid to nearest transmission line", "US", "public, HIFLD terms"),
    "road_distance_km": _p(
        "road_distance_km", "US Census TIGER/Line roads", "2023", "line vector",
        "distance from parcel centroid to nearest road", "US", "public domain"),
    "protected_overlap_frac": _p(
        "protected_overlap_frac", "USGS PAD-US", "2024", "polygon overlay",
        "fraction of parcel overlapping a protected area", "US", "public domain"),
    "habitat_sensitivity_0_100": _p(
        "habitat_sensitivity_0_100", "USGS GAP / NatureServe proxy", "2023", "30 m",
        "ecological sensitivity index, rescaled 0-100", "CONUS", "mixed, see source terms"),
    "developed_cover_frac": _p(
        "developed_cover_frac", "USGS NLCD", "2021", "30 m",
        "fraction of parcel in developed land-cover classes", "CONUS", "public domain"),
    "market_support_status": _p(
        "market_support_status", "public planning documents (LLM-extracted evidence)", "varies",
        "document", "structured evidence extraction for HUMAN REVIEW, never an autonomous decision",
        "where documents exist", "source-document terms"),
}


def provenance_for(feature: str, status: EvidenceStatus | None = None) -> Provenance:
    """Return the manifest provenance for a feature, optionally overriding the status to
    reflect what actually happened for a given parcel (observed, derived, assumed, ...)."""
    base = FEATURE_MANIFEST[feature]
    if status is None or status == base.status:
        return base
    return Provenance(feature=base.feature, source=base.source, vintage=base.vintage,
                      resolution=base.resolution, method=base.method, coverage=base.coverage,
                      license=base.license, status=status)
