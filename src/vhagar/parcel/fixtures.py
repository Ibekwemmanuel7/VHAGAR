"""Clearly-labeled synthetic demo fixtures so the engine, tests, and UI work offline.

Nothing here is a real measurement. Every feature value is marked ``SYNTHETIC`` and
carries the manifest provenance (with its status overridden to synthetic), so any output
built from these fixtures truthfully reports itself as demo data. To go to production you
replace ``synthetic_feature`` calls with a real ingest that sets ``OBSERVED``/``DERIVED``.
"""
from __future__ import annotations

from vhagar.parcel.manifest import provenance_for
from vhagar.parcel.schemas import EvidenceStatus, FeatureValue, Parcel, ProposedUse

__all__ = ["synthetic_feature", "missing_feature", "demo_parcels", "demo_feature_sets"]

_UNITS = {
    "slope_pct": "percent", "irradiance_kwh_m2_day": "kWh/m2/day",
    "mean_wind_ms_100m": "m/s", "wildfire_exposure_0_100": "index",
    "floodplain_frac": "fraction", "transmission_distance_km": "km",
    "road_distance_km": "km", "protected_overlap_frac": "fraction",
    "habitat_sensitivity_0_100": "index", "developed_cover_frac": "fraction",
    "market_support_status": "category",
}


def synthetic_feature(name: str, value) -> FeatureValue:
    """A demo feature value, explicitly marked synthetic with manifest provenance."""
    return FeatureValue(name=name, value=value, unit=_UNITS.get(name, ""),
                        status=EvidenceStatus.SYNTHETIC,
                        provenance=provenance_for(name, EvidenceStatus.SYNTHETIC),
                        note="synthetic demo fixture")


def missing_feature(name: str) -> FeatureValue:
    """An explicitly-missing input (no silent gap-filling)."""
    return FeatureValue(name=name, value=None, unit=_UNITS.get(name, ""),
                        status=EvidenceStatus.MISSING,
                        provenance=provenance_for(name, EvidenceStatus.MISSING),
                        note="not available for this parcel")


def _square(clon, clat, half):
    return [[clon - half, clat - half], [clon + half, clat - half],
            [clon + half, clat + half], [clon - half, clat + half]]


def demo_parcels() -> dict[str, Parcel]:
    return {
        "P-SOLAR-01": Parcel("P-SOLAR-01", _square(-119.70, 35.10, 0.010), name="Kern County flat (demo)"),
        "P-CONSV-01": Parcel("P-CONSV-01", _square(-121.90, 39.60, 0.012), name="Sierra foothill tract (demo)"),
        "P-DC-01": Parcel("P-DC-01", _square(-121.30, 38.60, 0.006), name="Sacramento edge parcel (demo)"),
        "P-EXCL-01": Parcel("P-EXCL-01", _square(-118.90, 35.40, 0.010),
                            name="Flat, sunny, but inside a protected area (demo)"),
        "P-THIN-01": Parcel("P-THIN-01", _square(-120.00, 39.00, 0.0008), name="Tiny parcel, thin evidence (demo)"),
    }


def demo_feature_sets() -> dict[str, tuple[ProposedUse, dict[str, FeatureValue]]]:
    """Per-parcel (proposed use, feature values). Chosen to exercise a strong score, a
    conservation inversion, a data-center access case, a hard exclusion, and an
    insufficient-evidence case."""
    def fs(**kw):
        return {k: synthetic_feature(k, v) for k, v in kw.items()}

    sets = {
        "P-SOLAR-01": (ProposedUse.SOLAR, fs(
            slope_pct=2.1, irradiance_kwh_m2_day=6.1, transmission_distance_km=3.2,
            road_distance_km=0.8, wildfire_exposure_0_100=28, floodplain_frac=0.0,
            protected_overlap_frac=0.0, market_support_status="mixed")),
        "P-CONSV-01": (ProposedUse.CONSERVATION, fs(
            habitat_sensitivity_0_100=78, protected_overlap_frac=0.35, road_distance_km=1.1,
            developed_cover_frac=0.04, wildfire_exposure_0_100=55, market_support_status="supportive")),
        "P-DC-01": (ProposedUse.DATA_CENTER, fs(
            slope_pct=3.5, transmission_distance_km=1.4, road_distance_km=0.5,
            wildfire_exposure_0_100=22, floodplain_frac=0.05, protected_overlap_frac=0.0,
            market_support_status="unknown")),
        # excellent solar physics and access, but almost entirely inside a protected area:
        # the hard exclusion must return "ineligible", not a high score
        "P-EXCL-01": (ProposedUse.SOLAR, fs(
            slope_pct=1.5, irradiance_kwh_m2_day=6.4, transmission_distance_km=2.0,
            road_distance_km=0.5, wildfire_exposure_0_100=15, floodplain_frac=0.0,
            protected_overlap_frac=0.95, market_support_status="mixed")),
        # thin: missing market and access inputs -> components and overall should refuse
        "P-THIN-01": (ProposedUse.RESIDENTIAL, {
            "slope_pct": synthetic_feature("slope_pct", 9.0),
            "developed_cover_frac": missing_feature("developed_cover_frac"),
            "road_distance_km": missing_feature("road_distance_km"),
            "wildfire_exposure_0_100": missing_feature("wildfire_exposure_0_100"),
            "floodplain_frac": missing_feature("floodplain_frac"),
            "protected_overlap_frac": missing_feature("protected_overlap_frac"),
            "market_support_status": missing_feature("market_support_status")}),
    }
    return sets
