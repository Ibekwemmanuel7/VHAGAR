"""Thin service layer: look up or build a parcel, score it, and serialise the result to a
plain JSON-ready dict for the API and the parcel page.

Two entry points:

* ``ParcelSuitabilityService.score``: a parcel held in a ``ParcelStore`` (the demo store,
  or PostGIS in a real deployment).
* ``ParcelSuitabilityService.score_geometry``: a caller-supplied polygon plus
  caller-supplied feature values. No feature pipeline is connected in this build, so
  nothing is looked up, estimated, or filled: inputs the caller does not supply are missing,
  and the engine returns ``insufficient_evidence``. A caller cannot make its own values
  count as evidence: values declared ``observed`` or ``derived`` are stored as
  ``caller_declared`` (the declared status is kept in provenance), and the result grade is
  ``unverified`` until a server-side provenance check exists.

Serialisation lives here (not in the engine) so the schemas stay pure dataclasses and there
is exactly one place that decides the wire shape of a result.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from vhagar.parcel.config import SCORING_CONFIG
from vhagar.parcel.engine import score_parcel
from vhagar.parcel.manifest import FEATURE_MANIFEST
from vhagar.parcel.schemas import (
    EvidenceStatus,
    FeatureValue,
    Parcel,
    ProposedUse,
    Provenance,
    SuitabilityResult,
    validate_ring,
)
from vhagar.parcel.store import ParcelStore

__all__ = [
    "result_to_dict",
    "ParcelSuitabilityService",
    "parcel_from_geometry",
    "features_from_payload",
    "CALLER_INPUT_NOTE",
]

CALLER_INPUT_NOTE = (
    "No feature pipeline is connected in this build. Only caller-supplied inputs are used; "
    "nothing is looked up, estimated, or filled. Caller-declared evidence statuses are "
    "not independently verified: values declared observed or derived are graded "
    "caller_declared, and the result is unverified, not evidence based.")

_CATEGORY_FEATURES = {"market_support_status"}


def result_to_dict(result: SuitabilityResult, parcel: Parcel | None = None) -> dict[str, Any]:
    """JSON-ready view of a suitability result. Component scores stay separate, and
    provenance, completeness, evidence grade, and exclusions are included in full."""
    out: dict[str, Any] = {
        "parcel_id": result.parcel_id,
        "use": result.use.value,
        "overall_score": result.overall_score,
        "overall_status": result.overall_status,
        "evidence_grade": result.evidence_grade,
        "exclusions": list(result.exclusions),
        "missing_required": list(result.missing_required),
        "market_evidence": {"value": result.market_value, "status": result.market_status,
                            "in_overall": False},
        "components": [
            {
                "component": c.component, "score": c.score, "weight": c.weight,
                "status": c.status, "in_overall": c.in_overall,
                "reason_codes": list(c.reason_codes), "inputs_used": list(c.inputs_used),
                "missing": list(c.missing),
            } for c in result.components
        ],
        "completeness": asdict(result.completeness),
        "provenance": [
            {
                "feature": p.feature, "source": p.source, "vintage": p.vintage,
                "resolution": p.resolution, "method": p.method, "coverage": p.coverage,
                "license": p.license, "status": p.status.value,
            } for p in result.provenance
        ],
        "reason_codes": list(result.reason_codes),
        "explanation": result.explanation,
        "config_version": result.config_version,
        "disclaimer": result.disclaimer,
    }
    if parcel is not None:
        out["parcel"] = {"name": parcel.name, "area_ha": parcel.area_ha,
                         "area_method": parcel.area_method, "centroid": parcel.centroid}
    return out


def parcel_from_geometry(geometry: Any, parcel_id: str = "custom",
                         name: str | None = None) -> Parcel:
    """Build a ``Parcel`` from a GeoJSON Polygon (outer ring used; holes are not supported
    in this slice) or a bare ``[[lon, lat], ...]`` ring. Raises ``ValueError`` with the
    validation problems when the polygon is not usable."""
    ring = geometry
    if isinstance(geometry, dict):
        if geometry.get("type") != "Polygon":
            raise ValueError("geometry must be a GeoJSON Polygon or a [[lon, lat], ...] ring")
        coords = geometry.get("coordinates") or []
        if not coords:
            raise ValueError("geometry has no coordinates")
        if len(coords) > 1:
            raise ValueError("polygons with holes are not supported in this slice")
        ring = coords[0]
    if not isinstance(ring, list):
        raise ValueError("geometry must be a GeoJSON Polygon or a [[lon, lat], ...] ring")
    problems = validate_ring(ring)
    if problems:
        raise ValueError("invalid geometry: " + ", ".join(problems))
    return Parcel(parcel_id=parcel_id, geometry=ring, name=name)


def features_from_payload(payload: dict[str, Any] | None) -> dict[str, FeatureValue]:
    """Parse caller-supplied features: ``{name: {"value": ..., "status": ..., "source"?,
    "vintage"?}}``. ``status`` is required and must be an ``EvidenceStatus``. Unknown
    feature names and wrong value types raise ``ValueError``; nothing is defaulted."""
    out: dict[str, FeatureValue] = {}
    for name, spec in (payload or {}).items():
        if name not in FEATURE_MANIFEST:
            raise ValueError(f"unknown feature: {name}")
        if not isinstance(spec, dict) or "status" not in spec:
            raise ValueError(f"feature {name} needs an object with a 'status'")
        try:
            status = EvidenceStatus(spec["status"])
        except ValueError as exc:
            raise ValueError(f"feature {name}: invalid status {spec['status']!r}") from exc
        declared = status
        if status in (EvidenceStatus.OBSERVED, EvidenceStatus.DERIVED):
            status = EvidenceStatus.CALLER_DECLARED  # no server-side provenance check yet
        value = spec.get("value")
        if status != EvidenceStatus.MISSING:
            if name in _CATEGORY_FEATURES:
                if not isinstance(value, str):
                    raise ValueError(f"feature {name} must be a string category")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"feature {name} must be numeric")
        base = FEATURE_MANIFEST[name]
        prov = Provenance(
            feature=name, source=str(spec.get("source") or "caller-supplied (unverified)"),
            vintage=str(spec.get("vintage") or "unknown"), resolution="unknown",
            method=f"supplied by the caller (declared {declared.value}; not verified)",
            coverage="this parcel",
            license="caller responsibility", status=status)
        out[name] = FeatureValue(name=name, value=None if status == EvidenceStatus.MISSING
                                 else value, unit="", status=status, provenance=prov,
                                 note=f"caller-supplied; manifest source would be {base.source}")
    return out


class ParcelSuitabilityService:
    """Scores parcels. The store and the scoring config are injected, so the same service
    runs over the in-memory demo store or a real PostGIS one."""

    def __init__(self, store: ParcelStore, config: dict = SCORING_CONFIG) -> None:
        self._store = store
        self._config = config

    def parcel_ids(self) -> list[str]:
        return self._store.list_ids()

    def score(self, parcel_id: str, use: ProposedUse | str,
              features: dict[str, FeatureValue]) -> dict[str, Any]:
        """Score a stored parcel. Raises ``KeyError`` if the parcel id is unknown."""
        parcel = self._store.get(parcel_id)
        if parcel is None:
            raise KeyError(parcel_id)
        result = score_parcel(parcel, ProposedUse(use), features, self._config)
        return result_to_dict(result, parcel)

    def score_geometry(self, geometry: Any, use: ProposedUse | str,
                       features_payload: dict[str, Any] | None,
                       parcel_id: str = "custom") -> dict[str, Any]:
        """Score a caller-supplied polygon with caller-supplied features only. Raises
        ``ValueError`` for an invalid geometry, use, or feature payload."""
        parcel = parcel_from_geometry(geometry, parcel_id=parcel_id)
        features = features_from_payload(features_payload)
        result = score_parcel(parcel, ProposedUse(use), features, self._config)
        out = result_to_dict(result, parcel)
        out["input_mode"] = "caller_supplied"
        out["input_note"] = CALLER_INPUT_NOTE
        if not features:
            out["reason_codes"].append("no_feature_pipeline")
        return out
