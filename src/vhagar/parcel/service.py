"""Thin service layer: look a parcel up in a store, score it, and serialise the result to a
plain JSON-ready dict for the API and the console panel.

Keeping serialisation here (not in the engine) means the schemas stay pure dataclasses and
there is exactly one place that decides the wire shape of a suitability result.
"""
from __future__ import annotations

from dataclasses import asdict

from vhagar.parcel.config import SCORING_CONFIG
from vhagar.parcel.engine import score_parcel
from vhagar.parcel.schemas import FeatureValue, ProposedUse, SuitabilityResult
from vhagar.parcel.store import ParcelStore

__all__ = ["result_to_dict", "ParcelSuitabilityService"]


def result_to_dict(result: SuitabilityResult) -> dict:
    """JSON-ready view of a suitability result. Component scores stay separate, provenance
    and completeness are included in full, and nothing is flattened into a single number."""
    return {
        "parcel_id": result.parcel_id,
        "use": result.use.value,
        "overall_score": result.overall_score,
        "overall_status": result.overall_status,
        "components": [
            {
                "component": c.component, "score": c.score, "weight": c.weight,
                "status": c.status, "reason_codes": list(c.reason_codes),
                "inputs_used": list(c.inputs_used), "missing": list(c.missing),
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


class ParcelSuitabilityService:
    """Scores parcels held in a ``ParcelStore``. The store and the scoring config are
    injected, so the same service runs over the in-memory demo store or a real PostGIS one."""

    def __init__(self, store: ParcelStore, config: dict = SCORING_CONFIG) -> None:
        self._store = store
        self._config = config

    def parcel_ids(self) -> list[str]:
        return self._store.list_ids()

    def score(self, parcel_id: str, use: ProposedUse | str,
              features: dict[str, FeatureValue]) -> dict:
        """Score a stored parcel. Raises ``KeyError`` if the parcel id is unknown."""
        parcel = self._store.get(parcel_id)
        if parcel is None:
            raise KeyError(parcel_id)
        if isinstance(use, str):
            use = ProposedUse(use)
        result = score_parcel(parcel, use, features, self._config)
        return result_to_dict(result)
