"""Parcel Evidence & Suitability: a small, rigorous parcel-intelligence slice.

VHAGAR began in wildfire. This subpackage carries the same discipline, sourced data
with provenance, spatially-valid evaluation, explicit uncertainty, and customer-readable
explanations, into a parcel-suitability workflow. It is deliberately NOT a land AVM, a
bankable valuation, or a real community-sentiment product; see ``docs/25_PARCEL_INTELLIGENCE.md``
for the honest scope. The wildfire tiers and the parcel slice are kept separate on
purpose, and wildfire exposure enters here only as one explicit hazard input.
"""
from vhagar.parcel.schemas import (
    Completeness,
    ComponentScore,
    EvidenceStatus,
    FeatureValue,
    Parcel,
    ProposedUse,
    Provenance,
    SuitabilityResult,
    parcel_area_ha,
    parcel_centroid,
)

__all__ = [
    "ProposedUse",
    "EvidenceStatus",
    "Provenance",
    "FeatureValue",
    "Parcel",
    "ComponentScore",
    "Completeness",
    "SuitabilityResult",
    "parcel_area_ha",
    "parcel_centroid",
]
