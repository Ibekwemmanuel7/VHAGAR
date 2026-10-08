"""Typed schemas for the parcel-suitability slice, plus a couple of geometry helpers.

Pure stdlib (dataclasses, enum, math), so everything here runs in the core CI env and is
unit-tested without a geospatial stack. Every value that flows into a score carries an
``EvidenceStatus`` and a ``Provenance`` record, so the output can always say which inputs
were observed, derived, assumed, or synthetic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "ProposedUse",
    "EvidenceStatus",
    "Provenance",
    "FeatureValue",
    "Parcel",
    "ComponentScore",
    "Completeness",
    "SuitabilityResult",
    "parcel_centroid",
    "parcel_area_ha",
]

_EARTH_R_M = 6_371_000.0


class ProposedUse(str, Enum):
    """The proposed land use a parcel is scored against."""

    SOLAR = "solar"
    WIND = "wind"
    CONSERVATION = "conservation"
    RESIDENTIAL = "residential"
    DATA_CENTER = "data_center"


class EvidenceStatus(str, Enum):
    """Where a feature value came from. This is the honesty backbone: a score is only as
    trustworthy as the status of the inputs behind it."""

    OBSERVED = "observed"        # a real measurement or authoritative dataset value
    DERIVED = "derived"          # computed from observed inputs
    ASSUMED = "assumed"          # a documented default standing in for a missing input
    SYNTHETIC = "synthetic"      # demo/fixture value, never real
    MISSING = "missing"          # not available for this parcel


@dataclass(frozen=True)
class Provenance:
    """A data-contract record for one feature input: enough to reproduce and audit it."""

    feature: str
    source: str
    vintage: str
    resolution: str
    method: str
    coverage: str
    license: str
    status: EvidenceStatus


@dataclass
class FeatureValue:
    """One feature for one parcel, with its status and (optionally) its provenance."""

    name: str
    value: float | None
    unit: str
    status: EvidenceStatus
    provenance: Provenance | None = None
    note: str | None = None


@dataclass
class Parcel:
    """A parcel geometry (outer lon/lat ring) and light metadata."""

    parcel_id: str
    geometry: list                       # outer ring: [[lon, lat], ...]
    name: str | None = None
    centroid: tuple[float, float] | None = None
    area_ha: float | None = None

    def __post_init__(self):
        if self.geometry and self.centroid is None:
            self.centroid = parcel_centroid(self.geometry)
        if self.geometry and self.area_ha is None:
            self.area_ha = round(parcel_area_ha(self.geometry), 3)


@dataclass
class ComponentScore:
    """A per-dimension suitability score (0..100), or ``None`` when evidence is too thin.
    Components are returned separately on purpose; there is no opaque single aggregate."""

    component: str                       # physical | access | hazard | market
    score: float | None
    weight: float
    status: str                          # scored | insufficient_evidence
    reason_codes: list[str] = field(default_factory=list)
    inputs_used: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


@dataclass
class Completeness:
    """A completeness and uncertainty statement over the inputs that fed a result."""

    observed: int
    derived: int
    assumed: int
    synthetic: int
    missing: int
    total: int
    completeness_pct: float               # (observed + derived) / total
    has_synthetic: bool


@dataclass
class SuitabilityResult:
    """The full, auditable suitability output for one parcel and one proposed use."""

    parcel_id: str
    use: ProposedUse
    overall_score: float | None
    overall_status: str                   # scored | insufficient_evidence
    components: list[ComponentScore]
    completeness: Completeness
    provenance: list[Provenance]
    reason_codes: list[str]
    explanation: str
    config_version: str
    disclaimer: str


def parcel_centroid(ring) -> tuple[float, float]:
    """Vertex-mean centroid (lon, lat) of a ring. Adequate for a parcel-scale polygon."""
    n = len(ring)
    if n == 0:
        return (0.0, 0.0)
    return (sum(c[0] for c in ring) / n, sum(c[1] for c in ring) / n)


def parcel_area_ha(ring) -> float:
    """Area in hectares of a lon/lat ring, via a local equal-area metric projection
    anchored at the centroid (shoelace in metres). Mirrors VHAGAR's practice of doing
    area in an equal-area frame rather than in raw degrees."""
    if len(ring) < 3:
        return 0.0
    lon0, lat0 = parcel_centroid(ring)
    coslat = math.cos(math.radians(lat0))
    pts = [(math.radians(c[0] - lon0) * _EARTH_R_M * coslat,
            math.radians(c[1] - lat0) * _EARTH_R_M) for c in ring]
    a = 0.0
    m = len(pts)
    for i in range(m):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % m]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0 / 10_000.0
