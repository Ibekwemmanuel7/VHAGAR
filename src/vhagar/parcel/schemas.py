"""Typed schemas for the parcel-suitability slice, plus geometry helpers.

Pure stdlib (dataclasses, enum, math) so everything here runs in the core CI env. Every
value that flows into a score carries an ``EvidenceStatus`` and a ``Provenance`` record, so
the output can always say which inputs were observed, derived, assumed, or synthetic.

Geometry. Area is computed geodesically with ``pyproj.Geod`` when the optional ``geo``
extra is installed, and otherwise with a local equal-area projection anchored at the
centroid (accurate to well under 1% for parcel-scale polygons). The method used is
recorded on the parcel (``area_method``). ``validate_ring`` rejects rings that are not a
usable simple polygon before anything is scored. Production ingest of real parcel layers
belongs in PostGIS or GDAL/OGR (see ``store.PostGISParcelStore``); these helpers are for
caller-supplied polygons and the demo.
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
    "normalize_ring",
    "validate_ring",
    "parcel_centroid",
    "parcel_area_ha",
    "MAX_RING_VERTICES",
]

_EARTH_R_M = 6_371_008.8  # mean Earth radius (IUGG)
MAX_RING_VERTICES = 1_000

Ring = list[list[float]]


class ProposedUse(str, Enum):
    """The proposed land use a parcel is scored against."""

    SOLAR = "solar"
    WIND = "wind"
    CONSERVATION = "conservation"
    RESIDENTIAL = "residential"
    DATA_CENTER = "data_center"


class EvidenceStatus(str, Enum):
    """Where a feature value came from. A score is only as trustworthy as the status of
    the inputs behind it."""

    OBSERVED = "observed"  # a real measurement or authoritative dataset value
    DERIVED = "derived"  # computed from observed inputs
    ASSUMED = "assumed"  # a documented default standing in for a missing input
    SYNTHETIC = "synthetic"  # demo/fixture value, never real
    MISSING = "missing"  # not available for this parcel


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
    value: float | str | None
    unit: str
    status: EvidenceStatus
    provenance: Provenance | None = None
    note: str | None = None


@dataclass
class Parcel:
    """A parcel geometry (outer lon/lat ring, EPSG:4326) and light metadata. A closing
    vertex equal to the first is dropped, so open and closed rings behave identically."""

    parcel_id: str
    geometry: Ring
    name: str | None = None
    centroid: tuple[float, float] | None = None
    area_ha: float | None = None
    area_method: str | None = None

    def __post_init__(self) -> None:
        self.geometry = normalize_ring(self.geometry)
        if self.geometry and self.centroid is None:
            self.centroid = parcel_centroid(self.geometry)
        if self.geometry and self.area_ha is None:
            area, method = _area_ha_with_method(self.geometry)
            self.area_ha = round(area, 3)
            self.area_method = method


@dataclass
class ComponentScore:
    """A per-dimension suitability score (0..100), or ``None`` when evidence is too thin.
    Components are returned separately on purpose. ``in_overall`` is False for components
    reported as evidence status only (market/community)."""

    component: str  # physical | access | hazard | market
    score: float | None
    weight: float
    status: str  # scored | insufficient_evidence
    reason_codes: list[str] = field(default_factory=list)
    inputs_used: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    in_overall: bool = True


@dataclass
class Completeness:
    """A completeness and uncertainty statement over the inputs that fed a result."""

    observed: int
    derived: int
    assumed: int
    synthetic: int
    missing: int
    total: int
    completeness_pct: float  # (observed + derived) / total
    has_synthetic: bool


@dataclass
class SuitabilityResult:
    """The full, auditable suitability output for one parcel and one proposed use.

    ``overall_status`` is one of ``scored``, ``ineligible`` (a hard exclusion or the area
    gate fired; no score), or ``insufficient_evidence`` (a required input or component is
    missing; no score). ``evidence_grade`` is ``evidence_based`` only when every required
    input present is observed or derived; ``provisional`` when any is assumed;
    ``illustrative`` when any is synthetic; ``no_evidence`` when no required input is present.
    """

    parcel_id: str
    use: ProposedUse
    overall_score: float | None
    overall_status: str
    components: list[ComponentScore]
    completeness: Completeness
    provenance: list[Provenance]
    reason_codes: list[str]
    explanation: str
    config_version: str
    disclaimer: str
    evidence_grade: str = "illustrative"
    exclusions: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    market_value: str | None = None  # supportive | mixed | opposed | unknown, or None
    market_status: str = "missing"  # EvidenceStatus of the market input


def normalize_ring(ring) -> Ring:
    """Return the ring as ``[[lon, lat], ...]`` floats without a duplicated closing vertex."""
    pts = [[float(c[0]), float(c[1])] for c in (ring or [])]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def validate_ring(ring) -> list[str]:
    """Problems that make ``ring`` unusable as a parcel polygon (empty list means valid):
    malformed coordinates, out-of-range lon/lat, fewer than 3 distinct vertices, too many
    vertices, self-intersection, or zero area."""
    try:
        pts = normalize_ring(ring)
    except (TypeError, ValueError, IndexError):
        return ["malformed_coordinates"]
    problems: list[str] = []
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in pts):
        return ["non_finite_coordinates"]
    if any(not (-180.0 <= x <= 180.0 and -90.0 <= y <= 90.0) for x, y in pts):
        problems.append("coordinates_out_of_range")
    if len({(x, y) for x, y in pts}) < 3:
        return problems + ["fewer_than_3_vertices"]
    if len(pts) > MAX_RING_VERTICES:
        return problems + ["too_many_vertices"]
    if _self_intersects(pts):
        problems.append("self_intersection")
    if _local_area_ha(pts) <= 0.0:
        problems.append("zero_area")
    return problems


def parcel_centroid(ring) -> tuple[float, float]:
    """Area-weighted polygon centroid (lon, lat), computed in a local metric plane. Falls
    back to the vertex mean for a degenerate ring."""
    pts = normalize_ring(ring)
    n = len(pts)
    if n == 0:
        return (0.0, 0.0)
    lon0 = sum(p[0] for p in pts) / n
    lat0 = sum(p[1] for p in pts) / n
    if n < 3:
        return (lon0, lat0)
    xy = _to_local(pts, lon0, lat0)
    a2 = cx = cy = 0.0
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        a2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a2) < 1e-9:
        return (lon0, lat0)
    cx /= 3.0 * a2
    cy /= 3.0 * a2
    coslat = math.cos(math.radians(lat0))
    return (lon0 + math.degrees(cx / (_EARTH_R_M * coslat)), lat0 + math.degrees(cy / _EARTH_R_M))


def parcel_area_ha(ring) -> float:
    """Area in hectares (geodesic via pyproj when available, else local equal-area)."""
    return _area_ha_with_method(normalize_ring(ring))[0]


# ----------------------------------------------------------------------------- internals


def _to_local(pts: Ring, lon0: float, lat0: float) -> list[tuple[float, float]]:
    coslat = math.cos(math.radians(lat0))
    return [(math.radians(x - lon0) * _EARTH_R_M * coslat, math.radians(y - lat0) * _EARTH_R_M)
            for x, y in pts]


def _local_area_ha(pts: Ring) -> float:
    if len(pts) < 3:
        return 0.0
    lon0 = sum(p[0] for p in pts) / len(pts)
    lat0 = sum(p[1] for p in pts) / len(pts)
    xy = _to_local(pts, lon0, lat0)
    a = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % len(xy)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0 / 10_000.0


def _area_ha_with_method(pts: Ring) -> tuple[float, str]:
    if len(pts) < 3:
        return 0.0, "degenerate"
    try:
        from pyproj import Geod  # optional 'geo' extra
    except ImportError:
        return _local_area_ha(pts), "local_equal_area"
    area_m2, _perim = Geod(ellps="WGS84").polygon_area_perimeter(
        [p[0] for p in pts], [p[1] for p in pts])
    return abs(area_m2) / 10_000.0, "geodesic_wgs84"


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r) -> bool:
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0) and 0 not in (o1, o2, o3, o4):
        return True
    return ((o1 == 0 and on_seg(a, c, b)) or (o2 == 0 and on_seg(a, d, b))
            or (o3 == 0 and on_seg(c, a, d)) or (o4 == 0 and on_seg(c, b, d)))


def _self_intersects(pts: Ring) -> bool:
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # adjacent edges share a vertex by construction
            if _segments_cross(*edges[i], *edges[j]):
                return True
    return False
