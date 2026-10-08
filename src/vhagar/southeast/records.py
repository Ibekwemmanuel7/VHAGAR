"""Burn records and a bitemporal registry.

A burn record is a declared or permitted burn: a customer unit polygon, a state
permit, or a federal prescribed-fire record. Sources differ a lot. Most state
permits give a point (often the holder's address), an acreage and a one-day
validity; customer uploads give a unit polygon. Every source is normalised to
:class:`BurnRecord`.

The registry is **bitemporal**. Each record version keeps its valid time
(``valid_from``/``valid_to``) and its ingest time (``ingested_at``), so the
platform can always answer "what did we know at the moment of the decision".
That is what makes a shadow-season evaluation honest: a permit that arrived
after the detection must not be allowed to explain the detection in replay.

Source freshness is tracked per polled source. A stale source must lower
confidence, never raise it: :mod:`vhagar.southeast.assess` refuses to
downgrade an alert on the strength of a stale source.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

__all__ = [
    "BURN_TYPES",
    "CUSTOMER_SOURCE",
    "GEOMETRY_ERROR_M",
    "GEOMETRY_KINDS",
    "STATUSES",
    "BurnRecord",
    "BurnRegistry",
    "records_from_csv",
    "records_from_geojson",
]

#: Source name for customer-declared burn plans. Other sources are free-form
#: names such as ``"GA_GFC"`` or ``"FL_FFS_OBA"``.
CUSTOMER_SOURCE = "CUSTOMER"

GEOMETRY_KINDS = ("unit_polygon", "address_point", "centroid")
STATUSES = ("planned", "active", "completed", "cancelled", "revoked_burn_ban")
BURN_TYPES = (
    "broadcast", "pile", "agricultural", "land_clearing", "silviculture", "other",
)
#: Statuses that withdraw an authorisation. A withdrawn record never explains
#: a fire.
WITHDRAWN = frozenset({"cancelled", "revoked_burn_ban"})

#: Default location error of the record geometry itself, in metres. A unit
#: polygon is exact by definition; a centroid is roughly placed; an address
#: point can be kilometres from the burn unit. These are starting values to be
#: replaced by per-source measurements from the first season.
GEOMETRY_ERROR_M: dict[str, float] = {
    "unit_polygon": 0.0,
    "centroid": 500.0,
    "address_point": 1_500.0,
}

_ACRE_M2 = 4_046.8564224


def _aware(dt: datetime | None, name: str) -> None:
    if dt is not None and dt.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class BurnRecord:
    """One version of a declared or permitted burn."""

    record_id: str
    source: str
    burn_type: str
    status: str
    geometry_kind: str
    valid_from: datetime
    valid_to: datetime
    ingested_at: datetime
    #: Lon/lat ring for ``unit_polygon`` records.
    ring: tuple[tuple[float, float], ...] | None = None
    #: Lon/lat point for ``address_point`` and ``centroid`` records.
    point: tuple[float, float] | None = None
    acres: float | None = None
    #: Holder or land-owner key. Used to group correlated records and for the
    #: owner-parcel candidate rule. Never evidence of authorisation on its own.
    holder_id: str | None = None
    customer_id: str | None = None
    certified_burner_id: str | None = None
    ignition_reported_at: datetime | None = None
    completion_reported_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id is required")
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")
        if self.geometry_kind not in GEOMETRY_KINDS:
            raise ValueError(f"geometry_kind {self.geometry_kind!r} not in {GEOMETRY_KINDS}")
        if self.burn_type not in BURN_TYPES:
            raise ValueError(f"burn_type {self.burn_type!r} not in {BURN_TYPES}")
        for name in ("valid_from", "valid_to", "ingested_at",
                     "ignition_reported_at", "completion_reported_at"):
            _aware(getattr(self, name), name)
        if self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        if self.geometry_kind == "unit_polygon":
            if not self.ring or len(self.ring) < 3:
                raise ValueError("unit_polygon records need a ring of at least 3 vertices")
        elif self.point is None:
            raise ValueError(f"{self.geometry_kind} records need a point")
        if self.acres is not None and (not math.isfinite(self.acres) or self.acres < 0):
            raise ValueError("acres must be a non-negative finite number")

    @property
    def is_customer(self) -> bool:
        return self.source == CUSTOMER_SOURCE

    @property
    def is_polygon(self) -> bool:
        return self.geometry_kind == "unit_polygon"

    @property
    def withdrawn(self) -> bool:
        return self.status in WITHDRAWN

    def anchor(self) -> tuple[float, float]:
        """Representative lon/lat: the point, or the vertex mean of the ring."""
        if self.ring:
            lons = [p[0] for p in self.ring]
            lats = [p[1] for p in self.ring]
            return sum(lons) / len(lons), sum(lats) / len(lats)
        assert self.point is not None
        return self.point

    def unit_radius_m(self) -> float:
        """Radius of a circle with the declared area. Zero when acres is unknown."""
        if not self.acres:
            return 0.0
        return math.sqrt(self.acres * _ACRE_M2 / math.pi)

    def geometry_error_m(self, table: dict[str, float] | None = None) -> float:
        return (table or GEOMETRY_ERROR_M)[self.geometry_kind]

    def group_key(self) -> tuple[str, str]:
        """Records with the same holder on the same valid day are correlated:
        one burn often produces several permits or several versions. They are
        scored as one candidate group so that their number is not evidence."""
        who = self.holder_id or self.record_id
        return who, self.valid_from.astimezone(UTC).date().isoformat()


class BurnRegistry:
    """Bitemporal store of burn record versions, with per-source freshness.

    >>> from datetime import datetime, UTC
    >>> t0 = datetime(2027, 1, 15, 8, tzinfo=UTC)
    >>> reg = BurnRegistry()
    >>> r = BurnRecord("P1", "GA_GFC", "silviculture", "active", "address_point",
    ...                t0, t0.replace(hour=23), t0, point=(-82.4, 31.0), acres=100)
    >>> reg.upsert(r)
    >>> len(reg.known_at(t0)), len(reg.known_at(t0.replace(hour=7)))
    (1, 0)
    """

    def __init__(self) -> None:
        self._versions: dict[str, list[BurnRecord]] = {}
        self._polls: dict[str, list[datetime]] = {}

    def upsert(self, record: BurnRecord) -> None:
        """Add a record version. Versions are kept, never overwritten."""
        versions = self._versions.setdefault(record.record_id, [])
        versions.append(record)
        versions.sort(key=lambda r: r.ingested_at)
        self.heartbeat(record.source, record.ingested_at)

    def heartbeat(self, source: str, at: datetime) -> None:
        """Record a successful poll of ``source`` at ``at``, with or without
        changes. Freshness is judged from these times."""
        _aware(at, "at")
        polls = self._polls.setdefault(source, [])
        polls.append(at)
        polls.sort()

    def known_at(self, t: datetime) -> list[BurnRecord]:
        """The latest version of every record ingested at or before ``t``."""
        out = []
        for versions in self._versions.values():
            known = [v for v in versions if v.ingested_at <= t]
            if known:
                out.append(known[-1])
        return out

    def versions(self, record_id: str) -> list[BurnRecord]:
        return list(self._versions.get(record_id, []))

    def last_poll(self, source: str, at: datetime) -> datetime | None:
        """Most recent successful poll of ``source`` at or before ``at``."""
        polls = [p for p in self._polls.get(source, []) if p <= at]
        return polls[-1] if polls else None

    def is_fresh(self, source: str, at: datetime, max_age: timedelta | None) -> bool:
        """True when the source was polled within ``max_age`` before ``at``.

        ``max_age=None`` means the source is not polled (customer uploads):
        the presence of the record is the whole of the evidence."""
        if max_age is None:
            return True
        last = self.last_poll(source, at)
        return last is not None and at - last <= max_age

    def sources(self) -> list[str]:
        return sorted(set(self._polls) | {v[-1].source for v in self._versions.values()})

    def __len__(self) -> int:
        return len(self._versions)


# --- loaders -----------------------------------------------------------------

def _parse_dt(s, name: str) -> datetime | None:
    if s in (None, ""):
        return None
    dt = s if isinstance(s, datetime) else datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"{name} {s!r} has no timezone offset")
    return dt


def _float_or_none(v) -> float | None:
    if v in (None, ""):
        return None
    return float(v)


def records_from_geojson(
    fc: dict,
    *,
    source: str = CUSTOMER_SOURCE,
    ingested_at: datetime,
) -> list[BurnRecord]:
    """Burn records from a GeoJSON FeatureCollection.

    Polygon features become ``unit_polygon`` records (outer ring only).
    MultiPolygon features are rejected, so that a unit is never silently
    truncated: split multi-part units into separate records. Point features
    become ``address_point`` records unless ``properties.geometry_kind`` says
    ``centroid``.

    Required properties: ``record_id``, ``valid_from``, ``valid_to`` (ISO 8601
    with offset). Optional: ``burn_type``, ``status``, ``acres``, ``holder_id``,
    ``customer_id``, ``certified_burner_id``, ``ignition_reported_at``,
    ``completion_reported_at``.
    """
    out = []
    for i, feat in enumerate(fc.get("features", [])):
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        ring = point = None
        if gtype == "Polygon":
            kind = "unit_polygon"
            ring = tuple((float(x), float(y)) for x, y, *_ in geom["coordinates"][0])
            if len(ring) > 3 and ring[0] == ring[-1]:
                ring = ring[:-1]
        elif gtype == "Point":
            kind = props.get("geometry_kind") or "address_point"
            x, y, *_ = geom["coordinates"]
            point = (float(x), float(y))
        else:
            raise ValueError(f"feature {i}: unsupported geometry type {gtype!r}")
        out.append(BurnRecord(
            record_id=str(props.get("record_id") or f"{source}-{i}"),
            source=source,
            burn_type=props.get("burn_type") or "broadcast",
            status=props.get("status") or "planned",
            geometry_kind=kind,
            valid_from=_parse_dt(props.get("valid_from"), "valid_from"),
            valid_to=_parse_dt(props.get("valid_to"), "valid_to"),
            ingested_at=ingested_at,
            ring=ring,
            point=point,
            acres=_float_or_none(props.get("acres")),
            holder_id=props.get("holder_id"),
            customer_id=props.get("customer_id"),
            certified_burner_id=props.get("certified_burner_id"),
            ignition_reported_at=_parse_dt(props.get("ignition_reported_at"), "ignition_reported_at"),
            completion_reported_at=_parse_dt(
                props.get("completion_reported_at"), "completion_reported_at"),
        ))
    return out


def records_from_csv(text: str, *, source: str, ingested_at: datetime) -> list[BurnRecord]:
    """Point records from a CSV with columns ``record_id, lon, lat, valid_from,
    valid_to`` and optional ``acres, burn_type, status, holder_id,
    geometry_kind, certified_burner_id``. This is the shape most state permit
    exports reduce to."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        out.append(BurnRecord(
            record_id=row["record_id"],
            source=source,
            burn_type=row.get("burn_type") or "other",
            status=row.get("status") or "active",
            geometry_kind=row.get("geometry_kind") or "address_point",
            valid_from=_parse_dt(row["valid_from"], "valid_from"),
            valid_to=_parse_dt(row["valid_to"], "valid_to"),
            ingested_at=ingested_at,
            point=(float(row["lon"]), float(row["lat"])),
            acres=_float_or_none(row.get("acres")),
            holder_id=row.get("holder_id") or None,
            certified_burner_id=row.get("certified_burner_id") or None,
        ))
    return out
