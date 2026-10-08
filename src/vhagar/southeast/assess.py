"""Association, control, threat and shadow routing for one fire event.

Design rules (docs/25 sections 6 and 7), enforced here in code:

1. **Three separate outputs.** A permit match never means "safe". A fire with
   no matching record is not automatically a wildfire.
2. **No combination over candidates.** The association score is the best
   single candidate *group*. Noisy-OR is not used: correlated candidates make
   it overconfident (five candidates at 0.4 give about 0.92). Correlated
   records (same holder, same day) are merged before scoring.
3. **Ownership is not authorisation.** The owner-parcel rule only finds
   candidates. Its score is capped below the strong threshold, so it always
   ends in analyst review.
4. **Stale sources never support a downgrade.**
5. **UNKNOWN control never counts as IN_BOUNDS.**
6. **Location error is kept whole.** Distances subtract the detection's
   location error only to *find* candidates; escape tests require a detection
   to be outside the unit by more than its error.

The association score here is the release 1 **rule baseline**. It is a
documented heuristic, not a calibrated probability. The learned model replaces
it only after it beats this baseline on a held-out season (docs/25 section 7).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from vhagar.intersect import distance_to_ring_m, haversine_m, point_in_ring
from vhagar.southeast.records import GEOMETRY_ERROR_M, BurnRecord, BurnRegistry

__all__ = [
    "ALERTING_ACTIONS",
    "SENSOR_LOCATION_ERROR_M",
    "AssessConfig",
    "Asset",
    "Assessment",
    "Association",
    "Candidate",
    "Control",
    "Obs",
    "ObservedEvent",
    "Routing",
    "Threat",
    "assess_control",
    "assess_event",
    "assess_threat",
    "associate",
    "find_candidates",
    "route",
]

#: Default 1-sigma-ish location error of one detection, metres. Roughly half
#: the pixel diagonal plus geolocation error. Override per detection when the
#: product gives scan/track sizes.
SENSOR_LOCATION_ERROR_M: dict[str, float] = {
    "viirs": 400.0,
    "modis": 1_000.0,
    "goes": 3_000.0,
    "landsat": 60.0,
    "sentinel2": 40.0,
}

#: Shadow actions that put the event in front of a person.
ALERTING_ACTIONS = frozenset({"ALERT", "ALERT_ESCAPE", "ANALYST", "HOLD"})


# --- inputs ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Obs:
    """One detection in lon/lat, with its own location error."""

    lon: float
    lat: float
    when: datetime
    sensor: str = "viirs"
    frp_mw: float | None = None
    location_error_m: float | None = None
    #: Ground area of the pixel in m2 when the product gives it (VIIRS scan x
    #: track). Used only for the declared-area escape test.
    pixel_area_m2: float | None = None
    static_anomaly: bool = False

    @property
    def error_m(self) -> float:
        if self.location_error_m is not None:
            return self.location_error_m
        return SENSOR_LOCATION_ERROR_M.get(self.sensor.lower(), 1_000.0)


@dataclass(slots=True)
class ObservedEvent:
    event_id: str
    detections: list[Obs] = field(default_factory=list)

    @property
    def start(self) -> datetime:
        return min(d.when for d in self.detections)

    @property
    def end(self) -> datetime:
        return max(d.when for d in self.detections)

    @property
    def static_fraction(self) -> float:
        return sum(d.static_anomaly for d in self.detections) / len(self.detections)


@dataclass(frozen=True, slots=True)
class Asset:
    """A customer asset: a point, or a polygon ring in lon/lat."""

    asset_id: str
    lon: float
    lat: float
    ring: tuple[tuple[float, float], ...] | None = None
    customer_id: str | None = None


@dataclass(frozen=True, slots=True)
class AssessConfig:
    #: Accept detections this long after ``valid_to``: residual heat and
    #: next-morning detections of a completed burn.
    residual_h: float = 72.0
    #: Accept detections this long before ``valid_from`` as a soft early start.
    early_h: float = 2.0
    #: Score at or above which a DECLARED or PERMIT candidate is "strong".
    strong_score: float = 0.8
    #: Ceiling for the owner-parcel rule. Must stay below ``strong_score``.
    owner_only_cap: float = 0.5
    #: Temporal weights.
    early_weight: float = 0.5
    residual_weight: float = 0.75
    geometry_error_m: dict[str, float] = field(default_factory=lambda: dict(GEOMETRY_ERROR_M))
    #: Escape when the observed footprint exceeds declared acres by this factor.
    escape_area_factor: float = 1.5
    #: Optional escape test on peak observed FRP (MW). Off by default: a
    #: defensible surface-fire FRP ceiling for Southern pine is a research item.
    escape_peak_frp_mw: float | None = None
    #: Control is UNKNOWN when the newest detection is older than this and the
    #: burn is not reported complete.
    stale_observation_h: float = 6.0
    #: Threat distance bands from the nearest detection (less its error) to an
    #: asset edge.
    threat_high_m: float = 2_000.0
    threat_elevated_m: float = 8_000.0
    #: Maximum poll age per polled source. Unlisted non-customer sources use
    #: ``default_source_max_age``. Customer uploads are not polled.
    source_max_age: dict[str, timedelta] = field(default_factory=dict)
    default_source_max_age: timedelta = timedelta(hours=1)
    #: Static-source events (flares, plants) are logged, not associated.
    static_fraction_limit: float = 0.5
    shadow_mode: bool = True

    def __post_init__(self) -> None:
        if not self.owner_only_cap < self.strong_score:
            raise ValueError("owner_only_cap must be below strong_score")

    def max_age(self, source: str, is_customer: bool) -> timedelta | None:
        if is_customer:
            return None
        return self.source_max_age.get(source, self.default_source_max_age)

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), default=str, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --- outputs -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Candidate:
    record: BurnRecord
    rule: str                 # DECLARED | PERMIT | OWNER_ONLY
    spatial: float
    temporal: float
    score: float
    distance_m: float
    group: tuple[str, str]
    source_fresh: bool


@dataclass(frozen=True, slots=True)
class Association:
    level: str                # DECLARED_MATCH | PERMIT_MATCH | WEAK_MATCH | OWNER_ONLY | NONE
    score: float
    best: Candidate | None
    n_candidates: int
    n_groups: int

    @property
    def strong(self) -> bool:
        return self.level in ("DECLARED_MATCH", "PERMIT_MATCH")


@dataclass(frozen=True, slots=True)
class Control:
    status: str               # IN_BOUNDS | ESCAPE_SUSPECTED | UNKNOWN | NOT_APPLICABLE
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Threat:
    level: str                # LOW | ELEVATED | HIGH
    nearest_asset_id: str | None
    distance_m: float | None
    note: str = ""


@dataclass(frozen=True, slots=True)
class Routing:
    shadow_action: str        # DOWNGRADE | ALERT | ALERT_ESCAPE | ANALYST | HOLD | LOG_STATIC
    operational_action: str   # PASS_THROUGH in shadow mode
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Assessment:
    event_id: str
    assessed_at: datetime
    association: Association
    control: Control
    threat: Threat
    routing: Routing
    config_fingerprint: str

    def to_record(self) -> dict:
        """JSON-ready audit record: inputs that decided, outputs, config."""
        best = self.association.best
        return {
            "event_id": self.event_id,
            "assessed_at": self.assessed_at.isoformat(),
            "config": self.config_fingerprint,
            "association": {
                "level": self.association.level,
                "score": round(self.association.score, 4),
                "n_candidates": self.association.n_candidates,
                "n_groups": self.association.n_groups,
                "best_record_id": best.record.record_id if best else None,
                "best_record_version": best.record.ingested_at.isoformat() if best else None,
                "best_source": best.record.source if best else None,
                "best_rule": best.rule if best else None,
                "best_distance_m": round(best.distance_m, 1) if best else None,
                "best_source_fresh": best.source_fresh if best else None,
            },
            "control": {"status": self.control.status, "reasons": list(self.control.reasons)},
            "threat": {
                "level": self.threat.level,
                "nearest_asset_id": self.threat.nearest_asset_id,
                "distance_m": None if self.threat.distance_m is None
                else round(self.threat.distance_m, 1),
                "note": self.threat.note,
            },
            "routing": {
                "shadow_action": self.routing.shadow_action,
                "operational_action": self.routing.operational_action,
                "reasons": list(self.routing.reasons),
            },
        }


# --- geometry helpers ----------------------------------------------------------

def _dist_to_record_m(o: Obs, rec: BurnRecord) -> float:
    """Distance from a detection to the record geometry (0 inside a polygon)."""
    if rec.ring:
        return distance_to_ring_m(o.lon, o.lat, rec.ring)
    lon, lat = rec.anchor()
    return haversine_m(o.lat, o.lon, lat, lon)


def _point_search_radius_m(o: Obs, rec: BurnRecord, cfg: AssessConfig) -> float:
    return o.error_m + rec.geometry_error_m(cfg.geometry_error_m) + rec.unit_radius_m()


# --- candidates and association --------------------------------------------------

def _temporal(event: ObservedEvent, rec: BurnRecord, cfg: AssessConfig) -> float:
    """1 inside the valid window; reduced weight for early or residual; 0 outside."""
    early = timedelta(hours=cfg.early_h)
    residual = timedelta(hours=cfg.residual_h)
    lo, hi = event.start, event.end
    if hi < rec.valid_from - early or lo > rec.valid_to + residual:
        return 0.0
    if lo >= rec.valid_from and hi <= rec.valid_to:
        return 1.0
    w = 1.0
    if lo < rec.valid_from:
        w = min(w, cfg.early_weight)
    if hi > rec.valid_to:
        w = min(w, cfg.residual_weight)
    return w


def find_candidates(
    event: ObservedEvent,
    registry: BurnRegistry,
    now: datetime,
    cfg: AssessConfig,
    owner_of=None,
) -> list[Candidate]:
    """Candidate records for an event, using only what was known at ``now``.

    ``owner_of(lon, lat) -> holder_id | None`` is an optional parcel lookup for
    the OWNER_ONLY rule.
    """
    out: list[Candidate] = []
    for rec in registry.known_at(now):
        if rec.withdrawn:
            continue
        t = _temporal(event, rec, cfg)
        if t == 0.0:
            continue
        dists = [_dist_to_record_m(o, rec) for o in event.detections]
        dmin = min(dists)
        rule = "DECLARED" if rec.is_customer else "PERMIT"
        if rec.is_polygon:
            near = [d <= o.error_m for d, o in zip(dists, event.detections, strict=True)]
            spatial = sum(near) / len(near)
            hit = any(near)
        else:
            radii = [_point_search_radius_m(o, rec, cfg) for o in event.detections]
            ratios = [d / r for d, r in zip(dists, radii, strict=True)]
            hit = min(ratios) <= 1.0
            spatial = max(0.0, 1.0 - min(ratios)) if hit else 0.0
            # A point record cannot place the unit well: map the closest
            # sensible match to the strong band only when it is well inside.
            spatial = min(1.0, spatial * 1.25)
        if not hit:
            if owner_of is None or rec.holder_id is None:
                continue
            if not any(owner_of(o.lon, o.lat) == rec.holder_id for o in event.detections):
                continue
            rule = "OWNER_ONLY"
            spatial = cfg.owner_only_cap
        score = spatial * t
        if rule == "OWNER_ONLY":
            score = min(score, cfg.owner_only_cap)
        fresh = registry.is_fresh(rec.source, now, cfg.max_age(rec.source, rec.is_customer))
        out.append(Candidate(rec, rule, spatial, t, score, dmin, rec.group_key(), fresh))
    return out


def associate(candidates: list[Candidate], cfg: AssessConfig) -> Association:
    """Best single candidate group. Never a combination of candidates."""
    if not candidates:
        return Association("NONE", 0.0, None, 0, 0)
    groups: dict[tuple[str, str], Candidate] = {}
    for c in candidates:
        cur = groups.get(c.group)
        if cur is None or c.score > cur.score:
            groups[c.group] = c
    best = max(groups.values(), key=lambda c: (c.score, c.rule == "DECLARED"))
    if best.rule == "OWNER_ONLY":
        level = "OWNER_ONLY"
    elif best.score >= cfg.strong_score:
        level = "DECLARED_MATCH" if best.rule == "DECLARED" else "PERMIT_MATCH"
    else:
        level = "WEAK_MATCH"
    return Association(level, best.score, best, len(candidates), len(groups))


# --- control -----------------------------------------------------------------------

def assess_control(
    event: ObservedEvent,
    association: Association,
    now: datetime,
    cfg: AssessConfig,
    weather_outside_prescription: bool | None = None,
) -> Control:
    best = association.best
    if association.level == "NONE" or best is None:
        return Control("NOT_APPLICABLE")
    rec = best.record
    reasons: list[str] = []

    # Spatial escape: a detection outside the unit by more than its own error.
    # Skipped for OWNER_ONLY: the record geometry is known not to locate the
    # unit, so "outside" carries no information there.
    geometry_trusted = best.rule != "OWNER_ONLY"
    outside = 0
    if geometry_trusted:
        for o in event.detections:
            d = _dist_to_record_m(o, rec)
            limit = o.error_m if rec.is_polygon else _point_search_radius_m(o, rec, cfg)
            if d > limit:
                outside += 1
    if outside:
        reasons.append(f"{outside} detection(s) outside the declared area beyond location error")

    # Area escape, only when pixel areas are known and acres declared.
    areas = [o.pixel_area_m2 for o in event.detections if o.pixel_area_m2]
    if rec.acres and areas:
        # Distinct pixels approximated by distinct rounded positions.
        seen: dict[tuple[float, float], float] = {}
        for o in event.detections:
            if o.pixel_area_m2:
                seen[(round(o.lon, 3), round(o.lat, 3))] = o.pixel_area_m2
        observed_acres = sum(seen.values()) / 4_046.8564224
        if observed_acres > cfg.escape_area_factor * rec.acres:
            reasons.append(
                f"observed pixel area {observed_acres:.0f} ac exceeds "
                f"{cfg.escape_area_factor} x declared {rec.acres:.0f} ac")

    if cfg.escape_peak_frp_mw is not None:
        frps = [o.frp_mw for o in event.detections if o.frp_mw is not None]
        if frps and max(frps) > cfg.escape_peak_frp_mw:
            reasons.append(f"peak FRP {max(frps):.0f} MW above {cfg.escape_peak_frp_mw:.0f} MW")

    if weather_outside_prescription:
        reasons.append("weather outside the burn prescription")

    if reasons:
        return Control("ESCAPE_SUSPECTED", tuple(reasons))
    if not geometry_trusted:
        return Control("UNKNOWN", ("record geometry does not locate the burn unit",))

    completed = rec.completion_reported_at is not None and rec.completion_reported_at <= now
    if not completed and now - event.end > timedelta(hours=cfg.stale_observation_h):
        return Control("UNKNOWN", (f"no detection for more than {cfg.stale_observation_h:g} h",))
    return Control("IN_BOUNDS")


# --- threat --------------------------------------------------------------------------

def assess_threat(event: ObservedEvent, assets: list[Asset], cfg: AssessConfig) -> Threat:
    """Distance from the event to the nearest asset, less location error.

    Independent of association: a declared burn next to a substation is still
    a threat to the substation."""
    if not assets:
        return Threat("LOW", None, None, "no assets registered")
    best_id, best_d = None, math.inf
    for a in assets:
        for o in event.detections:
            if a.ring:
                inside = point_in_ring(o.lon, o.lat, a.ring)
                d = 0.0 if inside else distance_to_ring_m(o.lon, o.lat, a.ring)
            else:
                d = haversine_m(o.lat, o.lon, a.lat, a.lon)
            d = max(0.0, d - o.error_m)
            if d < best_d:
                best_id, best_d = a.asset_id, d
    if best_d <= cfg.threat_high_m:
        level = "HIGH"
    elif best_d <= cfg.threat_elevated_m:
        level = "ELEVATED"
    else:
        level = "LOW"
    return Threat(level, best_id, best_d)


# --- routing --------------------------------------------------------------------------

def route(
    event: ObservedEvent,
    association: Association,
    control: Control,
    threat: Threat,
    cfg: AssessConfig,
) -> Routing:
    """docs/25 section 6.7. The order matters: safety rules first."""
    reasons: list[str] = []
    if event.static_fraction >= cfg.static_fraction_limit:
        action = "LOG_STATIC"
        reasons.append("static thermal source")
    elif control.status == "ESCAPE_SUSPECTED":
        action = "ALERT_ESCAPE"
        reasons.extend(control.reasons)
    elif threat.level == "HIGH":
        action = "ALERT"
        reasons.append(f"asset {threat.nearest_asset_id} within {cfg.threat_high_m:.0f} m")
    elif association.level == "NONE":
        action = "ALERT"
        reasons.append("no consistent burn record")
    elif association.level in ("OWNER_ONLY", "WEAK_MATCH"):
        action = "ANALYST"
        reasons.append(f"association {association.level}")
    elif control.status == "UNKNOWN":
        action = "HOLD"
        reasons.extend(control.reasons)
    elif association.strong and control.status == "IN_BOUNDS":
        assert association.best is not None
        if association.best.source_fresh:
            action = "DOWNGRADE"
            reasons.append(f"{association.level} {association.best.record.record_id}, in bounds")
        else:
            action = "ANALYST"
            reasons.append(f"source {association.best.record.source} is stale")
    else:  # pragma: no cover - every combination is handled above
        action = "ANALYST"
        reasons.append("unhandled combination")
    operational = "PASS_THROUGH" if cfg.shadow_mode else action
    return Routing(action, operational, tuple(reasons))


def assess_event(
    event: ObservedEvent,
    registry: BurnRegistry,
    now: datetime,
    *,
    assets: list[Asset] | None = None,
    cfg: AssessConfig | None = None,
    owner_of=None,
    weather_outside_prescription: bool | None = None,
) -> Assessment:
    """Full assessment of one event with what was known at ``now``."""
    if not event.detections:
        raise ValueError("event has no detections")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    cfg = cfg or AssessConfig()
    if event.static_fraction >= cfg.static_fraction_limit:
        assoc = Association("NONE", 0.0, None, 0, 0)
    else:
        assoc = associate(find_candidates(event, registry, now, cfg, owner_of), cfg)
    control = assess_control(event, assoc, now, cfg, weather_outside_prescription)
    threat = assess_threat(event, assets or [], cfg)
    routing = route(event, assoc, control, threat, cfg)
    return Assessment(event.event_id, now, assoc, control, threat, routing, cfg.fingerprint())
