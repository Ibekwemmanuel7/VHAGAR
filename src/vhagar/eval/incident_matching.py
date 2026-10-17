"""Retrospective incident association, not ignition warning or pixel accuracy.

References are scoring-only inputs. Call this after prediction, with deduplicated
reference incidents and canonical event lineage IDs from a frozen run. Never pass
its matches back into prediction features. All observations, including masked
ones, must be supplied so misses remain auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction

from vhagar.intersect import haversine_m


def _time(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")


def _location(lon: float, lat: float) -> None:
    if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180
            and -90 <= lat <= 90):
        raise ValueError("invalid longitude/latitude")


@dataclass(frozen=True)
class ReferenceIncident:
    incident_id: str
    lon: float
    lat: float
    discovery_at: datetime
    source_revision: str
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.incident_id or not self.source_revision:
            raise ValueError("reference ID and source revision are required")
        _time(self.discovery_at)
        _location(self.lon, self.lat)


@dataclass(frozen=True)
class IncidentObservation:
    observation_id: str
    lineage_id: str
    lon: float
    lat: float
    observed_at: datetime
    known_at: datetime | None
    source_revision: str
    masked: bool = False

    def __post_init__(self):
        if not self.observation_id or not self.lineage_id or not self.source_revision:
            raise ValueError("observation ID, lineage ID and source revision are required")
        _time(self.observed_at)
        if self.known_at is not None:
            _time(self.known_at)
            if self.known_at < self.observed_at:
                raise ValueError("known_at precedes observation")
        _location(self.lon, self.lat)


def _assignment(pairs):
    """Unit-capacity min-cost flow with exact lexicographic tuple costs.

    Augment until no path remains (maximum cardinality), minimising total
    absolute time difference, then total distance. Sorted stable IDs and strict
    relaxation break remaining ties reproducibly, without floating big-M costs.
    Reverse edges allow an earlier assignment to be reconsidered.
    """
    graph = {}

    def edge(a, b, cost):
        forward = [b, cost, 1, None]
        reverse = [a, (-cost[0], -cost[1]), 0, forward]
        forward[3] = reverse
        graph.setdefault(a, []).append(forward)
        graph.setdefault(b, []).append(reverse)
        return forward

    source, sink = (0, ""), (3, "")
    for rid in sorted({r for r, _ in pairs}):
        edge(source, (1, rid), (0, 0))
    links = {}
    for rid, lid in sorted(pairs):
        witness = pairs[rid, lid]
        # Preserve the computed distance exactly during residual subtraction;
        # floating cancellation must not introduce negative-cost cycles.
        links[rid, lid] = edge((1, rid), (2, lid), (witness[0], Fraction(witness[1])))
    for lid in sorted({e for _, e in pairs}):
        edge((2, lid), sink, (0, 0))
    while source in graph:
        dist, previous = {source: (0, 0)}, {}
        for _ in range(len(graph) - 1):
            changed = False
            for node in sorted(graph):
                if node not in dist:
                    continue
                for link in graph[node]:
                    target, cost, capacity, _reverse = link
                    candidate = (dist[node][0] + cost[0], dist[node][1] + cost[1])
                    if capacity and (target not in dist or candidate < dist[target]):
                        dist[target] = candidate
                        previous[target] = (node, link)
                        changed = True
            if not changed:
                break
        if sink not in previous:
            break
        node = sink
        while node != source:
            parent, link = previous[node]
            link[2] = 0
            link[3][2] = 1
            node = parent
    return {rid: lid for (rid, lid), link in links.items() if link[2] == 0}


def score_incidents(
    references: list[ReferenceIncident],
    observations: list[IncidentObservation],
    *,
    as_of: datetime,
    prediction_run: str,
    radius_m: float = 5_000,
    window: timedelta = timedelta(hours=6),
) -> dict:
    """Score the entire caller-frozen eligible incident frame.

    Availability is conservative: unknown or later known_at never qualifies.
    Final references may postdate as_of because only scoring reads them. An
    unmatched lineage is unresolved, never automatically a false alarm. No
    independence, reference completeness, or operational gate is inferred.
    """
    _time(as_of)
    if not prediction_run:
        raise ValueError("prediction_run must identify frozen code/config/input manifest")
    if not math.isfinite(radius_m) or radius_m < 0 or window < timedelta(0):
        raise ValueError("matching tolerances must be finite and non-negative")
    refs = sorted(references, key=lambda r: r.incident_id)
    obs = sorted(observations, key=lambda o: o.observation_id)
    if len({r.incident_id for r in refs}) != len(refs):
        raise ValueError("deduplicate reference incident IDs before scoring")
    if len({o.observation_id for o in obs}) != len(obs):
        raise ValueError("deduplicate observation IDs before scoring")
    pairs, reasons = {}, {}
    for ref in refs:
        nearby = available = masked = False
        for item in obs:
            delta = abs(item.observed_at - ref.discovery_at)
            distance = haversine_m(ref.lat, ref.lon, item.lat, item.lon)
            if delta > window or distance > radius_m:
                continue
            nearby = True
            if item.known_at is None or item.known_at > as_of:
                continue
            available = True
            if item.masked:
                masked = True
                continue
            key = (ref.incident_id, item.lineage_id)
            # Integer microseconds avoid time rounding in the optimisation.
            micros = delta // timedelta(microseconds=1)
            witness = (micros, distance, item.observation_id)
            if key not in pairs or witness < pairs[key]:
                pairs[key] = witness
        reasons[ref.incident_id] = (
            "lineage_conflict" if any(r == ref.incident_id for r, _ in pairs)
            else "masked_only" if available and masked
            else "availability_unknown_or_after_cutoff" if nearby
            else "no_observation_within_tolerances"
        )
    matches = _assignment(pairs)
    ledger = []
    for ref in refs:
        lid = matches.get(ref.incident_id)
        row = {"incident_id": ref.incident_id, "source_revision": ref.source_revision,
               "quality_flags": list(ref.quality_flags), "lineage_id": lid,
               "miss_reason": None if lid else reasons[ref.incident_id]}
        if lid:
            micros, distance, oid = pairs[ref.incident_id, lid]
            row.update(observation_id=oid, absolute_time_seconds=micros / 1e6,
                       distance_m=distance)
        ledger.append(row)

    def encode(value):
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat(timespec="microseconds")
        raise TypeError(type(value).__name__)

    manifest = {"schema_version": 1, "prediction_run": prediction_run, "as_of": as_of,
                "radius_m": radius_m, "window_microseconds": window // timedelta(microseconds=1),
                "references": [asdict(r) for r in refs],
                "observations": [asdict(o) for o in obs]}
    manifest = json.loads(json.dumps(manifest, default=encode))
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    used = set(matches.values())
    return {"manifest": manifest, "manifest_sha256": fingerprint,
            "n_reference": len(refs), "n_matched": len(matches),
            "n_missed": len(refs) - len(matches),
            "detection_fraction": len(matches) / len(refs) if refs else None,
            "ledger": ledger,
            "unmatched_lineage_ids": sorted({o.lineage_id for o in obs} - used),
            "claim_limit": "Retrospective incident association only; no false-alarm, "
                           "burned-pixel accuracy, ignition lead-time or service claim."}
