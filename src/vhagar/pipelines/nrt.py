"""Near-real-time hot path.

Target: satellite file landing on S3 to alert in **under 60 seconds**, against
a product SLA of 5-15 minutes.

    GOES-19 ABI L2 FDC -> s3://noaa-goes19
      -> SNS NewGOES19Object
        -> SQS (prefix filter on ABI-L2-FDCC/FDCF, DLQ attached)
          -> worker: decode -> temporal anomaly -> fuse -> classify
            -> PostGIS/TimescaleDB -> Redis pub/sub -> SSE -> browser

Why SQS rather than SNS straight to Lambda: a downstream outage should queue
work, not drop it. Fire detections are not replayable.

Why this is not Dagster: Dagster owns the *cold* path (backfills, cube
materialisation, retraining, reconciliation of the NRT stream against
later-arriving authoritative products). Putting a 5-minute-cadence stream
through an orchestrator adds scheduling latency for no benefit.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from vhagar.harmonize.fusion import Detection, FireEvent, cluster_detections, event_features

log = logging.getLogger(__name__)

__all__ = ["NRTConfig", "NRTPipeline"]


@dataclass(slots=True)
class NRTConfig:
    region: str = "conus"
    #: How far back to look when re-clustering on each new granule.
    lookback_hours: float = 24.0
    #: Extra spatial slack for GEO/LEO matching, on top of per-sensor tolerance.
    parallax_tolerance_m: float = 2_000.0
    max_gap_hours: float = 12.0
    #: Minimum calibrated event probability to raise an alert.
    alert_threshold: float = 0.7
    #: Suppress events whose detections are dominated by known persistent
    #: industrial/volcanic heat sources.
    static_anomaly_reject: float = 0.6


class NRTPipeline:
    """Stateful incremental event tracker.

    Each granule is folded into the running detection buffer, events are
    re-clustered over the lookback window, and any event crossing the alert
    threshold is emitted as an *update to an existing incident object* rather
    than as an independent point detection. Downstream consumers subscribe to
    incident IDs, not pixels.
    """

    def __init__(self, config: NRTConfig | None = None, classifier=None) -> None:
        self.config = config or NRTConfig()
        self.classifier = classifier
        self._buffer: list[Detection] = []
        self._alerted: set[str] = set()

    def ingest(self, detections: Iterable[Detection], now: datetime | None = None) -> list[FireEvent]:
        """Fold new detections in and return the current event set."""
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(hours=self.config.lookback_hours)
        self._buffer = [d for d in self._buffer if d.when >= cutoff]
        self._buffer.extend(d for d in detections if d.when >= cutoff)
        return cluster_detections(
            self._buffer,
            max_gap_hours=self.config.max_gap_hours,
            extra_tolerance_m=self.config.parallax_tolerance_m,
        )

    def score(self, event: FireEvent) -> float:
        """Calibrated wildfire probability for an event."""
        feats = event_features(event)
        if feats["static_anomaly_fraction"] >= self.config.static_anomaly_reject:
            return 0.0
        if self.classifier is None:
            # Transparent fallback until a trained classifier is registered:
            # multi-sensor agreement plus persistence, not a hidden heuristic.
            score = 0.35
            if feats["multi_sensor_agreement"]:
                score += 0.35
            if feats["n_detections"] >= 3:
                score += 0.2
            if feats.get("frp_growth_mw_per_h", 0) and feats["frp_growth_mw_per_h"] > 0:
                score += 0.1
            return min(score, 0.99)
        import numpy as np

        keys = sorted(feats)
        x = np.array([[feats[k] for k in keys]], dtype=np.float64)
        return float(self.classifier.predict_proba(x)[0, 1])

    def alerts(self, events: list[FireEvent]) -> list[dict]:
        """Events crossing the threshold, as alert payloads."""
        out = []
        for ev in events:
            p = self.score(ev)
            if p < self.config.alert_threshold:
                continue
            x, y = ev.centroid()
            out.append(
                {
                    "event_id": ev.event_id,
                    "probability": round(p, 3),
                    "first_seen": ev.start.isoformat(),
                    "last_seen": ev.end.isoformat(),
                    "duration_h": round(ev.duration_h, 2),
                    "n_detections": len(ev.detections),
                    "sensors": sorted(ev.sensors),
                    "centroid_projected": [x, y],
                    "region": self.config.region,
                    "is_update": ev.event_id in self._alerted,
                }
            )
            self._alerted.add(ev.event_id)
        return out
