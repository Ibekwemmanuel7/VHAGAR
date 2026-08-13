"""FastAPI serving layer.

Design notes that matter more than the code:

* **SSE, not WebSockets**, for the alert stream. Traffic is one-directional
  server -> client, SSE auto-reconnects, and it survives corporate proxies
  without extra protocol handling. Reserve WebSockets for collaborative
  annotation features.
* **Raster tiles come from TiTiler, not from this process.** Serving COG and
  Zarr tiles through the API process couples model serving latency to map
  panning. Vector tiles come from Martin (live PostGIS) and PMTiles on object
  storage (static layers: fuels, WUI, historical perimeters).
* **The channel spec is enforced at load time.** A model refuses to run
  against an input stack it was not trained on -- otherwise that failure shows
  up as quietly wrong predictions rather than an error.
"""

from __future__ import annotations

from datetime import date
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field

    _FASTAPI = True
except ImportError:  # pragma: no cover
    _FASTAPI = False


if _FASTAPI:

    class DangerRequest(BaseModel):
        region: str = Field("conus", pattern="^(conus|canada|europe)$")
        target_date: date
        lead_days: int = Field(1, ge=0, le=15)

    class DangerResponse(BaseModel):
        region: str
        target_date: date
        lead_days: int
        # Three quantities, never collapsed into one "risk" number.
        fwi: float | None = None
        fwi_percentile: float | None = None
        ignition_probability: float | None = None
        expected_burned_area_ha: float | None = None
        model_version: str
        calibration_ece: float | None = None

    class EventResponse(BaseModel):
        event_id: str
        probability: float
        first_seen: str
        last_seen: str
        sensors: list[str]
        n_detections: int
        is_update: bool

    app = FastAPI(
        title="VHAGAR API",
        version="0.1.0",
        description="Multi-sensor wildfire intelligence: detection, burned area, danger, spread.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        from vhagar import __version__

        return {"status": "ok", "version": __version__}

    @app.get("/v1/events", response_model=list[EventResponse])
    def list_events(
        region: str = Query("conus", pattern="^(conus|canada|europe)$"),
        min_probability: float = Query(0.7, ge=0.0, le=1.0),
        hours: int = Query(24, ge=1, le=168),
    ) -> list[Any]:
        """Active fire events. Backed by the TimescaleDB hypertable."""
        raise HTTPException(status_code=501, detail="wire to PostGIS/TimescaleDB")

    @app.post("/v1/danger", response_model=DangerResponse)
    def danger(_req: DangerRequest) -> Any:
        """Fire danger, ignition probability and expected burned area."""
        raise HTTPException(status_code=501, detail="wire to the T3 model registry")

    @app.get("/v1/stream/alerts")
    def stream_alerts(region: str = Query("conus")) -> Any:
        """Server-sent event stream of new and updated incidents."""
        raise HTTPException(status_code=501, detail="wire to Redis pub/sub")

else:  # pragma: no cover
    app = None

    def _missing(*_a, **_k):
        raise ImportError("vhagar.serve.api requires: pip install 'vhagar[serve]'")
