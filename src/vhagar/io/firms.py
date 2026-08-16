"""NASA FIRMS active fire API.

FIRMS is consumed, not competed with. Its **ultra real-time (URT)** feed
delivers MODIS detections ~25 s and VIIRS ~50 s after observation over CONUS,
Puerto Rico and Hawaii via a direct-broadcast antenna network -- an end-to-end
latency under a minute that no independent pipeline will beat. URT/RT records
are replaced by the standard NRT product after 6 hours.

Latency tiers, for planning:

    URT   < 60 s          direct broadcast, CONUS/PR/HI only
    RT    60-90 min faster than global NRT
    NRT   ~3 h            LANCE, global
    SP    ~2 months       standard science product, best geolocation

Set the ``FIRMS_MAP_KEY`` environment variable (free registration).
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime

__all__ = ["FIRMS_SOURCES", "FirmsClient", "FirmsRecord"]

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

#: Source identifiers accepted by the area API.
FIRMS_SOURCES = {
    "viirs_snpp_nrt": "VIIRS_SNPP_NRT",
    "viirs_noaa20_nrt": "VIIRS_NOAA20_NRT",
    "viirs_noaa21_nrt": "VIIRS_NOAA21_NRT",
    "modis_nrt": "MODIS_NRT",
    "viirs_snpp_sp": "VIIRS_SNPP_SP",
    "modis_sp": "MODIS_SP",
}


@dataclass(frozen=True, slots=True)
class FirmsRecord:
    latitude: float
    longitude: float
    brightness: float          # BT of channel I4/band 21, kelvin
    scan: float
    track: float
    acq_datetime: datetime
    satellite: str
    instrument: str
    confidence: str            # 'l'/'n'/'h' for VIIRS, 0-100 for MODIS
    version: str
    bright_t31: float
    frp: float                 # MW
    daynight: str

    @property
    def is_night(self) -> bool:
        return self.daynight.upper().startswith("N")

    @property
    def dt_mir_tir(self) -> float:
        """Brightness temperature difference, the core contextual fire signal."""
        return self.brightness - self.bright_t31


class FirmsClient:
    """Minimal FIRMS area-API client.

    >>> client = FirmsClient(map_key="demo")            # doctest: +SKIP
    >>> recs = client.area(                              # doctest: +SKIP
    ...     source="viirs_noaa20_nrt",
    ...     bbox=(-125.0, 32.0, -114.0, 42.0),
    ...     day_range=1,
    ... )
    """

    def __init__(self, map_key: str | None = None, timeout: float = 60.0) -> None:
        self.map_key = map_key or os.environ.get("FIRMS_MAP_KEY")
        if not self.map_key:
            raise ValueError(
                "FIRMS map key required. Register free at "
                "https://firms.modaps.eosdis.nasa.gov/api/map_key/ and set FIRMS_MAP_KEY."
            )
        self.timeout = timeout

    def _url(
        self,
        source: str,
        bbox: tuple[float, float, float, float],
        day_range: int,
        start: date | None,
    ) -> str:
        src = FIRMS_SOURCES.get(source, source)
        if not 1 <= day_range <= 10:
            raise ValueError("day_range must be 1..10")
        west, south, east, north = bbox
        area = f"{west},{south},{east},{north}"
        url = f"{BASE_URL}/{self.map_key}/{src}/{area}/{day_range}"
        if start is not None:
            url += f"/{start.isoformat()}"
        return url

    def area_csv(
        self,
        source: str,
        bbox: tuple[float, float, float, float],
        day_range: int = 1,
        start: date | None = None,
    ) -> str:
        """Raw FIRMS CSV text for a bbox/day-range. Kept separate from :meth:`area`
        so a caller can persist exactly what FIRMS returned and re-parse it later
        with :func:`parse_firms_csv`, no lossy round-trip through the dataclass."""
        import urllib.request

        url = self._url(source, bbox, day_range, start)
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8")

    def area(
        self,
        source: str,
        bbox: tuple[float, float, float, float],
        day_range: int = 1,
        start: date | None = None,
    ) -> list[FirmsRecord]:
        """Fetch detections within a bounding box (west, south, east, north)."""
        return parse_firms_csv(self.area_csv(source, bbox, day_range, start))


def parse_firms_csv(text: str) -> list[FirmsRecord]:
    """Parse the FIRMS CSV payload.

    Column names differ slightly between MODIS and VIIRS products (``bright_ti4``
    vs ``brightness``); both are handled.
    """
    out: list[FirmsRecord] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            acq = datetime.strptime(
                f"{row['acq_date']} {row['acq_time'].zfill(4)}", "%Y-%m-%d %H%M"
            ).replace(tzinfo=UTC)
            out.append(
                FirmsRecord(
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    brightness=float(row.get("bright_ti4") or row.get("brightness") or "nan"),
                    scan=float(row.get("scan", "nan")),
                    track=float(row.get("track", "nan")),
                    acq_datetime=acq,
                    satellite=row.get("satellite", ""),
                    instrument=row.get("instrument", ""),
                    confidence=str(row.get("confidence", "")),
                    version=row.get("version", ""),
                    bright_t31=float(row.get("bright_ti5") or row.get("bright_t31") or "nan"),
                    frp=float(row.get("frp", "nan")),
                    daynight=row.get("daynight", ""),
                )
            )
        except (KeyError, ValueError):
            continue
    return out
