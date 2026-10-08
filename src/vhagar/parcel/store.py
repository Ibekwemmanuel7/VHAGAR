"""Parcel persistence behind a narrow protocol, with an in-memory default and an optional,
documented PostGIS adapter that is NOT a hard dependency.

The engine and API depend only on the ``ParcelStore`` protocol -- ``get`` a parcel by id and
``list_ids``. The in-memory implementation is all the tests and the demo need. A real
deployment would implement the same protocol over PostGIS (geometry stored as ``geography``,
queried with ``ST_*``), which is sketched in ``PostGISParcelStore`` but imports psycopg only
when actually constructed, so importing this module never requires a database driver.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from vhagar.parcel.schemas import Parcel

__all__ = ["ParcelStore", "InMemoryParcelStore", "PostGISParcelStore"]


@runtime_checkable
class ParcelStore(Protocol):
    """Minimal persistence surface the suitability service needs. Keeping it this small
    means the in-memory store and a real spatial database are interchangeable."""

    def get(self, parcel_id: str) -> Parcel | None:
        ...

    def list_ids(self) -> list[str]:
        ...


class InMemoryParcelStore:
    """Dict-backed store. Enough for tests, the demo, and single-process serving."""

    def __init__(self, parcels: dict[str, Parcel] | None = None) -> None:
        self._parcels: dict[str, Parcel] = dict(parcels or {})

    def put(self, parcel: Parcel) -> None:
        self._parcels[parcel.parcel_id] = parcel

    def get(self, parcel_id: str) -> Parcel | None:
        return self._parcels.get(parcel_id)

    def list_ids(self) -> list[str]:
        return sorted(self._parcels)


class PostGISParcelStore:
    """Optional PostGIS-backed store, satisfying the same protocol. ``psycopg`` is imported
    lazily in ``__init__`` so this module stays importable without a database driver and the
    test suite never needs one.

    Expected schema (documented, not created here)::

        CREATE TABLE parcels (
            parcel_id text PRIMARY KEY,
            name      text,
            geom      geography(Polygon, 4326) NOT NULL
        );

    ``get`` reads the ring back as GeoJSON via ``ST_AsGeoJSON`` and rebuilds a ``Parcel``,
    which recomputes centroid and area the same way as every other code path.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # noqa: F401  (imported lazily, optional dependency)
        except ImportError as exc:  # pragma: no cover - only hit without the optional driver
            raise RuntimeError(
                "PostGISParcelStore needs the optional 'psycopg' driver; install it, or use "
                "InMemoryParcelStore.") from exc
        self._psycopg = psycopg
        self._dsn = dsn

    def get(self, parcel_id: str) -> Parcel | None:  # pragma: no cover - needs a live DB
        import json
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT parcel_id, name, ST_AsGeoJSON(geom::geometry) "
                "FROM parcels WHERE parcel_id = %s", (parcel_id,))
            row = cur.fetchone()
        if row is None:
            return None
        pid, name, geojson = row
        ring = json.loads(geojson)["coordinates"][0]
        # drop the closing coordinate GeoJSON repeats; Parcel stores an open ring
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        return Parcel(parcel_id=pid, geometry=[list(pt) for pt in ring], name=name)

    def list_ids(self) -> list[str]:  # pragma: no cover - needs a live DB
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT parcel_id FROM parcels ORDER BY parcel_id")
            return [r[0] for r in cur.fetchall()]
