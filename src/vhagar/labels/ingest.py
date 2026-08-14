"""Ingestion adapters: raw label sources normalised into the event registry.

Each source has its own field names, date formats and quirks. The job here is to
map them onto the common :class:`~vhagar.labels.registry.FireEventRecord` so that
splits and datasets never see a source-specific schema.

The design keeps the fallible part testable. Every ``normalize_*`` function takes
already-parsed rows (plain mappings) and is pure: no file IO, no network, no geo
dependency. A thin ``read_*`` wrapper does the actual file reading (GeoParquet or
shapefile via pyogrio, lazily imported) and hands rows to the normaliser. So the
field mapping, the date parsing and the quality assignment are unit-tested with a
handful of synthetic rows, while the heavy IO stays at the edge.

MTBS first
----------
MTBS is the T2 training source: US, 1984-present, analyst-QC, and crucially it
carries the **continuous dNBR/RBR severity raster**, not just a boundary. That
severity path is what makes a record trainable for a pixel model; a perimeter
alone is not (see :meth:`FireEventRecord.assert_trainable`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date

from vhagar.labels.registry import FireEventRecord, LabelSource

__all__ = ["normalize_mtbs", "read_mtbs"]


def _parse_date(value) -> date | None:
    """Parse the date formats MTBS extracts appear in, or None."""
    if value in (None, "", "0", 0):
        return None
    if isinstance(value, date):
        return value
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            from datetime import datetime

            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _first(row: Mapping, *keys, default=None):
    """First present, non-empty value among candidate keys (case-insensitive)."""
    lower = {k.lower(): v for k, v in row.items()}
    for k in keys:
        v = lower.get(k.lower())
        if v not in (None, ""):
            return v
    return default


def normalize_mtbs(
    rows: Iterable[Mapping],
    region: str = "conus",
    geometry_dir: str | None = None,
    severity_dir: str | None = None,
) -> list[FireEventRecord]:
    """Normalise MTBS burned-area records into :class:`FireEventRecord`.

    ``rows`` are mappings with MTBS attribute fields (from the bundle's shapefile
    or a GeoParquet export). Handles the common field-name variants across MTBS
    vintages. ``geometry_dir``/``severity_dir`` are prefixes for the per-fire
    geometry and dNBR severity rasters, so ``severity_path`` is populated and the
    record is trainable.
    """
    out: list[FireEventRecord] = []
    for row in rows:
        fire_id = _first(row, "Event_ID", "event_id", "MTBS_ID", "Fire_ID")
        if fire_id is None:
            continue
        lon = _first(row, "BurnBndLon", "lon", "longitude", "X")
        lat = _first(row, "BurnBndLat", "lat", "latitude", "Y")
        if lon is None or lat is None:
            continue
        acres = _first(row, "BurnBndAc", "Acres", "acres")
        area_ha = float(acres) * 0.404686 if acres not in (None, "") else None
        ignition = _parse_date(_first(row, "Ig_Date", "ignition_date", "StartDate", "Ig_Year"))
        ftype = _first(row, "Incid_Type", "fire_type")
        fire_type = _mtbs_fire_type(ftype)

        sev = f"{severity_dir}/{fire_id}_dnbr.tif" if severity_dir else None
        geom = f"{geometry_dir}/{fire_id}.parquet" if geometry_dir else None

        out.append(
            FireEventRecord(
                event_id=f"mtbs:{fire_id}",
                source=LabelSource.MTBS,
                region=region,
                ignition_date=ignition,
                containment_date=None,
                area_ha=area_ha,
                lon=float(lon),
                lat=float(lat),
                geometry_path=geom,
                interior_mask_path=sev,   # dNBR severity IS the interior mask
                severity_path=sev,
                continent="north_america",
                fire_type=fire_type,
                attributes={"raw_incid_type": ftype} if ftype else {},
            )
        )
    return out


def _mtbs_fire_type(incid_type) -> str | None:
    """Map MTBS Incid_Type to the registry's wildland/prescribed/agricultural."""
    if not incid_type:
        return None
    t = str(incid_type).strip().lower()
    if "wildfire" in t or "wildland" in t:
        return "wildland"
    if "prescribed" in t or "rx" in t:
        return "prescribed"
    if "agric" in t:
        return "agricultural"
    return None


def read_mtbs(
    path,
    region: str = "conus",
    geometry_dir: str | None = None,
    severity_dir: str | None = None,
) -> list[FireEventRecord]:
    """Read an MTBS shapefile or GeoParquet and normalise it. Needs pyogrio.

    The heavy IO edge: pyogrio's raw reader returns the attribute table as arrays
    (geometry dropped, since only the representative point is kept), which are
    zipped into row mappings and passed to :func:`normalize_mtbs`. The raw reader
    is used deliberately so no geopandas dependency is pulled in.
    """
    try:
        from pyogrio.raw import read as _raw_read
    except ImportError as exc:  # pragma: no cover
        raise ImportError("read_mtbs requires pyogrio: pip install pyogrio") from exc

    result = _raw_read(path, read_geometry=False)
    meta, field_data = result[0], result[-1]
    field_names = list(meta["fields"])
    rows = (dict(zip(field_names, values, strict=True)) for values in zip(*field_data, strict=True))
    return normalize_mtbs(
        rows, region=region, geometry_dir=geometry_dir, severity_dir=severity_dir
    )
