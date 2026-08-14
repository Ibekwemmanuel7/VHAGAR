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

__all__ = [
    "build_emsr_record",
    "normalize_mtbs",
    "read_emsr",
    "read_emsr_burned_geometries",
    "read_mtbs",
]


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


def build_emsr_record(
    activation_id: str,
    event_date,
    geometries,
    src_crs,
    delineation_path: str,
) -> FireEventRecord:
    """Build one European fire record from a Copernicus EMS burnt-area delineation.

    Area and representative point are derived from the union of the burnt-area
    polygons in an equal-area CRS (EPSG:3035), so hectares are unbiased. The
    delineation path is kept in ``attributes`` for the reference reader that
    rasterises it. Marked ``COPERNICUS_EMS``, which the registry reserves as
    evaluation-only, exactly right for the held-out continent. Pure: needs
    shapely and pyproj, no file IO.
    """
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    from shapely.ops import unary_union

    to_ea = Transformer.from_crs(src_crs, "EPSG:3035", always_xy=True)
    to_wgs = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

    def _to_ea(x, y, z=None):
        return to_ea.transform(x, y)

    ea = [shp_transform(_to_ea, g) for g in geometries if g is not None]
    if not ea:
        raise ValueError(f"{activation_id}: no burnt-area geometries")
    union = unary_union(ea)
    lon, lat = to_wgs.transform(union.centroid.x, union.centroid.y)

    if isinstance(event_date, str):
        event_date = _parse_date(event_date)

    return FireEventRecord(
        event_id=f"emsr:{activation_id}",
        source=LabelSource.COPERNICUS_EMS,
        region="europe",
        ignition_date=event_date,
        containment_date=None,
        area_ha=union.area / 1e4,
        lon=float(lon),
        lat=float(lat),
        geometry_path=str(delineation_path),
        continent="europe",
        attributes={"delineation_path": str(delineation_path)},
    )


def read_emsr_burned_geometries(delineation_path):
    """Read an EMS observed-event shapefile, keeping only burnt-area polygons.

    An EMS ``observedEventA`` layer can carry a "Burnt area" polygon alongside
    other classifications (e.g. "Not applicable"), so this filters on the
    ``notation``/``event_type`` fields to the burnt class. If no such field is
    present, all polygons are returned. Returns ``(geometries, crs)``. Needs
    pyogrio and shapely.
    """
    from pyogrio.raw import read as _raw_read
    from shapely import wkb

    result = _raw_read(delineation_path, read_geometry=True)
    meta, geom_wkb, field_data = result[0], result[2], result[-1]
    fields = list(meta["fields"])

    # Find a classification column, preferring the precise EMS notation field.
    lower = [f.lower() for f in fields]
    class_col = next((lower.index(c) for c in ("notation", "class", "obj_desc") if c in lower), None)
    keep = range(len(geom_wkb))
    if class_col is not None:
        labels = [str(v).lower() for v in field_data[class_col]]
        burnt = [i for i, s in enumerate(labels) if "burnt" in s or "burned" in s]
        if burnt:  # only filter if we actually matched; otherwise keep all
            keep = burnt

    geoms = [wkb.loads(bytes(geom_wkb[i])) for i in keep if geom_wkb[i] is not None]
    return geoms, meta["crs"]


def read_emsr(delineation_path, event_date, activation_id: str | None = None) -> FireEventRecord:
    """Read an EMS delineation shapefile into a fire record. Needs pyogrio, shapely."""
    geoms, crs = read_emsr_burned_geometries(delineation_path)
    if activation_id is None:
        from pathlib import Path

        activation_id = Path(delineation_path).stem
    return build_emsr_record(activation_id, event_date, geoms, crs, delineation_path)
