"""Extract HRRR 10 m wind and 2 m RH to a declared T4-M analysis grid.

This accepts a locally cached HRRR surface GRIB2 file. The caller must provide
both valid and availability times: a historical archive timestamp is not proof
of what was operationally available at a forecast issue time.
"""

from __future__ import annotations

import argparse
import json
import os
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.datasets.spread_ingest import GridSpec


def _bbox(value: str) -> tuple[float, float, float, float]:
    try:
        bbox = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise argparse.ArgumentTypeError("bbox must have positive area")
    return bbox


def _utc_timestamp(value: str) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone, e.g. 2026-08-04T00:00:00Z")
    return timestamp.tz_convert("UTC").isoformat()


def _configure_rasterio_proj() -> None:
    rasterio_spec = find_spec("rasterio")
    if rasterio_spec is None or rasterio_spec.origin is None:
        return
    bundled_proj = Path(rasterio_spec.origin).parent / "proj_data"
    if bundled_proj.is_dir():
        os.environ["PROJ_LIB"] = str(bundled_proj)
        os.environ["PROJ_DATA"] = str(bundled_proj)


def _band_number(dataset, *, element: str, short_name: str) -> int:
    for band in range(1, dataset.count + 1):
        tags = dataset.tags(band)
        if tags.get("GRIB_ELEMENT") == element and tags.get("GRIB_SHORT_NAME") == short_name:
            return band
    raise ValueError(f"HRRR file lacks {element} at {short_name}")


def _read_on_grid(grib_path: str, spec: GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _configure_rasterio_proj()
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject
    except ImportError as exc:  # pragma: no cover
        raise ImportError("HRRR ingest needs vhagar[geo] (rasterio with GRIB support)") from exc

    destination_transform = from_bounds(*spec.bbox, width=spec.shape[1], height=spec.shape[0])
    fields = []
    with rasterio.open(grib_path) as source:
        if source.crs is None:
            raise ValueError("HRRR GRIB has no CRS")
        bands = (
            _band_number(source, element="UGRD", short_name="10-HTGL"),
            _band_number(source, element="VGRD", short_name="10-HTGL"),
            _band_number(source, element="RH", short_name="2-HTGL"),
        )
        for band in bands:
            destination = np.full(spec.shape, np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(source, band),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=destination_transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            if not np.isfinite(destination).all():
                raise ValueError("HRRR field does not cover the complete analysis grid")
            fields.append(destination)
    return tuple(fields)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hrrr_grib2")
    parser.add_argument("--bbox", type=_bbox, required=True)
    parser.add_argument("--cell-deg", type=float, required=True)
    parser.add_argument("--valid-time-utc", type=_utc_timestamp, required=True)
    parser.add_argument("--available-time-utc", type=_utc_timestamp, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cell_deg <= 0:
        parser.error("--cell-deg must be positive")
    if pd.Timestamp(args.available_time_utc) > pd.Timestamp(args.valid_time_utc):
        parser.error("availability cannot be later than valid time for this prior-run forecast input")

    spec = GridSpec.from_bbox_res(args.bbox, args.cell_deg)
    u10, v10, relative_humidity_percent = _read_on_grid(args.hrrr_grib2, spec)
    relative_humidity = np.clip(relative_humidity_percent / 100.0, 0.0, 1.0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        u10_ms=u10,
        v10_ms=v10,
        relative_humidity=relative_humidity,
        valid_time_utc=np.array(args.valid_time_utc),
        available_time_utc=np.array(args.available_time_utc),
    )
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "source": "NOAA HRRR surface GRIB2",
                "grib2": str(Path(args.hrrr_grib2)),
                "bbox": args.bbox,
                "shape": spec.shape,
                "valid_time_utc": args.valid_time_utc,
                "available_time_utc": args.available_time_utc,
                "variables": {"u10_ms": "UGRD:10-HTGL", "v10_ms": "VGRD:10-HTGL", "relative_humidity": "RH:2-HTGL / 100"},
                "availability_note": "Caller-declared availability; verify against operational ingestion logs for scored replay.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}: HRRR 10 m wind and 2 m RH on {spec.shape[0]}x{spec.shape[1]} grid")
    print(f"wrote {metadata_path}: source and time provenance")


if __name__ == "__main__":
    main()
