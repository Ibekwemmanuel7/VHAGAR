"""Resample a real DEM GeoTIFF onto one T4-M grid and derive slope/aspect.

Download a DEM before running this command. For US fires, use a USGS 3DEP
GeoTIFF/COG; this script deliberately does not manufacture a flat DEM when the
source is missing. The output grid has the same bbox/cell-deg contract as
``t4m_build_physics_grid.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from vhagar.datasets.spread_ingest import GridSpec
from vhagar.datasets.t4m_terrain import slope_aspect_from_dem


def _bbox(value: str) -> tuple[float, float, float, float]:
    try:
        bbox = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise argparse.ArgumentTypeError("bbox must have positive area")
    return bbox


def _configure_rasterio_proj() -> None:
    """Avoid a process-wide, ABI-incompatible PROJ install on Windows."""
    rasterio_spec = find_spec("rasterio")
    if rasterio_spec is None or rasterio_spec.origin is None:
        return
    bundled_proj = Path(rasterio_spec.origin).parent / "proj_data"
    if bundled_proj.is_dir():
        os.environ["PROJ_LIB"] = str(bundled_proj)
        os.environ["PROJ_DATA"] = str(bundled_proj)


def _resample_dem(dem_paths: list[str], spec: GridSpec) -> np.ndarray:
    _configure_rasterio_proj()
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject
    except ImportError as exc:  # pragma: no cover - exercised only with geo extra
        raise ImportError("DEM ingestion needs vhagar[geo] (rasterio)") from exc

    height, width = spec.shape
    destination = np.full(spec.shape, np.nan, dtype=np.float32)
    for dem_path in dem_paths:
        with rasterio.open(dem_path) as source:
            if source.crs is None:
                raise ValueError(f"DEM has no coordinate reference system: {dem_path}")
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=from_bounds(*spec.bbox, width=width, height=height),
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                init_dest_nodata=False,
                resampling=Resampling.bilinear,
            )
    if not np.isfinite(destination).all():
        missing = int((~np.isfinite(destination)).sum())
        raise ValueError(
            f"DEM does not cover the analysis grid ({missing} cells missing); "
            "choose a DEM that covers the entire bbox"
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dem_geotiff",
        nargs="+",
        help="one or more real DEM GeoTIFF/COGs, not a fabricated .npy array",
    )
    parser.add_argument("--bbox", type=_bbox, required=True)
    parser.add_argument("--cell-deg", type=float, required=True)
    parser.add_argument("--source", required=True, help="e.g. USGS 3DEP 1/3 arc-second")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cell_deg <= 0:
        parser.error("--cell-deg must be positive")

    spec = GridSpec.from_bbox_res(args.bbox, args.cell_deg)
    elevation = _resample_dem(args.dem_geotiff, spec)
    slope_tan, aspect_rad = slope_aspect_from_dem(elevation, cell_size_m=spec.cell_size_m())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, elevation_m=elevation, slope_tan=slope_tan, aspect_rad=aspect_rad)
    metadata = {
        "source": args.source,
        "dem_geotiff": [str(Path(path)) for path in args.dem_geotiff],
        "bbox": args.bbox,
        "cell_deg": args.cell_deg,
        "shape": spec.shape,
        "cell_size_m": spec.cell_size_m(),
        "horizontal_distance_note": (
            "regular EPSG:4326 analysis grid; cell distances are approximated "
            "at bbox midpoint. Use a projected analysis grid for a large domain."
        ),
        "derivatives": {
            "slope_tan": "rise/run",
            "aspect_rad": "downslope aspect; 0=north, pi/2=east",
        },
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {out}: {spec.shape[0]}x{spec.shape[1]} terrain grid")
    print(f"wrote {metadata_path}: terrain provenance")


if __name__ == "__main__":
    main()
