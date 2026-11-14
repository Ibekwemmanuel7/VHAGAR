"""Fetch the latest USGS 3DEP one-arc-second DEM tiles covering a fire bbox.

The USGS TNM Access API is queried at execution time. The script selects the
newest GeoTIFF for each returned one-degree footprint, records its catalog
metadata, and never silently falls back to a coarser product. One arc-second
(about 30 m) is adequate for VHAGAR's current 333 m analysis grid; use a
finer, explicitly selected product when the analysis grid warrants it.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
DATASET = "National Elevation Dataset (NED) 1 arc-second"


def _bbox(value: str) -> tuple[float, float, float, float]:
    try:
        bbox = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise argparse.ArgumentTypeError("bbox must have positive area")
    return bbox


def _request_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "VHAGAR/0.1 terrain-ingest"})
    with urlopen(request, timeout=60) as response:  # nosec B310: fixed official HTTPS endpoint
        return json.load(response)


def _latest_tile_per_footprint(items: list[dict]) -> list[dict]:
    latest: dict[tuple[float, float, float, float], dict] = {}
    for item in items:
        bounds = item.get("boundingBox") or {}
        raw_key = tuple(bounds.get(name) for name in ("minX", "minY", "maxX", "maxY"))
        # TNM versions of the same one-degree tile differ by sub-arcsecond
        # boundary padding. Round that catalogue noise away before choosing the
        # newest acquisition; do not download historical duplicates.
        key = tuple(round(value, 2) for value in raw_key) if None not in raw_key else raw_key
        if None in key or not item.get("downloadURL"):
            continue
        current = latest.get(key)
        if current is None or item.get("publicationDate", "") > current.get("publicationDate", ""):
            latest[key] = item
    return sorted(latest.values(), key=lambda item: item["downloadURL"])


def _download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "VHAGAR/0.1 terrain-ingest"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:  # nosec B310
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=_bbox, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    bbox_text = ",".join(str(value) for value in args.bbox)
    query = urlencode({"datasets": DATASET, "bbox": bbox_text, "prodFormats": "GeoTIFF"})
    response = _request_json(f"{TNM_PRODUCTS_URL}?{query}")
    tiles = _latest_tile_per_footprint(response.get("items", []))
    if not tiles:
        raise SystemExit("USGS TNM returned no one-arc-second GeoTIFF tile for this bbox")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_tiles = []
    for tile in tiles:
        url = tile["downloadURL"]
        destination = out_dir / Path(url).name
        # Some TNM catalogue entries expose a zero/empty ``sizeInBytes``.  A
        # cancelled transfer can therefore look like a valid zero-byte file if
        # we compare only to the metadata.  A GeoTIFF DEM tile is materially
        # larger than 1 MiB; require that lower bound before reuse.
        if (destination.exists() and destination.stat().st_size > 1_000_000
                and destination.stat().st_size == tile.get("sizeInBytes", -1)):
            print(f"reusing {destination}")
        else:
            print(f"downloading {tile['title']} -> {destination}")
            _download(url, destination)
        manifest_tiles.append(
            {
                "title": tile["title"],
                "publication_date": tile.get("publicationDate"),
                "download_url": url,
                "size_bytes": destination.stat().st_size,
                "path": str(destination),
                "bounding_box": tile.get("boundingBox"),
            }
        )
    manifest_path = out_dir / "3dep_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "requested_bbox": args.bbox,
                "retrieved_utc": datetime.now().astimezone().isoformat(),
                "tiles": manifest_tiles,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}; {len(manifest_tiles)} tile(s) ready")


if __name__ == "__main__":
    main()
