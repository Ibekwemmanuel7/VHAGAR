"""Extract every HRRR frame in a fetch manifest into same-grid weather packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from t4m_hrrr_to_weather import _bbox, _read_on_grid

from vhagar.datasets.spread_ingest import GridSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--bbox", type=_bbox, required=True)
    parser.add_argument("--cell-deg", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.cell_deg <= 0:
        parser.error("--cell-deg must be positive")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    if not frames:
        parser.error("manifest has no frames")
    spec = GridSpec.from_bbox_res(args.bbox, args.cell_deg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_frames = []
    for frame in frames:
        grib_path = Path(frame["path"])
        if not grib_path.exists():
            parser.error(f"missing HRRR GRIB: {grib_path}")
        u10, v10, humidity_percent = _read_on_grid(str(grib_path), spec)
        out = out_dir / f"{grib_path.stem}.npz"
        np.savez_compressed(
            out,
            u10_ms=u10,
            v10_ms=v10,
            relative_humidity=np.clip(humidity_percent / 100.0, 0, 1),
            valid_time_utc=np.array(frame["valid_time_utc"]),
            available_time_utc=np.array(frame["available_time_utc"]),
        )
        output_frames.append({**frame, "weather_path": str(out)})
        print(f"wrote {out}")
    output_manifest = out_dir / "weather_manifest.json"
    output_manifest.write_text(
        json.dumps(
            {
                "source_manifest": str(Path(args.manifest)),
                "bbox": args.bbox,
                "cell_deg": args.cell_deg,
                "shape": spec.shape,
                "frames": output_frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_manifest}: {len(output_frames)} weather package(s)")


if __name__ == "__main__":
    main()
