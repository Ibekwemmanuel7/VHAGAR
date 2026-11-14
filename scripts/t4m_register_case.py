"""Register one complete historical fire case in the T4-M corpus.

This is intentionally a *curation gate*, not a downloader.  A case enters the
registry only after its observation, fuel/physics, weather, and terrain products
have been built and their sidecar provenance has been retained.  It prevents a
later training run from treating an unlabelled example or a copied file as an
independent historical incident.

Example (inspect before mutating the registry)::

    python scripts/t4m_register_case.py configs/t4m_corpus.yaml \
      --fire-id creek-2020 --region pacific_northwest \
      --issue-time-utc 2020-09-08T00:00:00Z \
      --observations data/t4m/creek-2020.parquet \
      --physics-grid data/t4m/creek-2020_grids.npz \
      --physics-provenance data/t4m/creek-2020_grids.json \
      --weather data/t4m/creek-2020_weather.npz \
      --weather-provenance data/t4m/creek-2020_weather.json \
      --terrain data/t4m/creek-2020_terrain.npz \
      --terrain-provenance data/t4m/creek-2020_terrain.json --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml


def _relative_existing_path(value: str, root: Path, label: str) -> str:
    """Return a repository-relative, existing regular file path."""
    path = Path(value)
    candidate = path if path.is_absolute() else root / path
    if not candidate.is_file():
        raise argparse.ArgumentTypeError(f"{label} does not exist or is not a file: {value}")
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be inside the repository: {value}") from exc


def _utc_timestamp(value: str) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise argparse.ArgumentTypeError("--issue-time-utc must include a timezone, for example ...Z")
    return timestamp.tz_convert("UTC").isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("registry", nargs="?", default="configs/t4m_corpus.yaml")
    parser.add_argument("--fire-id", required=True, help="stable incident identifier, e.g. creek-2020")
    parser.add_argument("--region", required=True, help="evaluation geography, e.g. pacific_northwest")
    parser.add_argument("--issue-time-utc", required=True, type=_utc_timestamp)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--observations-provenance", help="optional manifest for the raw observation export")
    parser.add_argument("--physics-grid", required=True)
    parser.add_argument("--physics-provenance", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--weather-provenance", required=True)
    parser.add_argument("--terrain", required=True)
    parser.add_argument("--terrain-provenance", required=True)
    parser.add_argument("--thermal-sensor", default="unknown", help="e.g. VIIRS NOAA-20 or ORBIT thermal")
    parser.add_argument("--weather-availability", choices=("verified", "pending"), default="pending")
    parser.add_argument("--status", default="ready_for_replay")
    parser.add_argument("--note", default="")
    parser.add_argument("--dry-run", action="store_true", help="print the candidate entry without modifying YAML")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.fire_id):
        parser.error("--fire-id must use lowercase letters, digits, underscores, or hyphens")
    if not args.region.strip():
        parser.error("--region cannot be blank")

    root = Path.cwd()
    registry_path = root / args.registry
    if not registry_path.is_file():
        parser.error(f"registry does not exist: {args.registry}")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    fires = list(registry.get("fires", []))
    if any(fire.get("fire_id") == args.fire_id for fire in fires):
        parser.error(f"fire_id already registered: {args.fire_id}")

    try:
        paths = {
            "observations": _relative_existing_path(args.observations, root, "observations"),
            "physics_grid": _relative_existing_path(args.physics_grid, root, "physics grid"),
            "physics_provenance": _relative_existing_path(args.physics_provenance, root, "physics provenance"),
            "weather": _relative_existing_path(args.weather, root, "weather"),
            "weather_provenance": _relative_existing_path(args.weather_provenance, root, "weather provenance"),
            "terrain": _relative_existing_path(args.terrain, root, "terrain"),
            "terrain_provenance": _relative_existing_path(args.terrain_provenance, root, "terrain provenance"),
        }
        if args.observations_provenance:
            paths["observations_provenance"] = _relative_existing_path(
                args.observations_provenance, root, "observations provenance"
            )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    # Prevent a superficially larger corpus assembled by relabelling the same
    # detection export or physical prior. The source data, not merely the ID,
    # must be incident-independent for leave-one-fire-out evaluation.
    for key in ("observations", "physics_grid"):
        if any(fire.get(key) == paths[key] for fire in fires):
            parser.error(
                f"{key} is already registered for another fire; "
                "an independent case needs its own observation and physics products"
            )

    entry = {
        "fire_id": args.fire_id,
        "region": args.region.strip(),
        "status": args.status,
        "observations": paths["observations"],
        "physics_grid": paths["physics_grid"],
        "physics_provenance": paths["physics_provenance"],
        "issue_time_utc": args.issue_time_utc,
        "thermal_sensor": args.thermal_sensor,
        # A cached historical weather file is not proof that the forecast was
        # available at issue time.  Preserve it, but keep replay/ML blocked
        # until the availability audit is declared verified.
        "weather": paths["weather"] if args.weather_availability == "verified" else "pending_operational_availability_audit",
        "weather_cached": paths["weather"],
        "weather_provenance": paths["weather_provenance"],
        "terrain": paths["terrain"],
        "terrain_provenance": paths["terrain_provenance"],
    }
    if "observations_provenance" in paths:
        entry["observations_provenance"] = paths["observations_provenance"]
    if args.note.strip():
        entry["note"] = args.note.strip()

    rendered = yaml.safe_dump({"fires": [entry]}, sort_keys=False, allow_unicode=True)
    if args.dry_run:
        print("Candidate registry entry (not written):")
        print(rendered, end="")
        return

    registry["fires"] = [*fires, entry]
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"registered {args.fire_id} in {args.registry}; weather availability={args.weather_availability}")


if __name__ == "__main__":
    main()
