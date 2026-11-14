"""Validate a T4-M corpus registry before training or held-out evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", nargs="?", default="configs/t4m_corpus.yaml")
    parser.add_argument("--for-ml-training", action="store_true")
    parser.add_argument(
        "--retrospective-research", action="store_true",
        help="allow cached historical weather with unverified operational availability; never an operational replay",
    )
    args = parser.parse_args()
    root = Path.cwd()
    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    fires = registry.get("fires", [])
    required = (
        "fire_id", "region", "observations", "physics_grid", "physics_provenance",
        "issue_time_utc", "weather", "terrain",
    )
    errors = []
    # A different ``fire_id`` is not evidence of an independent event.  Reusing
    # one FIRMS export or one physics package under several IDs would make a
    # leave-one-fire-out score leak the held-out incident into training.
    independent_sources: dict[str, dict[str, str]] = {
        "observations": {},
        "physics_grid": {},
    }
    for fire in fires:
        missing = [key for key in required if not fire.get(key)]
        if missing:
            errors.append(f"{fire.get('fire_id', '?')}: missing {', '.join(missing)}")
        for key in ("physics_grid", "physics_provenance", "weather", "terrain"):
            value = str(fire.get(key, ""))
            if value and not value.startswith("pending") and not (root / value).exists():
                errors.append(f"{fire['fire_id']}: missing file {value}")
        for key in ("weather", "terrain"):
            if str(fire.get(key, "")).startswith("pending"):
                if key == "weather" and args.retrospective_research:
                    cached = str(fire.get("weather_cached", ""))
                    provenance = str(fire.get("weather_provenance", ""))
                    if not cached or not provenance or not (root / cached).exists() or not (root / provenance).exists():
                        errors.append(
                            f"{fire['fire_id']}: retrospective weather needs cached product and provenance"
                        )
                else:
                    errors.append(f"{fire['fire_id']}: {key} is pending")
        for key, seen in independent_sources.items():
            value = str(fire.get(key, ""))
            if not value:
                continue
            previous = seen.get(value)
            if previous is not None:
                errors.append(
                    f"{fire['fire_id']}: {key} is shared with {previous}; "
                    "cases are not independent"
                )
            else:
                seen[value] = str(fire.get("fire_id", "?"))
    if args.for_ml_training:
        # The registry may also contain raw discovery exports or otherwise
        # blocked cases.  They are useful provenance, but only a validated
        # T4-M event table is an incident-level ML example.
        ml_fires = [
            fire for fire in fires
            if str(fire.get("observations", "")).lower().endswith(".parquet")
        ]
        regions = {fire.get("region") for fire in ml_fires if fire.get("region")}
        if len(ml_fires) < 3:
            errors.append(
                "ML training needs at least 3 independent, contract-complete fires; "
                f"registry has {len(ml_fires)}"
            )
        if len(regions) < 2:
            errors.append("ML training needs at least 2 regions for leave-one-region-out transfer evaluation")
    if errors:
        print("T4-M corpus NOT trainable:")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    scope = "retrospective research only" if args.retrospective_research else "operational-availability audit"
    if args.for_ml_training:
        print(f"T4-M corpus valid: {len(ml_fires)} contract-complete fires ({scope})")
    else:
        print(f"T4-M corpus valid: {len(fires)} registry fires ({scope})")


if __name__ == "__main__":
    main()
