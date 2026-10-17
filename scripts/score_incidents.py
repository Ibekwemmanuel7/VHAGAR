"""Score a frozen JSON incident manifest; see docs/30_SOUTHERN_EVALUATION.md."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vhagar.eval.incident_matching import (  # noqa: E402
    IncidentObservation,
    ReferenceIncident,
    score_incidents,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    if raw["schema_version"] != 1:
        raise ValueError("unsupported manifest schema")
    refs, observations = [], []
    for row in raw["references"]:
        row = dict(row)
        row["discovery_at"] = datetime.fromisoformat(row["discovery_at"])
        row["quality_flags"] = tuple(row.get("quality_flags", ()))
        refs.append(ReferenceIncident(**row))
    for row in raw["observations"]:
        row = dict(row)
        row["observed_at"] = datetime.fromisoformat(row["observed_at"])
        row["known_at"] = datetime.fromisoformat(row["known_at"]) if row["known_at"] else None
        observations.append(IncidentObservation(**row))
    report = score_incidents(
        refs, observations, as_of=datetime.fromisoformat(raw["as_of"]),
        prediction_run=raw["prediction_run"], radius_m=raw["radius_m"],
        window=timedelta(microseconds=raw["window_microseconds"]),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
