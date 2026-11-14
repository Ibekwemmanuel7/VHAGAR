"""Run leakage-safe leave-one-fire-out T4-M residual replays from a registry.

This command is deliberately limited to *retrospective research* when cases
use archived weather whose incident-time availability has not been audited.
It measures whether a bounded learned ROS residual improves on the physics
prior at later sparse thermal observations. It does not produce a perimeter
forecast or an operational certification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vhagar.eval.t4m_replay import run_t4m_time_forward_replay


def _contract_cases(registry: dict, root: Path) -> list[dict]:
    cases = []
    for fire in registry.get("fires", []):
        observation_path = root / str(fire.get("observations", ""))
        if observation_path.suffix.lower() == ".parquet" and observation_path.is_file():
            cases.append(fire)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", nargs="?", default="configs/t4m_corpus.yaml")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-quality", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.alpha < 0 or not 0 <= args.min_quality <= 1:
        parser.error("alpha must be non-negative and min-quality must be in [0, 1]")

    root = Path.cwd()
    registry = yaml.safe_load((root / args.registry).read_text(encoding="utf-8")) or {}
    cases = _contract_cases(registry, root)
    if len(cases) < 3:
        parser.error("need at least 3 contract-complete Parquet fire cases for leave-one-fire-out replay")
    results = []
    for target in cases:
        target_path = root / target["observations"]
        historical = pd.concat(
            [pd.read_parquet(root / fire["observations"]) for fire in cases if fire["fire_id"] != target["fire_id"]],
            ignore_index=True,
        )
        result = run_t4m_time_forward_replay(
            historical, pd.read_parquet(target_path), target["issue_time_utc"],
            alpha=args.alpha, min_quality=args.min_quality,
        )
        result["region"] = target["region"]
        results.append(result)
        print(
            f"{result['fire_id']}: physics RMSE={result['physics_log_ros_rmse']:.3f}, "
            f"residual RMSE={result['residual_log_ros_rmse']:.3f}, n={result['n_evaluation']}"
        )
    aggregate = {
        "n_fires": len(results),
        "mean_physics_log_ros_rmse": float(np.mean([row["physics_log_ros_rmse"] for row in results])),
        "mean_residual_log_ros_rmse": float(np.mean([row["residual_log_ros_rmse"] for row in results])),
        "mean_physics_arrival_mae_hours": float(np.mean([row["physics_arrival_mae_hours"] for row in results])),
        "mean_residual_arrival_mae_hours": float(np.mean([row["residual_arrival_mae_hours"] for row in results])),
    }
    output = {
        "scope": (
            "retrospective research-only leave-one-fire-out thermal-observation replay; "
            "weather operational availability is not audited and this is not a perimeter validation"
        ),
        "registry": args.registry,
        "alpha": args.alpha,
        "min_quality": args.min_quality,
        "aggregate": aggregate,
        "fires": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
