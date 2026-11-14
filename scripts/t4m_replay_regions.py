"""Run retrospective leave-one-region-out T4-M thermal-observation replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vhagar.eval.t4m_replay import run_t4m_time_forward_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", nargs="?", default="configs/t4m_corpus.yaml")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-quality", type=float, default=0.5)
    parser.add_argument("--minimum-interval-coverage", type=float, default=0.8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 0 <= args.minimum_interval_coverage <= 1:
        parser.error("--minimum-interval-coverage must be in [0, 1]")
    root = Path.cwd()
    registry = yaml.safe_load((root / args.registry).read_text(encoding="utf-8")) or {}
    cases = [
        fire for fire in registry.get("fires", [])
        if str(fire.get("observations", "")).lower().endswith(".parquet")
        and (root / str(fire["observations"])).is_file()
    ]
    if len({fire.get("region") for fire in cases}) < 3:
        parser.error("need at least three regions for leave-one-region-out evaluation")

    rows = []
    for region in sorted({str(fire["region"]) for fire in cases}):
        target_cases = [fire for fire in cases if fire["region"] == region]
        train_cases = [fire for fire in cases if fire["region"] != region]
        if len(train_cases) < 2:
            parser.error(f"holding out {region} leaves fewer than two training fires")
        historical = pd.concat([pd.read_parquet(root / fire["observations"]) for fire in train_cases], ignore_index=True)
        for target in target_cases:
            result = run_t4m_time_forward_replay(
                historical, pd.read_parquet(root / target["observations"]), target["issue_time_utc"],
                alpha=args.alpha, min_quality=args.min_quality,
            )
            result["held_out_region"] = region
            result["n_training_fires_outside_region"] = len(train_cases)
            result["interval_coverage_target"] = args.minimum_interval_coverage
            result["interval_coverage_pass"] = bool(
                result["interval_10_90_coverage"] >= args.minimum_interval_coverage
            )
            rows.append(result)
            print(f"{region} -> {result['fire_id']}: physics={result['physics_log_ros_rmse']:.3f}, residual={result['residual_log_ros_rmse']:.3f}")

    aggregate = {
        "n_regions": len({row["held_out_region"] for row in rows}),
        "n_fires": len(rows),
        "mean_physics_log_ros_rmse": float(np.mean([row["physics_log_ros_rmse"] for row in rows])),
        "mean_residual_log_ros_rmse": float(np.mean([row["residual_log_ros_rmse"] for row in rows])),
        "mean_ood_trust": float(np.mean([row["mean_ood_trust"] for row in rows])),
        "minimum_interval_coverage": args.minimum_interval_coverage,
        "n_interval_coverage_pass": int(sum(row["interval_coverage_pass"] for row in rows)),
        "intervals_calibrated_for_decision_use": bool(all(row["interval_coverage_pass"] for row in rows)),
    }
    output = {
        "scope": "retrospective research-only LORO thermal-observation replay; not operational or perimeter validation",
        "aggregate": aggregate,
        "fires": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
