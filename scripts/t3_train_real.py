"""Train the REAL T3 danger models on FPA-FOD and export compact serving
artifacts + an honest metrics report.

Run this once, locally, against the full FPA-FOD SQLite (Short, RDS-2013-0009;
the bundled demo has neither FIRE_SIZE nor DISCOVERY_DOY):

    python scripts/t3_train_real.py \
        --sqlite /path/to/FPA_FOD.sqlite \
        --out data/t3_real --years 2015 2016 2017 2018 2019 2020

It writes into ``--out``:
  * ``ba_model.joblib``          fitted conditional burned-area size model (small)
  * ``t3_real_serving.json``     compact numbers the API needs (per-month ignition
                                 climatology + lightning fraction + provenance)
  * ``t3_real_metrics.json``     the leakage-safe evaluation (CRPS/pinball for
                                 burned area; temporal + spatial AUPRC/Brier for
                                 ignition), each against its baseline

The multi-GB SQLite stays on your machine; only these artifacts are committed, so
the deployed API serves real numbers without shipping the raw database. Nothing
here fabricates: if a model does not beat its baseline, the metrics say so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vhagar.eval.burned_area import expected_burned_area
from vhagar.eval.burned_area_real import (
    evaluate_real_burned_area,
    fit_real_burned_area_model,
    read_fpa_fod_sizes,
)
from vhagar.eval.ignition_climatology import (
    climatology_frequency,
    evaluate_ignition_climatology,
    read_fpa_fod_occurrence,
)

# CONUS bbox by default; override for a region.
CONUS = (-125.0, 24.0, -66.5, 49.5)


def _month_mean_freq(freq: dict, base: float) -> dict:
    """Average historical per-cell occurrence frequency by calendar month."""
    by_m: dict[int, list] = {m: [] for m in range(1, 13)}
    for (_cx, _cy, m), v in freq.items():
        by_m[m].append(v)
    return {m: (float(np.mean(v)) if v else base) for m, v in by_m.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--out", default="data/t3_real")
    ap.add_argument("--bbox", nargs=4, type=float, default=list(CONUS),
                    metavar=("W", "S", "E", "N"))
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--cell-deg", type=float, default=0.5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(args.bbox)

    import joblib

    # ---- Burned area: fit for serving + leakage-safe evaluation --------------
    ba_model, ba_feats, ba_sum = fit_real_burned_area_model(
        args.sqlite, bbox=bbox, years=args.years)
    joblib.dump({"model": ba_model, "features": ba_feats}, out / "ba_model.joblib")
    ba_metrics = evaluate_real_burned_area(args.sqlite, bbox=bbox, years=args.years)

    # ---- Ignition: climatology for serving + two-way evaluation --------------
    occ = read_fpa_fod_occurrence(args.sqlite, bbox=bbox, years=args.years)
    freq, base, _seen = climatology_frequency(occ, args.cell_deg)
    month_freq = _month_mean_freq(freq, base)
    lightning_fraction = float(np.mean(
        read_fpa_fod_sizes(args.sqlite, bbox=bbox, years=args.years)["cause"].eq("lightning")))
    ig_metrics = evaluate_ignition_climatology(
        args.sqlite, bbox=bbox, years=args.years, cell_deg=args.cell_deg)

    # ---- Precompute E[BA|ignition] per month for the serving header ----------
    eba_by_month = {}
    for m in range(1, 13):
        doy = (m - 0.5) / 12.0 * 365.0
        ang = 2 * np.pi * (doy - 1) / 365.0
        # marginal over cause, weighted by the observed lightning fraction
        Xl = np.array([[1.0, np.sin(ang), np.cos(ang)]])
        Xh = np.array([[0.0, np.sin(ang), np.cos(ang)]])
        eba_l = float(expected_burned_area([1.0], ba_model.predict_quantiles(Xl))[0])
        eba_h = float(expected_burned_area([1.0], ba_model.predict_quantiles(Xh))[0])
        eba_by_month[m] = lightning_fraction * eba_l + (1 - lightning_fraction) * eba_h

    serving = {
        "data_source": "real-fpa-fod",
        "years": ba_metrics["years"], "n_fires": ba_sum["n_fires"],
        "cell_deg": args.cell_deg, "base_rate": base,
        "lightning_fraction": lightning_fraction,
        "ignition_prob_by_month": {str(k): v for k, v in month_freq.items()},
        "eba_given_ignition_ha_by_month": {str(k): v for k, v in eba_by_month.items()},
        "provenance": (f"Ignition = FPA-FOD occurrence climatology; burned area = "
                       f"conditional size model on {ba_sum['n_fires']} real fires "
                       f"({ba_metrics['years'][0]}-{ba_metrics['years'][-1]}). "
                       f"Weather-driven ML ignition is future work."),
    }
    (out / "t3_real_serving.json").write_text(json.dumps(serving, indent=2), encoding="utf-8")
    (out / "t3_real_metrics.json").write_text(json.dumps(
        {"burned_area": ba_metrics, "ignition": ig_metrics}, indent=2, default=float),
        encoding="utf-8")

    t = ig_metrics["temporal_holdout"]
    print(f"burned area: {ba_sum['n_fires']} fires, CRPS {ba_metrics['crps']:.3f} vs "
          f"climatology {ba_metrics['crps_climatology']:.3f} "
          f"(skill {ba_metrics['crps_skill_vs_climatology']:+.3f}); "
          f"RMSE {ba_metrics['rmse_mean']:.1f}+-{ba_metrics['rmse_std']:.1f}")
    print(f"ignition temporal: AUPRC {t['auprc_climatology']:.3f} vs base "
          f"{t['auprc_baserate']:.3f} (base rate {t['base_rate']:.3f}); "
          f"spatial AUPRC {ig_metrics['spatial_block']['auprc_climatology_mean']:.3f} vs "
          f"{ig_metrics['spatial_block']['auprc_baserate_mean']:.3f}")
    print(f"wrote {out}/ba_model.joblib, t3_real_serving.json, t3_real_metrics.json")


if __name__ == "__main__":
    main()
