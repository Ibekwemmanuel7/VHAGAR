"""T5 catastrophe loss on real CAL FIRE DINS data.

Reads the DINS "POSTFIRE_MASTER_DATA_SHARE" export (structures inspected in/near CA
wildfire perimeters since 2013, with a damage class and an assessed parcel value),
builds a per-fire ground-up structure-loss catalogue, and runs the T5 loss tier
(:mod:`vhagar.models.loss`) to produce Average Annual Loss and the occurrence /
aggregate exceedance-probability curves.

This validates T5 on real loss data. Two honest data-handling choices, both
reported: the raw "Assessed Improved Value (parcel)" column contains extreme
erroneous values (a handful of parcels at 10-100x plausible), so values are
winsorized to the [1st, 99th] percentile of positive values, and missing/zero
values are imputed with the median; and the annual rate is empirical, each recorded
fire treated as ~once per the record span, which is a record-based estimate, not a
stochastic event-set rate model.

Usage:
    python scripts/dins_t5_loss.py POSTFIRE_MASTER_DATA_SHARE_*.csv --out dins_t5_summary.json

The DINS CSV is not committed (60 MB); download it from
https://data.ca.gov/dataset/cal-fire-damage-inspection-dins-data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.models import loss

DESTROYED = "Destroyed (>50%)"


def build_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """Per-fire destroyed count and winsorized ground-up structure loss."""
    dmg = df["* Damage"].astype(str).str.strip()
    df = df.assign(_destroyed=dmg.eq(DESTROYED))
    val = pd.to_numeric(df["Assessed Improved Value (parcel)"], errors="coerce")
    pos = val[val > 0]
    lo, hi, med = pos.quantile(0.01), pos.quantile(0.99), float(pos.median())
    df = df.assign(_value=val.clip(lower=lo, upper=hi).where(val > 0, med))
    cat = (df.groupby("* Incident Name")
             .apply(lambda d: pd.Series({
                 "destroyed": int(d["_destroyed"].sum()),
                 "loss_usd": float(d.loc[d["_destroyed"], "_value"].sum())}),
                    include_groups=False)
             .reset_index()
             .rename(columns={"* Incident Name": "fire"}))
    cat.attrs["winsor_lo"], cat.attrs["winsor_hi"], cat.attrs["median"] = float(lo), float(hi), med
    return cat[cat.loss_usd > 0].sort_values("loss_usd", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dins_csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("dins_t5_summary.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.dins_csv, low_memory=False)
    years_series = pd.to_datetime(df["Incident Start Date"], errors="coerce").dt.year.dropna()
    span_years = int(years_series.max() - years_series.min() + 1)

    cat = build_catalogue(df)
    losses = cat.loss_usd.to_numpy()
    rates = np.full(losses.size, 1.0 / span_years)      # empirical record-based rate
    aal = loss.average_annual_loss(losses, rates)
    oep = loss.oep_curve(losses, rates)
    aep = loss.aep_curve(losses, rates, n_years=200_000, seed=0)

    # single-fire loss at a few occurrence return periods (largest event / year)
    oep_rp = {}
    for rp in (2, 5, 10, 20):
        idx = int(np.argmin(np.abs(oep["oep"] - 1.0 / rp)))
        oep_rp[f"{rp}yr"] = float(oep["threshold"][idx])

    summary = {
        "record_years": span_years,
        "record_span": [int(years_series.min()), int(years_series.max())],
        "n_fires_with_loss": int(losses.size),
        "total_destroyed": int((df["* Damage"].astype(str).str.strip() == DESTROYED).sum()),
        "total_structure_loss_usd": float(losses.sum()),
        "aal_usd_per_yr": float(aal),
        "aep_aal_check_usd_per_yr": float(aep["aal"]),
        "oep_single_fire_loss_at_return_period_usd": oep_rp,
        "winsorize": {"lo": cat.attrs["winsor_lo"], "hi": cat.attrs["winsor_hi"],
                      "median": cat.attrs["median"]},
        "top_fires": cat.head(10).to_dict(orient="records"),
    }
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"DINS record {summary['record_span'][0]}-{summary['record_span'][1]} "
          f"({span_years} yr); {summary['total_destroyed']:,} structures destroyed "
          f"across {losses.size} fires")
    print(f"Total ground-up structure loss: ${losses.sum():,.0f}")
    print(f"AAL (analytic): ${aal:,.0f}/yr   (Monte-Carlo check: ${aep['aal']:,.0f}/yr)")
    print("Top fires by winsorized structure loss:")
    for r in summary["top_fires"][:8]:
        print(f"  {r['fire']:<18} {int(r['destroyed']):>6,} destroyed  ${r['loss_usd']:,.0f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
