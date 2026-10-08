"""Locations, design constants, and the study's pre-registered choices.

Changing any constant here changes the study; bump ``STUDY_VERSION`` when you do, so that
every output records the design that produced it.
"""
from __future__ import annotations

from pathlib import Path

STUDY_VERSION = "tx-solar-1.0"

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "parcel"
RAW = DATA / "raw"
WORK = DATA / "work"
OUT = ROOT / "outputs" / "parcel_study"

EQUAL_AREA = "EPSG:5070"  # CONUS Albers equal-area, metres
GEOGRAPHIC = "EPSG:4326"

SNAPSHOT_YEAR = 2016  # features describe the land as of this year
LABEL_FIRST_YEAR = 2017  # cases: facilities installed in this window
LABEL_LAST_YEAR = 2025
TRAIN_LAST_YEAR = 2020  # temporal holdout: train 2017-2020, test 2021-2025

MIN_PARCEL_HA = 2.0  # population: parcels at least this large
CASE_MIN_OVERLAP_HA = 2.0  # a parcel is a case when overlap >= 2 ha ...
CASE_MIN_OVERLAP_FRAC = 0.25  # ... or >= 25% of the parcel
PRE_SNAPSHOT_EXCLUDE_HA = 0.5  # drop parcels already covered by a pre-2017 facility

N_CONTROLS = 30_000  # simple random sample of non-case parcels
SEED = 20261008

NFHL_CELL_DEG = 0.25  # floodplain fetch cells around sampled parcels
