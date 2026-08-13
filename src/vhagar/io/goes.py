"""GOES-R ABI Fire Detection and Characterization (FDC) from AWS Open Data.

This is the **hot path**. GOES L2 FDC lands on S3 roughly 1-2 minutes after
scan end, giving a 5-minute effective detection cadence over CONUS and 10
minutes full disk. Latency here is the product; do not route it through Earth
Engine.

Event-driven ingest
-------------------
NOAA publishes an SNS topic per satellite for new-object notifications::

    arn:aws:sns:us-east-1:123901341784:NewGOES19Object

Subscribe an **SQS queue** (not a Lambda directly) with a prefix filter on
``ABI-L2-FDCC``/``ABI-L2-FDCF`` and a dead-letter queue attached, so a
downstream outage queues work rather than dropping it.

Constellation status (2026)
---------------------------
* GOES-19 -- operational GOES-East at 75.2W since April 2025
* GOES-18 -- operational GOES-West
* GOES-16 -- backup; GOES-17 in on-orbit storage, no new data
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

__all__ = [
    "FDC_MASK_MEANINGS",
    "GOES_BUCKETS",
    "GOES_SNS_TOPICS",
    "GoesProduct",
    "fdc_key_prefix",
    "is_fire_mask_value",
    "parse_goes_key",
]

GOES_BUCKETS = {
    16: "noaa-goes16",
    17: "noaa-goes17",  # storage; historical only
    18: "noaa-goes18",
    19: "noaa-goes19",
}

GOES_SNS_TOPICS = {
    16: "arn:aws:sns:us-east-1:123901341784:NewGOES16Object",
    18: "arn:aws:sns:us-east-1:123901341784:NewGOES18Object",
    19: "arn:aws:sns:us-east-1:123901341784:NewGOES19Object",
}

#: FDC ``Mask`` band codes. 10-15 are unfiltered detections; 30-35 are the
#: same categories after the Part II 12-hour temporal filter. VHAGAR keeps
#: both: the unfiltered stream is the low-latency signal, the filtered one is
#: the high-precision confirmation.
FDC_MASK_MEANINGS = {
    10: "good_quality_fire",
    11: "saturated_fire",
    12: "cloud_contaminated_fire",
    13: "high_probability_fire",
    14: "medium_probability_fire",
    15: "low_probability_fire",
    30: "good_quality_fire_temporally_filtered",
    31: "saturated_fire_temporally_filtered",
    32: "cloud_contaminated_fire_temporally_filtered",
    33: "high_probability_fire_temporally_filtered",
    34: "medium_probability_fire_temporally_filtered",
    35: "low_probability_fire_temporally_filtered",
}

FIRE_MASK_VALUES = frozenset(FDC_MASK_MEANINGS)
HIGH_CONFIDENCE_MASK_VALUES = frozenset({10, 11, 13, 30, 31, 33})


@dataclass(frozen=True, slots=True)
class GoesProduct:
    """A located GOES L2 product file on S3."""

    satellite: int
    product: str          # e.g. "ABI-L2-FDCC"
    start: datetime
    key: str

    @property
    def bucket(self) -> str:
        return GOES_BUCKETS[self.satellite]

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def domain(self) -> str:
        return {"C": "conus", "F": "full_disk", "M": "mesoscale"}.get(self.product[-1], "unknown")


def fdc_key_prefix(satellite: int, when: datetime, domain: str = "C") -> str:
    """S3 key prefix for FDC files in a given hour.

    Layout is ``ABI-L2-FDC{C,F}/YYYY/DDD/HH/``.
    """
    if satellite not in GOES_BUCKETS:
        raise ValueError(f"unknown GOES satellite {satellite}")
    if domain not in {"C", "F", "M1", "M2"}:
        raise ValueError(f"unknown ABI domain {domain!r}")
    w = when.astimezone(UTC)
    return f"ABI-L2-FDC{domain}/{w.year:04d}/{w.timetuple().tm_yday:03d}/{w.hour:02d}/"


def parse_goes_key(key: str, satellite: int) -> GoesProduct:
    """Parse an ABI L2 S3 key into a :class:`GoesProduct`.

    Filenames look like::

        OR_ABI-L2-FDCC-M6_G19_s20261931200204_e20261931202577_c20261931203089.nc

    ``s`` is the scan start: ``YYYYDDDHHMMSSt`` (tenths of a second).

    >>> p = parse_goes_key(
    ...     "ABI-L2-FDCC/2026/193/12/"
    ...     "OR_ABI-L2-FDCC-M6_G19_s20261931200204_e20261931202577_c20261931203089.nc", 19)
    >>> p.domain
    'conus'
    """
    name = key.rsplit("/", 1)[-1]
    parts = name.split("_")
    product = parts[1].rsplit("-", 1)[0] if len(parts) > 1 else "unknown"
    stamp = next((p for p in parts if p.startswith("s") and p[1:].isdigit()), None)
    if stamp is None:
        raise ValueError(f"no scan-start token in {name!r}")
    s = stamp[1:]
    start = datetime(
        year=int(s[0:4]),
        month=1,
        day=1,
        hour=int(s[7:9]),
        minute=int(s[9:11]),
        second=int(s[11:13]),
        tzinfo=UTC,
    ).replace() + _days(int(s[4:7]) - 1)
    return GoesProduct(satellite=satellite, product=product, start=start, key=key)


def _days(n: int):
    from datetime import timedelta

    return timedelta(days=n)


def is_fire_mask_value(value: int, high_confidence_only: bool = False) -> bool:
    """Whether an FDC ``Mask`` code denotes a fire pixel."""
    return value in (HIGH_CONFIDENCE_MASK_VALUES if high_confidence_only else FIRE_MASK_VALUES)


def list_fdc_files(
    satellite: int,
    start: datetime,
    end: datetime,
    domain: str = "C",
    anonymous: bool = True,
) -> Iterator[GoesProduct]:
    """List FDC files in a time window. Requires ``s3fs``.

    The NOAA buckets are public and not requester-pays, so anonymous access is
    the default and costs nothing.
    """
    try:
        import s3fs
    except ImportError as exc:  # pragma: no cover
        raise ImportError("list_fdc_files requires s3fs: pip install s3fs") from exc

    fs = s3fs.S3FileSystem(anon=anonymous)
    bucket = GOES_BUCKETS[satellite]
    cursor = start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    while cursor <= end:
        prefix = fdc_key_prefix(satellite, cursor, domain)
        try:
            keys = fs.ls(f"{bucket}/{prefix}", detail=False)
        except FileNotFoundError:
            keys = []
        for full in keys:
            key = full.split("/", 1)[1]
            try:
                product = parse_goes_key(key, satellite)
            except ValueError:
                continue
            if start <= product.start <= end:
                yield product
        cursor += _days(0) + _hours(1)


def _hours(n: int):
    from datetime import timedelta

    return timedelta(hours=n)
