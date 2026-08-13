"""Fire event registry, the common label spine across regions and tasks.

Every label source (MTBS, NBAC, EFFIS, Copernicus EMS, WFIGS, FEDS, FPA-FOD)
is normalised into :class:`FireEventRecord`. Downstream, splits and datasets
speak only this vocabulary, so adding a region or a label source does not
touch the training code.

Two rules encoded here:

1. **``LabelQuality`` is explicit and travels with the record.** Copernicus EMS
   activations are human-verified VHR delineations; VIIRS-derived perimeters
   have a documented 0.71-0.93 F1 ceiling against agency perimeters and ~9%
   unburned islands inside them. Treating those as equally trustworthy is how
   an evaluation becomes meaningless.

2. **Perimeter-only records are flagged.** Training a pixel model on a
   rasterised perimeter without an interior severity mask injects a large,
   spatially structured commission-error prior. ``has_interior_mask`` makes
   that refusable rather than accidental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

__all__ = ["FireEventRecord", "LabelQuality", "LabelSource", "SOURCE_QUALITY", "EventRegistry"]


class LabelQuality(str, Enum):
    """How much a label can be trusted, in descending order."""

    FIELD = "field"                # CBI plots, ground survey
    AIRBORNE_IR = "airborne_ir"    # NIROPS: metre-scale, the best ground truth
    VHR_HUMAN = "vhr_human"        # Copernicus EMS: human-verified VHR
    ANALYST_QC = "analyst_qc"      # MTBS, NOAA HMS, EFFIS with manual QA
    HIGHRES_AUTO = "highres_auto"  # Landsat/Sentinel-2 automated (NBAC, LC2 L3 BA)
    COARSE_AUTO = "coarse_auto"    # MODIS/VIIRS derived (FEDS, MCD64A1, GlobFire)
    ADMINISTRATIVE = "administrative"  # reported polygons/points of mixed provenance


class LabelSource(str, Enum):
    MTBS = "mtbs"
    LANDSAT_C2_L3_BA = "landsat_c2_l3_ba"
    WELTY_JEFFRIES = "welty_jeffries"
    WFIGS = "wfigs"
    NIROPS = "nirops"
    FPA_FOD = "fpa_fod"
    NBAC = "nbac"
    CNFDB = "cnfdb"
    EFFIS = "effis"
    COPERNICUS_EMS = "copernicus_ems"
    FEDS = "feds"
    MCD64A1 = "mcd64a1"
    GLOBFIRE = "globfire"
    CBI = "cbi"


SOURCE_QUALITY: dict[LabelSource, LabelQuality] = {
    LabelSource.CBI: LabelQuality.FIELD,
    LabelSource.NIROPS: LabelQuality.AIRBORNE_IR,
    LabelSource.COPERNICUS_EMS: LabelQuality.VHR_HUMAN,
    LabelSource.MTBS: LabelQuality.ANALYST_QC,
    LabelSource.EFFIS: LabelQuality.ANALYST_QC,
    LabelSource.NBAC: LabelQuality.HIGHRES_AUTO,
    LabelSource.LANDSAT_C2_L3_BA: LabelQuality.HIGHRES_AUTO,
    LabelSource.FEDS: LabelQuality.COARSE_AUTO,
    LabelSource.MCD64A1: LabelQuality.COARSE_AUTO,
    LabelSource.GLOBFIRE: LabelQuality.COARSE_AUTO,
    LabelSource.WFIGS: LabelQuality.ADMINISTRATIVE,
    LabelSource.WELTY_JEFFRIES: LabelQuality.ADMINISTRATIVE,
    LabelSource.CNFDB: LabelQuality.ADMINISTRATIVE,
    LabelSource.FPA_FOD: LabelQuality.ADMINISTRATIVE,
}

#: Sources reserved as evaluation-only. Copernicus EMS is the highest-quality
#: European geometry available and is worth far more as an unseen test set
#: than as marginal extra training data.
EVALUATION_ONLY: frozenset[LabelSource] = frozenset(
    {LabelSource.COPERNICUS_EMS, LabelSource.NIROPS, LabelSource.CBI}
)


@dataclass(slots=True)
class FireEventRecord:
    """One fire event, normalised across all label sources."""

    event_id: str
    source: LabelSource
    region: str                      # conus | canada | europe
    ignition_date: date | None
    containment_date: date | None
    #: Reported/mapped final area, hectares.
    area_ha: float | None
    #: Representative point, EPSG:4326.
    lon: float
    lat: float
    #: Path to the geometry (GeoParquet) and, when present, the interior mask.
    geometry_path: str | None = None
    interior_mask_path: str | None = None
    severity_path: str | None = None
    ecoregion: str | None = None
    continent: str | None = None
    cause: str | None = None         # human | lightning | unknown
    fire_type: str | None = None     # wildland | agricultural | prescribed
    tile_ids: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

    @property
    def quality(self) -> LabelQuality:
        return SOURCE_QUALITY[self.source]

    @property
    def has_interior_mask(self) -> bool:
        return self.interior_mask_path is not None

    @property
    def is_evaluation_only(self) -> bool:
        return self.source in EVALUATION_ONLY

    def assert_trainable(self) -> None:
        """Raise if this record must not be used as pixel-level training data."""
        if self.is_evaluation_only:
            raise ValueError(
                f"{self.source.value} is reserved for evaluation "
                "(see vhagar.labels.registry.EVALUATION_ONLY)"
            )
        if self.geometry_path and not self.has_interior_mask:
            raise ValueError(
                f"event {self.event_id} has a perimeter but no interior mask. "
                "Rasterised perimeters contain ~9% unburned islands; training a "
                "pixel model on one injects a structured commission-error prior. "
                "Derive an interior severity mask first, or use this record for "
                "event-level tasks only."
            )


class EventRegistry:
    """In-memory registry with the selection rules the protocol requires."""

    def __init__(self, records: list[FireEventRecord] | None = None) -> None:
        self._records: dict[str, FireEventRecord] = {}
        for r in records or []:
            self.add(r)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records.values())

    def add(self, record: FireEventRecord) -> None:
        if record.event_id in self._records:
            raise ValueError(f"duplicate event_id {record.event_id!r}")
        self._records[record.event_id] = record

    def training_records(self, require_interior_mask: bool = True) -> list[FireEventRecord]:
        out = []
        for r in self._records.values():
            if r.is_evaluation_only:
                continue
            if require_interior_mask and r.geometry_path and not r.has_interior_mask:
                continue
            out.append(r)
        return out

    def evaluation_records(self) -> list[FireEventRecord]:
        return [r for r in self._records.values() if r.is_evaluation_only]

    def to_split_units(self):
        """Convert to :class:`vhagar.eval.splits.SplitUnit` objects."""
        from vhagar.eval.splits import SplitUnit

        units = []
        for r in self._records.values():
            when = r.ignition_date
            if when is None:
                continue
            units.append(
                SplitUnit(
                    uid=r.event_id,
                    lon=r.lon,
                    lat=r.lat,
                    when=when,
                    group=r.event_id,
                    ecoregion=r.ecoregion,
                    continent=r.continent,
                    tile_id=r.tile_ids[0] if r.tile_ids else None,
                )
            )
        return units

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self._records.values():
            key = f"{r.region}/{r.source.value}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))
