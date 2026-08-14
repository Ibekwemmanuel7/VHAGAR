"""T2 burned-area samples: predictor plus reference, aligned and masked.

A Stage-0 sample pairs a continuous burn-severity **predictor** (dNBR or RBR)
with a boolean **reference** burned mask, on one grid, plus a **valid** mask of
pixels where both are usable. The valid mask is the whole point: the most common
silent EO bug is nodata quietly becoming 0 and then "unburned ground" (or, on the
predictor side, a spuriously low dNBR). Every statistic downstream is computed
over ``valid`` only, and a nodata pixel on either side is excluded here, once, so
nothing later has to remember to.

MTBS first
----------
MTBS ships, per fire, a dNBR raster and a thematic burn-severity raster on the
**same grid**, so the predictor and the reference are already co-registered; no
regridding is needed for the first number. The thematic classes map to a burned
mask via :func:`mtbs_burned_mask`. This shares lineage with the map being
evaluated (MTBS computes that dNBR), which is fine for standing the pipeline up
and getting a per-fold Olofsson number, and is flagged wherever the number is
reported. Swapping in independent Sentinel-2/Landsat composites later changes only
the predictor source, not this module's shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MTBS_BURNED_CLASSES",
    "MTBS_MAPPED_CLASSES",
    "T2Sample",
    "make_sample",
    "mtbs_burned_mask",
]

#: MTBS thematic severity codes. 1 unburned-to-low, 2 low, 3 moderate, 4 high,
#: 5 increased greenness, 0 background / 6 non-processing are outside the mapped
#: assessment. Burned is low/moderate/high; increased greenness is mapped but not
#: burned, so it is a valid negative, not nodata.
MTBS_BURNED_CLASSES = (2, 3, 4)
MTBS_MAPPED_CLASSES = (1, 2, 3, 4, 5)


@dataclass(slots=True)
class T2Sample:
    """One burned-area sample: predictor, reference, and the valid mask."""

    event_id: str
    tile_id: str | None
    predictor: np.ndarray   # continuous, e.g. dNBR (higher = more burned)
    reference: np.ndarray   # bool, True = burned (truth)
    valid: np.ndarray       # bool, True = usable in both predictor and reference

    @property
    def shape(self) -> tuple[int, ...]:
        return self.predictor.shape

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def burned_fraction(self) -> float:
        """Fraction of valid pixels that are burned in the reference."""
        n = self.n_valid
        if n == 0:
            return float("nan")
        return float(np.count_nonzero(self.reference & self.valid) / n)


def mtbs_burned_mask(
    severity: np.ndarray,
    burned_classes: tuple[int, ...] = MTBS_BURNED_CLASSES,
    mapped_classes: tuple[int, ...] = MTBS_MAPPED_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn an MTBS thematic severity raster into ``(burned, valid)`` boolean masks.

    ``burned`` is True for the burned classes; ``valid`` is True for any mapped
    class (so increased-greenness pixels are valid negatives, and background /
    non-processing pixels are excluded rather than counted as unburned).
    """
    sev = np.asarray(severity)
    burned = np.isin(sev, burned_classes)
    valid = np.isin(sev, mapped_classes)
    return burned, valid


def make_sample(
    event_id: str,
    predictor: np.ndarray,
    reference: np.ndarray,
    reference_valid: np.ndarray | None = None,
    predictor_nodata: float | None = None,
    tile_id: str | None = None,
) -> T2Sample:
    """Assemble a :class:`T2Sample`, propagating nodata into the valid mask.

    ``predictor`` and ``reference`` must share a shape (co-registered). A pixel
    is valid only where the predictor is finite (and not ``predictor_nodata`` if
    given) and the reference is valid (``reference_valid``, defaulting to all).
    """
    # float32 predictor: RBR/dNBR precision is ample at 32-bit and it halves the
    # memory of a large fire window.
    predictor = np.asarray(predictor, dtype=np.float32)
    reference = np.asarray(reference).astype(bool)
    if predictor.shape != reference.shape:
        raise ValueError(
            f"predictor shape {predictor.shape} does not match reference {reference.shape}"
        )

    valid = np.isfinite(predictor)
    if predictor_nodata is not None:
        valid &= predictor != predictor_nodata
    if reference_valid is not None:
        rv = np.asarray(reference_valid).astype(bool)
        if rv.shape != predictor.shape:
            raise ValueError(
                f"reference_valid shape {rv.shape} does not match predictor {predictor.shape}"
            )
        valid &= rv

    return T2Sample(
        event_id=event_id,
        tile_id=tile_id,
        predictor=predictor,
        reference=reference,
        valid=valid,
    )


def read_mtbs_sample(record, dnbr_path: str, severity_path: str) -> T2Sample:
    """Read a fire's dNBR and thematic severity rasters into a sample. Needs rasterio.

    MTBS keeps both on the same grid, so no regridding: read both, derive the
    burned/valid masks from the thematic raster, and mask the dNBR predictor.
    The rasterio read is the lazily-imported IO edge.
    """
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("read_mtbs_sample requires rasterio: pip install rasterio") from exc

    with rasterio.open(dnbr_path) as ds:
        dnbr = ds.read(1).astype(np.float64)
        nodata = ds.nodata
    with rasterio.open(severity_path) as ds:
        severity = ds.read(1)

    if nodata is not None:
        dnbr = np.where(dnbr == nodata, np.nan, dnbr)
    burned, valid = mtbs_burned_mask(severity)
    return make_sample(
        record.event_id, dnbr, burned, reference_valid=valid,
        tile_id=record.tile_ids[0] if record.tile_ids else None,
    )
