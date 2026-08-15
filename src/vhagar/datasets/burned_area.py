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
from pathlib import Path

import numpy as np

__all__ = [
    "MTBS_BURNED_CLASSES",
    "MTBS_MAPPED_CLASSES",
    "MTBS_NONPROCESSING_CLASS",
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
#: MTBS class 6 is the non-processing / non-mapping mask: never a valid label.
MTBS_NONPROCESSING_CLASS = 6


@dataclass(slots=True)
class T2Sample:
    """One burned-area sample: predictor, reference, and the valid mask."""

    event_id: str
    tile_id: str | None
    predictor: np.ndarray   # continuous, e.g. RBR (higher = more burned)
    reference: np.ndarray   # bool, True = burned (truth)
    valid: np.ndarray       # bool, True = usable in both predictor and reference
    #: Optional multi-channel feature stack ``[C, H, W]`` for deep models (e.g.
    #: pre-NBR, post-NBR, dNBR). The threshold baseline uses only ``predictor`` so
    #: the two stay comparable; ``stack`` is extra input the segmenter can use.
    stack: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.predictor.shape

    @property
    def features(self) -> np.ndarray:
        """Model input as ``[C, H, W]``: the stack if present, else predictor as 1 channel."""
        return self.stack if self.stack is not None else self.predictor[None]

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

    @property
    def is_usable(self) -> bool:
        """Whether a threshold can be fit and tested on this sample.

        Needs some valid pixels of both classes; a window that is all-cloud
        (no valid predictor) or entirely inside or outside the burn is useless
        for calibration and would otherwise crash a fold.
        """
        f = self.burned_fraction
        return self.n_valid > 0 and np.isfinite(f) and 0.0 < f < 1.0

    def save(self, path) -> Path:
        """Persist to a compressed ``.npz`` so an expensive imagery pull is reused."""
        path = Path(path)
        arrays = dict(
            predictor=self.predictor, reference=self.reference, valid=self.valid,
            event_id=np.array(self.event_id), tile_id=np.array(self.tile_id or ""),
        )
        if self.stack is not None:
            arrays["stack"] = self.stack.astype(np.float32)
        np.savez_compressed(path, **arrays)
        return path if path.suffix else path.with_suffix(".npz")

    @classmethod
    def load(cls, path) -> T2Sample:
        with np.load(path, allow_pickle=False) as z:
            tile = str(z["tile_id"])
            return cls(
                event_id=str(z["event_id"]),
                tile_id=tile or None,
                predictor=z["predictor"],
                reference=z["reference"].astype(bool),
                valid=z["valid"].astype(bool),
                stack=z["stack"] if "stack" in z.files else None,
            )


def mtbs_burned_mask(
    severity: np.ndarray,
    burned_classes: tuple[int, ...] = MTBS_BURNED_CLASSES,
    mapped_classes: tuple[int, ...] = MTBS_MAPPED_CLASSES,
    include_background: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn an MTBS thematic severity raster into ``(burned, valid)`` boolean masks.

    ``burned`` is True for the burned classes. ``valid`` marks pixels with a known
    label. Two framings:

    * ``include_background=False`` (default): valid is only the mapped in-perimeter
      classes (1-5). This measures burn-severity discrimination *within* a fire, a
      task whose base rate is ~90% burned. Kept for backward compatibility.
    * ``include_background=True``: background (class 0) counts as a genuine unburned
      negative, so valid is every class except the non-processing mask (6). This is
      the burned-area *detection* task against an unburned landscape, at a realistic
      base rate, and is the correct framing when the window carries unburned context
      (docs/11). Caveat: the MTBS CONUS mosaic uses 0 for both background and out-of-
      footprint nodata, so a coastal window can count ocean as unburned; acceptable
      for interior fires, a caveat for coastal ones.
    """
    sev = np.asarray(severity)
    burned = np.isin(sev, burned_classes)
    # include_background: everything with a real thematic label is valid, excluding
    # only the non-processing mask (6), so background (0) becomes an unburned
    # negative. Otherwise only the in-perimeter mapped classes (1-5) are valid.
    valid = (sev != MTBS_NONPROCESSING_CLASS) if include_background else np.isin(sev, mapped_classes)
    return burned, valid


def make_sample(
    event_id: str,
    predictor: np.ndarray,
    reference: np.ndarray,
    reference_valid: np.ndarray | None = None,
    predictor_nodata: float | None = None,
    tile_id: str | None = None,
    stack: np.ndarray | None = None,
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

    stack_arr = None
    if stack is not None:
        stack_arr = np.asarray(stack, dtype=np.float32)
        if stack_arr.ndim != 3 or stack_arr.shape[1:] != predictor.shape:
            raise ValueError(
                f"stack shape {stack_arr.shape} must be (C, {predictor.shape[0]}, "
                f"{predictor.shape[1]})"
            )

    return T2Sample(
        event_id=event_id,
        tile_id=tile_id,
        predictor=predictor,
        reference=reference,
        valid=valid,
        stack=stack_arr,
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
