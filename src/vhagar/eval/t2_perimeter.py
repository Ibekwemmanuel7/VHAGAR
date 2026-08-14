"""T2 perimeter-vs-severity: how much a rasterised perimeter overstates burned area.

The architecture warns (section 4.1) never to train a pixel model on a rasterised
perimeter without an interior severity mask, because a large, spatially
structured fraction of the area inside a perimeter is unburned islands. This
quantifies that fraction directly from the MTBS thematic burn-severity mosaic: the
mapped extent (the assessed perimeter interior) is the "all burned" claim a
rasterised perimeter makes, and the per-pixel severity classes say which of those
pixels actually burned.

This is a census, not a sample: every pixel of the severity product is counted,
so the commission fraction is exact with respect to that product. The residual
uncertainty is in the MTBS classification itself, which shares lineage with the
perimeter, so this is a pipeline and data-quality number, not an independent
accuracy claim. It is still the honest first T2 number, and it is the one the
architecture explicitly asks for.

MTBS thematic classes
---------------------
0 background (outside any perimeter, also the raster nodata), 1 unburned to low,
2 low, 3 moderate, 4 high, 5 increased greenness, 6 non-processing / nodata
within a perimeter. The mapped perimeter interior is classes 1-5; class 6 is
dropped. Whether class 1 ("unburned to low") counts as burned is a real
definitional choice that moves the commission fraction a lot, so it is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MTBS_BACKGROUND",
    "MTBS_NODATA_WITHIN",
    "PerimeterSeverityResult",
    "class_histogram",
    "perimeter_vs_severity",
]

#: MTBS pixel is 30 m.
MTBS_PIXEL_AREA_HA = 0.09
#: Classes to exclude from the assessed perimeter interior entirely.
MTBS_BACKGROUND = 0
MTBS_NODATA_WITHIN = 6


@dataclass(slots=True)
class PerimeterSeverityResult:
    """Census comparison of a rasterised perimeter against per-pixel severity."""

    perimeter_ha: float          # area a rasterised perimeter calls burned
    severity_burned_ha: float    # area the severity classes call burned
    unburned_within_ha: float    # unburned islands inside the perimeter
    commission_fraction: float   # unburned_within / perimeter
    burned_classes: tuple[int, ...]
    mapped_classes: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"perimeter {self.perimeter_ha:,.0f} ha, "
            f"severity-burned {self.severity_burned_ha:,.0f} ha, "
            f"commission {100 * self.commission_fraction:.1f}% "
            f"(burned classes {self.burned_classes})"
        )


def perimeter_vs_severity(
    histogram,
    pixel_area_ha: float = MTBS_PIXEL_AREA_HA,
    burned_classes: tuple[int, ...] = (2, 3, 4),
    mapped_classes: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> PerimeterSeverityResult:
    """Commission analysis from a class histogram (index = class value, value = count).

    ``mapped_classes`` is the assessed perimeter interior (the rasterised-perimeter
    "burned" claim); ``burned_classes`` is what actually burned. The difference is
    the unburned-islands commission.
    """
    h = np.asarray(histogram, dtype=np.float64)

    def area(classes) -> float:
        return float(sum(h[c] for c in classes if c < h.size)) * pixel_area_ha

    perimeter_ha = area(mapped_classes)
    severity_ha = area(burned_classes)
    unburned_within = perimeter_ha - severity_ha
    commission = unburned_within / perimeter_ha if perimeter_ha > 0 else float("nan")
    return PerimeterSeverityResult(
        perimeter_ha=perimeter_ha,
        severity_burned_ha=severity_ha,
        unburned_within_ha=unburned_within,
        commission_fraction=commission,
        burned_classes=tuple(burned_classes),
        mapped_classes=tuple(mapped_classes),
    )


def class_histogram(mosaic_path, max_class: int = 255) -> np.ndarray:
    """Stream a thematic mosaic and return per-class pixel counts. Needs rasterio.

    Reads block by block, so a CONUS-scale mosaic that is many gigabytes
    uncompressed is counted in constant memory. The rasterio read is the
    lazily-imported IO edge.
    """
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("class_histogram requires rasterio: pip install rasterio") from exc

    hist = np.zeros(max_class + 1, dtype=np.int64)
    with rasterio.open(mosaic_path) as ds:
        for _, window in ds.block_windows(1):
            block = ds.read(1, window=window)
            hist += np.bincount(block.ravel(), minlength=max_class + 1)[: max_class + 1]
    return hist
