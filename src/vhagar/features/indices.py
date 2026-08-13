"""Spectral indices for burned area and vegetation state.

Everything here is pure NumPy and works on arrays of any shape, so it can be
unit-tested without a raster stack. Band arguments are surface reflectance in
[0, 1] unless stated otherwise.

Band conventions
----------------
Sentinel-2 : NIR = **B8A** (865 nm, 20 m), SWIR1 = B11 (1610 nm), SWIR2 = B12 (2190 nm)
             Use B8A, not B8, it matches the SWIR bandpass and native 20 m grid.
Landsat 8/9: NIR = B5, SWIR1 = B6, SWIR2 = B7

Severity metric choice
----------------------
``RBR`` is the primary continuous severity metric. Against 1,681 CBI field
plots it outperformed RdNBR and dNBR (pooled R² 0.705 / 0.677 / 0.646), avoids
RdNBR's divergence as ``nbr_pre -> 0``, and preserves the sign of pre-fire NBR.
``dNBR`` is retained for MTBS/EFFIS comparability, not because it is better.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SEVERITY_BREAKPOINTS_KEY_BENSON",
    "classify_severity",
    "dnbr",
    "nbr",
    "ndmi",
    "ndvi",
    "ndwi",
    "normalized_difference",
    "rbr",
    "rdnbr",
]


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """``(a - b) / (a + b)`` with a guarded denominator.

    NaNs propagate. Pixels where ``|a + b| < eps`` become NaN rather than inf,
    so that downstream masks catch them instead of silently producing extreme
    values.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = a + b
    out = np.full(np.broadcast(a, b).shape, np.nan, dtype=np.float64)
    valid = np.abs(denom) >= eps
    np.divide(a - b, denom, out=out, where=valid)
    return out


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio."""
    return normalized_difference(nir, swir2)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return normalized_difference(nir, red)


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Normalized Difference Moisture Index (a.k.a. NDWI-Gao)."""
    return normalized_difference(nir, swir1)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI, open water, used for masking."""
    return normalized_difference(green, nir)


def dnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray, scaled: bool = True) -> np.ndarray:
    """Differenced NBR. ``scaled`` multiplies by 1000 (the field convention)."""
    d = np.asarray(nbr_pre, dtype=np.float64) - np.asarray(nbr_post, dtype=np.float64)
    return d * 1000.0 if scaled else d


def rdnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray, scaled: bool = True) -> np.ndarray:
    """Relativized dNBR (Miller & Thode 2007).

    ``dNBR / sqrt(|NBR_pre|)``. Diverges as ``nbr_pre -> 0``; the conventional
    0.001 floor is applied. Prefer :func:`rbr`.
    """
    pre = np.asarray(nbr_pre, dtype=np.float64)
    d = dnbr(pre, nbr_post, scaled=scaled)
    denom = np.sqrt(np.maximum(np.abs(pre), 0.001))
    return d / denom


def rbr(nbr_pre: np.ndarray, nbr_post: np.ndarray, scaled: bool = True) -> np.ndarray:
    """Relativized Burn Ratio (Parks, Dillon & Miller 2014).

    ``dNBR / (NBR_pre + 1.001)``. The +1.001 offset removes the singularity and
    preserves the sign of pre-fire NBR. This is VHAGAR's primary severity metric.
    """
    pre = np.asarray(nbr_pre, dtype=np.float64)
    return dnbr(pre, nbr_post, scaled=scaled) / (pre + 1.001)


#: Key & Benson (2006) dNBR x1000 breakpoints, adopted by EFFIS.
#:
#: These are a *reference*, not a universal calibration. VHAGAR derives
#: operational breakpoints per ecoregion from CBI regression; see
#: ``vhagar.eval.severity``. They are provided here for comparability only.
SEVERITY_BREAKPOINTS_KEY_BENSON: tuple[float, ...] = (100.0, 270.0, 440.0, 660.0)

_SEVERITY_LABELS = (
    "unburned_or_regrowth",
    "low",
    "moderate_low",
    "moderate_high",
    "high",
)


def classify_severity(
    index: np.ndarray,
    breakpoints: tuple[float, ...] = SEVERITY_BREAKPOINTS_KEY_BENSON,
) -> np.ndarray:
    """Bin a continuous severity index into ordinal classes 0..len(breakpoints).

    NaN inputs map to ``-1``.
    """
    idx = np.asarray(index, dtype=np.float64)
    binned = np.digitize(idx, np.asarray(breakpoints, dtype=np.float64), right=False)
    # np.where rather than masked assignment so 0-d inputs (a single pixel or
    # a scalar index) work identically to arrays.
    return np.where(np.isnan(idx), -1, binned).astype(np.int16)


def severity_labels() -> tuple[str, ...]:
    return _SEVERITY_LABELS
