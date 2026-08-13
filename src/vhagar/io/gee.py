"""Google Earth Engine access.

Scope, deliberately narrow
--------------------------
GEE is used for **label generation**, decade-scale compositing, and the
AlphaEarth / Satellite Embedding feature layer. It is **not** on the
near-real-time path: interactive requests are capped at roughly 5 minutes of
compute and tens of megabytes, batch tasks queue behind other tenants, and the
free noncommercial tiers (150-1,000 EECU-hr/month) will not carry production.
Anything with a latency SLA reads directly from S3 -- see :mod:`vhagar.io.goes`.

Export strategy
---------------
``ee.data.computePixels(..., format="NUMPY_NDARRAY")`` against the
**high-volume endpoint**, a process pool, exponential-backoff retry, results
repacked into chunked Zarr.

TFRecord export is rejected: it coerces every band to float32 (2x inflation on
uint16 reflectance), is not randomly seekable for shuffling, and is a
TensorFlow-shaped format in a PyTorch stack. ``getDownloadURL`` is rejected
for bulk work: hard 32 MB per-request ceiling.

.. warning::
   Commercial operation of VHAGAR requires a paid Earth Engine Cloud plan.
   Price your actual EECU profile before letting a GEE dependency reach the
   product path.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

HIGH_VOLUME_ENDPOINT = "https://earthengine-highvolume.googleapis.com"

#: Documented Earth Engine limits (2026). Kept here so the retry/pool sizing
#: has a single source of truth.
GEE_LIMITS = {
    "concurrent_requests": 40,
    "requests_per_second": 100,
    "max_aggregation_result_mb": 100,
    "max_request_payload_mb": 10,
    "getdownloadurl_max_mb": 32,
    "interactive_timeout_s": 300,
    "max_ready_tasks": 3000,
    "asset_storage_gb": 250,
}

#: Collections VHAGAR reads. Verify IDs against the catalog before first use;
#: NOAA-21 in particular is inferred from the S-NPP/NOAA-20 naming pattern.
COLLECTIONS = {
    "viirs_af_snpp": "NASA/LANCE/SNPP_VIIRS/C2",     # ends 2026-11-01
    "viirs_af_noaa20": "NASA/LANCE/NOAA20_VIIRS/C2",
    "goes19_fdcc": "NOAA/GOES/19/FDCC",
    "goes19_fdcf": "NOAA/GOES/19/FDCF",
    "goes18_fdcc": "NOAA/GOES/18/FDCC",
    "firms": "FIRMS",
    "mtbs": "USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1",
    "mcd64a1": "MODIS/061/MCD64A1",
    "firecci51": "ESA/CCI/FireCCI/5_1",
    "globfire_daily": "JRC/GWIS/GlobFire/v2/DailyPerimeters",
    "s2_sr": "COPERNICUS/S2_SR_HARMONIZED",
    "landsat8_l2": "LANDSAT/LC08/C02/T1_L2",
    "landsat9_l2": "LANDSAT/LC09/C02/T1_L2",
    "gridmet": "IDAHO_EPSCOR/GRIDMET",
    "gridmet_drought": "GRIDMET/DROUGHT",
    "era5_land_hourly": "ECMWF/ERA5_LAND/HOURLY",
    "smap_l4": "NASA/SMAP/SPL4SMGP/007",
    "modis_ndvi": "MODIS/061/MOD13A1",
    "modis_lst": "MODIS/061/MOD11A1",
    "ghsl_pop": "JRC/GHSL/P2023A/GHS_POP",
    "satellite_embedding": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
}


def initialize(project: str | None = None, high_volume: bool = True) -> None:
    """Authenticate and initialise Earth Engine against the right endpoint.

    Always pass ``high_volume=True`` for programmatic fan-out. The standard
    endpoint caches intermediates and is tuned for interactive human use;
    hammering it with a process pool is both slower and antisocial.
    """
    import ee  # lazy: heavy optional dependency

    opts: dict[str, Any] = {}
    if project:
        opts["project"] = project
    if high_volume:
        opts["opt_url"] = HIGH_VOLUME_ENDPOINT
    try:
        ee.Initialize(**opts)
    except Exception:  # noqa: BLE001 - first-run auth flow
        log.info("Earth Engine not initialised; starting authentication flow")
        ee.Authenticate()
        ee.Initialize(**opts)


def with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs: Any,
) -> Any:
    """Exponential backoff with full jitter.

    Earth Engine returns transient 429/500 under concurrency; without this the
    failure rate on a 25-worker pool is high enough to poison a whole export.
    """
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - EE raises a wide surface
            last = exc
            if attempt == max_attempts - 1:
                break
            delay = min(max_delay, base_delay * 2**attempt)
            time.sleep(random.uniform(0, delay))
            log.debug("EE retry %d/%d after %s", attempt + 1, max_attempts, exc)
    raise RuntimeError(f"Earth Engine call failed after {max_attempts} attempts") from last


@dataclass(frozen=True, slots=True)
class PatchRequest:
    """One chip to extract."""

    uid: str
    #: (xmin, ymin, xmax, ymax) in ``crs``.
    bounds: tuple[float, float, float, float]
    crs: str
    bands: tuple[str, ...]
    scale_m: float = 375.0


def fetch_patch(image: Any, req: PatchRequest) -> np.ndarray:
    """Fetch a single chip as a structured NumPy array via ``computePixels``.

    Returns a structured array with one field per band -- note that this
    *preserves dtypes*, which is the whole point of choosing this path over
    TFRecord export.
    """
    import ee

    x0, y0, x1, y1 = req.bounds
    width = int(round((x1 - x0) / req.scale_m))
    height = int(round((y1 - y0) / req.scale_m))
    request = {
        "expression": image.select(list(req.bands)),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {
                "scaleX": req.scale_m,
                "shearX": 0,
                "translateX": x0,
                "shearY": 0,
                "scaleY": -req.scale_m,
                "translateY": y1,
            },
            "crsCode": req.crs,
        },
    }
    return with_retry(ee.data.computePixels, request)


def fetch_patches(
    image_factory: Callable[[], Any],
    requests: Sequence[PatchRequest],
    n_workers: int = 25,
) -> Iterable[tuple[str, np.ndarray]]:
    """Fetch many chips concurrently.

    ``image_factory`` is called *inside* each worker, because ``ee`` objects
    do not survive pickling across processes. ``n_workers`` should stay at or
    below the documented 40 concurrent-request limit; 25 leaves headroom for
    retries.

    Yields ``(uid, array)`` as results arrive. Failures are logged and skipped
    rather than aborting the export -- the caller is expected to diff requested
    against received uids and re-run, which is why the whole pipeline is
    resumable.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if n_workers > GEE_LIMITS["concurrent_requests"]:
        raise ValueError(
            f"n_workers={n_workers} exceeds the documented concurrent request limit "
            f"({GEE_LIMITS['concurrent_requests']})"
        )

    def _worker(req: PatchRequest) -> tuple[str, np.ndarray]:
        initialize(high_volume=True)
        return req.uid, fetch_patch(image_factory(), req)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, r): r for r in requests}
        for fut in as_completed(futures):
            req = futures[fut]
            try:
                yield fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("patch %s failed: %s", req.uid, exc)


def structured_to_stack(arr: np.ndarray, bands: Sequence[str]) -> np.ndarray:
    """Convert a ``computePixels`` structured array to ``(C, H, W)`` float32."""
    return np.stack([arr[b].astype(np.float32) for b in bands], axis=0)
