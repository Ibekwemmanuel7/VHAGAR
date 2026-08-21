"""Next-day fire spread samples (WildfireSpreadTS-style) for a physics-informed,
calibrated forecaster.

The task: given day-t multimodal rasters plus the day-t active-fire mask, predict
the day-(t+1) active-fire mask. The public benchmark is WildfireSpreadTS (Gerard
et al. 2023), ~600 fires as daily 23-channel GeoTIFF stacks. This module gives:

* :class:`WFSample`, the per-day sample the forecaster and eval consume;
* :func:`synthetic_wfs_fire`, a labelled synthetic generator whose next-day truth
  is driven by the physics *plus a barrier channel the physics prior cannot see*,
  so a learned corrector has something real to add and the harness is testable
  offline;
* :func:`load_wfs_geotiff_pair`, a thin reader for the real day-t / day-(t+1)
  GeoTIFFs (rasterio, lazy), parameterised by a channel map because the real
  stack's channel order differs from ours.

Design follows the repo convention: the pure sample schema + synthetic generator
are numpy and unit-tested; file IO lives only in the thin reader at the edge.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["CHANNELS", "WFSample", "synthetic_wfs_fire", "load_wfs_geotiff_pair"]

#: Feature channel order used by the synthetic data and the physics prior. The
#: prior reads fuel/wind/slope (isotropic ROS); ``active_fire`` is the day-t seed;
#: ``barrier`` is a suppression layer (roads, firebreaks, water) that the physics
#: prior ignores but a learned corrector can exploit.
CHANNELS: dict[str, int] = {"fuel": 0, "wind": 1, "slope": 2, "active_fire": 3, "barrier": 4}


@dataclass(slots=True)
class WFSample:
    """One next-day spread example on a single fire's grid."""

    fire_id: str
    features: np.ndarray          # [C, H, W] float, channel order per CHANNELS
    fire_t: np.ndarray            # [H, W] bool, active fire on day t (the seed)
    fire_t1: np.ndarray           # [H, W] bool, active fire on day t+1 (target)
    valid: np.ndarray             # [H, W] bool, usable pixels
    date: str | None = None
    channels: Mapping[str, int] = field(default_factory=lambda: dict(CHANNELS))

    def _ch(self, name: str) -> np.ndarray:
        return self.features[self.channels[name]]

    @property
    def fuel(self) -> np.ndarray:
        return self._ch("fuel")

    @property
    def wind(self) -> np.ndarray:
        return self._ch("wind")

    @property
    def slope(self) -> np.ndarray:
        return self._ch("slope")

    @property
    def new_burn(self) -> np.ndarray:
        """Cells that ignite between t and t+1 (the region the forecast is judged
        on: predicting already-burning cells is trivial persistence)."""
        return self.fire_t1 & ~self.fire_t & self.valid

    @property
    def is_usable(self) -> bool:
        """Usable only if there is genuinely new burn to predict and some spare
        unburned area to be wrong about."""
        incr = ~self.fire_t & self.valid
        return bool(self.new_burn.any() and incr.sum() > self.new_burn.sum())


def _smooth_field(rng: np.random.Generator, h: int, w: int, k: int = 3) -> np.ndarray:
    """A smooth [0,1] field: low-frequency sinusoids so neighbours correlate."""
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    f = np.zeros((h, w))
    for _ in range(k):
        a, b = rng.uniform(1.5, 5.0, size=2)
        ph = rng.uniform(0, 2 * np.pi, size=2)
        f += np.sin(a * 2 * np.pi * yy + ph[0]) * np.cos(b * 2 * np.pi * xx + ph[1])
    f = (f - f.min()) / (np.ptp(f) + 1e-9)
    return f


def synthetic_wfs_fire(
    rng: np.random.Generator, fire_id: str = "syn", H: int = 48, W: int = 48,
    t0_q: float = 0.15, dt_q: float = 0.40, label_noise: float = 0.05,
) -> tuple[WFSample, float]:
    """One synthetic next-day sample. Returns ``(sample, true_horizon)``.

    The next-day truth is the fast-marching front on a ROS field that a *barrier*
    layer suppresses; the physics prior (built from fuel/wind/slope only) cannot
    see the barrier, so a corrector that reads the barrier channel can beat it.
    ``true_horizon`` is the arrival-time increment from t to t+1, a reasonable
    default for the physics prior's horizon."""
    from vhagar.models.spread import fast_marching_arrival, rate_of_spread

    fuel = _smooth_field(rng, H, W)
    wind = _smooth_field(rng, H, W)
    slope = _smooth_field(rng, H, W)
    barrier = (_smooth_field(rng, H, W) > 0.75).astype(np.float64)   # sparse firebreaks

    ros_true = rate_of_spread(fuel, wind, slope) * (1.0 - 0.85 * barrier)
    seed = np.zeros((H, W), dtype=bool)
    cy, cx = rng.integers(H // 4, 3 * H // 4), rng.integers(W // 4, 3 * W // 4)
    seed[cy, cx] = True
    arrival = fast_marching_arrival(np.maximum(ros_true, 1e-3), seed)
    reach = np.isfinite(arrival)
    tv = arrival[reach]
    t0 = float(np.quantile(tv, t0_q))
    t1 = float(np.quantile(tv, dt_q))
    fire_t = reach & (arrival <= t0)
    fire_t1 = reach & (arrival <= t1)
    if label_noise > 0:
        flip = rng.random((H, W)) < label_noise
        fire_t1 = np.where(flip, ~fire_t1, fire_t1) & reach
    fire_t1 = fire_t1 | fire_t   # already-burning stays burning

    feats = np.zeros((len(CHANNELS), H, W), dtype=np.float64)
    feats[CHANNELS["fuel"]] = fuel
    feats[CHANNELS["wind"]] = wind
    feats[CHANNELS["slope"]] = slope
    feats[CHANNELS["active_fire"]] = fire_t.astype(np.float64)
    feats[CHANNELS["barrier"]] = barrier
    valid = np.ones((H, W), dtype=bool)
    return WFSample(fire_id, feats, fire_t, fire_t1, valid), float(t1 - t0)


def load_wfs_geotiff_pair(
    day_t_tif: str | Path, day_t1_tif: str | Path, *, channel_map: Mapping[str, int],
    fire_channel: int, fire_threshold: float = 0.0, fire_id: str | None = None,
) -> WFSample:
    """Thin reader for a real WildfireSpreadTS day-t / day-(t+1) pair.

    ``channel_map`` maps our names (fuel, wind, slope, active_fire, barrier) to
    the source stack's band indices; ``fire_channel`` is the active-fire band in
    each daily stack (a detection is a value above ``fire_threshold``). Needs
    rasterio. The real stacks live on Zenodo / Hugging Face (WildfireSpreadTS)."""
    import rasterio

    dt, dt1 = Path(day_t_tif), Path(day_t1_tif)
    for p in (dt, dt1):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download WildfireSpreadTS (Gerard et al. 2023) from "
                "its Zenodo/Hugging Face release.")
    with rasterio.open(dt) as src:
        stack_t = src.read().astype(np.float64)
    with rasterio.open(dt1) as src:
        stack_t1 = src.read().astype(np.float64)
    _, H, W = stack_t.shape
    feats = np.zeros((len(CHANNELS), H, W), dtype=np.float64)
    for name, idx in CHANNELS.items():
        if name == "active_fire":
            continue
        src_idx = channel_map.get(name)
        if src_idx is not None:
            band = stack_t[src_idx]
            rng = np.ptp(band)
            feats[idx] = (band - band.min()) / (rng + 1e-9)
    fire_t = stack_t[fire_channel] > fire_threshold
    fire_t1 = stack_t1[fire_channel] > fire_threshold
    feats[CHANNELS["active_fire"]] = fire_t.astype(np.float64)
    valid = np.isfinite(stack_t).all(axis=0) & np.isfinite(stack_t1[fire_channel])
    return WFSample(fire_id or dt.stem, feats, fire_t, fire_t1 | fire_t, valid)
