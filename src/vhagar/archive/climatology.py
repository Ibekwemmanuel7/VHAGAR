"""Diurnal climatology reducer for the Tier B radiance archive.

What this is for
----------------
The persistence and diurnal-baseline features are statistics, not frames:
``mu(pixel, hour)`` and ``sigma(pixel, hour)`` per channel. A pixel that always
runs warm at 2 pm local is not anomalous at 2 pm; the anomaly is the departure
from its own diurnal baseline. Storing every frame to compute that later would
cost terabytes, so this reduces frames online into per-pixel, per-hour running
mean and variance and keeps only the statistics.

Why UTC-hour bins are the right diurnal bins here
-------------------------------------------------
GOES is geostationary, so every pixel has a fixed longitude and therefore a fixed
offset between UTC and local solar time. Binning a pixel's samples by UTC hour is
the same as binning by its own local time, shifted by a constant that depends
only on longitude. So ``mu(pixel, utc_hour)`` already is that pixel's diurnal
baseline; the local-time relabelling, if wanted, is a per-pixel constant applied
at use time. No per-pixel scatter into different bins is needed, which keeps the
update a single vectorised pass.

Numerics
--------
Welford's online algorithm, vectorised over the pixel grid, so mean and variance
are computed in one pass without holding the samples and without the catastrophic
cancellation of the sum-of-squares formula. Updates are NaN-aware: a fill,
bad-DQF or saturated pixel (all NaN out of the CMIP decoder) simply does not
update that pixel's accumulators, so each pixel's count reflects only its valid
samples. Two accumulators merge with the parallel form (Chan et al.), so a
backfill can reduce shards concurrently and combine them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

__all__ = ["DiurnalClimatology"]


class DiurnalClimatology:
    """Running per-pixel, per-hour mean and variance for a set of channels.

    Memory is ``n_bins * n_channels * pixels * 3`` float64 values, so this is
    sized for a tile (a 96 km tile is 48 x 48 pixels at 2 km), not full CONUS.
    The Tier B backfill runs it per tile.
    """

    def __init__(
        self,
        channels: Sequence[str],
        shape: tuple[int, int],
        n_bins: int = 24,
    ) -> None:
        if n_bins <= 0 or 1440 % n_bins != 0:
            # Bins must divide the day into whole minutes: 24 is hourly, 48 is
            # 30-minute, 96 is 15-minute. 7 would not tile the day evenly.
            raise ValueError(f"n_bins must divide 1440 (minutes per day), got {n_bins}")
        self.channels = tuple(channels)
        self.shape = tuple(shape)
        self.n_bins = int(n_bins)
        full = (self.n_bins, *self.shape)
        self._count = {c: np.zeros(full, dtype=np.float64) for c in self.channels}
        self._mean = {c: np.zeros(full, dtype=np.float64) for c in self.channels}
        self._m2 = {c: np.zeros(full, dtype=np.float64) for c in self.channels}

    # -- binning ---------------------------------------------------------

    def bin_for(self, when: datetime) -> int:
        """UTC diurnal bin index for a timestamp."""
        hour = when.hour + when.minute / 60.0 + when.second / 3600.0
        return int(hour * self.n_bins / 24.0) % self.n_bins

    # -- update ----------------------------------------------------------

    def update_frame(self, bin_index: int, values: Mapping[str, np.ndarray]) -> None:
        """Fold one frame of per-channel arrays into a given bin, NaN-aware.

        Vectorised Welford. Pixels that are NaN in ``values`` do not update, so
        each pixel's count is the number of valid samples it has seen.
        """
        if not 0 <= bin_index < self.n_bins:
            raise ValueError(f"bin_index {bin_index} out of range [0, {self.n_bins})")
        for channel, arr in values.items():
            if channel not in self._count:
                continue
            x = np.asarray(arr, dtype=np.float64)
            if x.shape != self.shape:
                raise ValueError(
                    f"channel {channel} shape {x.shape} does not match {self.shape}"
                )
            valid = np.isfinite(x)
            count = self._count[channel][bin_index]
            mean = self._mean[channel][bin_index]
            m2 = self._m2[channel][bin_index]

            new_count = count + valid
            delta = np.where(valid, x - mean, 0.0)
            denom = np.where(new_count == 0, 1.0, new_count)
            new_mean = mean + np.where(valid, delta / denom, 0.0)
            delta2 = np.where(valid, x - new_mean, 0.0)

            self._count[channel][bin_index] = new_count
            self._mean[channel][bin_index] = new_mean
            self._m2[channel][bin_index] = m2 + delta * delta2

    def update(self, stack) -> None:
        """Fold a :class:`~vhagar.io.cmip_reader.CMIPStack` into its UTC-hour bin.

        Duck-typed on ``scan_start`` and ``bt_k`` so this module does not depend
        on the io layer.
        """
        values = {c: stack.bt_k[c] for c in self.channels if c in stack.bt_k}
        self.update_frame(self.bin_for(stack.scan_start), values)

    # -- readouts --------------------------------------------------------

    def count(self, channel: str) -> np.ndarray:
        """Valid-sample count per bin and pixel."""
        return self._count[channel]

    def mean(self, channel: str) -> np.ndarray:
        """Mean per bin and pixel, NaN where no samples were seen."""
        count = self._count[channel]
        return np.where(count > 0, self._mean[channel], np.nan)

    def variance(self, channel: str, ddof: int = 1) -> np.ndarray:
        """Variance per bin and pixel, NaN where fewer than ``ddof + 1`` samples."""
        count = self._count[channel]
        denom = count - ddof
        return np.where(denom > 0, self._m2[channel] / np.where(denom > 0, denom, 1.0), np.nan)

    def std(self, channel: str, ddof: int = 1) -> np.ndarray:
        """Standard deviation per bin and pixel."""
        return np.sqrt(self.variance(channel, ddof=ddof))

    # -- combine ---------------------------------------------------------

    def merge(self, other: DiurnalClimatology) -> DiurnalClimatology:
        """Return the accumulator equal to having seen both streams.

        Parallel Welford (Chan, Golub, LeVeque), so shards reduced concurrently
        combine exactly, with no dependence on order.
        """
        if (self.channels, self.shape, self.n_bins) != (
            other.channels, other.shape, other.n_bins
        ):
            raise ValueError("cannot merge climatologies with different layout")
        out = DiurnalClimatology(self.channels, self.shape, self.n_bins)
        for c in self.channels:
            ca, cb = self._count[c], other._count[c]
            count = ca + cb
            denom = np.where(count == 0, 1.0, count)
            delta = other._mean[c] - self._mean[c]
            out._count[c] = count
            out._mean[c] = self._mean[c] + delta * np.where(count == 0, 0.0, cb / denom)
            out._m2[c] = (
                self._m2[c] + other._m2[c] + delta**2 * np.where(count == 0, 0.0, ca * cb / denom)
            )
        return out

    # -- persistence -----------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Write the accumulator to a ``.npz`` file."""
        path = Path(path)
        payload: dict[str, np.ndarray] = {
            "__channels__": np.array(self.channels),
            "__shape__": np.array(self.shape),
            "__n_bins__": np.array([self.n_bins]),
        }
        for c in self.channels:
            payload[f"{c}::count"] = self._count[c]
            payload[f"{c}::mean"] = self._mean[c]
            payload[f"{c}::m2"] = self._m2[c]
        np.savez(path, **payload)
        return path if path.suffix else path.with_suffix(".npz")

    @classmethod
    def load(cls, path: Path | str) -> DiurnalClimatology:
        """Read an accumulator written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as z:
            channels = [str(c) for c in z["__channels__"]]
            shape = tuple(int(v) for v in z["__shape__"])
            n_bins = int(z["__n_bins__"][0])
            obj = cls(channels, shape, n_bins)
            for c in channels:
                obj._count[c] = z[f"{c}::count"]
                obj._mean[c] = z[f"{c}::mean"]
                obj._m2[c] = z[f"{c}::m2"]
        return obj
