"""T1 Stage-1 differentiator: temporal-anomaly early detection on the 3.9 um series.

The geostationary contextual algorithms (GOES FDC, SLSTR FRP) flag a fire when its
brightness temperature crosses an *absolute* contextual threshold. That is late: the
threshold must sit above the midday diurnal peak to avoid false alarms, so a fire, and
especially a night fire starting from a cold baseline, is not flagged until it is well
developed. The architecture's one learned Stage-1 component forecasts the *expected*
per-pixel BT from recent history plus solar geometry, and flags **residual** excursions.
Because the residual is measured against each pixel's own diurnal baseline, a fire is
caught as soon as it lifts BT above that baseline, not when it crosses a global cut, so
detection is earlier at the same false-alarm rate. Published evidence: porting a
rapid-scan temporal algorithm to 5-min geostationary data roughly doubled mean lead time
(35 -> 65 min) ahead of official reporting.

This module makes that concrete. The numpy pieces, a per-pixel diurnal forecaster, the
residual and absolute-threshold detectors, matched-FAR calibration, and the lead-time
experiment, run anywhere and are unit-tested on synthetic BT series with an injected
fire. The production forecaster is ``models.TemporalAnomalyNet`` (a 3D-conv TCN trained
on clear-sky 3.9 um cubes); ``train_temporal_net`` wires it for real data and needs the
torch extra. Only the forecaster changes; the residual/lead-time protocol is identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DiurnalForecaster",
    "HourlyBaselineForecaster",
    "RealLeadResult",
    "synthetic_bt_series",
    "calibrate_threshold_to_far",
    "early_detection_experiment",
    "climatology_diurnal_amplitude",
    "hourly_baseline_residual",
    "real_lead_experiment",
    "train_temporal_net",
]


def climatology_diurnal_amplitude(npz_path, channel: str = "C07", min_bins: int = 8) -> dict:
    """Real diurnal amplitude of a channel from a saved :class:`DiurnalClimatology`.

    Loads the per-pixel, per-UTC-hour mean/variance ``.npz`` (``{ch}::mean``, ``m2``,
    ``count``) and returns the per-pixel diurnal amplitude ``max_hour(mean) -
    min_hour(mean)`` summary for the fire channel. This is the concrete sensitivity an
    *absolute* contextual threshold sacrifices: it must sit roughly one amplitude above
    the night baseline to avoid daytime false alarms, which a residual detector recovers.
    Pure numpy. Returns median/p25/p90 amplitude (K), median per-hour sigma, and pixel
    count.
    """
    z = np.load(npz_path)
    mean = z[f"{channel}::mean"]
    m2 = z[f"{channel}::m2"]
    cnt = z[f"{channel}::count"]
    valid = (cnt > 0).sum(axis=0) >= min_bins
    mu = np.where(cnt > 0, mean, np.nan)[:, valid]
    amp = np.nanmax(mu, axis=0) - np.nanmin(mu, axis=0)
    var = np.where(cnt > 1, m2 / np.maximum(cnt - 1, 1), np.nan)[:, valid]
    return {
        "channel": channel,
        "n_pixels": int(valid.sum()),
        "amplitude_k_median": float(np.nanmedian(amp)),
        "amplitude_k_p25": float(np.nanpercentile(amp, 25)),
        "amplitude_k_p90": float(np.nanpercentile(amp, 90)),
        "sigma_k_median": float(np.nanmedian(np.sqrt(var))),
    }


def synthetic_bt_series(
    n_days: int = 3, cadence_min: int = 5, n_pixels: int = 200,
    diurnal_amp: float = 12.0, base_bt: float = 285.0, noise_sd: float = 1.5,
    fire_onset_frac: float = 0.75, fire_ramp_k_per_h: float = 25.0,
    fire_at_night: bool = True, seed: int = 0,
):
    """Synthetic 3.9 um brightness-temperature series with one injected fire.

    Every pixel has a diurnal cosine (peak mid-afternoon) plus Gaussian noise; the last
    pixel gets a linear BT ramp starting at ``fire_onset_frac`` of the record, placed at
    night by default (a cold baseline, where an absolute threshold is slowest). Returns
    ``(hours, bt, fire_pixel, onset_idx)`` with ``bt`` shaped ``(n_pixels, n_t)``.
    """
    rng = np.random.default_rng(seed)
    steps = int(n_days * 24 * 60 / cadence_min)
    hours = np.arange(steps) * (cadence_min / 60.0)
    diurnal = base_bt + diurnal_amp * np.cos(2 * np.pi * (hours % 24 - 14.0) / 24.0)
    bt = diurnal[None, :] + rng.normal(0.0, noise_sd, size=(n_pixels, steps))

    fire_pixel = n_pixels - 1
    # place onset near a chosen fraction, nudged to a night hour if requested
    onset = int(fire_onset_frac * steps)
    if fire_at_night:
        while (hours[onset] % 24) > 5 and (hours[onset] % 24) < 20:
            onset = (onset + 1) % steps
    ramp = np.maximum(0.0, (hours - hours[onset])) * fire_ramp_k_per_h
    bt[fire_pixel] = bt[fire_pixel] + ramp
    return hours, bt, fire_pixel, onset


@dataclass(slots=True)
class DiurnalForecaster:
    """Per-pixel expected-BT baseline: a harmonic fit of BT on hour-of-day.

    Fit on a pixel's own (fire-free) history, then the residual ``BT - predicted`` is the
    anomaly signal. Harmonic-in-hour rather than a global threshold is the whole point:
    it removes the diurnal cycle, so a fire stands out at any time of day.
    """

    coef: np.ndarray            # [n_pixels, 2*n_harmonics+1]
    n_harmonics: int

    @staticmethod
    def _design(hours: np.ndarray, n_harmonics: int) -> np.ndarray:
        cols = [np.ones_like(hours)]
        for k in range(1, n_harmonics + 1):
            cols.append(np.sin(2 * np.pi * k * hours / 24.0))
            cols.append(np.cos(2 * np.pi * k * hours / 24.0))
        return np.column_stack(cols)

    @classmethod
    def fit(cls, hours: np.ndarray, bt: np.ndarray, n_harmonics: int = 3) -> DiurnalForecaster:
        A = cls._design(hours, n_harmonics)
        coef, *_ = np.linalg.lstsq(A, bt.T, rcond=None)   # [n_feat, n_pixels]
        return cls(coef=coef.T, n_harmonics=n_harmonics)

    def predict(self, hours: np.ndarray) -> np.ndarray:
        A = self._design(hours, self.n_harmonics)
        return (A @ self.coef.T).T                        # [n_pixels, n_t]

    def residual(self, hours: np.ndarray, bt: np.ndarray) -> np.ndarray:
        return bt - self.predict(hours)


def calibrate_threshold_to_far(scores_fire_free: np.ndarray, target_far: float) -> float:
    """Threshold whose exceedance rate on fire-free samples equals ``target_far``.

    Both detectors are calibrated this way so the lead-time comparison is at *equal*
    false-alarm rate, the only fair way to compare an early detector to a late one.
    """
    q = 100.0 * (1.0 - target_far)
    return float(np.percentile(scores_fire_free, q))


@dataclass(frozen=True, slots=True)
class EarlyDetectionResult:
    lead_minutes: float
    residual_detect_min_after_onset: float
    absolute_detect_min_after_onset: float
    target_far: float


def early_detection_experiment(
    hours: np.ndarray, bt: np.ndarray, fire_pixel: int, onset_idx: int,
    forecaster: DiurnalForecaster, target_far: float = 0.01, cadence_min: int = 5,
) -> EarlyDetectionResult:
    """At a matched false-alarm rate, how many minutes earlier does the residual
    detector flag the fire than an absolute-BT threshold?

    Fire-free pixels calibrate both thresholds to ``target_far``. On the fire pixel, the
    first post-onset exceedance of each detector is found; the lead is
    ``absolute_time - residual_time`` (positive = residual detector is earlier).
    """
    free = np.ones(bt.shape[0], dtype=bool)
    free[fire_pixel] = False

    resid = forecaster.residual(hours, bt)
    thr_res = calibrate_threshold_to_far(resid[free].ravel(), target_far)
    thr_abs = calibrate_threshold_to_far(bt[free].ravel(), target_far)

    def first_after(scores_row, thr):
        idx = np.flatnonzero((scores_row > thr) & (np.arange(len(scores_row)) >= onset_idx))
        return int(idx[0]) if idx.size else None

    r_idx = first_after(resid[fire_pixel], thr_res)
    a_idx = first_after(bt[fire_pixel], thr_abs)
    big = 10**9
    r_min = (r_idx - onset_idx) * cadence_min if r_idx is not None else big
    a_min = (a_idx - onset_idx) * cadence_min if a_idx is not None else big
    lead = (a_min - r_min) if (r_idx is not None or a_idx is not None) else 0.0
    return EarlyDetectionResult(
        lead_minutes=float(lead),
        residual_detect_min_after_onset=float(r_min),
        absolute_detect_min_after_onset=float(a_min),
        target_far=target_far,
    )


@dataclass(slots=True)
class HourlyBaselineForecaster:
    """NaN-safe diurnal baseline for the *real* cube: per-pixel, per-hour-bin mean BT.

    The harmonic :class:`DiurnalForecaster` needs a clean series; a real 3.9 um cube is
    full of NaN (cloud, fill, saturation), which ``lstsq`` cannot take. This forecaster is
    the same diurnal-baseline idea expressed as the on-the-fly counterpart of
    :class:`~vhagar.archive.climatology.DiurnalClimatology`: bin frames by UTC hour and
    take the per-pixel ``nanmean``. The residual ``BT - baseline[hour]`` is the anomaly
    score, NaN wherever the pixel itself is NaN, so nodata never masquerades as an anomaly.
    """

    baseline: np.ndarray        # [n_pixels, n_bins] per-pixel per-hour-bin mean BT
    n_bins: int

    @classmethod
    def fit(
        cls, hours: np.ndarray, bt: np.ndarray, n_bins: int = 24,
        clear_mask: np.ndarray | None = None,
    ) -> HourlyBaselineForecaster:
        """Fit on (clear-sky) history. ``bt`` is ``[n_pixels, n_t]``; ``hours`` is the
        UTC hour-of-day per frame. ``clear_mask`` (``[n_t]``) selects the frames used as
        the baseline (default: all)."""
        n_pix = bt.shape[0]
        base = np.full((n_pix, n_bins), np.nan, dtype=np.float64)
        bins = np.floor((hours % 24) / (24.0 / n_bins)).astype(int) % n_bins
        sel = np.ones(bt.shape[1], dtype=bool) if clear_mask is None else clear_mask
        for b in range(n_bins):
            cols = np.flatnonzero((bins == b) & sel)
            if cols.size:
                with np.errstate(invalid="ignore"):
                    base[:, b] = np.nanmean(bt[:, cols], axis=1)
        return cls(baseline=base, n_bins=n_bins)

    def predict(self, hours: np.ndarray) -> np.ndarray:
        bins = np.floor((hours % 24) / (24.0 / self.n_bins)).astype(int) % self.n_bins
        return self.baseline[:, bins]           # [n_pixels, n_t]

    def residual(self, hours: np.ndarray, bt: np.ndarray) -> np.ndarray:
        return bt - self.predict(hours)


def hourly_baseline_residual(
    hours: np.ndarray, bt2d: np.ndarray, n_bins: int = 24,
    clear_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience: fit :class:`HourlyBaselineForecaster` and return the residual array."""
    fc = HourlyBaselineForecaster.fit(hours, bt2d, n_bins=n_bins, clear_mask=clear_mask)
    return fc.residual(hours, bt2d)


@dataclass(frozen=True, slots=True)
class RealLeadResult:
    n_fire_pixels: int
    n_residual_led: int
    frac_residual_led: float
    median_lead_min: float
    p25_lead_min: float
    p75_lead_min: float
    target_far: float
    residual_threshold_k: float


def real_lead_experiment(
    residual2d: np.ndarray, fdc_first_idx: np.ndarray, target_far: float = 0.01,
    cadence_min: int = 5, min_valid_frac: float = 0.5,
) -> RealLeadResult:
    """On a real cube, how much earlier does the residual detector fire than GOES FDC?

    ``residual2d`` is ``[n_pixels, n_t]`` (a flattened cube's residual); ``fdc_first_idx``
    is the per-pixel first FDC-detection frame index (``-1`` where FDC never fired), from
    :func:`vhagar.archive.temporal_cube.fdc_first_detection_grid`, flattened to
    ``[n_pixels]``. The residual threshold is calibrated to ``target_far`` on the
    **fire-free** pixels (those FDC never flagged and that are mostly valid), the matched
    false-alarm rate that makes the comparison fair. For each fire pixel, the lead is
    ``(fdc_first_idx - first residual exceedance) * cadence_min`` minutes; positive means
    the residual detector was earlier. Pure numpy.
    """
    fire = fdc_first_idx >= 0
    valid_frac = np.isfinite(residual2d).mean(axis=1)
    free = (~fire) & (valid_frac >= min_valid_frac)
    if not free.any():
        raise ValueError("no fire-free, mostly-valid pixels to calibrate the threshold on")

    free_scores = residual2d[free]
    thr = float(np.nanpercentile(free_scores, 100.0 * (1.0 - target_far)))

    leads: list[float] = []
    n_led = 0
    fire_idx = np.flatnonzero(fire)
    for p in fire_idx:
        row = residual2d[p]
        exceed = np.flatnonzero(np.nan_to_num(row, nan=-np.inf) > thr)
        if exceed.size == 0:
            continue                       # residual never fired on this pixel
        r_first = int(exceed[0])
        lead = (int(fdc_first_idx[p]) - r_first) * cadence_min
        leads.append(float(lead))
        if lead > 0:
            n_led += 1
    if not leads:
        return RealLeadResult(
            n_fire_pixels=int(fire.sum()), n_residual_led=0, frac_residual_led=0.0,
            median_lead_min=0.0, p25_lead_min=0.0, p75_lead_min=0.0,
            target_far=target_far, residual_threshold_k=thr,
        )
    a = np.array(leads)
    return RealLeadResult(
        n_fire_pixels=len(leads),
        n_residual_led=n_led,
        frac_residual_led=float(n_led / len(leads)),
        median_lead_min=float(np.median(a)),
        p25_lead_min=float(np.percentile(a, 25)),
        p75_lead_min=float(np.percentile(a, 75)),
        target_far=target_far,
        residual_threshold_k=thr,
    )


def train_temporal_net(
    cube: np.ndarray, covariates: np.ndarray | None = None, window: int = 6,
    epochs: int = 10, lr: float = 1e-3, seed: int = 0, device: str | None = None,
):
    """Train ``TemporalAnomalyNet`` to forecast the next BT frame from a window of past
    frames (+ optional covariates), on clear-sky history. Needs torch.

    ``cube`` is ``[T, H, W]`` brightness temperature; ``covariates`` is optional
    ``[T, C-1, H, W]`` exogenous channels (solar zenith, day-of-year encoding). Returns
    the trained model; residuals at inference are the anomaly score, fed to the same
    matched-FAR / lead-time protocol as the numpy path.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("train_temporal_net needs torch; install the torch extra") from exc
    import torch.nn.functional as F  # noqa: N812

    from vhagar.models.segmentation import TemporalAnomalyNet
    from vhagar.train.train import set_seeds

    set_seeds(seed, deterministic=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    T, H, W = cube.shape
    x = torch.from_numpy(cube.astype(np.float32))[:, None]            # [T,1,H,W]
    if covariates is not None:
        x = torch.cat([x, torch.from_numpy(covariates.astype(np.float32))], dim=1)
    x = x.to(dev)
    in_ch = x.shape[1]
    model = TemporalAnomalyNet(in_channels=in_ch, window=window).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for t in range(window, T):
            win = x[t - window:t][None]                              # [1,window,C,H,W]
            target = x[t, 0][None, None]                             # [1,1,H,W] next BT
            loss = F.smooth_l1_loss(model(win), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model
