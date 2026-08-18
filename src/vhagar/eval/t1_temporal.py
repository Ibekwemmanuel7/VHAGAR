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

import warnings
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
    "baseline_contamination",
    "hourly_baseline_residual",
    "real_lead_experiment",
    "cohort_lead_summary",
    "train_temporal_net",
    "temporal_net_residuals",
    "learned_residuals",
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
                # An hour bin with no valid sample for some pixel yields an all-NaN
                # slice; that is expected (partial coverage) and leaves the baseline NaN
                # there, so silence only that benign warning rather than the array.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    base[:, b] = np.nanmean(bt[:, cols], axis=1)
        return cls(baseline=base, n_bins=n_bins)

    def predict(self, hours: np.ndarray) -> np.ndarray:
        bins = np.floor((hours % 24) / (24.0 / self.n_bins)).astype(int) % self.n_bins
        return self.baseline[:, bins]           # [n_pixels, n_t]

    def residual(self, hours: np.ndarray, bt: np.ndarray) -> np.ndarray:
        return bt - self.predict(hours)


def baseline_contamination(fdc_first_idx: np.ndarray, clear_mask: np.ndarray) -> float:
    """Fraction of fire pixels whose FIRST FDC detection falls inside the clear-sky window.

    The diurnal baseline is only a clean reference if it is fit on fire-free frames. If a
    fire ignites early, the "clear" frames used for its baseline already contain its hot
    brightness temperature, so the baseline is contaminated and the residual is
    meaningless (it can trip on baseline error, inflating apparent lead at a loose FAR, or
    be suppressed, lagging at a strict one). This returns the share of fire pixels for
    which that happened, so a caller can refuse to trust the lead-time table. Pure numpy.
    """
    fire = np.flatnonzero(fdc_first_idx >= 0)
    if fire.size == 0:
        return 0.0
    clear_idx = np.flatnonzero(clear_mask)
    if clear_idx.size == 0:
        return 0.0
    last_clear = int(clear_idx[-1])
    contaminated = np.count_nonzero(fdc_first_idx[fire] <= last_clear)
    return float(contaminated / fire.size)


def hourly_baseline_residual(
    hours: np.ndarray, bt2d: np.ndarray, n_bins: int = 24,
    clear_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience: fit :class:`HourlyBaselineForecaster` and return the residual array."""
    fc = HourlyBaselineForecaster.fit(hours, bt2d, n_bins=n_bins, clear_mask=clear_mask)
    return fc.residual(hours, bt2d)


@dataclass(frozen=True, slots=True)
class RealLeadResult:
    n_fire_pixels: int                     # fire pixels that the residual actually detected
    n_fire_pixels_total: int               # all fire pixels (detected or not)
    detection_rate: float                  # n_fire_pixels / n_fire_pixels_total
    n_residual_led: int
    frac_residual_led: float               # among detected pixels
    median_lead_min: float                 # among detected pixels; NaN if none detected
    p25_lead_min: float
    p75_lead_min: float
    target_far: float
    residual_threshold_k: float
    leads_min: tuple[float, ...] = ()      # detected pixels' leads, for pooled aggregation


def _per_frame_threshold(
    free_scores: np.ndarray, target_far: float, hours: np.ndarray | None, far_bins: int,
) -> np.ndarray:
    """Residual threshold per frame, calibrated on fire-free pixels.

    With ``far_bins == 1`` (or no ``hours``) this is one global percentile, the original
    behaviour. With ``far_bins > 1`` the day is split into that many time-of-day bins and
    each gets its own percentile, pooled across all fire-free pixels in that bin. That is
    the fix for the night-blindness of a global cut: daytime 3.9 um residuals have much
    larger variance (solar reflectance), so a single percentile is set by daytime and
    desensitises night. A per-time-of-day threshold judges a night fire against the night
    distribution, which is the entire point of removing the diurnal cycle. A bin too sparse
    to calibrate falls back to the global threshold. Returns a ``[n_t]`` threshold vector.
    """
    n_t = free_scores.shape[1]
    q = 100.0 * (1.0 - target_far)
    global_thr = float(np.nanpercentile(free_scores, q))
    if hours is None or far_bins <= 1:
        return np.full(n_t, global_thr)
    bins = np.floor((hours % 24) / (24.0 / far_bins)).astype(int) % far_bins
    thr_by_bin = np.full(far_bins, global_thr)
    for b in range(far_bins):
        cols = np.flatnonzero(bins == b)
        if cols.size == 0:
            continue
        pool = free_scores[:, cols]
        if np.isfinite(pool).sum() >= 100:      # enough samples for a stable percentile
            thr_by_bin[b] = float(np.nanpercentile(pool, q))
    return thr_by_bin[bins]


def _first_persistent(exceed: np.ndarray, k: int) -> int | None:
    """Index of the end of the first run of ``k`` consecutive True values, or None.

    A real fire drives a *sustained* residual excursion; an isolated frame over threshold
    is a false alarm. Requiring ``k`` consecutive exceedances before declaring a detection
    is how contextual detectors confirm across scans, and it is what stops a single pre-fire
    night blip from being scored as a huge early "lead". The end of the run is returned
    because that is when the alarm could actually be raised (you cannot confirm until the
    k-th frame). ``k <= 1`` is the plain first-exceedance behaviour.
    """
    if k <= 1:
        idx = np.flatnonzero(exceed)
        return int(idx[0]) if idx.size else None
    run = 0
    for i, v in enumerate(exceed):
        run = run + 1 if v else 0
        if run >= k:
            return i
    return None


def real_lead_experiment(
    residual2d: np.ndarray, fdc_first_idx: np.ndarray, target_far: float = 0.01,
    cadence_min: int = 5, min_valid_frac: float = 0.5,
    hours: np.ndarray | None = None, far_bins: int = 1, min_consec: int = 1,
    eval_start: int = 0,
) -> RealLeadResult:
    """On a real cube, how much earlier does the residual detector fire than GOES FDC?

    ``residual2d`` is ``[n_pixels, n_t]`` (a flattened cube's residual); ``fdc_first_idx``
    is the per-pixel first FDC-detection frame index (``-1`` where FDC never fired), from
    :func:`vhagar.archive.temporal_cube.fdc_first_detection_grid`, flattened to
    ``[n_pixels]``. The threshold is calibrated to ``target_far`` on the **fire-free**
    pixels (those FDC never flagged and that are mostly valid), the matched false-alarm
    rate that makes the comparison fair. Pass ``hours`` (per-frame UTC hour-of-day) and
    ``far_bins > 1`` to calibrate a separate threshold per time-of-day bin, so a night fire
    is not desensitised by daytime residual variance (see :func:`_per_frame_threshold`).
    ``min_consec`` requires that many consecutive exceedances before a detection is
    declared (see :func:`_first_persistent`).

    ``eval_start`` is the train/test split: residual detections are only counted from that
    frame onward. This is essential, the baseline was *fit* on the leading (clear) frames,
    so a "detection" there is scoring on training data and, because a fire pixel can differ
    from the fire-free calibration pixels for ordinary reasons (a warmer surface), it fires
    pre-ignition and manufactures a huge false lead. Set ``eval_start`` to the end of the
    baseline window so detection is measured only on held-out frames, the same period in
    which FDC first flags the fire. For each fire pixel the lead is ``(fdc_first_idx -
    detection) * cadence_min`` minutes; positive means the residual detector was earlier.
    Pure numpy.
    """
    fire = fdc_first_idx >= 0
    valid_frac = np.isfinite(residual2d).mean(axis=1)
    free = (~fire) & (valid_frac >= min_valid_frac)
    if not free.any():
        raise ValueError("no fire-free, mostly-valid pixels to calibrate the threshold on")

    thr_per_frame = _per_frame_threshold(residual2d[free], target_far, hours, far_bins)
    thr_report = float(np.median(thr_per_frame))

    n_total = int(fire.sum())
    leads: list[float] = []
    n_led = 0
    fire_idx = np.flatnonzero(fire)
    for p in fire_idx:
        row = np.nan_to_num(residual2d[p], nan=-np.inf)
        exceed = row > thr_per_frame
        if eval_start > 0:
            exceed[:eval_start] = False              # held-out period only; no scoring on the
            #                                          frames the baseline was fit on
        r_first = _first_persistent(exceed, min_consec)
        if r_first is None:
            continue                       # residual never persistently fired: a non-detection,
            #                                NOT a zero lead. It is excluded from the lead stats
            #                                and lowers the detection rate.
        lead = (int(fdc_first_idx[p]) - r_first) * cadence_min
        leads.append(float(lead))
        if lead > 0:
            n_led += 1
    n_det = len(leads)
    det_rate = float(n_det / n_total) if n_total else 0.0
    if n_det == 0:
        return RealLeadResult(
            n_fire_pixels=0, n_fire_pixels_total=n_total, detection_rate=0.0,
            n_residual_led=0, frac_residual_led=0.0, median_lead_min=float("nan"),
            p25_lead_min=float("nan"), p75_lead_min=float("nan"),
            target_far=target_far, residual_threshold_k=thr_report,
        )
    a = np.array(leads)
    return RealLeadResult(
        n_fire_pixels=n_det,
        n_fire_pixels_total=n_total,
        detection_rate=det_rate,
        n_residual_led=n_led,
        frac_residual_led=float(n_led / n_det),
        median_lead_min=float(np.median(a)),
        p25_lead_min=float(np.percentile(a, 25)),
        p75_lead_min=float(np.percentile(a, 75)),
        target_far=target_far,
        residual_threshold_k=thr_report,
        leads_min=tuple(float(x) for x in leads),
    )


def cohort_lead_summary(
    per_fire: list[tuple[str, RealLeadResult]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-fire lead results into a per-stratum verdict.

    ``per_fire`` is a list of ``(stratum, RealLeadResult)`` at one FAR. For each stratum it
    reports, first and most importantly, the **detection rate** (fraction of fire pixels the
    residual detected at all in the held-out window), because a fire the residual never
    flags is a non-detection, not a zero-lead tie, and must not be hidden. Then, *among
    detected pixels*, the lead over FDC: a fire-level median (one vote per fire that detected
    anything) and a pooled per-pixel median (robust to a fire with a handful of pixels). A
    fire that detected nothing counts as not-led at the fire level. The cohort question a
    single fire cannot answer: does the detector lead FDC *where theory says it should*
    (night cold-starts) and not elsewhere (day)?
    """
    out: dict[str, dict[str, float]] = {}
    by_stratum: dict[str, list[RealLeadResult]] = {}
    for stratum, r in per_fire:
        by_stratum.setdefault(stratum, []).append(r)
    for stratum, results in by_stratum.items():
        # fire-level lead: NaN (non-detecting) fires count as not-led
        fire_leads = np.array([r.median_lead_min for r in results])
        detected_fire_leads = fire_leads[~np.isnan(fire_leads)]
        pooled = np.array([x for r in results for x in r.leads_min])
        n_det = sum(r.n_fire_pixels for r in results)
        n_tot = sum(r.n_fire_pixels_total for r in results)
        out[stratum] = {
            "n_fires": float(len(results)),
            "frac_fires_detected": float(np.mean(~np.isnan(fire_leads))),
            "detection_rate": float(n_det / n_tot) if n_tot else 0.0,
            "frac_fires_led": float(np.mean(np.nan_to_num(fire_leads, nan=-1.0) > 0)),
            "median_fire_lead_min": (float(np.median(detected_fire_leads))
                                     if detected_fire_leads.size else float("nan")),
            "pooled_pixel_median_lead_min": float(np.median(pooled)) if pooled.size else float("nan"),
            "pooled_pixel_frac_led": float(np.mean(pooled > 0)) if pooled.size else 0.0,
            "total_fire_pixels": float(n_tot),
        }
    return out


def _prep_cube_tensor(cube: np.ndarray, covariates: np.ndarray | None, bt_mean: float):
    """Mean-centre BT, fill NaN with 0, stack covariates -> torch ``[T, C, H, W]``.

    A real 3.9 um cube has NaN (cloud/fill/saturation); NaN would poison a conv and its
    gradient. Centring by the clear-sky mean and filling holes with 0 (the mean, after
    centring) gives the network a benign, in-distribution value where data is missing,
    while the finite mask is kept so the loss and the residual never score a filled pixel.
    """
    import torch

    bt_centred = np.nan_to_num(cube - bt_mean, nan=0.0).astype(np.float32)
    x = torch.from_numpy(bt_centred)[:, None]                        # [T,1,H,W]
    if covariates is not None:
        x = torch.cat([x, torch.from_numpy(covariates.astype(np.float32))], dim=1)
    return x


def train_temporal_net(
    cube: np.ndarray, covariates: np.ndarray | None = None, window: int = 6,
    epochs: int = 10, lr: float = 1e-3, seed: int = 0, device: str | None = None,
    bt_mean: float | None = None,
):
    """Train ``TemporalAnomalyNet`` to forecast the current BT frame from a window of past
    frames (+ optional covariates), on clear-sky history. Needs torch.

    ``cube`` is ``[T, H, W]`` brightness temperature; ``covariates`` is optional
    ``[T, C-1, H, W]`` exogenous channels (e.g. cosine of solar zenith). BT is mean-centred
    (by ``bt_mean`` or the cube's own nanmean) and NaN-filled; the forecasting loss is
    **masked to the finite target pixels**, so cloud/fill never trains the net. Returns the
    trained model; its residuals at inference are the anomaly score for the same matched-FAR
    / persistence / lead-time protocol as the numpy path.
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
    mu = float(np.nanmean(cube)) if bt_mean is None else float(bt_mean)
    T = cube.shape[0]
    x = _prep_cube_tensor(cube, covariates, mu).to(dev)
    finite = torch.from_numpy(np.isfinite(cube)).to(dev)             # [T,H,W]
    model = TemporalAnomalyNet(in_channels=x.shape[1], window=window).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for t in range(window, T):
            win = x[t - window:t][None]                              # [1,window,C,H,W]
            pred = model(win)[0, 0]                                  # [H,W] centred forecast
            m = finite[t]
            if not bool(m.any()):
                continue
            loss = F.smooth_l1_loss(pred[m], x[t, 0][m])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def temporal_net_residuals(
    cube: np.ndarray, model, window: int, covariates: np.ndarray | None = None,
    device: str | None = None, bt_mean: float | None = None,
) -> np.ndarray:
    """Per-pixel forecast residual ``BT - expected`` from a trained ``TemporalAnomalyNet``.

    Returns ``[n_pixels, T]`` (flattened grid), with the first ``window`` frames and any
    originally-NaN pixel left NaN so the lead-time protocol never scores a value the network
    filled. The residual is in kelvin (the centring cancels in the difference). Needs torch.
    """
    import torch

    mu = float(np.nanmean(cube)) if bt_mean is None else float(bt_mean)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    T, H, W = cube.shape
    x = _prep_cube_tensor(cube, covariates, mu).to(dev)
    model.eval()
    resid = np.full((T, H, W), np.nan, dtype=np.float32)
    with torch.no_grad():
        for t in range(window, T):
            pred = model(x[t - window:t][None])[0, 0].cpu().numpy()  # [H,W] centred forecast
            actual_centred = cube[t] - mu
            resid[t] = np.where(np.isfinite(cube[t]), actual_centred - pred, np.nan)
    return resid.reshape(T, H * W).T


def learned_residuals(
    cube: np.ndarray, clear_end: int, window: int = 6, epochs: int = 10, lr: float = 1e-3,
    covariates: np.ndarray | None = None, seed: int = 0, device: str | None = None,
) -> np.ndarray:
    """Train ``TemporalAnomalyNet`` on the leading clear-sky span, then return residuals
    over the whole cube. One call for the learned path; feeds ``real_lead_experiment``.

    ``clear_end`` is the frame index up to which the record is fire-free (the training
    span). The clear-sky mean is computed once and used to centre both training and
    inference, so the two walk the same normalisation. Needs torch.
    """
    mu = float(np.nanmean(cube[:clear_end]))
    cov_train = covariates[:clear_end] if covariates is not None else None
    model = train_temporal_net(
        cube[:clear_end], covariates=cov_train, window=window, epochs=epochs, lr=lr,
        seed=seed, device=device, bt_mean=mu,
    )
    return temporal_net_residuals(
        cube, model, window, covariates=covariates, device=device, bt_mean=mu
    )
