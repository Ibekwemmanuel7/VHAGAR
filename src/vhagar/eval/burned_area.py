"""T3 expected burned area: E[BA] = P(ignition) x E[BA | ignition].

The third of the three quantities the architecture refuses to collapse into one
"risk" number (`docs/00` sections 5.1, 5.4). The conditional distribution of
burned area given an ignition is heavy-tailed: a handful of wind-driven events
dominate the total, so squared-error training and RMSE evaluation are captured
by those extremes and are unstable fold to fold. The doc's prescription, applied
here:

* **Fit the distribution, not the mean.** Log-space quantile gradient boosting
  (the state of the art per ECMWF's fire work) gives a full predictive set of
  quantiles per cell.
* **A GPD tail for the extremes.** A peaks-over-threshold generalised-Pareto fit
  extrapolates the far quantiles (q99+) that the boosting cannot see enough of.
* **Score with CRPS and pinball, never RMSE.** CRPS via its quantile
  decomposition is proper and stable under the tail; RMSE is reported only to
  *show* its instability, not to select a model.

`lon`/`lat` are excluded from the model (the T1 leakage lesson); they only build
the 5-degree spatial blocks. `synthetic_burned_area_scenario` builds a
heavy-tailed world so the behaviour can be shown, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vhagar.eval.metrics import crps_from_quantiles, pinball_loss, skill_score

__all__ = [
    "DEFAULT_TAUS",
    "BurnedAreaModel",
    "synthetic_burned_area_scenario",
    "expected_burned_area",
    "evaluate_expected_ba",
]

DEFAULT_TAUS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def _hgbr_quantile(tau: float, seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="quantile", quantile=tau, max_depth=3, max_iter=200,
        learning_rate=0.06, l2_regularization=1.0, random_state=seed,
    )


@dataclass(slots=True)
class BurnedAreaModel:
    """Log-space quantile-boosting distribution of burned area, with a GPD tail.

    Fit on ``log1p(area_ha)``; predictions are returned in hectares. Quantiles
    are sorted per row so the predictive set is monotone. When ``use_tail`` the
    quantiles above ``tail_from`` are replaced by a peaks-over-threshold
    generalised-Pareto extrapolation fit on the training exceedances.
    """

    taus: tuple[float, ...] = DEFAULT_TAUS
    seed: int = 0
    use_tail: bool = True
    tail_from: float = 0.9
    _models: dict = None            # tau -> fitted regressor (log space)
    _gpd: tuple | None = None       # (shape c, scale) of the log-residual tail

    def fit(self, X, area):
        logy = np.log1p(np.asarray(area, dtype=np.float64))
        self._models = {t: _hgbr_quantile(t, self.seed).fit(X, logy) for t in self.taus}
        if self.use_tail and self.tail_from in self._models:
            from scipy.stats import genpareto

            thr = self._models[self.tail_from].predict(X)
            exceed = logy - thr
            exceed = exceed[exceed > 0]
            if exceed.size >= 30:
                try:
                    c, _loc, scale = genpareto.fit(exceed, floc=0.0)
                    if np.isfinite(c) and np.isfinite(scale) and scale > 0:
                        self._gpd = (float(c), float(scale))
                except Exception:
                    self._gpd = None
        return self

    def predict_quantiles(self, X):
        """Return ``(n, len(taus))`` burned-area quantiles in hectares."""
        from scipy.stats import genpareto

        log_q = np.column_stack([self._models[t].predict(X) for t in self.taus])
        if self._gpd is not None:
            c, scale = self._gpd
            j = self.taus.index(self.tail_from)
            thr = log_q[:, j]
            for k, t in enumerate(self.taus):
                if t > self.tail_from:
                    frac = (t - self.tail_from) / (1.0 - self.tail_from)
                    log_q[:, k] = thr + genpareto.ppf(frac, c, loc=0.0, scale=scale)
        log_q = np.sort(log_q, axis=1)              # enforce monotone quantiles
        return np.expm1(log_q)


def expected_burned_area(p_ignition, quantile_preds, taus=DEFAULT_TAUS):
    """E[BA] = P(ignition) x E[BA | ignition].

    The conditional mean is approximated from the predictive quantiles by
    trapezoidal integration over probability, which is robust to the tail.
    """
    taus = np.asarray(taus, dtype=np.float64)
    q = np.asarray(quantile_preds, dtype=np.float64)
    grid = np.concatenate([[0.0], taus, [1.0]])
    qext = np.column_stack([q[:, :1], q, q[:, -1:]])
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # noqa: NPY201
    mean_cond = trap(qext, grid, axis=1)            # E[BA | ignition] per row
    return np.asarray(p_ignition, dtype=np.float64).ravel() * mean_cond


def synthetic_burned_area_scenario(rng, n=4000, years=(2019, 2020, 2021, 2022)):
    """A heavy-tailed conditional burned-area world (one row per ignition).

    log burned area is driven by dryness / fuel / wind / slope, plus a rare
    wind-driven "blow-up" multiplier that produces the long right tail where RMSE
    breaks down. Returns ``(X, area_ha, year, lon, lat, feature_names)``.
    """
    fn = ["dryness", "fuel", "wind", "slope"]
    per_year = max(n // len(years), 1)
    X, area, yr, lon, lat = [], [], [], [], []
    for y in years:
        dryness = rng.beta(2, 2, per_year)
        fuel = rng.beta(2, 2, per_year)
        wind = rng.beta(2, 3, per_year)
        slope = rng.beta(2, 4, per_year)
        loga = (3.0 + 1.6 * dryness + 1.4 * fuel + 1.5 * wind + 0.5 * slope
                + rng.normal(0, 0.5, per_year))
        # rare wind-driven blow-ups: a heavy right tail, capped at a realistic
        # megafire size so the extremes are large but not absurd.
        blow = (rng.random(per_year) < 0.04 * (0.4 + wind))
        loga = loga + blow * rng.gamma(shape=2.0, scale=0.9, size=per_year)
        a = np.clip(np.expm1(loga), 1.0, 300_000.0)   # hectares
        X.append(np.column_stack([dryness, fuel, wind, slope]))
        area.append(a)
        yr.append(np.full(per_year, y))
        lon.append(-120.0 + 12.0 * rng.random(per_year))
        lat.append(34.0 + 12.0 * rng.random(per_year))
    return (np.vstack(X), np.concatenate(area), np.concatenate(yr),
            np.concatenate(lon), np.concatenate(lat), fn)


def _blocks(lon, lat, deg=5.0):
    return (np.floor(np.asarray(lon) / deg).astype(np.int64) * 100_000
            + np.floor(np.asarray(lat) / deg).astype(np.int64))


def evaluate_expected_ba(X, area, lon, lat, taus=DEFAULT_TAUS, n_folds: int = 4,
                         seed: int = 0, use_tail: bool = True) -> dict:
    """Blocked-CV evaluation of the conditional burned-area distribution.

    Returns CRPS and per-tau pinball for the model and for a climatology
    reference (constant training quantiles), the CRPS skill score, and RMSE mean
    and standard deviation across folds to expose its tail-driven instability.
    """
    from sklearn.model_selection import GroupKFold

    X = np.asarray(X, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    groups = _blocks(lon, lat)
    folds = int(min(n_folds, len(np.unique(groups))))
    if folds < 2:
        raise ValueError("need >= 2 spatial blocks")

    taus = tuple(taus)
    n = area.shape[0]
    oof = np.full((n, len(taus)), np.nan)
    oof_climo = np.full((n, len(taus)), np.nan)
    rmse_folds, rmse_climo_folds = [], []

    for tr, te in GroupKFold(n_splits=folds).split(X, area, groups):
        model = BurnedAreaModel(taus=taus, seed=seed, use_tail=use_tail).fit(X[tr], area[tr])
        oof[te] = model.predict_quantiles(X[te])
        # climatology: constant training quantiles for every held-out row
        cq = np.quantile(area[tr], taus)
        oof_climo[te] = np.tile(cq, (te.size, 1))
        # RMSE per fold on the median prediction (the natural point forecast)
        med = model.predict_quantiles(X[te])[:, taus.index(0.5)]
        rmse_folds.append(float(np.sqrt(np.mean((area[te] - med) ** 2))))
        rmse_climo_folds.append(float(np.sqrt(np.mean((area[te] - np.median(area[tr])) ** 2))))

    ok = ~np.isnan(oof[:, 0])
    crps_model = crps_from_quantiles(area[ok], oof[ok], taus)
    crps_climo = crps_from_quantiles(area[ok], oof_climo[ok], taus)
    return {
        "crps": crps_model,
        "crps_climatology": crps_climo,
        "crps_skill_vs_climatology": skill_score(crps_model, crps_climo, perfect=0.0),
        "pinball": {float(t): pinball_loss(area[ok], oof[ok][:, k], float(t))
                    for k, t in enumerate(taus)},
        "rmse_mean": float(np.mean(rmse_folds)),
        "rmse_std": float(np.std(rmse_folds)),
        "crps_note": "CRPS is proper and stable; RMSE std across folds shows the tail instability",
        "n": int(ok.sum()),
        "tail": use_tail,
    }
