"""Olofsson good-practice area estimation with confidence intervals.

Map pixel counts are **biased** estimators of area, because classifiers make
asymmetric commission and omission errors. Reporting "the model mapped
143,200 ha burned" is a methodological error, not a rounding issue: burned area
is typically <1% of the landscape, so even a 1% commission rate on the
unburned class can double the apparent burned area.

VHAGAR reports burned area only as an error-adjusted estimate with a 95%
confidence interval, computed here following Olofsson et al. (2013, 2014).

Estimator (stratified random sampling, map classes as strata)
-------------------------------------------------------------
    W_i          stratum weight = mapped area proportion of stratum i
    n_i          reference samples drawn in stratum i
    n_ik         samples mapped i and referenced k

    p_hat_ik  = W_i * n_ik / n_i
    p_hat_k   = sum_i p_hat_ik                       (error-adjusted proportion)
    A_hat_k   = A_total * p_hat_k                    (error-adjusted area)
    S(p_k)    = sqrt( sum_i W_i^2 * (n_ik/n_i)(1 - n_ik/n_i) / (n_i - 1) )
    CI_95     = A_hat_k +/- 1.96 * A_total * S(p_k)

User's accuracy (stratum i) = p_hat_ii / sum_k p_hat_ik
Producer's accuracy (class k) = p_hat_kk / p_hat_k
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AreaEstimate",
    "estimate_areas",
    "sample_size_stratified",
    "allocate_samples",
]


@dataclass(frozen=True, slots=True)
class AreaEstimate:
    """Error-adjusted area for one class."""

    class_index: int
    class_name: str
    mapped_area: float
    adjusted_area: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    users_accuracy: float
    producers_accuracy: float

    @property
    def margin_of_error(self) -> float:
        return 1.96 * self.standard_error

    @property
    def relative_margin(self) -> float:
        return self.margin_of_error / self.adjusted_area if self.adjusted_area else float("nan")

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"{self.class_name}: {self.adjusted_area:,.0f} "
            f"+/- {self.margin_of_error:,.0f} (95% CI) "
            f"[mapped {self.mapped_area:,.0f}]  "
            f"UA={self.users_accuracy:.3f} PA={self.producers_accuracy:.3f}"
        )


def estimate_areas(
    confusion: np.ndarray,
    mapped_areas: np.ndarray,
    class_names: list[str] | None = None,
) -> list[AreaEstimate]:
    """Error-adjusted areas from a stratified reference sample.

    Parameters
    ----------
    confusion
        ``(n_classes, n_classes)`` integer counts. **Rows are map classes
        (the strata), columns are reference classes.** Row sums are the number
        of reference samples drawn per stratum.
    mapped_areas
        ``(n_classes,)`` mapped area per class, in any consistent unit
        (hectares, km^2, pixel counts). Output is in the same unit.
    class_names
        Optional labels.

    Returns
    -------
    list[AreaEstimate]

    Notes
    -----
    Strata with fewer than 2 reference samples cannot yield a variance
    estimate and raise. Good practice is 50-100 samples in each rare stratum.

    >>> conf = np.array([[97, 3], [10, 90]])
    >>> est = estimate_areas(conf, np.array([200_000.0, 20_000.0]))
    >>> round(est[1].adjusted_area) > 0
    True
    """
    conf = np.asarray(confusion, dtype=np.float64)
    if conf.ndim != 2 or conf.shape[0] != conf.shape[1]:
        raise ValueError("confusion must be a square (n_classes, n_classes) matrix")
    areas = np.asarray(mapped_areas, dtype=np.float64)
    n_classes = conf.shape[0]
    if areas.shape != (n_classes,):
        raise ValueError(f"mapped_areas must have shape ({n_classes},)")
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    n_i = conf.sum(axis=1)
    if np.any(n_i < 2):
        bad = [class_names[i] for i in np.where(n_i < 2)[0]]
        raise ValueError(
            f"strata {bad} have <2 reference samples; variance is undefined. "
            "Good practice is 50-100 samples in each rare stratum."
        )

    a_total = float(areas.sum())
    w_i = areas / a_total

    # p_hat[i, k] = W_i * n_ik / n_i
    p_hat = w_i[:, None] * (conf / n_i[:, None])
    p_k = p_hat.sum(axis=0)

    # Variance of the class-k proportion estimate.
    prop = conf / n_i[:, None]
    var_k = np.sum((w_i**2)[:, None] * prop * (1.0 - prop) / (n_i[:, None] - 1.0), axis=0)
    se_k = np.sqrt(var_k)

    row_sums = p_hat.sum(axis=1)
    out: list[AreaEstimate] = []
    for k in range(n_classes):
        adjusted = a_total * float(p_k[k])
        se_area = a_total * float(se_k[k])
        ua = float(p_hat[k, k] / row_sums[k]) if row_sums[k] > 0 else float("nan")
        pa = float(p_hat[k, k] / p_k[k]) if p_k[k] > 0 else float("nan")
        out.append(
            AreaEstimate(
                class_index=k,
                class_name=class_names[k],
                mapped_area=float(areas[k]),
                adjusted_area=adjusted,
                standard_error=se_area,
                ci95_low=adjusted - 1.96 * se_area,
                ci95_high=adjusted + 1.96 * se_area,
                users_accuracy=ua,
                producers_accuracy=pa,
            )
        )
    return out


def sample_size_stratified(
    stratum_weights: np.ndarray,
    target_users_accuracy: np.ndarray,
    target_se_overall: float = 0.01,
) -> int:
    """Cochran's stratified sample size for a target overall-accuracy SE.

    ``n ~= [ sum_i W_i * S_i ]^2 / S(O)^2``  with ``S_i = sqrt(U_i (1 - U_i))``.
    """
    w = np.asarray(stratum_weights, dtype=np.float64)
    u = np.asarray(target_users_accuracy, dtype=np.float64)
    if w.shape != u.shape:
        raise ValueError("stratum_weights and target_users_accuracy must have the same shape")
    s_i = np.sqrt(np.clip(u * (1.0 - u), 0.0, None))
    n = (np.sum(w * s_i) / target_se_overall) ** 2
    return int(np.ceil(n))


def allocate_samples(
    n_total: int,
    stratum_weights: np.ndarray,
    rare_classes: list[int],
    min_per_rare: int = 75,
) -> np.ndarray:
    """Good-practice allocation: floor the rare strata, distribute the rest.

    Burned area is a rare class. Proportional allocation would put almost no
    samples in it, and the burned-class confidence interval would be useless.
    """
    w = np.asarray(stratum_weights, dtype=np.float64)
    n = np.zeros(w.shape, dtype=int)
    for k in rare_classes:
        n[k] = min_per_rare
    remaining = n_total - int(n.sum())
    if remaining < 0:
        raise ValueError(
            f"n_total={n_total} is below the rare-class floor "
            f"({len(rare_classes)} x {min_per_rare}); increase n_total"
        )
    common = [i for i in range(len(w)) if i not in set(rare_classes)]
    if common:
        w_common = w[common] / w[common].sum()
        alloc = np.floor(w_common * remaining).astype(int)
        alloc[int(np.argmax(w_common))] += remaining - int(alloc.sum())
        for i, c in enumerate(common):
            n[c] = alloc[i]
    return n
