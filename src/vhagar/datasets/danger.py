"""T3 ignition sampling design: the trap that ruins most ignition models.

Ignition databases are *presence-only* with strong reporting bias: the chance a
fire is recorded rises with population density and road proximity, which are the
very covariates an ignition model wants to use. A naive model with randomly
drawn pseudo-absences learns *where people report fires*, not where fires start,
and reports probabilities calibrated to a fictional base rate.

This module implements the four mitigations the architecture requires
(``docs/00`` section 5.6), each as a pure, testable function:

1. **Target-group background sampling.** Draw pseudo-absences from the same
   observation process that produced the presences (the pool of *all* reported
   events, across causes), so detection bias is shared by positives and
   negatives and cancels rather than being learned.
2. **Negative stratification.** Match the pseudo-absence land-cover / stratum
   distribution to the positives, so the model cannot separate classes on a
   land-cover prior that is an artefact of where the two samples were drawn.
3. **Rare-event prior correction.** If negatives are down-sampled to balance the
   classes, the King and Zeng (2001) intercept correction maps the model's
   probabilities from the down-sampled base rate back to the true one. Without
   it the reported probabilities are calibrated to a base rate that does not
   exist operationally.
4. **Cause stratification.** Human- and lightning-caused ignitions have nearly
   orthogonal covariate structure; they are carried as a label so the model can
   be fit and evaluated per cause.

Nothing here fetches data. :func:`assemble_ignition_samples` takes presence and
candidate records you supply; :func:`synthetic_reporting_scenario` builds a
biased presence-only world for tests and the CLI demo, so the sampling design's
effect can be shown, not just asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "IgnitionSamples",
    "rare_event_correction",
    "target_group_background",
    "stratify_negatives",
    "assemble_ignition_samples",
    "synthetic_reporting_scenario",
    "reporting_weight",
    "frames_to_records",
]

_EPS = 1e-12


def rare_event_correction(p, tau: float, ybar: float):
    """King-Zeng prior correction: probabilities from a down-sampled fit -> true base rate.

    A classifier trained on all positives plus a down-sample of negatives sees a
    positive fraction ``ybar`` far above the true base rate ``tau``. Its
    probabilities are calibrated to ``ybar``. Shifting the intercept by the log
    prior-odds ratio maps them back to ``tau``:

        odds_corrected = odds(p) * (tau / (1 - tau)) * ((1 - ybar) / ybar)

    >>> float(round(rare_event_correction(0.5, tau=0.01, ybar=0.5), 4))
    0.0101
    >>> # a balanced-trained 0.5 becomes ~tau; the ordering is preserved
    >>> rare_event_correction(0.9, 0.01, 0.5) > rare_event_correction(0.5, 0.01, 0.5)
    True
    """
    if not (0.0 < tau < 1.0) or not (0.0 < ybar < 1.0):
        raise ValueError("tau and ybar must be in (0, 1)")
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    factor = (tau / (1.0 - tau)) * ((1.0 - ybar) / ybar)
    odds = (p / (1.0 - p)) * factor
    return odds / (1.0 + odds)


def target_group_background(
    presence_ids,
    pool_ids,
    n: int,
    rng: np.random.Generator,
    pool_weight=None,
):
    """Sample ``n`` pseudo-absence cell-day ids from the observation pool.

    ``pool_ids`` is the set of cell-days the reporting process actually observed
    (e.g. every reported event of *any* cause, or every satellite active-fire
    cell-day). Backgrounds are drawn from it, excluding presences, optionally
    weighted by ``pool_weight`` (reporting intensity). Because backgrounds and
    presences come from the same process, a covariate that only predicts
    *observation* is equally present in both and cannot be learned as ignition
    signal.
    """
    pool = np.asarray(pool_ids)
    presence = set(np.asarray(presence_ids).tolist())
    keep = np.array([pid not in presence for pid in pool], dtype=bool)
    cand = pool[keep]
    if cand.size == 0:
        return np.array([], dtype=pool.dtype)
    w = None
    if pool_weight is not None:
        w = np.asarray(pool_weight, dtype=np.float64)[keep]
        s = w.sum()
        w = None if s <= 0 else w / s
    take = int(min(n, cand.size))
    return rng.choice(cand, size=take, replace=False, p=w)


def stratify_negatives(
    pos_strata,
    cand_ids,
    cand_strata,
    rng: np.random.Generator,
    ratio: float = 1.0,
    cand_weight=None,
):
    """Sample negatives so their stratum distribution matches the positives'.

    For each stratum, draw ``ratio x (positives in that stratum)`` candidates
    (capped by availability). This removes the land-cover prior that separates a
    presence sample from a naively drawn background, forcing the model onto
    weather and fuel-state signal instead of "fires are reported in this cover
    type".

    ``cand_weight`` (optional, per candidate) is reporting intensity: within each
    stratum the draw is weighted by it, so backgrounds are pulled from where the
    reporting process is active, exactly as :func:`target_group_background` does.
    Without it, or where a stratum's weights are degenerate, the draw is uniform.
    """
    pos_strata = np.asarray(pos_strata)
    cand_ids = np.asarray(cand_ids)
    cand_strata = np.asarray(cand_strata)
    weight = None if cand_weight is None else np.asarray(cand_weight, dtype=np.float64)
    want: dict = {}
    labels, counts = np.unique(pos_strata, return_counts=True)
    for lab, c in zip(labels, counts, strict=True):
        want[lab] = int(round(ratio * c))
    chosen: list = []
    for lab, target in want.items():
        mask = cand_strata == lab
        pool = cand_ids[mask]
        if pool.size == 0 or target <= 0:
            continue
        take = int(min(target, pool.size))
        p = None
        if weight is not None:
            w = weight[mask]
            s = w.sum()
            if np.isfinite(s) and s > 0:
                p = w / s                            # reporting-intensity weighted draw
        chosen.append(rng.choice(pool, size=take, replace=False, p=p))
    if not chosen:
        return np.array([], dtype=cand_ids.dtype)
    return np.concatenate(chosen)


@dataclass(slots=True)
class IgnitionSamples:
    """A cause-labelled ignition design matrix, ready for blocked evaluation.

    ``lon``/``lat`` are kept OUT of ``X`` on purpose (the T1 lesson: raw
    coordinates supply most of the gain in-region while destroying transfer).
    They are retained only to build spatial-block groups. ``tau`` is the true
    population base rate; ``ybar`` is the positive fraction in this (possibly
    down-sampled) design, so :func:`rare_event_correction` can recalibrate.
    """

    X: np.ndarray
    y: np.ndarray
    cause: np.ndarray            # "human" | "lightning" | "" (background)
    lon: np.ndarray
    lat: np.ndarray
    year: np.ndarray
    feature_names: list[str]
    tau: float
    ybar: float
    block_degrees: float = 5.0
    meta: dict = field(default_factory=dict)

    def block_group(self) -> np.ndarray:
        """Spatial-block id per row for GroupKFold / spatial-block CV."""
        bx = np.floor(self.lon / self.block_degrees).astype(np.int64)
        by = np.floor(self.lat / self.block_degrees).astype(np.int64)
        return bx * 100_000 + by

    def cause_mask(self, cause: str) -> np.ndarray:
        """Rows for one cause head: that cause's positives plus all backgrounds."""
        return (self.cause == cause) | (self.y == 0)


def assemble_ignition_samples(
    presence,
    candidates,
    feature_names,
    rng: np.random.Generator,
    tau: float,
    neg_per_pos: float = 3.0,
    use_target_group: bool = True,
    stratify: bool = True,
    block_degrees: float = 5.0,
):
    """Build an :class:`IgnitionSamples` from presence + candidate records.

    ``presence`` and ``candidates`` are dicts of equal-length arrays with keys:
    ``id``, ``lon``, ``lat``, ``year``, ``stratum``, ``weight`` (candidates
    only, reporting intensity), ``cause`` (presence only), and one entry per
    ``feature_names``. ``use_target_group`` / ``stratify`` toggle the two
    background mitigations so a caller can measure their effect.
    """
    pres_id = np.asarray(presence["id"])
    n_pos = pres_id.size
    n_neg = int(round(neg_per_pos * n_pos))

    cand_id = np.asarray(candidates["id"])
    cand_stratum = np.asarray(candidates.get("stratum", np.zeros(cand_id.size)))
    weight = candidates.get("weight")

    # A presence (a real ignition) must never be drawn as a label-0 background.
    # Exclude presence ids from the sampling pool in EVERY branch; the row-index
    # map below is still built over the full candidate set, so ids map correctly.
    avail = ~np.isin(cand_id, pres_id)
    pool_id = cand_id[avail]
    pool_stratum = cand_stratum[avail]
    pool_weight = None if weight is None else np.asarray(weight)[avail]

    if stratify:
        neg_ids = stratify_negatives(
            np.asarray(presence.get("stratum", np.zeros(n_pos))),
            pool_id, pool_stratum, rng, ratio=neg_per_pos, cand_weight=pool_weight,
        )
    elif use_target_group:
        neg_ids = target_group_background(pres_id, pool_id, n_neg, rng, pool_weight=pool_weight)
    else:
        # Naive random background: ignores the observation process entirely.
        neg_ids = rng.choice(pool_id, size=int(min(n_neg, pool_id.size)), replace=False)

    cand_pos = {cid: k for k, cid in enumerate(cand_id)}
    neg_rows = np.array([cand_pos[i] for i in neg_ids], dtype=np.int64)

    def stack(src, rows, keys):
        return np.column_stack([np.asarray(src[k])[rows] for k in keys])

    Xp = np.column_stack([np.asarray(presence[k]) for k in feature_names])
    Xn = stack(candidates, neg_rows, feature_names)
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(n_pos, int), np.zeros(neg_rows.size, int)])
    cause = np.concatenate([np.asarray(presence["cause"]).astype(object),
                            np.full(neg_rows.size, "", dtype=object)])
    lon = np.concatenate([np.asarray(presence["lon"]), np.asarray(candidates["lon"])[neg_rows]])
    lat = np.concatenate([np.asarray(presence["lat"]), np.asarray(candidates["lat"])[neg_rows]])
    year = np.concatenate([np.asarray(presence["year"]), np.asarray(candidates["year"])[neg_rows]])
    ybar = float(n_pos / max(n_pos + neg_rows.size, 1))
    return IgnitionSamples(
        X=X.astype(np.float64), y=y, cause=cause, lon=lon, lat=lat, year=year,
        feature_names=list(feature_names), tau=float(tau), ybar=ybar,
        block_degrees=block_degrees,
        meta={"n_pos": n_pos, "n_neg": int(neg_rows.size),
              "use_target_group": use_target_group, "stratify": stratify},
    )


def reporting_weight(cand_lon, cand_lat, pres_lon, pres_lat,
                     bandwidth_deg: float = 0.25, floor: float = 1e-3):
    """Target-group reporting intensity per candidate cell, from occurrence density.

    When you do not have an explicit reporting-effort surface, the density of
    *reported* ignitions is itself an estimate of where the reporting process is
    active. Each candidate cell is weighted by how many presences fall in its
    ``bandwidth_deg`` neighbourhood, so backgrounds are drawn preferentially from
    the same observed footprint as the presences. Returns a normalised weight.
    """
    from collections import Counter

    def cell(lo, la):
        return (np.floor(np.asarray(lo, float) / bandwidth_deg).astype(np.int64),
                np.floor(np.asarray(la, float) / bandwidth_deg).astype(np.int64))

    px, py = cell(pres_lon, pres_lat)
    counts = Counter(zip(px.tolist(), py.tolist(), strict=True))
    cx, cy = cell(cand_lon, cand_lat)
    w = np.array([counts.get((x, y), 0) for x, y in zip(cx.tolist(), cy.tolist(), strict=True)],
                 dtype=np.float64) + floor
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / w.size)


def frames_to_records(presence_df, candidate_df, feature_cols, *,
                      id_col: str = "id", lon_col: str = "lon", lat_col: str = "lat",
                      year_col: str = "year", stratum_col: str = "stratum",
                      cause_col: str = "cause", weight_col: str = "weight",
                      bandwidth_deg: float = 0.25):
    """Turn two dataframes into the presence/candidate records ``assemble_ignition_samples``
    expects.

    ``presence_df`` is one row per reported ignition cell-day; ``candidate_df`` is
    the target-group pool of observable cell-days (e.g. every cell-day in the
    region-period, or every cell that ever reported a fire). Both must carry the
    ``feature_cols`` covariates and lon/lat/year; ``presence_df`` also carries a
    cause. A missing candidate ``weight`` column is filled from
    :func:`reporting_weight` (occurrence density), giving target-group sampling
    for free. ``stratum`` defaults to a single stratum if absent.
    """
    def base(df):
        d = {"id": np.asarray(df[id_col]),
             "lon": np.asarray(df[lon_col], dtype=np.float64),
             "lat": np.asarray(df[lat_col], dtype=np.float64),
             "year": np.asarray(df[year_col])}
        d["stratum"] = (np.asarray(df[stratum_col]) if stratum_col in df.columns
                        else np.zeros(len(df), dtype=np.int64))
        for c in feature_cols:
            d[c] = np.asarray(df[c], dtype=np.float64)
        return d

    presence = base(presence_df)
    presence["cause"] = (np.asarray(presence_df[cause_col]).astype(object)
                         if cause_col in presence_df.columns
                         else np.full(len(presence_df), "unknown", dtype=object))
    candidates = base(candidate_df)
    if weight_col in candidate_df.columns:
        candidates["weight"] = np.asarray(candidate_df[weight_col], dtype=np.float64)
    else:
        candidates["weight"] = reporting_weight(
            candidates["lon"], candidates["lat"], presence["lon"], presence["lat"],
            bandwidth_deg=bandwidth_deg)
    return presence, candidates, list(feature_cols)


def synthetic_reporting_scenario(
    rng: np.random.Generator,
    n_cells: int = 4000,
    years=(2019, 2020, 2021, 2022),
    tau: float = 0.02,
):
    """A biased presence-only world for tests and the demo.

    True ignition depends on weather/fuel-state (``dryness``, ``fuel``,
    ``wind``) and, weakly, on ``people`` (human-caused). But *reporting* depends
    strongly on ``people`` and ``roads``: a real ignition in the backcountry is
    often never recorded. So the observed presences over-represent populated
    cells. A model with random background sees ``people``/``roads`` as the top
    ignition predictor (the artefact); target-group background + stratification
    should suppress that and recover the weather/fuel signal.

    Returns ``(presence, candidates, feature_names, true_tau)``.
    """
    fn = ["dryness", "fuel", "wind", "people", "roads"]
    rows = []
    cid = 0
    for yr in years:
        for _ in range(n_cells):
            dryness = rng.beta(2, 2)
            fuel = rng.beta(2, 2)
            wind = rng.beta(2, 3)
            people = rng.beta(1.3, 3.0)    # human footprint, moderately spread
            roads = np.clip(people + rng.normal(0, 0.12), 0, 1)
            # True ignition depends on weather + fuel ONLY, and modestly (noisy,
            # partly driven by latent factors the features do not capture). People
            # does NOT cause ignition here.
            lin = 1.7 * dryness + 1.3 * fuel + 0.5 * wind - 2.5
            p_true = 1.0 / (1.0 + np.exp(-lin))
            ignited = rng.random() < p_true * (tau / 0.05)   # scale toward base rate
            cause = "lightning" if rng.random() < 0.5 + 0.3 * (0.5 - people) else "human"
            # Reporting depends STRONGLY on the human footprint: a backcountry
            # ignition is usually never recorded. This is the confounder.
            p_report = 1.0 / (1.0 + np.exp(-(4.6 * people + 3.4 * roads - 2.4)))
            reported = ignited and (rng.random() < p_report)
            rows.append((cid, yr, dryness, fuel, wind, people, roads, ignited, reported, cause))
            cid += 1
    arr = rows
    ids = np.array([r[0] for r in arr])
    yrs = np.array([r[1] for r in arr])
    feats = {fn[k]: np.array([r[2 + k] for r in arr], dtype=np.float64) for k in range(5)}
    lon = -120.0 + 12.0 * feats["roads"] + rng.normal(0, 0.4, ids.size)   # geography ~ human footprint
    lat = 34.0 + 12.0 * feats["people"] + rng.normal(0, 0.4, ids.size)
    stratum = np.digitize(feats["fuel"], [0.33, 0.66])                    # 3 land-cover-like strata
    reported = np.array([r[8] for r in arr])
    ignited = np.array([r[7] for r in arr])
    cause = np.array([r[9] for r in arr], dtype=object)

    presence = {"id": ids[reported], "lon": lon[reported], "lat": lat[reported],
                "year": yrs[reported], "stratum": stratum[reported], "cause": cause[reported]}
    for k in fn:
        presence[k] = feats[k][reported]
    # Candidate pool = every observed cell-day (the target group), weighted by
    # reporting intensity so backgrounds share the presences' detection bias.
    weight = 1.0 / (1.0 + np.exp(-(2.8 * feats["people"] + 2.2 * feats["roads"] - 1.8)))
    candidates = {"id": ids, "lon": lon, "lat": lat, "year": yrs,
                  "stratum": stratum, "weight": weight}
    for k in fn:
        candidates[k] = feats[k]
    candidates["_ignited"] = ignited      # truth, for scoring the recovered signal
    return presence, candidates, fn, float(ignited.mean())
