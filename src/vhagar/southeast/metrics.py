"""Shadow-season metrics (docs/25 section 7).

The central failure metric is the **dangerous-event downgrade rate**: wildfires
and escapes that the routing would have downgraded. It decides releases; a
nuisance reduction cannot compensate for it. Because dangerous events are
rare in one pilot season, every rate is reported with its count and an exact
95 percent upper bound. Zero failures in a small sample does not prove a low
failure rate, and the report says so.

Detection coverage is reported separately from classification: a classifier
cannot recover a fire that no sensor detected.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from vhagar.intersect import haversine_m
from vhagar.southeast.assess import ALERTING_ACTIONS

__all__ = [
    "DANGEROUS_LABELS",
    "analyst_load",
    "clopper_pearson_upper",
    "dangerous_downgrade",
    "detection_coverage",
    "nuisance_reduction",
    "recall_among_detected",
    "season_summary",
]

#: Ground-truth labels for evaluated events.
DANGEROUS_LABELS = frozenset({"WILDFIRE", "ESCAPE"})


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper bound on a binomial rate.

    >>> round(clopper_pearson_upper(0, 30), 3)   # about 3/n for zero failures
    0.095
    >>> clopper_pearson_upper(0, 0)
    1.0
    """
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(confidence, k + 1, n - k))


def dangerous_downgrade(actions: Mapping[str, str], labels: Mapping[str, str]) -> dict:
    """Count dangerous events (by label) that the shadow routing downgraded.

    ``actions`` maps event_id to shadow action; ``labels`` maps event_id to a
    reviewed label. Unlabelled events are excluded and counted."""
    dangerous = [e for e, lab in labels.items() if lab in DANGEROUS_LABELS and e in actions]
    downgraded = [e for e in dangerous if actions[e] == "DOWNGRADE"]
    n, k = len(dangerous), len(downgraded)
    return {
        "n_dangerous": n,
        "n_downgraded": k,
        "rate": (k / n) if n else None,
        "upper_95": clopper_pearson_upper(k, n),
        "downgraded_event_ids": sorted(downgraded),
        "n_actions_unlabelled": sum(1 for e in actions if e not in labels),
        "note": "small sample: zero failures does not prove a low rate" if n < 30 else "",
    }


def nuisance_reduction(actions: Mapping[str, str], labels: Mapping[str, str]) -> dict:
    """Share of declared-burn events that would not reach a person."""
    burns = [e for e, lab in labels.items() if lab == "DECLARED_BURN" and e in actions]
    removed = [e for e in burns if actions[e] == "DOWNGRADE"]
    return {
        "n_declared_burn_events": len(burns),
        "n_removed": len(removed),
        "share_removed": (len(removed) / len(burns)) if burns else None,
    }


def recall_among_detected(actions: Mapping[str, str], labels: Mapping[str, str]) -> float | None:
    dangerous = [e for e, lab in labels.items() if lab in DANGEROUS_LABELS and e in actions]
    if not dangerous:
        return None
    return sum(actions[e] in ALERTING_ACTIONS for e in dangerous) / len(dangerous)


def analyst_load(actions: Mapping[str, str], days: Mapping[str, str]) -> dict:
    """Analyst queue items per day. ``days`` maps event_id to an ISO date."""
    per_day = Counter(days[e] for e, a in actions.items() if a == "ANALYST" and e in days)
    if not per_day:
        return {"days": 0, "mean_per_day": 0.0, "peak_per_day": 0}
    return {
        "days": len(per_day),
        "mean_per_day": sum(per_day.values()) / len(per_day),
        "peak_per_day": max(per_day.values()),
    }


def detection_coverage(
    documented: Iterable[Mapping],
    event_points: Iterable[Mapping],
    *,
    radius_m: float = 3_000.0,
    days: int = 2,
) -> dict:
    """Share of independently documented fires with any satellite detection.

    ``documented`` items need ``id, lon, lat, date`` (ISO date) and optionally
    ``size_class``; ``event_points`` items need ``lon, lat, date``. A fire is
    covered when a detection lies within ``radius_m`` and ``days`` of its
    reported date. Returns overall and per size-class coverage."""
    from datetime import date as _date

    pts = [(p["lon"], p["lat"], _date.fromisoformat(str(p["date"])[:10])) for p in event_points]
    total: Counter = Counter()
    hit: Counter = Counter()
    for f in documented:
        d0 = _date.fromisoformat(str(f["date"])[:10])
        cls = f.get("size_class", "all")
        total[cls] += 1
        for lon, lat, d in pts:
            if abs((d - d0).days) <= days and haversine_m(f["lat"], f["lon"], lat, lon) <= radius_m:
                hit[cls] += 1
                break
    n, k = sum(total.values()), sum(hit.values())
    return {
        "n_documented": n,
        "n_detected": k,
        "coverage": (k / n) if n else None,
        "by_size_class": {c: {"n": total[c], "detected": hit[c],
                              "coverage": hit[c] / total[c]} for c in sorted(total)},
    }


def season_summary(actions: Mapping[str, str], labels: Mapping[str, str],
                   days: Mapping[str, str]) -> dict:
    return {
        "dangerous_downgrade": dangerous_downgrade(actions, labels),
        "recall_among_detected": recall_among_detected(actions, labels),
        "nuisance_reduction": nuisance_reduction(actions, labels),
        "analyst_load": analyst_load(actions, days),
        "shadow_action_counts": dict(Counter(actions.values())),
    }
