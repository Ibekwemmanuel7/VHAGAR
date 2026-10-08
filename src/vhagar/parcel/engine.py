"""The explainable, configuration-driven parcel-suitability engine.

Given a parcel, a proposed use, and a set of ``FeatureValue`` inputs, it computes the four
component scores (physical, access, hazard, market) and an overall score, but only when
minimum evidence requirements are met. It never silently fills gaps: a missing or unknown
input yields a reason code and, if a component falls below its minimum inputs, that
component and possibly the overall return an explicit "insufficient_evidence" outcome. All
weights and thresholds come from the versioned config, not from constants in this file.
"""
from __future__ import annotations

from vhagar.parcel.config import SCORING_CONFIG, min_area_ha_for, score_feature
from vhagar.parcel.manifest import provenance_for
from vhagar.parcel.schemas import (
    Completeness,
    ComponentScore,
    EvidenceStatus,
    FeatureValue,
    Parcel,
    ProposedUse,
    Provenance,
    SuitabilityResult,
)

__all__ = ["score_parcel", "DISCLAIMER"]

DISCLAIMER = (
    "Transparent, configuration-driven suitability screening, not a validated land "
    "valuation (AVM), a bankable number, or a measured community-sentiment product. "
    "Component scores are shown separately on purpose. Inputs marked synthetic are demo "
    "fixtures, not real measurements.")

_PHRASE = {
    "slope_pct": "terrain slope",
    "irradiance_kwh_m2_day": "solar irradiance",
    "mean_wind_ms_100m": "wind resource",
    "wildfire_exposure_0_100": "wildfire exposure",
    "floodplain_frac": "floodplain overlap",
    "transmission_distance_km": "distance to transmission",
    "road_distance_km": "road access",
    "protected_overlap_frac": "protected-area overlap",
    "habitat_sensitivity_0_100": "habitat value",
    "developed_cover_frac": "developed land cover",
    "market_support_status": "community/market evidence",
}


def _band(score: float) -> str:
    if score >= 75:
        return "strong"
    if score >= 55:
        return "moderate"
    if score >= 40:
        return "marginal"
    return "poor"


def _score_component(name, cfg, features: dict[str, FeatureValue]):
    """Score one component: weighted mean of available feature sub-scores, renormalised
    over what is present, gated by min_inputs. Returns a ComponentScore."""
    used, missing, reasons = [], [], []
    num = den = 0.0
    for feat, fweight, spec in cfg["features"]:
        fv = features.get(feat)
        sub = None
        if fv is not None and fv.status != EvidenceStatus.MISSING:
            sub = score_feature(fv.value, spec)
        if sub is None:
            missing.append(feat)
            reasons.append(f"missing:{feat}")
            continue
        used.append(feat)
        num += fweight * sub
        den += fweight
        if sub < 40:
            reasons.append(f"weak:{feat}")
        elif sub >= 80:
            reasons.append(f"strong:{feat}")
    if len(used) < cfg["min_inputs"] or den == 0:
        return ComponentScore(component=name, score=None, weight=cfg["weight"],
                              status="insufficient_evidence",
                              reason_codes=[f"insufficient_evidence:{name}", *reasons],
                              inputs_used=used, missing=missing)
    return ComponentScore(component=name, score=round(num / den, 1), weight=cfg["weight"],
                          status="scored", reason_codes=reasons, inputs_used=used, missing=missing)


def _completeness(features: dict[str, FeatureValue], referenced: list[str]) -> Completeness:
    counts = {s: 0 for s in ("observed", "derived", "assumed", "synthetic", "missing")}
    for feat in referenced:
        fv = features.get(feat)
        status = fv.status.value if fv is not None else "missing"
        counts[status] = counts.get(status, 0) + 1
    total = len(referenced)
    real = counts["observed"] + counts["derived"]
    return Completeness(observed=counts["observed"], derived=counts["derived"],
                        assumed=counts["assumed"], synthetic=counts["synthetic"],
                        missing=counts["missing"], total=total,
                        completeness_pct=round(100.0 * real / total, 1) if total else 0.0,
                        has_synthetic=counts["synthetic"] > 0)


def _explain(parcel, use, overall, overall_status, comps, completeness) -> str:
    use_label = use.value.replace("_", " ")
    if overall_status != "scored":
        return (f"{parcel.parcel_id}: cannot produce an overall {use_label} suitability "
                f"score yet. {_insufficiency_reason(parcel, use, comps)} "
                f"{_evidence_sentence(completeness)}")
    strengths = [c.component for c in comps if c.score is not None and c.score >= 70]
    weaks = [c.component for c in comps if c.score is not None and c.score < 45]
    s = f"{parcel.parcel_id} scores {overall} out of 100 for {use_label} use ({_band(overall)})."
    if strengths:
        s += " Strengths: " + ", ".join(strengths) + "."
    if weaks:
        s += " Main constraints: " + ", ".join(weaks) + "."
    drivers = _top_drivers(comps)
    if drivers:
        s += " Key drivers: " + "; ".join(drivers) + "."
    s += " " + _evidence_sentence(completeness)
    return s


def _top_drivers(comps) -> list[str]:
    out = []
    for c in comps:
        for rc in c.reason_codes:
            if rc.startswith("weak:"):
                out.append("low " + _PHRASE.get(rc.split(":", 1)[1], rc.split(":", 1)[1]))
            elif rc.startswith("strong:"):
                out.append("good " + _PHRASE.get(rc.split(":", 1)[1], rc.split(":", 1)[1]))
    return out[:5]


def _insufficiency_reason(parcel, use, comps) -> str:
    if parcel.area_ha is not None and parcel.area_ha < min_area_ha_for(use.value):
        return (f"The parcel ({parcel.area_ha} ha) is below the {min_area_ha_for(use.value)} ha "
                f"minimum this screen uses for {use.value.replace('_', ' ')}.")
    unscored = [c.component for c in comps if c.status != "scored"]
    if unscored:
        return "Not enough evidence in: " + ", ".join(unscored) + "."
    return "Too few components could be scored."


def _evidence_sentence(c: Completeness) -> str:
    bits = f"Evidence: {c.observed} observed, {c.derived} derived, {c.assumed} assumed, " \
           f"{c.synthetic} synthetic, {c.missing} missing of {c.total} inputs " \
           f"({c.completeness_pct}% real)."
    if c.has_synthetic:
        bits += " Synthetic inputs are demo fixtures, not real measurements."
    return bits


def score_parcel(parcel: Parcel, use: ProposedUse, features: dict[str, FeatureValue],
                 config: dict = SCORING_CONFIG) -> SuitabilityResult:
    """Score a parcel for a proposed use. Returns a fully auditable SuitabilityResult:
    component scores (or insufficient-evidence), completeness, provenance, reason codes,
    and a plain-language explanation. The overall score is withheld unless the parcel
    passes the minimum-area gate, the physical component scored, and scored components
    cover at least half the use's component weight."""
    if isinstance(use, str):
        use = ProposedUse(use)
    use_cfg = config["uses"][use.value]
    referenced = [f for comp in use_cfg.values() for (f, _w, _s) in comp["features"]]

    comps = [_score_component(name, cfg, features) for name, cfg in use_cfg.items()]
    completeness = _completeness(features, referenced)
    provenance = _provenance(features, referenced)

    # hard area gate
    area_ok = parcel.area_ha is None or parcel.area_ha >= min_area_ha_for(use.value, config)
    physical = next((c for c in comps if c.component == "physical"), None)
    scored = [c for c in comps if c.status == "scored"]
    scored_weight = sum(c.weight for c in scored)

    overall, status, reasons = None, "insufficient_evidence", []
    if not area_ok:
        reasons.append(f"parcel_below_min_area_for_{use.value}")
    elif physical is None or physical.status != "scored":
        reasons.append("physical_component_insufficient_evidence")
    elif scored_weight < 0.5:
        reasons.append("insufficient_component_coverage")
    else:
        num = sum(c.weight * c.score for c in scored)
        overall = round(num / scored_weight, 1)
        status = "scored"
    for c in comps:
        reasons.extend(c.reason_codes)

    return SuitabilityResult(
        parcel_id=parcel.parcel_id, use=use, overall_score=overall, overall_status=status,
        components=comps, completeness=completeness, provenance=provenance,
        reason_codes=list(dict.fromkeys(reasons)),
        explanation=_explain(parcel, use, overall, status, comps, completeness),
        config_version=config["version"], disclaimer=DISCLAIMER)


def _provenance(features: dict[str, FeatureValue], referenced: list[str]) -> list[Provenance]:
    out = []
    for feat in referenced:
        fv = features.get(feat)
        if fv is not None and fv.provenance is not None:
            out.append(fv.provenance)
        else:
            status = fv.status if fv is not None else EvidenceStatus.MISSING
            try:
                out.append(provenance_for(feat, status))
            except KeyError:
                continue
    return out
