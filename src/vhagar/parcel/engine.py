"""The explainable, configuration-driven parcel-suitability engine.

Given a parcel, a proposed use, and ``FeatureValue`` inputs, it returns separate component
scores and, only when the evidence contract in ``vhagar.parcel.config`` is met, an overall
score. The decision order is:

1. **Area gate.** A parcel below the use's minimum size is ``ineligible``.
2. **Hard exclusions.** A triggered exclusion (for example protected-area overlap above the
   configured threshold for solar) is ``ineligible``, with the reason and no score, even if
   other inputs are missing.
3. **Required evidence.** Every feature of every required component, and every exclusion
   input, must be present. Otherwise the result is ``insufficient_evidence`` and names the
   missing inputs. Nothing is renormalised around a gap, so missing evidence can never
   raise a score.
4. **Score.** Fixed-weight mean of the required components.

Market/community evidence is scored as its own component for display but never enters
the overall score. Every result carries an ``evidence_grade`` (``evidence_based``,
``provisional``, ``illustrative``) so a number computed from synthetic or assumed inputs
is never presented as a real assessment.
"""
from __future__ import annotations

from typing import TypeGuard

from vhagar.parcel.config import (
    SCORING_CONFIG,
    exclusion_triggered,
    min_area_ha_for,
    required_features_for,
    score_feature,
)
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
_EXCLUSION_PHRASE = {
    "protected_area_overlap": "protected-area overlap",
    "floodplain_overlap": "floodplain overlap",
    "predominantly_developed": "developed land cover",
}


# (phrase when the sub-score is strong, phrase when it is weak)
_DRIVER = {
    "slope_pct": ("gentle terrain", "steep terrain"),
    "irradiance_kwh_m2_day": ("high solar irradiance", "low solar irradiance"),
    "mean_wind_ms_100m": ("strong wind resource", "weak wind resource"),
    "wildfire_exposure_0_100": ("low wildfire exposure", "high wildfire exposure"),
    "floodplain_frac": ("little floodplain overlap", "significant floodplain overlap"),
    "transmission_distance_km": ("close to transmission", "far from transmission"),
    "road_distance_km": ("good road access", "poor road access"),
    "habitat_sensitivity_0_100": ("high habitat value", "low habitat value"),
}
# Fractions whose desirable direction depends on the use (protected overlap is good for
# conservation and a constraint for solar), so the phrase follows the value, not the score.
_FRACTION_NOUN = {"protected_overlap_frac": "protected-area overlap",
                  "developed_cover_frac": "developed land cover"}
_STRENGTH = {"physical": "site physical conditions", "access": "access to infrastructure",
             "hazard": "low hazard and constraint exposure"}
_WEAKNESS = {"physical": "site physical conditions", "access": "access to infrastructure",
             "hazard": "hazard and constraint exposure"}


def _phrase(feature: str) -> str:
    return _PHRASE.get(feature, feature)


def _band(score: float) -> str:
    if score >= 75:
        return "strong"
    if score >= 55:
        return "moderate"
    if score >= 40:
        return "marginal"
    return "poor"


def _present(fv: FeatureValue | None) -> TypeGuard[FeatureValue]:
    return fv is not None and fv.status != EvidenceStatus.MISSING and fv.value is not None


def _score_component(name: str, cfg: dict, features: dict[str, FeatureValue]) -> ComponentScore:
    """Score one component. Every listed feature is required; weights are fixed."""
    used: list[str] = []
    missing: list[str] = []
    reasons: list[str] = []
    num = 0.0
    subs: dict[str, float] = {}
    for feat, fweight, spec in cfg["features"]:
        fv = features.get(feat)
        sub = score_feature(fv.value, spec) if _present(fv) else None
        if sub is None:
            missing.append(feat)
            reasons.append(f"missing:{feat}")
            continue
        used.append(feat)
        subs[feat] = sub
        num += fweight * sub
    in_overall = bool(cfg.get("in_overall", True))
    if missing:
        return ComponentScore(component=name, score=None, weight=cfg["weight"],
                              status="insufficient_evidence",
                              reason_codes=[f"insufficient_evidence:{name}", *reasons],
                              inputs_used=used, missing=missing, in_overall=in_overall)
    for feat, sub in subs.items():
        if sub < 40:
            reasons.append(f"weak:{feat}")
        elif sub >= 80:
            reasons.append(f"strong:{feat}")
    total_w = sum(w for (_f, w, _s) in cfg["features"])
    return ComponentScore(component=name, score=round(num / total_w, 1), weight=cfg["weight"],
                          status="scored", reason_codes=reasons, inputs_used=used,
                          missing=missing, in_overall=in_overall)


def _completeness(features: dict[str, FeatureValue], referenced: list[str]) -> Completeness:
    counts = {s.value: 0 for s in EvidenceStatus}
    for feat in referenced:
        fv = features.get(feat)
        status = fv.status.value if _present(fv) else EvidenceStatus.MISSING.value
        counts[status] += 1
    total = len(referenced)
    real = counts["observed"] + counts["derived"]
    return Completeness(observed=counts["observed"], derived=counts["derived"],
                        assumed=counts["assumed"], synthetic=counts["synthetic"],
                        missing=counts["missing"], total=total,
                        completeness_pct=round(100.0 * real / total, 1) if total else 0.0,
                        has_synthetic=counts["synthetic"] > 0,
                        caller_declared=counts["caller_declared"])


def _evidence_grade(features: dict[str, FeatureValue], required: list[str]) -> str:
    statuses = {features[f].status for f in required if _present(features.get(f))}
    if not statuses:
        return "no_evidence"
    if EvidenceStatus.SYNTHETIC in statuses:
        return "illustrative"
    if EvidenceStatus.ASSUMED in statuses:
        return "provisional"
    if EvidenceStatus.CALLER_DECLARED in statuses:
        return "unverified"
    return "evidence_based"


def _grade_prefix(grade: str) -> str:
    if grade == "illustrative":
        return "Illustrative demo result (synthetic inputs, not a real assessment). "
    if grade == "provisional":
        return "Provisional result (some inputs are assumed defaults). "
    if grade == "unverified":
        return ("Unverified result (inputs were declared by the caller and not checked "
                "against source data). ")
    return ""


def score_parcel(parcel: Parcel, use: ProposedUse | str, features: dict[str, FeatureValue],
                 config: dict = SCORING_CONFIG) -> SuitabilityResult:
    """Score a parcel for a proposed use and return a fully auditable result."""
    if isinstance(use, str):
        use = ProposedUse(use)
    use_cfg = config["uses"][use.value]
    comp_cfgs: dict = use_cfg["components"]
    required = required_features_for(use.value, config)
    referenced = list(dict.fromkeys(
        required + [f for c in comp_cfgs.values() for (f, _w, _s) in c["features"]]))

    comps = [_score_component(name, cfg, features) for name, cfg in comp_cfgs.items()]
    completeness = _completeness(features, referenced)
    provenance = _provenance(features, referenced)
    grade = _evidence_grade(features, required)

    reasons: list[str] = []
    exclusions: list[str] = []
    min_area = min_area_ha_for(use.value, config)
    if parcel.area_ha is not None and parcel.area_ha < min_area:
        exclusions.append(f"parcel_below_min_area_for_{use.value}")
    for rule in use_cfg.get("exclusions", []):
        fv = features.get(rule["feature"])
        if _present(fv) and exclusion_triggered(float(fv.value), rule):  # type: ignore[arg-type]
            exclusions.append(rule["code"])
    missing_required = [f for f in required if not _present(features.get(f))]

    overall: float | None = None
    if exclusions:
        status = "ineligible"
        reasons += [f"ineligible:{code}" for code in exclusions]
    elif missing_required:
        status = "insufficient_evidence"
        reasons += [f"missing_required:{f}" for f in missing_required]
    else:
        scored = {c.component: c for c in comps}
        req = use_cfg["required_components"]
        wsum = sum(scored[n].weight for n in req)
        overall = round(sum(scored[n].weight * float(scored[n].score or 0.0) for n in req)
                        / wsum, 1)
        status = "scored"
    for c in comps:
        reasons.extend(c.reason_codes)
    if grade not in ("evidence_based", "no_evidence"):
        reasons.append(f"evidence_grade:{grade}")

    mfv = features.get("market_support_status")
    market_value = str(mfv.value) if _present(mfv) else None
    market_status = mfv.status.value if _present(mfv) else EvidenceStatus.MISSING.value
    explanation = _explain(parcel, use, overall, status, comps, completeness, grade,
                           exclusions, missing_required, use_cfg, features, min_area)
    return SuitabilityResult(
        parcel_id=parcel.parcel_id, use=use, overall_score=overall, overall_status=status,
        components=comps, completeness=completeness, provenance=provenance,
        reason_codes=list(dict.fromkeys(reasons)), explanation=explanation,
        config_version=config["version"], disclaimer=DISCLAIMER, evidence_grade=grade,
        exclusions=exclusions, missing_required=missing_required, market_value=market_value,
        market_status=market_status)


def _explain(parcel: Parcel, use: ProposedUse, overall: float | None, status: str,
             comps: list[ComponentScore], completeness: Completeness, grade: str,
             exclusions: list[str], missing_required: list[str], use_cfg: dict,
             features: dict[str, FeatureValue], min_area: float) -> str:
    use_label = use.value.replace("_", " ")
    s = _grade_prefix(grade)
    if status == "ineligible":
        why = []
        for code in exclusions:
            if code.startswith("parcel_below_min_area"):
                why.append(f"the parcel ({parcel.area_ha} ha) is below the {min_area} ha minimum")
                continue
            rule = next(r for r in use_cfg["exclusions"] if r["code"] == code)
            val = features[rule["feature"]].value
            why.append(f"{_EXCLUSION_PHRASE.get(code, code)} is {float(val):.0%}, above the "  # type: ignore[arg-type]
                       f"{float(rule['threshold']):.0%} limit")
        s += (f"{parcel.parcel_id} is ineligible for {use_label} under this screen: "
              + "; ".join(why) + ". No suitability score is produced.")
    elif status == "insufficient_evidence":
        s += (f"{parcel.parcel_id}: cannot produce an overall {use_label} suitability score. "
              "Missing required inputs: " + ", ".join(_phrase(f) for f in missing_required)
              + ". The screen does not estimate or skip required evidence.")
    else:
        assert overall is not None
        label = _band(overall) if grade == "evidence_based" else f"{_band(overall)}, {grade}"
        s += f"{parcel.parcel_id} scores {overall} out of 100 for {use_label} use ({label})."
        core = [c for c in comps if c.in_overall]
        strengths = [_STRENGTH.get(c.component, c.component) for c in core
                     if c.score is not None and c.score >= 70]
        weaks = [_WEAKNESS.get(c.component, c.component) for c in core
                 if c.score is not None and c.score < 45]
        if strengths:
            s += " Strengths: " + ", ".join(strengths) + "."
        if weaks:
            s += " Main constraints: " + ", ".join(weaks) + "."
        drivers = _top_drivers(core, features)
        if drivers:
            s += " Key drivers: " + "; ".join(drivers) + "."
    market = next((c for c in comps if not c.in_overall), None)
    if market is not None:
        fv = features.get("market_support_status")
        val = fv.value if _present(fv) else "unknown"
        s += (f" Community/market evidence: {val} (reported separately; it does not change the "
              "suitability score).")
    s += " " + _evidence_sentence(completeness)
    return s


def _top_drivers(comps: list[ComponentScore], features: dict[str, FeatureValue]) -> list[str]:
    out = []
    for c in comps:
        for rc in c.reason_codes:
            kind, _, feat = rc.partition(":")
            if kind not in ("weak", "strong"):
                continue
            if feat in _FRACTION_NOUN:
                level = "substantial" if float(features[feat].value) >= 0.5 else "little"  # type: ignore[arg-type]
                out.append(f"{level} {_FRACTION_NOUN[feat]}")
                continue
            good, bad = _DRIVER.get(feat, (f"favourable {_phrase(feat)}", f"poor {_phrase(feat)}"))
            if kind == "weak":
                out.append(bad)
            elif kind == "strong":
                out.append(good)
    return out[:5]


def _evidence_sentence(c: Completeness) -> str:
    bits = (f"Evidence: {c.observed} observed, {c.derived} derived, {c.assumed} assumed, "
            f"{c.synthetic} synthetic, {c.missing} missing of {c.total} inputs "
            f"({c.completeness_pct}% real).")
    if c.has_synthetic:
        bits += " Synthetic inputs are demo fixtures, not real measurements."
    return bits


def _provenance(features: dict[str, FeatureValue], referenced: list[str]) -> list[Provenance]:
    out = []
    for feat in referenced:
        fv = features.get(feat)
        if _present(fv) and fv.provenance is not None:
            out.append(fv.provenance)
            continue
        status = fv.status if _present(fv) else EvidenceStatus.MISSING
        try:
            out.append(provenance_for(feat, status))
        except KeyError:
            continue
    return out
