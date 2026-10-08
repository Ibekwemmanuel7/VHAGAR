"""Evaluation scaffold for parcel suitability, built for the labels this problem actually has.

Suitability labels are sparse, imbalanced, geographically autocorrelated, and often
contested (a "suitable" tag is a human judgement, sometimes a disputed one). Random
train/test splits leak spatial structure and flatter the model. This scaffold therefore:

  * reuses VHAGAR's spatial-block holdout (``vhagar.eval.splits``) so whole lat/lon blocks,
    not individual parcels, are held out, and a near-duplicate neighbour cannot sit on both
    sides of the split;
  * compares every candidate against two mandatory baselines -- a transparent rules
    baseline and a null/majority baseline -- so a score only means something relative to
    "predict the majority class" and "a simple rule";
  * reports skill over the majority baseline, calibration bins for probabilistic outputs,
    and the known failure modes of the labels themselves;
  * invents no performance numbers. The only data shipped here is CLEARLY-LABELLED SYNTHETIC
    fixtures, enough to exercise the harness end to end. Real metrics appear only when real
    labels are supplied.

Nothing in this module should be read as a measured accuracy claim for VHAGAR parcel
suitability. It is the apparatus that would produce such a claim once real, documented
labels exist.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date

from vhagar.eval.splits import SplitUnit, spatial_block_split
from vhagar.parcel.config import SCORING_CONFIG
from vhagar.parcel.engine import score_parcel
from vhagar.parcel.schemas import FeatureValue, Parcel, ProposedUse

__all__ = [
    "LabeledParcel",
    "LABEL_LIMITATIONS",
    "ConfusionMetrics",
    "BaselineResult",
    "EvaluationReport",
    "rules_baseline",
    "majority_baseline",
    "engine_prediction",
    "confusion_metrics",
    "evaluate_spatial_block",
    "synthetic_labeled_set",
]

#: Standing caveats about the labels. These are PROPERTIES OF THE PROBLEM, not of a run,
#: so they are stated up front and repeated in every report rather than discovered per run.
LABEL_LIMITATIONS = (
    "Labels are sparse and class-imbalanced; most parcels are never adjudicated.",
    "Labels are spatially autocorrelated; nearby parcels share drivers and reviewers.",
    "Suitability is partly contested: a label encodes a human/market judgement that "
    "another stakeholder might reverse, so disagreement is signal, not only noise.",
    "Positive labels can reflect what got proposed or permitted, not what was optimal "
    "(selection bias toward already-developed or already-protected land).",
    "Vintage drift: a parcel's label can flip as policy, grid, or hazard conditions change.",
)


@dataclass
class LabeledParcel:
    """A parcel with a proposed use, its feature inputs, and a SYNTHETIC suitability label.

    ``label`` is 1 (suitable) or 0 (not), as judged by some external process. ``lon``/``lat``
    locate it for spatial blocking; ``group`` is an optional grouping key (county, reviewer,
    campaign) for group holdout. ``when`` dates the label for temporal holdout.
    """

    parcel: Parcel
    use: ProposedUse
    features: dict[str, FeatureValue]
    label: int
    lon: float
    lat: float
    when: date
    group: str | None = None


@dataclass
class ConfusionMetrics:
    """Threshold metrics for a binary suitability decision. All derivable from the counts,
    which are kept so a reader can recompute anything and see the class balance."""

    n: int
    positives: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def base_rate(self) -> float:
        return self.positives / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n, "positives": self.positives, "base_rate": round(self.base_rate, 4),
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "accuracy": round(self.accuracy, 4), "precision": round(self.precision, 4),
            "recall": round(self.recall, 4), "f1": round(self.f1, 4),
        }


@dataclass
class BaselineResult:
    """One predictor's held-out performance, plus its skill over the majority baseline."""

    name: str
    metrics: ConfusionMetrics
    skill_vs_majority: float | None = None  # accuracy - majority accuracy; None for majority itself


@dataclass
class EvaluationReport:
    """The report format. Deliberately leads with the QUESTION and the LABEL CAVEATS, not a
    single headline number, so the result cannot be quoted out of context."""

    prediction_unit: str
    deployment_question: str
    split_method: str
    n_units: int
    label_limitations: tuple[str, ...]
    baselines: list[BaselineResult]
    calibration: list[dict] | None = None
    failure_modes: list[str] = field(default_factory=list)
    is_synthetic: bool = True
    notes: str = ""

    def summary(self) -> str:
        lines = [
            "PARCEL SUITABILITY EVALUATION REPORT",
            ("  [SYNTHETIC FIXTURE -- no real labels; numbers below exercise the harness only]"
             if self.is_synthetic else "  [real labels]"),
            f"  Prediction unit : {self.prediction_unit}",
            f"  Question        : {self.deployment_question}",
            f"  Split method    : {self.split_method}",
            f"  Units evaluated : {self.n_units}",
            "  Label limitations:",
        ]
        lines += [f"    - {lim}" for lim in self.label_limitations]
        lines.append("  Held-out results (vs baselines):")
        for b in self.baselines:
            m = b.metrics
            skill = "" if b.skill_vs_majority is None else f"  skill_vs_majority={b.skill_vs_majority:+.4f}"
            lines.append(
                f"    {b.name:<16} acc={m.accuracy:.3f} prec={m.precision:.3f} "
                f"rec={m.recall:.3f} f1={m.f1:.3f} (base_rate={m.base_rate:.3f}){skill}")
        if self.calibration:
            lines.append("  Calibration (predicted vs observed):")
            for bin_ in self.calibration:
                lines.append(
                    f"    bin {bin_['lo']:.2f}-{bin_['hi']:.2f}: "
                    f"n={bin_['n']} mean_pred={bin_['mean_pred']:.3f} observed={bin_['observed']:.3f}")
        if self.failure_modes:
            lines.append("  Known failure modes:")
            lines += [f"    - {fm}" for fm in self.failure_modes]
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- predictors

def engine_prediction(lp: LabeledParcel, threshold: float = 55.0,
                      config: dict = SCORING_CONFIG) -> int | None:
    """The engine as a classifier: suitable iff the overall score clears ``threshold``.
    Returns None when the engine refuses to score (insufficient evidence) -- an abstention,
    which the harness counts separately rather than scoring as a wrong guess."""
    result = score_parcel(lp.parcel, lp.use, lp.features, config)
    if result.overall_score is None:
        return None
    return 1 if result.overall_score >= threshold else 0


def rules_baseline(lp: LabeledParcel) -> int:
    """A transparent, deliberately dumb rule: suitable unless an obvious disqualifier is
    present (steep slope, high wildfire exposure, or notable floodplain). This is the bar a
    learned or weighted model must beat to justify its complexity."""
    def val(name):
        fv = lp.features.get(name)
        return None if fv is None else fv.value
    slope = val("slope_pct")
    fire = val("wildfire_exposure_0_100")
    flood = val("floodplain_frac")
    if slope is not None and slope > 15:
        return 0
    if fire is not None and fire > 70:
        return 0
    if flood is not None and flood > 0.2:
        return 0
    return 1


def majority_baseline(labels: list[int]) -> int:
    """Predict the training-set majority class for everything. The null model."""
    if not labels:
        return 0
    return 1 if sum(labels) * 2 >= len(labels) else 0


# --------------------------------------------------------------------------- metrics + CV

def confusion_metrics(y_true: list[int], y_pred: list[int]) -> ConfusionMetrics:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 0)
    return ConfusionMetrics(n=len(y_true), positives=sum(y_true),
                            tp=tp, fp=fp, tn=tn, fn=fn)


def _to_split_units(dataset: list[LabeledParcel]) -> list[SplitUnit]:
    return [SplitUnit(uid=lp.parcel.parcel_id, lon=lp.lon, lat=lp.lat, when=lp.when,
                      group=lp.group) for lp in dataset]


def evaluate_spatial_block(dataset: list[LabeledParcel], *, n_folds: int = 3,
                           block_degrees: float = 1.0, threshold: float = 55.0,
                           seed: int = 0, config: dict = SCORING_CONFIG) -> EvaluationReport:
    """Run spatial-block cross-validation for the engine and both baselines.

    Whole spatial blocks are held out per fold (via ``spatial_block_split``). For each fold
    the majority baseline is fit on that fold's TRAINING labels only; the engine and rules
    baseline are stateless. Engine abstentions (no overall score) are excluded from the
    engine's confusion counts and reported as a failure mode, never silently scored.
    """
    by_id = {lp.parcel.parcel_id: lp for lp in dataset}
    manifest = spatial_block_split(_to_split_units(dataset), n_folds=n_folds,
                                   block_degrees=block_degrees, seed=seed)

    eng_t: list[int] = []
    eng_p: list[int] = []
    rule_t: list[int] = []
    rule_p: list[int] = []
    maj_t: list[int] = []
    maj_p: list[int] = []
    abstentions = 0

    for fold in manifest.folds:
        train = [by_id[u] for u in fold["train"]]
        test = [by_id[u] for u in fold["test"]]
        maj = majority_baseline([lp.label for lp in train])
        for lp in test:
            pred = engine_prediction(lp, threshold=threshold, config=config)
            if pred is None:
                abstentions += 1
            else:
                eng_t.append(lp.label)
                eng_p.append(pred)
            rule_t.append(lp.label)
            rule_p.append(rules_baseline(lp))
            maj_t.append(lp.label)
            maj_p.append(maj)

    maj_metrics = confusion_metrics(maj_t, maj_p)
    eng_metrics = confusion_metrics(eng_t, eng_p)
    rule_metrics = confusion_metrics(rule_t, rule_p)
    maj_acc = maj_metrics.accuracy

    baselines = [
        BaselineResult("majority/null", maj_metrics, skill_vs_majority=None),
        BaselineResult("rules", rule_metrics, skill_vs_majority=rule_metrics.accuracy - maj_acc),
        BaselineResult("suitability_engine", eng_metrics,
                       skill_vs_majority=eng_metrics.accuracy - maj_acc),
    ]

    failure_modes = [
        "Spatial-block folds can leave a fold with only one class; small-sample metrics "
        "are unstable and must not be averaged into a single headline number.",
        f"Engine abstained on {abstentions} held-out parcel(s) for insufficient evidence; "
        "abstentions are excluded from engine metrics, not counted as correct or wrong.",
        "The engine threshold is a product decision, not a fitted parameter here; sweep it "
        "against real labels before deployment.",
    ]
    return EvaluationReport(
        prediction_unit="one parcel + proposed use",
        deployment_question=(
            "Would this parcel clear a first-pass suitability screen for the proposed use, "
            "subject to human review?"),
        split_method=(f"spatial-block holdout, {n_folds} folds, {block_degrees} deg blocks "
                      f"(seed {seed}); majority baseline fit on train labels per fold"),
        n_units=len(dataset),
        label_limitations=LABEL_LIMITATIONS,
        baselines=baselines,
        calibration=None,
        failure_modes=failure_modes,
        is_synthetic=all(
            any(fv.status.value == "synthetic" for fv in lp.features.values())
            for lp in dataset) if dataset else True,
        notes="Metrics are over SYNTHETIC fixtures and exercise the harness only.",
    )


# --------------------------------------------------------------------------- fixtures

def synthetic_labeled_set(n: int = 60, seed: int = 7) -> list[LabeledParcel]:
    """Build a clearly-synthetic labelled dataset spread over several lat/lon blocks, with a
    deliberately imbalanced, noisy, weak label so the harness has something to chew on. The
    label is a transparent synthetic rule plus noise -- it is NOT a real suitability judgement
    and must never be reported as one."""
    from vhagar.parcel.fixtures import synthetic_feature

    rng = random.Random(seed)
    out: list[LabeledParcel] = []
    # a handful of block centres so spatial blocking has real structure
    centres = [(-120.0, 39.0), (-119.0, 35.0), (-121.5, 38.5), (-118.0, 34.0), (-122.0, 40.0)]
    for i in range(n):
        clon, clat = centres[i % len(centres)]
        lon = clon + rng.uniform(-0.3, 0.3)
        lat = clat + rng.uniform(-0.3, 0.3)
        slope = round(rng.uniform(0.5, 20.0), 1)
        irr = round(rng.uniform(4.0, 6.8), 2)
        fire = round(rng.uniform(5, 90), 0)
        flood = round(rng.uniform(0.0, 0.4), 2)
        tx = round(rng.uniform(0.5, 20.0), 1)
        road = round(rng.uniform(0.1, 8.0), 1)
        feats = {
            "slope_pct": synthetic_feature("slope_pct", slope),
            "irradiance_kwh_m2_day": synthetic_feature("irradiance_kwh_m2_day", irr),
            "wildfire_exposure_0_100": synthetic_feature("wildfire_exposure_0_100", fire),
            "floodplain_frac": synthetic_feature("floodplain_frac", flood),
            "transmission_distance_km": synthetic_feature("transmission_distance_km", tx),
            "road_distance_km": synthetic_feature("road_distance_km", road),
            "protected_overlap_frac": synthetic_feature("protected_overlap_frac", 0.0),
            "market_support_status": synthetic_feature("market_support_status",
                                                       rng.choice(["supportive", "mixed", "unknown"])),
        }
        half = 0.01
        geom = [[lon - half, lat - half], [lon + half, lat - half],
                [lon + half, lat + half], [lon - half, lat + half]]
        parcel = Parcel(f"P-SYN-{i:03d}", geom, name=f"synthetic parcel {i}")
        # synthetic label: good solar site = low slope, high irradiance, low fire/flood, + noise
        score = (slope < 8) + (irr > 5.5) + (fire < 50) + (flood < 0.15) + (tx < 10)
        label = 1 if (score >= 4) != (rng.random() < 0.15) else 0  # ~15% label noise
        out.append(LabeledParcel(parcel=parcel, use=ProposedUse.SOLAR, features=feats,
                                 label=label, lon=lon, lat=lat,
                                 when=date(2025, 1, 1), group=f"block-{i % len(centres)}"))
    return out
