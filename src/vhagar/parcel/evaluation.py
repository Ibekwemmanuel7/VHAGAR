"""Evaluation scaffold for parcel suitability, built for the labels this problem has.

Suitability labels are sparse, imbalanced, geographically autocorrelated, and often
contested (a "suitable" tag is a human judgement, sometimes a disputed one). Random
train/test splits leak spatial structure and flatter the model. This scaffold therefore:

* reuses VHAGAR's leakage-safe splitters (``vhagar.eval.splits``): spatial-block holdout by
  default, or leave-one-group-out (county, reviewer, campaign) or leave-year-out;
* reports **coverage**: the engine may refuse to score (insufficient evidence), and a
  system that refuses hard cases can look accurate on what is left. Coverage and the
  abstention rate are reported next to every metric;
* compares the engine with a null/majority baseline and a transparent rules baseline on the
  **identical held-out parcels** (the "matched" set the engine decided), and again on all
  held-out parcels with an abstention counted as "not suitable" (the "full" set);
* reports precision, recall, F1, balanced accuracy, and the engine's average precision
  (PR-AUC) rather than accuracy alone, because accuracy rewards predicting the majority
  class when labels are imbalanced;
* invents no performance numbers. The only data shipped here are CLEARLY LABELLED
  SYNTHETIC fixtures that exercise the harness end to end.

Nothing in this module is a measured accuracy claim for parcel suitability.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from vhagar.eval.metrics import average_precision
from vhagar.eval.splits import (
    SplitManifest,
    SplitUnit,
    leave_one_group_out,
    leave_year_out,
    spatial_block_split,
)
from vhagar.parcel.config import SCORING_CONFIG
from vhagar.parcel.engine import score_parcel
from vhagar.parcel.schemas import EvidenceStatus, FeatureValue, Parcel, ProposedUse

__all__ = [
    "LabeledParcel",
    "LABEL_LIMITATIONS",
    "ConfusionMetrics",
    "PredictorResult",
    "EvaluationReport",
    "EngineOutcome",
    "rules_baseline",
    "majority_baseline",
    "engine_outcome",
    "engine_prediction",
    "confusion_metrics",
    "evaluate_suitability",
    "evaluate_spatial_block",
    "synthetic_labeled_set",
]

#: Standing caveats about the labels. These are properties of the problem, not of a run,
#: so they are stated in every report.
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
    """A parcel with a proposed use, its feature inputs, and a binary suitability label.

    ``label`` is 1 (suitable) or 0 (not). ``lon``/``lat`` locate it for spatial blocking,
    ``group`` is a grouping key (county, reviewer, campaign) for group holdout, ``when``
    dates the label for temporal holdout, and ``label_source`` records where the label came
    from ("synthetic" for fixtures).
    """

    parcel: Parcel
    use: ProposedUse
    features: dict[str, FeatureValue]
    label: int
    lon: float
    lat: float
    when: date
    group: str | None = None
    label_source: str = "synthetic"


@dataclass
class ConfusionMetrics:
    """Threshold metrics for a binary decision, with the raw counts kept so a reader can
    recompute anything and see the class balance."""

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
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def specificity(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2.0

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
            "precision": round(self.precision, 4), "recall": round(self.recall, 4),
            "f1": round(self.f1, 4), "balanced_accuracy": round(self.balanced_accuracy, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class PredictorResult:
    """One predictor's held-out performance on the matched and the full held-out sets."""

    name: str
    matched: ConfusionMetrics  # the parcels the engine decided (scored or ineligible)
    full: ConfusionMetrics  # every held-out parcel; engine abstention counted as 0
    average_precision: float | None = None  # engine only (it has a continuous score)


@dataclass
class EvaluationReport:
    """The report format. It leads with the question, the split, coverage, and the label
    caveats, not a single headline number, so the result cannot be quoted out of context."""

    prediction_unit: str
    deployment_question: str
    split_method: str
    n_units: int
    n_decided: int
    n_abstained: int
    label_limitations: tuple[str, ...]
    predictors: list[PredictorResult]
    skill: dict[str, dict[str, float]]
    calibration: list[dict] | None = None
    failure_modes: list[str] = field(default_factory=list)
    is_synthetic: bool = True
    notes: str = ""

    @property
    def coverage(self) -> float:
        return self.n_decided / self.n_units if self.n_units else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.n_abstained / self.n_units if self.n_units else 0.0

    def predictor(self, name: str) -> PredictorResult:
        return next(p for p in self.predictors if p.name == name)

    def summary(self) -> str:
        lines = [
            "PARCEL SUITABILITY EVALUATION REPORT",
            ("  [SYNTHETIC FIXTURE: no real labels; numbers below exercise the harness only]"
             if self.is_synthetic else "  [real labels]"),
            f"  Prediction unit : {self.prediction_unit}",
            f"  Question        : {self.deployment_question}",
            f"  Split method    : {self.split_method}",
            f"  Held-out units  : {self.n_units}",
            f"  Coverage        : engine decided {self.n_decided} of {self.n_units} "
            f"({self.coverage:.1%}); abstained on {self.n_abstained} "
            f"({self.abstention_rate:.1%}) for insufficient evidence",
            "  Label limitations:",
        ]
        lines += [f"    - {lim}" for lim in self.label_limitations]
        for which in ("matched", "full"):
            title = ("Matched set (identical parcels the engine decided)" if which == "matched"
                     else "Full set (all held-out parcels; engine abstention counted as 0)")
            lines.append(f"  {title}:")
            for p in self.predictors:
                m: ConfusionMetrics = getattr(p, which)
                ap = (f" AP={p.average_precision:.3f}"
                      if which == "matched" and p.average_precision is not None
                      and not math.isnan(p.average_precision) else "")
                lines.append(
                    f"    {p.name:<18} n={m.n:<4} prec={m.precision:.3f} rec={m.recall:.3f} "
                    f"f1={m.f1:.3f} bal_acc={m.balanced_accuracy:.3f} "
                    f"(base_rate={m.base_rate:.3f}){ap}")
        lines.append("  Skill on the matched set (differences; positive is better):")
        for name, sk in self.skill.items():
            lines.append(f"    {name:<18} " + "  ".join(f"{k}={v:+.3f}" for k, v in sk.items()))
        if self.calibration:
            lines.append("  Calibration (predicted vs observed):")
            for b in self.calibration:
                lines.append(f"    bin {b['lo']:.2f}-{b['hi']:.2f}: n={b['n']} "
                             f"mean_pred={b['mean_pred']:.3f} observed={b['observed']:.3f}")
        if self.failure_modes:
            lines.append("  Known failure modes:")
            lines += [f"    - {fm}" for fm in self.failure_modes]
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- predictors


@dataclass(frozen=True)
class EngineOutcome:
    """What the engine did for one parcel: ``scored``, ``ineligible`` (a definitive "not
    suitable"), or ``insufficient_evidence`` (an abstention)."""

    status: str
    score: float | None  # 0..100 ranking score; 0.0 for ineligible, None for abstention
    prediction: int | None  # 1/0, or None for abstention


def engine_outcome(lp: LabeledParcel, threshold: float = 55.0,
                   config: dict = SCORING_CONFIG) -> EngineOutcome:
    """The engine as a classifier: suitable iff the overall score clears ``threshold``.
    ``ineligible`` is a definitive 0; ``insufficient_evidence`` is an abstention."""
    r = score_parcel(lp.parcel, lp.use, lp.features, config)
    if r.overall_status == "ineligible":
        return EngineOutcome("ineligible", 0.0, 0)
    if r.overall_score is None:
        return EngineOutcome(r.overall_status, None, None)
    return EngineOutcome("scored", r.overall_score, 1 if r.overall_score >= threshold else 0)


def engine_prediction(lp: LabeledParcel, threshold: float = 55.0,
                      config: dict = SCORING_CONFIG) -> int | None:
    """Binary engine decision for one parcel, or None when the engine abstains."""
    return engine_outcome(lp, threshold, config).prediction


def rules_baseline(lp: LabeledParcel) -> int:
    """A transparent, deliberately simple rule: suitable unless an obvious disqualifier is
    present (steep slope, high wildfire exposure, or notable floodplain). Missing inputs
    are ignored. This is the bar a weighted model must beat to justify its complexity."""
    def val(name: str) -> float | None:
        fv = lp.features.get(name)
        if fv is None or fv.status == EvidenceStatus.MISSING or fv.value is None:
            return None
        return float(fv.value)

    slope, fire, flood = val("slope_pct"), val("wildfire_exposure_0_100"), val("floodplain_frac")
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
    pairs = list(zip(y_true, y_pred, strict=True))
    return ConfusionMetrics(
        n=len(pairs), positives=sum(y_true),
        tp=sum(1 for t, p in pairs if t == 1 and p == 1),
        fp=sum(1 for t, p in pairs if t == 0 and p == 1),
        tn=sum(1 for t, p in pairs if t == 0 and p == 0),
        fn=sum(1 for t, p in pairs if t == 1 and p == 0))


def _to_split_units(dataset: list[LabeledParcel]) -> list[SplitUnit]:
    return [SplitUnit(uid=lp.parcel.parcel_id, lon=lp.lon, lat=lp.lat, when=lp.when,
                      group=lp.group) for lp in dataset]


def _manifest(dataset: list[LabeledParcel], split: str, n_folds: int, block_degrees: float,
              seed: int) -> tuple[SplitManifest, str]:
    units = _to_split_units(dataset)
    if split == "spatial_block":
        return (spatial_block_split(units, n_folds=n_folds, block_degrees=block_degrees,
                                    seed=seed),
                f"spatial-block holdout, {n_folds} folds, {block_degrees} deg blocks (seed {seed})")
    if split == "group":
        return leave_one_group_out(units, by="group"), "leave-one-group-out (group key)"
    if split == "year":
        return leave_year_out(units), "leave-year-out (label year)"
    raise ValueError(f"unknown split: {split!r} (use spatial_block, group, or year)")


def evaluate_suitability(dataset: list[LabeledParcel], *, split: str = "spatial_block",
                         n_folds: int = 3, block_degrees: float = 1.0,
                         threshold: float = 55.0, seed: int = 0,
                         config: dict = SCORING_CONFIG) -> EvaluationReport:
    """Leakage-safe evaluation of the engine against majority and rules baselines.

    The majority baseline is fit on each fold's TRAINING labels only; the engine and the
    rules baseline are stateless. Each held-out parcel is evaluated once (in the fold that
    holds it out).
    """
    by_id = {lp.parcel.parcel_id: lp for lp in dataset}
    manifest, split_desc = _manifest(dataset, split, n_folds, block_degrees, seed)

    rows: list[tuple[int, EngineOutcome, int, int]] = []  # label, engine, rules, majority
    seen: set[str] = set()
    for fold in manifest.folds:
        train = [by_id[u] for u in fold["train"]]
        maj = majority_baseline([lp.label for lp in train])
        for uid in fold["test"]:
            if uid in seen:
                continue
            seen.add(uid)
            lp = by_id[uid]
            rows.append((lp.label, engine_outcome(lp, threshold, config), rules_baseline(lp), maj))

    decided = [r for r in rows if r[1].prediction is not None]
    y_m = [r[0] for r in decided]
    y_f = [r[0] for r in rows]

    eng_m = confusion_metrics(y_m, [r[1].prediction for r in decided])  # type: ignore[misc]
    eng_f = confusion_metrics(y_f, [r[1].prediction or 0 for r in rows])
    rul_m = confusion_metrics(y_m, [r[2] for r in decided])
    rul_f = confusion_metrics(y_f, [r[2] for r in rows])
    maj_m = confusion_metrics(y_m, [r[3] for r in decided])
    maj_f = confusion_metrics(y_f, [r[3] for r in rows])
    ap = (average_precision(y_m, [r[1].score for r in decided]) if decided else float("nan"))

    predictors = [
        PredictorResult("majority/null", maj_m, maj_f),
        PredictorResult("rules", rul_m, rul_f),
        PredictorResult("suitability_engine", eng_m, eng_f, average_precision=ap),
    ]

    def diff(a: ConfusionMetrics, b: ConfusionMetrics) -> dict[str, float]:
        return {"f1": round(a.f1 - b.f1, 4),
                "bal_acc": round(a.balanced_accuracy - b.balanced_accuracy, 4)}

    skill = {
        "rules": {f"{k}_vs_majority": v for k, v in diff(rul_m, maj_m).items()},
        "suitability_engine": {
            **{f"{k}_vs_majority": v for k, v in diff(eng_m, maj_m).items()},
            **{f"{k}_vs_rules": v for k, v in diff(eng_m, rul_m).items()},
        },
    }

    n_abst = len(rows) - len(decided)
    failure_modes = [
        f"Engine abstained on {n_abst} of {len(rows)} held-out parcels. Matched-set metrics "
        "describe only the parcels it decided; read them with the coverage line.",
        "Folds can leave a held-out block with one class; small-sample metrics are unstable "
        "and must not be collapsed into a single headline number.",
        "The decision threshold is a product choice, not a fitted parameter here; sweep it "
        "against real labels before deployment.",
        "The engine output is a 0-100 score, not a probability, so calibration is not "
        "reported. Map scores to probabilities on real labels before quoting any risk.",
    ]
    is_synth = (not dataset) or any(
        lp.label_source == "synthetic"
        or any(fv.status == EvidenceStatus.SYNTHETIC for fv in lp.features.values())
        for lp in dataset)
    return EvaluationReport(
        prediction_unit="one parcel + proposed use",
        deployment_question=("Would this parcel clear a first-pass suitability screen for the "
                             "proposed use, subject to human review?"),
        split_method=f"{split_desc}; majority baseline fit on train labels per fold",
        n_units=len(rows), n_decided=len(decided), n_abstained=n_abst,
        label_limitations=LABEL_LIMITATIONS, predictors=predictors, skill=skill,
        calibration=None, failure_modes=failure_modes, is_synthetic=is_synth,
        notes=("Metrics are over SYNTHETIC fixtures and exercise the harness only."
               if is_synth else ""))


def evaluate_spatial_block(dataset: list[LabeledParcel], *, n_folds: int = 3,
                           block_degrees: float = 1.0, threshold: float = 55.0,
                           seed: int = 0, config: dict = SCORING_CONFIG) -> EvaluationReport:
    """Spatial-block holdout shortcut for :func:`evaluate_suitability`."""
    return evaluate_suitability(dataset, split="spatial_block", n_folds=n_folds,
                                block_degrees=block_degrees, threshold=threshold, seed=seed,
                                config=config)


# --------------------------------------------------------------------------- fixtures


def synthetic_labeled_set(n: int = 60, seed: int = 7,
                          missing_rate: float = 0.1) -> list[LabeledParcel]:
    """A clearly synthetic labelled solar dataset spread over several lat/lon blocks and
    label years, with an imbalanced, noisy label and some missing inputs so coverage and
    abstention are exercised. The label is a transparent synthetic rule plus noise. It is
    NOT a real suitability judgement and must never be reported as one."""
    from vhagar.parcel.fixtures import missing_feature, synthetic_feature

    rng = random.Random(seed)
    out: list[LabeledParcel] = []
    centres = [(-120.0, 39.0), (-119.0, 35.0), (-121.5, 38.5), (-118.0, 34.0), (-122.0, 40.0)]
    for i in range(n):
        clon, clat = centres[i % len(centres)]
        lon = clon + rng.uniform(-0.3, 0.3)
        lat = clat + rng.uniform(-0.3, 0.3)
        values: dict[str, Any] = {
            "slope_pct": round(rng.uniform(0.5, 20.0), 1),
            "irradiance_kwh_m2_day": round(rng.uniform(4.0, 6.8), 2),
            "wildfire_exposure_0_100": round(rng.uniform(5, 90), 0),
            "floodplain_frac": round(rng.uniform(0.0, 0.4), 2),
            "transmission_distance_km": round(rng.uniform(0.5, 20.0), 1),
            "road_distance_km": round(rng.uniform(0.1, 8.0), 1),
            "protected_overlap_frac": round(rng.choice([0.0, 0.0, 0.0, 0.05, 0.3]), 2),
            "market_support_status": rng.choice(["supportive", "mixed", "unknown"]),
        }
        feats = {}
        for k, v in values.items():
            dropped = k != "market_support_status" and rng.random() < missing_rate
            feats[k] = missing_feature(k) if dropped else synthetic_feature(k, v)
        half = 0.01
        geom = [[lon - half, lat - half], [lon + half, lat - half],
                [lon + half, lat + half], [lon - half, lat + half]]
        parcel = Parcel(f"P-SYN-{i:03d}", geom, name=f"synthetic parcel {i}")
        good = ((values["slope_pct"] < 8) + (values["irradiance_kwh_m2_day"] > 5.5)
                + (values["wildfire_exposure_0_100"] < 50) + (values["floodplain_frac"] < 0.15)
                + (values["transmission_distance_km"] < 10)
                + (values["protected_overlap_frac"] <= 0.1))
        label = 1 if (good >= 4) != (rng.random() < 0.08) else 0  # about 8% label noise
        out.append(LabeledParcel(parcel=parcel, use=ProposedUse.SOLAR, features=feats,
                                 label=label, lon=lon, lat=lat,
                                 when=date(2021 + i % 5, 1, 1), group=f"county-{i % 7}",
                                 label_source="synthetic"))
    return out
