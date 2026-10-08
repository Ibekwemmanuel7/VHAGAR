"""LLM-assisted extraction of community/planning evidence, built production-minded.

This turns public documents (meeting notes, planning records, public comments) into
STRUCTURED, CITED evidence for a human reviewer. It is deliberately conservative:

  * provider-agnostic: an ``EvidenceExtractor`` protocol, with a deterministic local
    ``MockEvidenceExtractor`` so tests and the demo need no API key;
  * every extraction must carry a source citation, the supporting text span, the document
    date, a bounded confidence, and a review status;
  * the output is EVIDENCE FOR HUMAN REVIEW, never ground truth and never an autonomous
    sentiment decision. Nothing here writes a market score by itself; only a reviewer's
    acceptance turns an extraction into a scorable ``market_support_status`` feature;
  * guardrails treat the source document as untrusted data: instruction-like text inside
    a document is flagged as possible prompt injection and never followed, and a claim
    without a citation, or with certainty the span does not support, is downgraded.

Pure stdlib, CI-safe, fully unit-tested on the mock.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

__all__ = [
    "SourceDocument",
    "ExtractedEvidence",
    "EvidenceExtractor",
    "MockEvidenceExtractor",
    "validate_extraction",
    "reviewed_market_status",
    "INJECTION_PATTERNS",
]

#: instruction-like patterns that should never be obeyed when found inside a source doc.
INJECTION_PATTERNS = [
    r"ignore (?:all |the )?previous", r"disregard (?:all |the )?(?:previous|above)",
    r"you are now", r"new instructions?:", r"system\s*:", r"assistant\s*:",
    r"respond with", r"output (?:the following|only)", r"set .*(?:confidence|sentiment) to",
    r"mark this as (?:supportive|approved)",
]
_SUPPORT = ("support", "in favor", "approve", "endorse", "benefit", "welcome")
_OPPOSE = ("oppose", "against", "reject", "concern", "deny", "object", "moratorium")


@dataclass
class SourceDocument:
    """One public document to extract from. ``text`` is untrusted content."""

    doc_id: str
    text: str
    date: str | None = None


@dataclass
class ExtractedEvidence:
    """Structured, cited evidence about community/market sentiment, for human review."""

    doc_id: str
    sentiment: str                 # supportive | mixed | opposed | unknown
    claim: str
    text_span: str                 # the quoted span supporting the claim ("" if none)
    document_date: str | None
    confidence: float              # 0..1, bounded
    status: str                    # needs_review | abstain
    flags: list[str] = field(default_factory=list)


class EvidenceExtractor(Protocol):
    """Provider-agnostic interface. A real adapter (OpenAI, Anthropic, local model) and the
    deterministic mock both implement this."""

    def extract(self, document: SourceDocument) -> ExtractedEvidence:
        ...


def _find_injection(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in INJECTION_PATTERNS)


def _best_span(text: str, keywords) -> str:
    for sent in re.split(r"(?<=[.!?])\s+", text):
        low = sent.lower()
        if any(k in low for k in keywords):
            return sent.strip()[:240]
    return ""


class MockEvidenceExtractor:
    """A deterministic, keyword-based stand-in for a real LLM extractor. It never needs a
    key, always routes to human review, bounds its own confidence, and refuses to follow
    instructions embedded in the document."""

    def extract(self, document: SourceDocument) -> ExtractedEvidence:
        text = document.text or ""
        flags: list[str] = []
        injected = _find_injection(text)
        if injected:
            flags.append("possible_prompt_injection")

        low = text.lower()
        sup = sum(low.count(k) for k in _SUPPORT)
        opp = sum(low.count(k) for k in _OPPOSE)
        if sup == 0 and opp == 0:
            sentiment, span, conf = "unknown", "", 0.2
        elif sup > 0 and opp > 0:
            sentiment = "mixed"
            span = _best_span(text, _SUPPORT + _OPPOSE)
            conf = 0.5
        elif sup > opp:
            sentiment, span, conf = "supportive", _best_span(text, _SUPPORT), 0.6
        else:
            sentiment, span, conf = "opposed", _best_span(text, _OPPOSE), 0.6

        if not span and sentiment != "unknown":
            flags.append("no_citation_span")
            conf = min(conf, 0.3)
        if injected:
            conf = min(conf, 0.3)        # never let injected text raise confidence

        status = "abstain" if (sentiment == "unknown" or "no_citation_span" in flags) else "needs_review"
        claim = f"Document {document.doc_id} reads as {sentiment} toward the proposed use."
        ev = ExtractedEvidence(doc_id=document.doc_id, sentiment=sentiment, claim=claim,
                               text_span=span, document_date=document.date,
                               confidence=round(min(conf, 0.8), 2), status=status, flags=flags)
        ev.flags.extend(validate_extraction(ev))
        ev.flags = list(dict.fromkeys(ev.flags))
        return ev


def validate_extraction(ev: ExtractedEvidence) -> list[str]:
    """Guardrail check on an extraction. Returns a list of problem flags (empty is clean).
    Enforces: a citation span for any non-unknown sentiment, bounded confidence, no
    over-certain claim without a span, a valid sentiment label, and a review-or-abstain
    status (the pipeline never emits an autonomous 'accepted' decision)."""
    problems = []
    if ev.sentiment not in ("supportive", "mixed", "opposed", "unknown"):
        problems.append("invalid_sentiment")
    if not (0.0 <= ev.confidence <= 0.8):
        problems.append("confidence_out_of_bounds")
    if ev.sentiment != "unknown" and not ev.text_span:
        problems.append("missing_citation")
    if ev.confidence > 0.5 and not ev.text_span:
        problems.append("unsupported_certainty")
    if ev.status not in ("needs_review", "abstain"):
        problems.append("non_review_status")
    return problems


def reviewed_market_status(ev: ExtractedEvidence, reviewer_accepted: bool):
    """Convert an extraction into a scorable ``market_support_status`` ONLY after a human
    has accepted it. Without acceptance, returns 'unknown', so the engine treats market
    evidence as pending rather than letting the LLM decide the score."""
    if not reviewer_accepted or ev.status != "needs_review":
        return "unknown"
    return ev.sentiment
