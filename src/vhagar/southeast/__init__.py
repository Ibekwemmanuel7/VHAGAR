"""VHAGAR Southeast: permit-aware event assessment for the US South.

Release 1 of the design in ``docs/25_US_SOUTHEAST_PLATFORM``. Three separate
outputs for every fire event, because they are different questions:

    association   is the event consistent with a declared or permitted burn?
    control       is it staying inside its expected bounds?
    threat        does it threaten a registered asset, whatever its source?

Routing combines the three. In release 1 routing runs in **shadow mode**: the
operational alert is never changed; the decision the rules *would* have made
is recorded for evaluation.

Pure numpy and stdlib, so it runs in the core CI environment.
"""

from __future__ import annotations

from vhagar.southeast.assess import (
    AssessConfig,
    Assessment,
    Asset,
    Association,
    Candidate,
    Control,
    Obs,
    ObservedEvent,
    Routing,
    Threat,
    assess_event,
    assess_threat,
    associate,
    find_candidates,
    route,
)
from vhagar.southeast.records import BurnRecord, BurnRegistry

__all__ = [
    "AssessConfig",
    "Asset",
    "Assessment",
    "Association",
    "BurnRecord",
    "BurnRegistry",
    "Candidate",
    "Control",
    "Obs",
    "ObservedEvent",
    "Routing",
    "Threat",
    "assess_event",
    "assess_threat",
    "associate",
    "find_candidates",
    "route",
]
