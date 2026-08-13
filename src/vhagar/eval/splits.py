"""Leakage-proof cross-validation splits.

This module is the single most important piece of code in VHAGAR.

Almost every wildfire ML result that looks too good is a leakage artifact.
Two documented examples that motivate the design:

* A FIRMS wildfire/non-wildfire classifier scored F1 **0.985** under a random
  split, **0.767** under an event-aware split, and **0.627** under a 5-degree
  spatial-block holdout. Raw lat/lon accounted for ~89% of model gain while
  *harming* out-of-region transfer.
* On next-day spread benchmarks, fold-to-fold standard deviation is
  ±0.08-0.10 AP -- comparable to the entire spread of model rankings. A
  reported 0.02 AP improvement without per-fold numbers is noise.

Consequently there is **no random-split function in this module and there
never will be**. :func:`random_split` exists solely to raise an error with a
pointer to the right alternative.

Every splitter returns a :class:`SplitManifest`, which is serialisable to
JSON/GeoParquet and is expected to be versioned alongside model artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

__all__ = [
    "SplitManifest",
    "SplitUnit",
    "leave_one_group_out",
    "leave_year_out",
    "random_split",
    "spatial_block_split",
    "verify_no_overlap",
]


@dataclass(frozen=True, slots=True)
class SplitUnit:
    """The atomic unit that gets assigned to a fold.

    Never a pixel and never a chip. Depending on task this is a fire event, a
    tile, or a (tile, year) pair -- anything whose members are *not*
    exchangeable with members of another unit.
    """

    uid: str
    #: Representative coordinate in EPSG:4326, used for spatial blocking.
    lon: float
    lat: float
    #: Observation date, used for temporal blocking.
    when: date
    #: Optional grouping keys: fire event id, ecoregion, continent, tile id.
    group: str | None = None
    ecoregion: str | None = None
    continent: str | None = None
    tile_id: str | None = None


@dataclass(slots=True)
class SplitManifest:
    """A concrete, reproducible assignment of units to folds.

    Serialise this next to every trained model. A model without its split
    manifest is an unfalsifiable claim.
    """

    scheme: str
    folds: list[dict[str, list[str]]]
    params: dict = field(default_factory=dict)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def __iter__(self) -> Iterator[tuple[list[str], list[str]]]:
        for f in self.folds:
            yield f["train"], f["test"]

    def fingerprint(self) -> str:
        """Stable hash of the assignment, for experiment tracking."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p

    @classmethod
    def from_json(cls, path: str | Path) -> SplitManifest:
        data = json.loads(Path(path).read_text())
        return cls(scheme=data["scheme"], folds=data["folds"], params=data.get("params", {}))


def random_split(*_args, **_kwargs):  # noqa: D401
    """Deliberately unavailable.

    Random splits leak spatially and temporally autocorrelated fire data and
    inflate every metric VHAGAR reports. Use :func:`spatial_block_split`,
    :func:`leave_one_group_out` (by fire event / ecoregion / continent), or
    :func:`leave_year_out`.
    """
    raise NotImplementedError(
        "Random splits are not supported. Fire data is spatially and temporally "
        "autocorrelated; a random split inflates metrics by 0.2-0.4 F1. Use "
        "spatial_block_split(), leave_one_group_out(), or leave_year_out()."
    )


def _require(units: Sequence[SplitUnit]) -> list[SplitUnit]:
    units = list(units)
    if not units:
        raise ValueError("no split units provided")
    uids = [u.uid for u in units]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate SplitUnit.uid values")
    return units


def spatial_block_split(
    units: Sequence[SplitUnit],
    n_folds: int = 5,
    block_degrees: float = 5.0,
    seed: int = 0,
) -> SplitManifest:
    """Assign whole spatial blocks to folds.

    Units are bucketed into ``block_degrees`` lat/lon cells; whole cells are
    then dealt to folds. Block size should be set from the empirical range of
    the residual variogram, not chosen for convenience. 5 degrees is a
    defensible default for continental fire-danger work; use larger blocks if
    your residuals remain correlated beyond that range.
    """
    units = _require(units)
    rng = np.random.default_rng(seed)

    def block_of(u: SplitUnit) -> tuple[int, int]:
        return (int(np.floor(u.lon / block_degrees)), int(np.floor(u.lat / block_degrees)))

    blocks: dict[tuple[int, int], list[str]] = {}
    for u in units:
        blocks.setdefault(block_of(u), []).append(u.uid)

    keys = sorted(blocks)
    if len(keys) < n_folds:
        raise ValueError(
            f"only {len(keys)} spatial blocks at {block_degrees} degrees but {n_folds} folds "
            "requested; reduce n_folds or block_degrees"
        )
    order = rng.permutation(len(keys))
    assign: dict[int, list[str]] = {k: [] for k in range(n_folds)}
    for rank, key_idx in enumerate(order):
        assign[rank % n_folds].extend(blocks[keys[key_idx]])

    folds = []
    for k in range(n_folds):
        test = sorted(assign[k])
        train = sorted(u.uid for u in units if u.uid not in set(test))
        folds.append({"train": train, "test": test})

    return SplitManifest(
        scheme="spatial_block",
        folds=folds,
        params={"n_folds": n_folds, "block_degrees": block_degrees, "seed": seed},
    )


def leave_one_group_out(
    units: Sequence[SplitUnit],
    by: str = "group",
    max_folds: int | None = None,
) -> SplitManifest:
    """Leave-one-fire-out / -ecoregion-out / -continent-out.

    ``by`` selects the attribute: ``group`` (fire event), ``ecoregion``,
    ``continent``, or ``tile_id``. Units with a missing key are dropped and
    reported in ``params['dropped']`` -- silently including them would put the
    same fire in train and test.
    """
    units = _require(units)
    if by not in {"group", "ecoregion", "continent", "tile_id"}:
        raise ValueError(f"unsupported grouping {by!r}")

    keyed = [(getattr(u, by), u.uid) for u in units]
    dropped = [uid for key, uid in keyed if key is None]
    groups: dict[str, list[str]] = {}
    for key, uid in keyed:
        if key is not None:
            groups.setdefault(str(key), []).append(uid)

    if len(groups) < 2:
        raise ValueError(f"need >= 2 distinct {by} values, found {len(groups)}")

    names = sorted(groups)
    if max_folds is not None:
        names = names[:max_folds]

    all_uids = {uid for key, uid in keyed if key is not None}
    folds = []
    for name in names:
        test = sorted(groups[name])
        train = sorted(all_uids - set(test))
        folds.append({"train": train, "test": test, "held_out": name})

    return SplitManifest(
        scheme=f"leave_one_{by}_out",
        folds=folds,
        params={"n_groups": len(groups), "dropped": dropped, "max_folds": max_folds},
    )


def leave_year_out(
    units: Sequence[SplitUnit],
    n_test_years: int = 1,
    n_val_years: int = 1,
) -> SplitManifest:
    """Year-permutation folds, the standard protocol for spread benchmarks.

    For each candidate test year (chronological), the preceding
    ``n_val_years`` become validation and everything else is training.
    Cross-year domain shift is the dominant error source in fire modelling;
    this is what exposes it.
    """
    units = _require(units)
    by_year: dict[int, list[str]] = {}
    for u in units:
        by_year.setdefault(u.when.year, []).append(u.uid)

    years = sorted(by_year)
    if len(years) < n_test_years + n_val_years + 1:
        raise ValueError(
            f"need at least {n_test_years + n_val_years + 1} distinct years, found {len(years)}"
        )

    folds = []
    for i in range(len(years) - n_test_years + 1):
        test_years = years[i : i + n_test_years]
        remaining = [y for y in years if y not in test_years]
        if len(remaining) <= n_val_years:
            continue
        val_years = remaining[-n_val_years:] if n_val_years else []
        train_years = [y for y in remaining if y not in val_years]
        folds.append(
            {
                "train": sorted(uid for y in train_years for uid in by_year[y]),
                "val": sorted(uid for y in val_years for uid in by_year[y]),
                "test": sorted(uid for y in test_years for uid in by_year[y]),
                "held_out": ",".join(str(y) for y in test_years),
            }
        )

    return SplitManifest(
        scheme="leave_year_out",
        folds=folds,
        params={"years": years, "n_test_years": n_test_years, "n_val_years": n_val_years},
    )


def verify_no_overlap(manifest: SplitManifest) -> None:
    """Assert train/val/test disjointness in every fold. Call this in CI."""
    for i, fold in enumerate(manifest.folds):
        sets = {k: set(v) for k, v in fold.items() if isinstance(v, list)}
        keys = sorted(sets)
        for a_idx, a in enumerate(keys):
            for b in keys[a_idx + 1 :]:
                overlap = sets[a] & sets[b]
                if overlap:
                    raise AssertionError(
                        f"fold {i}: {len(overlap)} unit(s) in both {a!r} and {b!r}, "
                        f"e.g. {sorted(overlap)[:3]}"
                    )


def summarise(manifest: SplitManifest) -> str:
    """One-line-per-fold human summary, for logs."""
    lines = [f"{manifest.scheme}  ({manifest.n_folds} folds, fp={manifest.fingerprint()})"]
    for i, fold in enumerate(manifest.folds):
        parts = [f"{k}={len(v)}" for k, v in fold.items() if isinstance(v, list)]
        held = fold.get("held_out")
        suffix = f"  held_out={held}" if held else ""
        lines.append(f"  fold {i:>2}: " + "  ".join(parts) + suffix)
    return "\n".join(lines)


def units_from_records(records: Iterable[dict]) -> list[SplitUnit]:
    """Build :class:`SplitUnit` objects from dict records (e.g. a GeoParquet read)."""
    out = []
    for r in records:
        when = r["when"]
        if isinstance(when, str):
            when = date.fromisoformat(when)
        out.append(
            SplitUnit(
                uid=str(r["uid"]),
                lon=float(r["lon"]),
                lat=float(r["lat"]),
                when=when,
                group=r.get("group"),
                ecoregion=r.get("ecoregion"),
                continent=r.get("continent"),
                tile_id=r.get("tile_id"),
            )
        )
    return out
