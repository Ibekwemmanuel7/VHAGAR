"""Dataset primitives shared by all four tasks.

The important invariant: **a dataset is always constructed from a
:class:`~vhagar.eval.splits.SplitManifest` fold, never from a directory
listing.** That is what makes leakage structurally impossible rather than a
matter of discipline.

Halo handling: chips are stored with a 32-cell halo for convolutional context.
Loss and metrics are computed on the *core* only, via
:attr:`vhagar.grid.Tile.core_slice`. Forgetting this double-counts overlapping
regions and inflates every metric slightly -- :func:`crop_to_core` exists so
you do not have to remember.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vhagar.grid import HALO_CELLS, TILE_CELLS

__all__ = ["ChipRecord", "VhagarDataset", "crop_to_core", "normalise"]


@dataclass(frozen=True, slots=True)
class ChipRecord:
    """Pointer to one training chip in the analysis store."""

    uid: str
    tile_id: str
    #: ISO date of the target time step.
    when: str
    #: Zarr group / path within the analysis store.
    path: str
    #: Optional fire-event id, used for event-blocked splitting.
    event_id: str | None = None
    ecoregion: str | None = None
    continent: str | None = None


def crop_to_core(arr: np.ndarray, halo: int = HALO_CELLS) -> np.ndarray:
    """Drop the halo from the last two axes."""
    if halo <= 0:
        return arr
    return arr[..., halo : halo + TILE_CELLS, halo : halo + TILE_CELLS]


def normalise(
    x: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-channel standardisation.

    Statistics must be computed on **training folds only**. Computing them on
    the whole dataset is a subtle but real leak: test-fold radiometry
    influences the normalisation the model sees at training time.
    """
    m = np.asarray(mean, dtype=np.float32).reshape(-1, *([1] * (x.ndim - 1)))
    s = np.asarray(std, dtype=np.float32).reshape(-1, *([1] * (x.ndim - 1)))
    return (x.astype(np.float32) - m) / (s + eps)


class VhagarDataset:
    """Base dataset. Torch-free so it can be tested without the torch extra.

    Subclasses implement :meth:`load_chip`, returning a dict of arrays.
    Wrap in ``torch.utils.data.Dataset`` via :meth:`as_torch` when needed.
    """

    def __init__(
        self,
        records: Sequence[ChipRecord],
        root: str | Path,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        crop_halo: bool = False,
    ) -> None:
        self.records = list(records)
        self.root = Path(root)
        self.transform = transform
        self.crop_halo = crop_halo

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        sample = self.load_chip(rec)
        if self.crop_halo:
            sample = {
                k: crop_to_core(v) if isinstance(v, np.ndarray) and v.ndim >= 2 else v
                for k, v in sample.items()
            }
        sample["uid"] = rec.uid
        sample["tile_id"] = rec.tile_id
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def load_chip(self, record: ChipRecord) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- split integration ------------------------------------------------

    @classmethod
    def from_fold(
        cls,
        records: Sequence[ChipRecord],
        manifest,
        fold: int,
        subset: str = "train",
        **kwargs: Any,
    ) -> VhagarDataset:
        """Build a dataset from one fold of a split manifest.

        This is the only supported construction path for training data.

        Parameters
        ----------
        manifest : vhagar.eval.splits.SplitManifest
        fold : int
        subset : {'train', 'val', 'test'}
        """
        if fold >= manifest.n_folds:
            raise IndexError(f"fold {fold} out of range ({manifest.n_folds} folds)")
        wanted = set(manifest.folds[fold].get(subset, []))
        if not wanted:
            raise ValueError(f"fold {fold} has no {subset!r} units")
        selected = [r for r in records if r.uid in wanted]
        missing = wanted - {r.uid for r in selected}
        if missing:
            raise ValueError(
                f"{len(missing)} unit(s) in the manifest have no chip record, "
                f"e.g. {sorted(missing)[:3]}. Refusing to silently train on a "
                "different set than the manifest describes."
            )
        return cls(selected, **kwargs)

    def as_torch(self):
        """Return a ``torch.utils.data.Dataset`` view."""
        try:
            from torch.utils.data import Dataset as TorchDataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError("as_torch requires torch: pip install 'vhagar[torch]'") from exc

        outer = self

        class _Wrapped(TorchDataset):
            def __len__(self) -> int:
                return len(outer)

            def __getitem__(self, i: int):
                import torch

                sample = outer[i]
                return {
                    k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v)
                    for k, v in sample.items()
                }

        return _Wrapped()


@dataclass(slots=True)
class ChannelSpec:
    """Declares the channels a model consumes, with provenance.

    Carrying provenance in the spec (rather than a bare list of names) is what
    lets the serving layer refuse to run a model against an input stack it was
    not trained on -- a failure mode that otherwise surfaces as quietly wrong
    predictions.
    """

    names: list[str]
    sources: dict[str, str] = field(default_factory=dict)
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mean and len(self.mean) != len(self.names):
            raise ValueError("mean must match names")
        if self.std and len(self.std) != len(self.names):
            raise ValueError("std must match names")

    @property
    def n_channels(self) -> int:
        return len(self.names)

    def assert_compatible(self, other: ChannelSpec) -> None:
        if self.names != other.names:
            raise ValueError(
                f"channel mismatch:\n  model expects {self.names}\n  input provides {other.names}"
            )
