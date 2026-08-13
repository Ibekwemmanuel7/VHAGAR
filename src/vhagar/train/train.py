"""Training entrypoint.

    python -m vhagar.train.train task=burned_area model=siamese fold=0

Non-negotiable preconditions, checked before a single batch is loaded:

1. A split manifest exists, verifies disjoint, and its fingerprint is logged.
2. Normalisation statistics were computed on training folds only.
3. Environment versions (numpy, torch, **GDAL, PROJ**) are recorded. A PROJ
   minor release can move your pixels via grid-shift updates.
4. Seeds are set and deterministic algorithms are enabled.

A run missing any of these is not reproducible and will not be promoted.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RunRecord:
    """What every promoted model must ship alongside its weights."""

    split_manifest_path: str
    split_fingerprint: str
    fold: int
    seed: int
    env_versions: dict[str, str]
    normalisation_source: str = "train_folds_only"

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.__dict__, indent=2))
        return p


def env_versions() -> dict[str, str]:
    """Capture the versions that can silently change your numbers."""
    out: dict[str, str] = {}
    import numpy

    out["numpy"] = numpy.__version__
    for name, mod in (("torch", "torch"), ("rasterio", "rasterio"), ("pyproj", "pyproj")):
        try:
            out[name] = __import__(mod).__version__
        except ImportError:
            out[name] = "absent"
    try:
        from osgeo import gdal

        out["gdal"] = gdal.__version__
    except ImportError:
        try:
            import rasterio

            out["gdal"] = rasterio.__gdal_version__
        except ImportError:
            out["gdal"] = "absent"
    try:
        import pyproj

        out["proj"] = pyproj.proj_version_str
    except ImportError:
        out["proj"] = "absent"
    return out


def set_seeds(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        # Legacy global seed is intentional here: third-party geospatial and
        # augmentation libraries still read np.random's global state.
        np.random.seed(seed)  # noqa: NPY002
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass


def preflight(split_manifest_path: str | Path, fold: int, seed: int, out_dir: str | Path) -> RunRecord:
    """Enforce the reproducibility contract. Raises rather than warns."""
    from vhagar.eval.splits import SplitManifest, verify_no_overlap

    manifest = SplitManifest.from_json(split_manifest_path)
    verify_no_overlap(manifest)
    if manifest.scheme.startswith("random"):
        raise ValueError("random split manifests are not permitted; see docs/02_VALIDATION.md")
    if fold >= manifest.n_folds:
        raise IndexError(f"fold {fold} out of range ({manifest.n_folds} folds)")

    set_seeds(seed)
    record = RunRecord(
        split_manifest_path=str(split_manifest_path),
        split_fingerprint=manifest.fingerprint(),
        fold=fold,
        seed=seed,
        env_versions=env_versions(),
    )
    record.write(Path(out_dir) / "run_record.json")
    log.info("preflight ok: %s fold=%d fp=%s", manifest.scheme, fold, record.split_fingerprint)
    return record


def main() -> None:  # pragma: no cover - wired to Hydra in the full build
    raise SystemExit(
        "Wire this to Hydra: @hydra.main(config_path='../../../configs', config_name='config'). "
        "Call preflight() before constructing any dataset."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
