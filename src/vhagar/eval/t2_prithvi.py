"""T2 Prithvi fine-tune glue: leakage-proof chipping, terratorch export, fair scoring.

The Prithvi fine-tune itself runs in TerraTorch on a GPU (docs/13). This module is the
VHAGAR-side plumbing that keeps it honest and comparable to the RBR threshold and the U-Net:

* :func:`grouped_split` assigns *whole fires* to train / val / test, so no fire's pixels
  appear in two splits (the same leave-fire-out discipline as ``t2_unet.grouped_folds``).
* :func:`chip_sample` tiles a six-band :class:`T2Sample` into fixed-size image/label chips
  in the HLS-Burn-Scars convention (label 0 unburned, 1 burned, -1 nodata), pure and
  testable; :func:`write_chip_geotiffs` writes them as the paired GeoTIFFs terratorch reads
  (rasterio-guarded).
* :func:`score_masks` pushes predicted masks back through the *same* skill-over-naive metric
  used for RBR and the U-Net, on the identical held-out fires, so "Prithvi beats +0.54" is a
  claim about the same code path, not a benchmark number from elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "grouped_split",
    "chip_sample",
    "write_chip_geotiffs",
    "export_prithvi_chips",
    "stitch_chip_predictions",
    "PrithviScore",
    "score_masks",
    "summarise_scores",
    "nbr_threshold_baseline",
    "nbr_threshold_transfer",
    "export_inference_chips",
]


def grouped_split(
    event_ids: Sequence[str], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> dict[str, list[str]]:
    """Assign whole fires to train / val / test. Leakage-proof: a fire is in exactly one.

    A foundation-model fine-tune is one training run, not k-fold, so the fair split is a
    single grouped partition by fire (not by pixel or chip). Deterministic in ``seed``.
    """
    ids = sorted(set(event_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(test_frac * n))) if n >= 3 else 0
    n_val = max(1, int(round(val_frac * n))) if n - n_test >= 2 else 0
    test = ids[:n_test]
    val = ids[n_test:n_test + n_val]
    train = ids[n_test + n_val:]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def chip_sample(
    sample, chip: int = 224, stride: int | None = None, min_valid_frac: float = 0.1,
    burn_balance: bool = False, max_bg_ratio: float = 1.0, seed: int = 0,
) -> list[dict]:
    """Tile one six-band :class:`T2Sample` into ``chip x chip`` image/label pairs.

    Returns a list of ``{"y0","x0","image","label"}`` where ``image`` is ``[C, chip, chip]``
    reflectance (NaN filled with 0) and ``label`` is ``[chip, chip]`` int8 in the
    HLS-Burn-Scars convention: 1 burned, 0 unburned, -1 nodata/invalid. Edge tiles are
    shifted inward to stay in bounds; a tile with too few valid pixels is dropped.

    ``burn_balance`` addresses the wide-window class imbalance that makes a plain fine-tune
    predict "not burned" everywhere: it keeps every tile that contains a burned pixel and
    subsamples the all-unburned tiles to at most ``max_bg_ratio`` times the burn tiles, so
    the model sees a roughly balanced burn/background chip set instead of one dominated by
    empty land (tiling stays non-overlapping, so there are no redundant near-duplicate
    chips). Pixel-level imbalance within a chip is handled by the Dice loss. Pure numpy.
    """
    feats = sample.features                      # [C, H, W]
    burned = np.asarray(sample.reference, dtype=bool)
    valid = np.asarray(sample.valid, dtype=bool)
    C, H, W = feats.shape
    stride = stride or chip                       # non-overlapping tiles (no redundant chips)
    if chip > H or chip > W:                      # pad up to one chip, marking pad invalid
        ph, pw = max(0, chip - H), max(0, chip - W)
        feats = np.pad(feats, ((0, 0), (0, ph), (0, pw)), mode="constant", constant_values=np.nan)
        burned = np.pad(burned, ((0, ph), (0, pw)), constant_values=False)
        valid = np.pad(valid, ((0, ph), (0, pw)), constant_values=False)
        C, H, W = feats.shape

    ys = list(range(0, max(1, H - chip + 1), stride))
    xs = list(range(0, max(1, W - chip + 1), stride))
    if ys[-1] != H - chip:
        ys.append(H - chip)
    if xs[-1] != W - chip:
        xs.append(W - chip)

    out: list[dict] = []
    for y0 in ys:
        for x0 in xs:
            v = valid[y0:y0 + chip, x0:x0 + chip]
            if v.mean() < min_valid_frac:
                continue
            img = np.nan_to_num(feats[:, y0:y0 + chip, x0:x0 + chip], nan=0.0).astype(np.float32)
            lab = np.where(v, burned[y0:y0 + chip, x0:x0 + chip].astype(np.int8), -1).astype(np.int8)
            out.append({"y0": int(y0), "x0": int(x0), "image": img, "label": lab})

    if burn_balance and out:
        burn = [c for c in out if bool((c["label"] == 1).any())]
        bg = [c for c in out if not bool((c["label"] == 1).any())]
        if burn:                                  # keep all burn tiles + capped background
            keep = min(len(bg), int(np.ceil(max_bg_ratio * len(burn))))
            rng = np.random.default_rng(seed)
            idx = sorted(rng.permutation(len(bg))[:keep].tolist())
            out = burn + [bg[i] for i in idx]
    return out


def write_chip_geotiffs(chips: Sequence[dict], data_dir, prefix: str) -> list[str]:
    """Write image/label chips in the HLS-Burn-Scars filename convention. Needs rasterio.

    For each chip writes ``{prefix}_{i}_merged.tif`` (``[C, chip, chip]`` float32 image) and
    ``{prefix}_{i}.mask.tif`` (``[chip, chip]`` int16 label, nodata -1) into ``data_dir``,
    which is the layout terratorch's ``GenericNonGeoSegmentationDataModule`` reads with
    ``img_grep: "*_merged.tif"`` / ``label_grep: "*.mask.tif"``. No georeferencing (the
    non-geo dataset reads arrays). Returns the per-chip stems for the split files.
    """
    from pathlib import Path

    import rasterio
    from rasterio.transform import from_origin

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tf = from_origin(0, 0, 1, 1)
    stems: list[str] = []
    for i, ch in enumerate(chips):
        img, lab = ch["image"], ch["label"]
        C, h, w = img.shape
        stem = f"{prefix}_{i}"
        with rasterio.open(
            data_dir / f"{stem}_merged.tif", "w", driver="GTiff", height=h, width=w,
            count=C, dtype="float32", transform=tf, crs="EPSG:4326",
        ) as dst:
            dst.write(img)
        with rasterio.open(
            data_dir / f"{stem}.mask.tif", "w", driver="GTiff", height=h, width=w,
            count=1, dtype="int16", nodata=-1, transform=tf, crs="EPSG:4326",
        ) as dst:
            dst.write(lab.astype("int16")[None])
        stems.append(stem)
    return stems


def export_prithvi_chips(
    samples_by_id: dict, out_dir, chip: int = 224, val_frac: float = 0.15,
    test_frac: float = 0.15, seed: int = 0, min_valid_frac: float = 0.1,
    burn_balance: bool = False, max_bg_ratio: float = 1.0,
) -> dict[str, int]:
    """Export a terratorch-ready chip dataset matching the published burn-scars layout.

    Fires are partitioned by :func:`grouped_split` (whole fires per split), every usable
    sample is chipped, and all chips are written into a single ``out_dir/data`` directory in
    the ``*_merged.tif`` / ``*.mask.tif`` convention. Split membership is written as
    ``out_dir/splits/{train,val,test}.txt`` (one chip stem per line), which is exactly what
    the model card's ``burn_scars_config.yaml`` consumes: point all three ``*_data_root`` at
    ``out_dir/data`` and the ``*_split`` at these files. Returns the chip count per split.
    Needs rasterio for the writes; the split and chipping are pure.
    """
    import json
    import shutil
    from pathlib import Path

    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    splits_dir = out_dir / "splits"
    # Clear any prior export so stale chips from an earlier run cannot linger in data/
    # (the split files get overwritten, but orphaned chip GeoTIFFs would bloat the dataset).
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(splits_dir, ignore_errors=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    usable = {i: s for i, s in samples_by_id.items() if getattr(s, "is_usable", True)}
    split = grouped_split(list(usable), val_frac=val_frac, test_frac=test_frac, seed=seed)
    counts: dict[str, int] = {}
    manifest: dict[str, dict] = {}
    for split_name, ids in split.items():
        stems: list[str] = []
        for eid in ids:
            s = usable[eid]
            # balance only the training split; val/test stay a faithful uniform tiling so the
            # score reflects real per-fire coverage, not an oversampled burn subset.
            balance = burn_balance and split_name == "train"
            chips = chip_sample(s, chip=chip, min_valid_frac=min_valid_frac,
                                burn_balance=balance, max_bg_ratio=max_bg_ratio, seed=seed)
            safe = eid.replace(":", "_").replace("/", "_")
            written = write_chip_geotiffs(chips, data_dir, safe)
            H, W = s.reference.shape
            for ch, stem in zip(chips, written, strict=True):
                manifest[stem] = {"event_id": eid, "y0": ch["y0"], "x0": ch["x0"],
                                  "H": int(H), "W": int(W)}
            stems.extend(written)
        (splits_dir / f"{split_name}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
        counts[split_name] = len(stems)
    (out_dir / "_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    # The manifest maps each chip stem back to its fire and pixel offset, so per-chip
    # predictions can be reassembled into per-fire masks for the fire-level skill scoring.
    (out_dir / "_chips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts


def stitch_chip_predictions(pred_by_stem: dict, manifest: dict) -> dict:
    """Reassemble per-chip predicted masks into per-fire masks using the export manifest.

    terratorch predicts one mask per chip; the fire-level skill metric needs one mask per
    fire. ``pred_by_stem`` maps a chip stem (``{fire}_{i}``) to its predicted ``[h, w]``
    burned mask; ``manifest`` is the ``_chips.json`` written by :func:`export_prithvi_chips`
    (stem -> event_id, y0, x0, H, W). Each chip is placed back at its pixel offset in the
    fire's grid (clipped to the original H x W, since edge chips overhang the pad), and
    overlaps take the burned-wins maximum. Returns ``{event_id: mask[H, W]}`` (uint8). Pure.
    """
    canvases: dict[str, np.ndarray] = {}
    for stem, pred in pred_by_stem.items():
        meta = manifest.get(stem)
        if meta is None:
            continue
        eid = meta["event_id"]
        H, W = int(meta["H"]), int(meta["W"])
        if eid not in canvases:
            canvases[eid] = np.zeros((H, W), dtype=np.uint8)
        y0, x0 = int(meta["y0"]), int(meta["x0"])
        p = (np.asarray(pred) > 0).astype(np.uint8)
        h = min(p.shape[0], H - y0)
        w = min(p.shape[1], W - x0)
        if h <= 0 or w <= 0:
            continue
        sl = canvases[eid][y0:y0 + h, x0:x0 + w]
        np.maximum(sl, p[:h, :w], out=sl)
    return canvases


@dataclass(frozen=True, slots=True)
class PrithviScore:
    event_id: str
    f1: float
    iou: float
    naive_f1: float

    @property
    def skill_f1(self) -> float:
        return self.f1 - self.naive_f1


def score_masks(pred_by_event: dict, samples_by_id: dict) -> list[PrithviScore]:
    """Skill-over-naive per held-out fire from predicted burned masks.

    ``pred_by_event`` maps event id to a boolean/0-1 predicted burned mask on the sample's
    grid. For each fire, F1/IoU are scored on the valid pixels against the MTBS reference,
    with the predict-all-burned naive F1 on the same pixels, exactly as the U-Net and RBR
    are scored (``eval.metrics.confusion_counts``), so the three are directly comparable.
    Pure numpy.
    """
    from vhagar.eval.metrics import confusion_counts

    out: list[PrithviScore] = []
    for eid, pred in pred_by_event.items():
        s = samples_by_id.get(eid)
        if s is None:
            continue
        v = np.asarray(s.valid, dtype=bool)
        if not v.any():
            continue
        truth = np.asarray(s.reference, dtype=bool)[v].astype(np.uint8)
        p = (np.asarray(pred) > 0)[v].astype(np.uint8)
        cc = confusion_counts(truth, p)
        naive = confusion_counts(truth, np.ones_like(truth))
        out.append(PrithviScore(event_id=eid, f1=float(cc.f1), iou=float(cc.iou),
                                naive_f1=float(naive.f1)))
    return out


def export_inference_chips(
    samples_by_id: dict, out_dir, chip: int = 224, min_valid_frac: float = 0.1,
) -> int:
    """Chip every sample into one flat dataset for inference (no train/val/test split).

    For the transfer test: chip a held-out cohort (e.g. the European fires) so a model
    trained elsewhere can predict them. Writes ``out_dir/data`` in the ``*_merged.tif`` /
    ``*.mask.tif`` convention, an ``out_dir/splits/all.txt``, and ``out_dir/_chips.json``
    (stem -> event_id, y0, x0, H, W) so per-chip predictions stitch back per fire. Returns the
    chip count. Needs rasterio for the writes.
    """
    import json
    import shutil
    from pathlib import Path

    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    splits_dir = out_dir / "splits"
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(splits_dir, ignore_errors=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    usable = {i: s for i, s in samples_by_id.items() if getattr(s, "is_usable", True)}
    stems: list[str] = []
    manifest: dict[str, dict] = {}
    for eid, s in usable.items():
        chips = chip_sample(s, chip=chip, min_valid_frac=min_valid_frac)
        safe = eid.replace(":", "_").replace("/", "_")
        written = write_chip_geotiffs(chips, data_dir, safe)
        H, W = s.reference.shape
        for ch, stem in zip(chips, written, strict=True):
            manifest[stem] = {"event_id": eid, "y0": ch["y0"], "x0": ch["x0"],
                              "H": int(H), "W": int(W)}
        stems.extend(written)
    (splits_dir / "all.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
    (out_dir / "_chips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(stems)


def _post_nbr(features: np.ndarray) -> np.ndarray:
    """Post-fire NBR from the six Prithvi bands: (NIR_narrow - SWIR2) / (NIR_narrow + SWIR2).

    Band order is :data:`vhagar.io.optical.PRITHVI_BAND_ASSETS` = (blue, green, red, nir08,
    swir16, swir22), so NIR-narrow is index 3 and SWIR2 is index 5. Burned ground has low NBR.
    """
    nir, swir = features[3].astype(np.float64), features[5].astype(np.float64)
    denom = nir + swir
    return np.where(np.abs(denom) > 1e-6, (nir - swir) / denom, np.nan)


def nbr_threshold_baseline(
    samples_by_id: dict, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.15,
) -> tuple[list[PrithviScore], float]:
    """Same-fire baseline: a post-fire NBR threshold scored like Prithvi, on the same split.

    Fits a single NBR cut (burned where ``NBR <= t``) by maximising F1 over the *train* fires'
    valid pixels, then scores each *test* fire with the identical skill-over-naive metric as
    :func:`score_masks`. Because the split, the fires, the reference and the metric are the
    same, the returned skills are directly comparable to the Prithvi predictions: does the
    foundation model beat a pointwise spectral threshold on these exact fires? Returns
    ``(per-fire scores, threshold)``. Pure numpy.
    """
    split = grouped_split(list(samples_by_id), val_frac=val_frac, test_frac=test_frac, seed=seed)
    thr = _tune_nbr([samples_by_id[e] for e in split["train"]])
    return _score_nbr([samples_by_id[e] for e in split["test"]], thr), thr


def _tune_nbr(train_samples: Sequence) -> float:
    """Fit the F1-optimal post-fire NBR cut (burned where NBR <= t) on train pixels."""
    from vhagar.eval.metrics import confusion_counts  # noqa: F401  (kept for symmetry)

    vals, labs = [], []
    for s in train_samples:
        v = np.asarray(s.valid, dtype=bool)
        nbr = _post_nbr(s.features)[v]
        m = np.isfinite(nbr)
        vals.append(nbr[m])
        labs.append(np.asarray(s.reference, dtype=bool)[v][m])
    nbr_all = np.concatenate(vals)
    y_all = np.concatenate(labs)
    best_t, best_f1 = float(np.nanmedian(nbr_all)), -1.0
    for t in np.quantile(nbr_all, np.linspace(0.02, 0.98, 60)):
        pred = nbr_all <= t
        tp = int(np.sum(pred & y_all))
        fp = int(np.sum(pred & ~y_all))
        fn = int(np.sum(~pred & y_all))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _score_nbr(test_samples: Sequence, thr: float) -> list[PrithviScore]:
    """Score a fixed NBR cut on each test fire with the skill-over-naive metric."""
    from vhagar.eval.metrics import confusion_counts

    out: list[PrithviScore] = []
    for s in test_samples:
        v = np.asarray(s.valid, dtype=bool)
        nbr = _post_nbr(s.features)
        pred = (np.nan_to_num(nbr, nan=1.0) <= thr)[v].astype(np.uint8)  # nodata -> unburned
        truth = np.asarray(s.reference, dtype=bool)[v].astype(np.uint8)
        cc = confusion_counts(truth, pred)
        naive = confusion_counts(truth, np.ones_like(truth))
        out.append(PrithviScore(getattr(s, "event_id", "?"), float(cc.f1), float(cc.iou),
                                float(naive.f1)))
    return out


def nbr_threshold_transfer(train_samples: Sequence, test_samples: Sequence) -> tuple[list[PrithviScore], float]:
    """Transfer baseline: tune the NBR cut on ``train_samples`` (e.g. CONUS), score
    ``test_samples`` (e.g. Europe). The spectral-threshold analogue of a model trained on one
    continent and applied to another, for the leave-one-continent-out comparison. Pure numpy.
    """
    thr = _tune_nbr(train_samples)
    return _score_nbr(test_samples, thr), thr


def summarise_scores(scores: Sequence[PrithviScore]) -> dict:
    """Mean skill over naive for the Prithvi predictions across the held-out fires."""
    if not scores:
        return {"fires": 0}
    skill = np.array([s.skill_f1 for s in scores])
    return {
        "fires": len(scores),
        "f1_mean": float(np.mean([s.f1 for s in scores])),
        "iou_mean": float(np.mean([s.iou for s in scores])),
        "skill_mean": float(skill.mean()),
        "fires_positive_skill": int(np.sum(skill > 0)),
    }
