# Running the Prithvi fine-tune on Google Colab

This is the operational runbook for the one part of the T2 experiment that needs a GPU:
fine-tuning Prithvi-EO-2.0-300M and running inference. Everything else (building the six-band
dataset, exporting chips, and the leakage-safe head-to-head scoring) runs on CPU and is better
done on your own machine. The split of labor:

| Step | Where | Why |
|------|-------|-----|
| `t2-prithvi-build`, `t2-prithvi-export` | Local (CPU + network) | Needs your `registry.parquet` + MTBS mosaic and the Sentinel-2 network, no GPU. |
| `terratorch fit` + inference | **Colab (GPU)** | The only GPU-bound step. |
| `t2-headtohead` (scoring) | Local (CPU) | Pure numpy; compares Prithvi vs U-Net vs RBR. |

So the flow is: **build + export locally → move the chip dataset to Colab → fine-tune + predict
on Colab → bring predictions back → score locally.**

## 0. Prepare inputs locally

```
vhagar t2-prithvi-build  --registry registry.parquet --mosaic mtbs_CONUS_2021.tif \
    --region conus --year 2021 --max-fires 20 --res-m 30 --cache-dir data/t2_prithvi
vhagar t2-prithvi-export --cache-dir data/t2_prithvi --out-dir data/t2_prithvi_chips --chip 224
```

You now have `data/t2_prithvi_chips/` (the chips terratorch trains on, plus `_split.json` and
`_chips.json`) and `prithvi_burnscars_vhagar.yaml` at the repo root. Put these three where Colab
can read them. The simplest path is Google Drive: copy `data/t2_prithvi_chips/` and
`prithvi_burnscars_vhagar.yaml` into a Drive folder, e.g. `MyDrive/vhagar/`.

## 1. Colab: enable the GPU

Open a new notebook at colab.research.google.com. **Runtime → Change runtime type → T4 GPU
→ Save.** Confirm with:

```python
!nvidia-smi
```

You want to see a T4 (or better). The free T4 is enough for 20-fire fine-tunes; large runs may
need Colab Pro for longer sessions and less frequent disconnects.

## 2. Colab: mount Drive and stage the dataset

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/work && cp -r /content/drive/MyDrive/vhagar/t2_prithvi_chips /content/work/data_chips
!cp /content/drive/MyDrive/vhagar/prithvi_burnscars_vhagar.yaml /content/work/
!ls /content/work/data_chips
```

The config's data roots are relative (`data/t2_prithvi_chips/...`). Either recreate that layout
or edit the four `*_data_root` / `*_split` paths in the YAML to point at
`/content/work/data_chips`. Recreating the layout is less error-prone:

```python
!mkdir -p /content/work/data/t2_prithvi_chips && \
 cp -r /content/work/data_chips/* /content/work/data/t2_prithvi_chips/
%cd /content/work
```

## 3. Colab: install TerraTorch and fetch weights

```python
!pip install -q terratorch
```

The pretrained backbone (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M`) is pulled from Hugging Face
automatically by `backbone_pretrained: true` on the first run. If the download is rate-limited,
authenticate:

```python
from huggingface_hub import login
login()   # paste a read token from huggingface.co/settings/tokens
```

## 4. Colab: fine-tune

```python
!terratorch fit -c prithvi_burnscars_vhagar.yaml
```

Checkpoints land under `lightning_logs/version_*/checkpoints/`. Training early-stops on
`val/loss` (patience 15, max 100 epochs); on a T4 with ~470 training chips this is roughly
20–40 minutes. If it errors on a datamodule argument, that is a TerraTorch version-schema drift
(an arg rename in `GenericNonGeoSegmentationDataModule`); `terratorch fit --help` shows the
current names. The band, decoder, and loss arguments are from the published model card and
should not need changing.

Note the best checkpoint path:

```python
import glob
ckpt = sorted(glob.glob('lightning_logs/version_*/checkpoints/*.ckpt'))[-1]
print(ckpt)
```

## 5. Colab: predict the test chips

Run inference over the held-out chips and write one GeoTIFF per chip:

```python
!terratorch predict -c prithvi_burnscars_vhagar.yaml --ckpt_path {ckpt} \
    --predict_output_dir /content/work/preds \
    --data.init_args.predict_data_root  /content/work/data/t2_prithvi_chips/data \
    --data.init_args.predict_split      /content/work/data/t2_prithvi_chips/splits/test.txt
!ls /content/work/preds | head
```

`predict` writes one mask per **chip**, not per fire. That is fine: the scorer stitches chips
back per fire using `_chips.json`. One thing to check before leaving Colab, the scorer matches a
prediction file to a chip by reducing its stem to the chip stem, tolerating a single trailing
suffix (`{stem}_pred`, `{stem}_merged`). Look at the actual output names:

```python
!ls /content/work/preds | head -3
```

If they are `{chipstem}_pred.tif` you are set. If TerraTorch stacked two suffixes (e.g.
`{chipstem}_merged_pred.tif`), rename so only the chip stem remains before one suffix:

```python
import os, re
for f in os.listdir('/content/work/preds'):
    new = re.sub(r'_merged(_pred)?\.tif$', '.tif', f)   # -> {chipstem}.tif
    if new != f: os.rename(f'/content/work/preds/{f}', f'/content/work/preds/{new}')
```

## 6. Bring predictions back

```python
!cp -r /content/work/preds /content/drive/MyDrive/vhagar/prithvi_preds
```

Then on your machine, sync the Drive folder (or download it) so you have `data/prithvi_preds/`.

## 7. Local: the decisive, leakage-safe comparison

One command, back in the repo:

```
vhagar t2-headtohead --cache-dir data/t2_prithvi --pred-dir data/prithvi_preds \
    --chips-manifest data/t2_prithvi_chips/_chips.json \
    --split data/t2_prithvi_chips/_split.json --out-json h2h_report.json
```

It trains the RBR threshold and the U-Net on the split's train+val fires, scores all three
models on its exact test fires, and prints each model's mean per-fire skill plus paired-bootstrap
differences (mean, CI, P(a>b), and whether the CI excludes zero). The split is authoritative and
any in-sample Prithvi mask is rejected, so the printed margins cannot be leakage artefacts. The
question it answers: does the foundation model beat the U-Net's +0.54 skill on the same held-out
fires, and does it degrade less out of region on the leave-one-continent-out transfer?

## Gotchas specific to Colab

- **Disconnects.** Free Colab reclaims idle sessions and caps runtime. Checkpoint to Drive
  (`trainer.default_root_dir` or copy `lightning_logs/` to Drive periodically) so a disconnect
  mid-fine-tune is resumable with `--ckpt_path`.
- **Reflectance scaling.** The config's `means`/`stds` assume Sentinel-2 L2A surface reflectance
  in 0–1. If your chips are in 0–10000 DN, they will not match; the export writes 0–1, so this is
  only a risk if you re-tile from a different source.
- **CPU steps in Colab.** You *can* run `t2-prithvi-build`/`export` and even `t2-headtohead` in
  Colab too (install the repo with `pip install -e .`), but the build needs your registry + MTBS
  files uploaded, which is usually more hassle than running those locally.
- **Version drift.** TerraTorch moves fast. If `fit` or `predict` argument names differ from the
  above, trust `--help` over this doc; the model/data *values* are stable, only the CLI plumbing
  drifts.
