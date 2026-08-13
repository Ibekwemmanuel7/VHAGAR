# VHAGAR: Should You Train Your Own Foundation Model?

**Short answer: yes, but a ~90M-parameter thermal-native one for under $25k,
in stages, and only after a tuned supervised baseline earns it. Not a
300-600M-parameter GeoFM.**

---

## 0. The number that should govern your budget

There is exactly one published foundation model pretrained on geostationary
MIR/TIR data. **HighFM**, a ~90M ViT-B/4 pretrained on 2.23 TB of MSG SEVIRI
(11 bands, 15-min, Mediterranean fire seasons 2014-19, 13.6 M 32×32 patches).
It was built for essentially your use case, and it works:

**Active fire detection, cross-entropy loss:**

| Model | Balanced acc. | Fire IoU | Fire recall |
|---|---|---|---|
| U-Net from scratch | 0.834 | 0.022 | 0.802 |
| ViT-B/4 from scratch | 0.883 | 0.048 | 0.829 |
| ViT-B/4 ImageNet | 0.856 | 0.049 | 0.769 |
| Copernicus-FM | 0.894 | 0.053 | 0.845 |
| **HighFM-ST** | 0.917 | 0.066 | 0.881 |
| **HighFM-MT** (multi-timestep) | **0.925** | **0.079** | **0.890** |

**Same task, Dice loss:**

| Model | Balanced acc. | Fire IoU | Fire recall |
|---|---|---|---|
| U-Net from scratch | 0.684 | **0.272** | 0.369 |
| ViT-B/4 from scratch | 0.709 | **0.286** | 0.419 |
| Copernicus-FM | 0.717 | **0.313** | 0.436 |

Read those two tables together.

**In-domain thermal pretraining genuinely helps**. HighFM-MT beats
Copernicus-FM by +3.1 pts balanced accuracy and +49 % relative fire IoU, and
multi-timestep beats single-timestep on every fire metric.

**And switching the loss function from cross-entropy to Dice took a
from-scratch U-Net's fire IoU from 0.022 to 0.272, a 12× gain that dwarfs
everything pretraining bought anyone.**

That asymmetry is the whole message. Fix your loss and your class-imbalance
handling before you spend a dollar on GPUs.

> ⚠️ **Unverified and decision-relevant:** HighFM's Dice-loss row was not
> extractable from the rendered paper, so **whether in-domain pretraining still
> leads under the loss you would actually deploy is unknown.** Read that table
> before committing to Stage 2. HighFM's pretraining compute is also unpublished.

---

## 1. What existing GeoFMs cost

| Model | Params | Pretraining data | Hardware | GPU-hours | Est. rental |
|---|---|---|---|---|---|
| Prithvi-EO-2.0-300M | ~300M | 4.2M HLS samples | 80× A100-40GB | ~21,000 | ~$17k-42k |
| Prithvi-EO-2.0-600M | ~600M | same, 400 epochs | 240× A100 | ~58,000 | ~$46k-115k |
| TerraMind-1.0-B | ViT-B | 9M samples, 500B tokens | 32× A100, 6 d | ~4,600 | ~$4k-9k |
| TerraMind, **all experiments** |. |. | A100 | **~30,000** | ~$24k-60k |
| Clay v1.5 | 632M | 70M chips | 160× L4 | ~128,000 | ~$50k-210k; 15.1 t CO₂e |
| **Galileo-Base** | **85M** | 127k instances | **1× H100** | **573** | **~$1-1.7k** |
| Galileo-Tiny / Nano | 5.3M / 0.8M | same | 1× H100 | 259 / 200 | ~$300-400 |
| **Presto** | **0.4M** | 21.5M pixel-timeseries | **1× V100, 43 hr** | 43 | **~$25-60** |
| TESSERA | 40M enc. | 0.8B d-pixels, >2 TB | 8× MI300X | ~3,000/epoch | ~$6k-12k |
| AnySat | shared | 171B pixels | H100 | 1,760 | ~$2.6k-5.3k |
| GAIA (15 yr merged GEO IR) | ~250M | GOES+Himawari+Meteosat 10.3 µm | 40× A10, >2 wk | ~13,400 | ~$6k-13k |
| **HighFM** | **~90M** | **2.23 TB SEVIRI** | not reported |. | *my estimate: 1k-4k A100-hr, $1k-8k* |

**The cost spread is four orders of magnitude and performance does not track
cost.** Galileo-Base (85M, 573 H100-hr, ~$1.5k) beats SatMAE, CROMA and DOFA on
eleven benchmarks. Presto reaches 95.3 % on EuroSat-MS against SatMAE ViT-L's
99.0 % with **765× fewer parameters**. The two most expensive models in the
table are not the two best.

⚠️ Note the TerraMind arithmetic: ~30,000 GPU-hours covers *all* pretraining
experiments, not one run. **Budget a 3-6× multiplier over your single-run
estimate.**

---

## 2. Does pretraining beat supervised-from-scratch at all?

Evenly: often not, and the fire benchmark is one of the cases where it doesn't.

- **PANGAEA** evaluated 9 GeoFMs against a plain U-Net. The **U-Net was top-2 on
  more datasets (6) than any GeoFM** (CROMA, Scale-MAE: 3 each). Directly on
  point: **on HLS Burn Scars, U-Net scored 84.51 % mIoU vs Prithvi's 83.62 %**. 
  the flagship fire benchmark Prithvi was pretrained in-domain for is won by a
  from-scratch U-Net.
- **Specialized Foundation Models Struggle to Beat Supervised Baselines**: a
  NAS-tuned CNN matched DOFA-Large within 1 point using 3-10× fewer parameters
  and **zero pretraining data**. Their charge, that FM papers benchmark FMs
  only against other FMs, is fair.
- **Curation beats volume.** Pruning **75 % of a 2M-image Sentinel-1 corpus
  improved accuracy** (ViT-B/8: 80.5 % → 83.6 %) while saving >40 % compute.
- **Early saturation.** At fixed compute, 13M images × 67 epochs *underperformed*
  3-4M images × 200-400 epochs.

Where GeoFMs do reliably win: the **low-label regime** (≈10 % of labels), and
**multi-task encoder reuse**.

**Practical crossover:** benefits appear at ≥1M in-domain samples, <~5k
annotated scenes, and ≥2 downstream tasks. Below ~100k pretraining samples,
pretraining is noise. Above ~5-10M, returns flatten hard. Your sweet spot is
**1M-20M patches**, exactly HighFM's regime.

---

## 3. Does optical pretraining transfer to 3.9 µm physics?

**Weakly negative for large gains, weakly positive for a floor.**

- In HighFM, **ImageNet pretraining was *worse* than from-scratch ViT** on fire
  balanced accuracy (0.856 vs 0.883).
- Copernicus-FM and Panopticon (Sentinel optical/SAR pretrained with spectral
  hypernetworks) beat from-scratch but lost decisively to thermal-native HighFM.
- Conversely **thermal→thermal transfers well**: Landsat-7 thermal → FOREST-2
  CubeSat achieved **79.6 % macro-F1 zero-shot** across very different sensors,
  and joint training with only ~500 thermal CubeSat crops beat the CubeSat-only
  baseline (0.877 vs 0.850 F1).

So: **thermal→thermal transfers; optical→thermal gives a generic-vision floor
and nothing physics-specific.** This is the empirical case for training your
own rather than fine-tuning Prithvi/TerraMind for the *active-fire* task. 
which is what you asked, and the answer is yes for T1, no for T2.

---

## 4. The objective: why not plain MAE

Your signal is not texture. At 2 km GOES / 3 km SEVIRI a fire is often **a
single pixel whose 3.9 µm brightness temperature departs from its own recent and
climatological diurnal baseline**, with positives <0.4 % of pixels. An MAE can
achieve near-perfect reconstruction MSE while being blind to a 20 K single-pixel
anomaly.

**The objective I would build: masked clear-sky forecasting.**

> Given frames t−k…t−1 plus timestamp, geolocation, solar zenith and
> day-of-year, **predict the clear-sky brightness temperature field at time t.**
> Fire is then, by construction, the large positive residual at 3.9 µm.

This is a learned, per-pixel, per-hour, per-season replacement for the
hand-engineered contextual thresholds in the operational algorithms, and
crucially **the pretext task and the operational task share the same physics**,
so transfer is not a leap of faith. It is also exactly what
`TemporalAnomalyNet` already does at small scale in
`vhagar/models/segmentation.py`. Withhold known-fire pixels from the forecast
loss so the model learns the *baseline*, not the fires.

Proposed batch mixture:

| Share | Objective |
|---|---|
| 50 % | masked clear-sky forecasting (Huber loss, fire pixels downweighted) |
| 30 % | tube masking, spatial patches masked across all timesteps, 75 % ratio |
| 20 % | channel masking, hide 3.9 µm entirely, reconstruct from TIR + geometry |

Use a **mixture** of structured masks, not uniform random. Presto and Galileo
both found this materially more stable, with Galileo reporting reduced training
collapse. **Do not use pure DINO/contrastive**: it optimises for scene-level
semantics, the opposite of single-pixel anomaly detection.

Also worth copying: **TESSERA's per-pixel temporal encoder** can beat image-patch
ViTs when the phenomenon is temporal rather than spatial, and it demonstrated
clear embedding separation for burned vs unburned pixels.

---

## 5. The staged plan

| Stage | Model | Data | Hardware | Wall clock | Cost |
|---|---|---|---|---|---|
| **0. Baseline** | GBM + tuned U-Net on `physics_features` | labelled only | 1× L40S / Colab Pro+ | 1-3 d | **$50-300** |
| **1. Pilot FM** | 10-30M ViT-B/4 | 2-5M GOES CONUS patches | 8× A100 | 1-3 d | **$400-1,500** |
| **2. Production FM** | 85-100M ViT-B/4, multi-sensor | 10-25M patches | 8× A100/H100 | 5-14 d | **$3k-20k** |
| **3. Scale-up** | 300-600M | 50M+ patches | 32-80× H100 | 10-20 d | **$40k-200k**. *don't* |

**Stage 0 (weeks 1-4).** Build the operational baseline: GOES ABI
C07/C11/C13/C14/C15 plus `vhagar.features.physics_features` → tuned GBM or
U-Net, with **Dice/Tversky/focal loss and explicit class-imbalance handling**.
Freeze the event-and-time-blocked split now. This is your bar and quite possibly
your product.

Separately, fine-tune **Prithvi-EO-2.0-600M-TL** for the *optical burn-scar*
task (T2), where it is already benchmarked at 83.2 % IoU. Free performance, no
training risk, and it does not compete with your thermal work.

**Stage 1 (weeks 5-10, ~$1,500).** 10-30M ViT-B/4 on 2-5M curated GOES CONUS
patches with the forecasting + tube-masking objective, 8× A100 for 2-3 days.
**Gate: does it beat Stage 0 by ≥3 pts PR-AUC and ≥2 min on time-to-detection,
on the frozen split?** HighFM's +4.2 pt balanced-accuracy gain over from-scratch
ViT is the plausible upper bound. If you see <1 point, stop and bank Stage 0.

**Stage 2 (months 3-5, $5k-20k).** Only if Stage 1 passes. ~90M params, 10-25M
patches, multi-sensor (ABI + SEVIRI + FCI) with wavelength-conditioned
embeddings and **viewing-geometry conditioning** (off-nadir GEO pixels are 3-5×
larger, see `physics.geometry.pixel_area_growth`). Multi-timestep, always.
Expect the encoder to pay for itself across detect / burned-area / spread rather
than on detection alone.

**Stage 3, don't.** Conditions that would flip it: >50M genuinely diverse
curated patches *and* a still-rising loss-vs-scale curve at Stage 2; ≥5 distinct
downstream products; a partner supplying compute (JUWELS-class, ESA Φ-lab, NASA
IMPACT) so marginal cost ≈ 0; or someone publishing a thermal scaling law
showing gains beyond 100M. None currently hold.

### Corpus construction

Target **10-25M patches of 32×32 to 64×64** at native 2 km, 3-5 fire seasons
over CONUS + Canada + Mediterranean. In priority order:

1. **Aggressive de-redundancy.** Consecutive 5-min frames are ~99 % identical.
   Hierarchical k-means in feature space, prune 50-75 %. Expect this to *improve*
   accuracy while halving compute.
2. **Stratify by biome × month × local solar hour.** Cover the full diurnal
   cycle including night, where MIR discrimination is easiest and where
   optical-derived GeoFMs are useless.
3. **Oversample fire-adjacent tiles 20-50×** but keep a large clean-negative
   pool, hot desert, solar farms, gas flares, sun glint and cloud edges are the
   operational failure mode and must be in pretraining.
4. **Keep cloudy patches** (unlike HighFM, which discarded them). Cloud and
   smoke occlusion is the deployment condition; cloud also makes a good natural
   mask augmentation.
5. **Store int16 brightness temperature** with per-band scale/offset. Never
   float32, you would double your storage for no precision that matters.

### Architecture

- ViT-B class, 85-100M, **patch 4×4** on 2-3 km data. 16×16 destroys
  single-pixel fires.
- **Wavelength-conditioned dynamic patch embedding** (DOFA / Copernicus-FM
  style) so one encoder ingests SEVIRI-11, ABI-16, FCI-16, VIIRS-M and SLSTR.
  Plus a sensor-ID embedding and a **viewing-geometry embedding**.
- Fine-grained timestamps, **local solar time**, solar zenith, day-of-year
  Fourier features. The diurnal cycle is the dominant confound, give the model
  the phase directly rather than making it infer it.
- 4-8 timesteps at 10-30 min spacing.

---

## 6. Training engineering

**Parallelism.** At ≤600M, **FSDP `SHARD_GRAD_OP`** (≈ ZeRO-2) is enough and
simpler than DeepSpeed. **bf16, not fp16**, brightness-temperature dynamic
range plus MSE makes fp16 loss-scaling fragile. FlashAttention-2/3,
`torch.compile`. Activation checkpointing only if memory-bound; with 64-256
token sequences you usually aren't, and it costs ~30 % throughput.

**Data loading is the real bottleneck.** Each 32×32 patch-4 sample is ~10 KB of
compute; the GPU will starve on NetCDF. Convert offline to **WebDataset tar
shards of 1-2 GB** (or sharded Zarr v3) of pre-cropped pre-normalised int16
tensors, stage to **local NVMe** not S3, 8-16 workers/GPU, `persistent_workers`,
prefetch 4. Run in **us-east-1** so GOES pulls are free.

**Throughput reference.** Standard MAE ViT-B at 224² is ~107 hr on 8× A100 per
1M-image run (≈856 GPU-hr). Your 64-token sequences should run ~3× faster per
sample, so **1,500-4,000 samples/s on 8× A100** if IO keeps up, a 20M-patch ×
200-epoch budget is ~10 days on one node. **Verify with a 1-hour throughput
probe before committing.**

**Checkpointing.** Every ~30 min, keep last 3 + best, store optimizer state
(`SHARDED_STATE_DICT`), include RNG and dataloader shard position so resume is
bit-reproducible. Spot instances halve cost only if resume actually works. 
test it deliberately on day one.

### How to avoid the classic failure

The failure mode is shipping a foundation model that never beats its supervised
baseline, and discovering that at the end. Prevent it structurally:

1. **Build and tune the supervised baseline first**, loss function included.
   The HighFM tables prove the loss is worth more than the encoder.
2. **Freeze the evaluation protocol before pretraining starts.**
3. **Split by fire event and by time, never randomly.** At 5-min cadence a
   random split leaks the *same fire* between train and test and will inflate
   metrics by tens of points. VHAGAR's `eval.splits` makes this structural.
4. **Track a "pretraining premium"**. Δ over baseline at matched fine-tuning
   budget, at every checkpoint. If it is <1 point at 25 % of planned compute,
   stop.

### Evaluating the encoder

**Linear probe is actively misleading here.** It measures linear separability of
pooled global features; your task is sparse per-pixel anomaly detection with
<0.4 % positives. A probe can score well while the model has no localisation
ability, and it systematically favours contrastive encoders over MAE encoders
whose useful information sits in intermediate patch tokens.

Use instead: attentive/multi-layer probe with a light decoder over frozen
features; **full fine-tune with a UPerNet head** (the number you will deploy);
k-NN only as a cheap sanity check.

**Report operational metrics, not IoU alone:** PR-AUC, **recall at a fixed
false-alarm rate per 10⁶ pixels**, **time-to-first-detection in minutes**,
**minimum detectable FRP in MW**, and per-biome / per-solar-hour breakdowns.
HighFM's 0.079 fire IoU at 0.890 recall is a perfect illustration of why a
single IoU tells you nothing about operational value.

---

## 7. The decision rule

> Train your own thermal encoder **only if** (i) you have ≥1M in-domain thermal
> patches you can curate, (ii) you have <5k labelled fire scenes, (iii) you need
> ≥2 downstream tasks from one encoder, and (iv) your tuned supervised baseline
> has plateaued.
>
> **If any of (i)-(iv) fails, don't.** If all four hold, budget Stage 2, not
> Stage 3.

For VHAGAR specifically: (i) yes. GOES ABI alone gives you more in-domain
thermal data than HighFM used, free, in us-east-1; (ii) yes; (iii) yes. 
detect + burned area + spread; (iv) not yet, which is why Stage 0 comes first.

**And keep Prithvi-EO-2.0-300M-BurnScars for T2.** It reports 87.5 % burned-area
IoU out of the box under Apache-2.0. Training your own optical burn-scar encoder
would be spending money to reproduce something free. The "train my own" decision
is task-specific, not ideological: **thermal from scratch, optical fine-tuned.**
