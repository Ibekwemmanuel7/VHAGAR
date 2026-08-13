# VHAGAR: Physics-Aware Detection

**How much physics to build into the model, in what order, and what will
actually reduce false alarms versus what is academically appealing.**

---

## 0. The headline

About **70 % of the achievable false-alarm reduction comes from Stages 1 and 2
below, which contain almost no interesting physics-ML.** Build those first and
resist the temptation to skip to Stage 5.

There is, as of 2026, **no published physics-informed neural network for
wildfire detection or FRP retrieval.** The PINN-wildfire intersection in the
literature is about fire *spread* parameter learning, not thermal remote
sensing. Mainstream ML fire detection. Sentinel-2 deep learning frameworks,
Himawari-8 CNNs, self-supervised temporal hotspot detection, is essentially
physics-free image classification.

So the gap is real and worth filling. It also means **there are no published
gain figures to promise**, and anyone quoting them is extrapolating. Structure
the programme so Stages 1-3 deliver value independently of whether Stage 5
works.

---

## 1. The forward physics, briefly

Implemented in `src/vhagar/physics/`, tested against published anchors in
`tests/test_physics.py`.

### 1.1 Why the mid-infrared channel exists

The local logarithmic sensitivity of the Planck function

```
b(λ, T) = ∂ln B / ∂ln T = x·eˣ / (eˣ − 1),    x = c₂ / (λT),  c₂ = 14388 µm·K
```

| λ | T | x | b |
|---|---|---|---|
| 3.9 µm | 300 K | 12.3 | **12.3** |
| 11 µm | 300 K | 4.36 | **4.4** |
| 3.9 µm | 1000 K | 3.69 | 3.8 |

The monochromatic radiance ratio of a 1000 K flame to a 300 K background is
**5616 at 3.9 µm versus 28.6 at 11 µm**, a factor of ~196.

A pixel 0.5 % covered by a 1000 K fire over 300 K background reads **413 K** at
3.9 µm and **307 K** at 11 µm. (Published summaries say ~394 K; that is the
constant-b linearisation, which under-reads by ~19 K because b falls with
temperature. `vhagar.physics.planck` is exact and the tests assert the exact
value, see the note in the module docstring so nobody "fixes" it back.)

### 1.2 The Dozier mixture and the Wooster FRP method

```
L_i = τ_i·[ p·ε_f·B(λ_i,T_f) + (1−p)·ε_b·B(λ_i,T_b) ] + L↑_atm,i + τ_i(1−ε)L↓_atm,i + L_sol,i
```

Because b ≈ 4 over 650-1350 K, the fire temperature and fraction can be
eliminated entirely:

```
FRP = A_pix·σ / (a·τ_MIR) · (L_MIR − L̄_MIR,bg)
```

Accuracy **±12 %** for T_f in 665-1365 K. Fire radiative energy → dry matter at
**0.368 ± 0.015 kg/MJ**. Sensor-specific `a` [×10⁻⁹ W m⁻² sr⁻¹ µm⁻¹ K⁻⁴]:
MODIS Terra 2.96, MODIS C6 operational 3.0, SEVIRI 3.06, GOES-8 Imager 3.07,
BIRD HSRS 3.33.

**Note the structure: `a`, `τ`, `ε` and the background all enter FRP
multiplicatively.** Errors are therefore lognormal. Predict FRP in log space or
predict a *correction factor*, never an additive offset. This is exactly what
`ConstrainedFRPHead` does.

### 1.3 Saturation is censoring, not noise

| Sensor / channel | MIR saturation |
|---|---|
| **SLSTR S7 (3.74 µm)** | **~311 K** → falls back to F1 |
| SEVIRI IR3.9 | ~335 K |
| MODIS band 22 | ~331 K (C6 substitutes band 21, ~500 K) |
| VIIRS M13 | ~650 K (dual gain) |
| VIIRS I4 | ~367 K |
| ABI Ch7 | ~400 K |

Above saturation the truth is known only to be `≥ observed`. A model trained
with an ordinary loss on clipped values systematically under-predicts the
largest fires, the ones that matter. Use `CensoredMSELoss`.

---

## 2. Atmosphere: the largest correctable systematic

**Published anchor:** at nadir, TCWV 20 kg/m², mid-latitude summer, the
radiance-difference-effective SEVIRI MIR transmittance is **0.69**. Using the
band-averaged value (0.74) instead makes BOA radiance ~10 % too low.

So a pipeline ignoring atmospheric correction, as MODIS Collection 6 L2 does,
setting τ = 1, carries a **~31 % low FRP bias at nadir**, needing a ×1.45
correction. And view angle compounds it deterministically:

```
τ(θ_v) ≈ τ(0)^sec(θ_v)
```

| θ_v | τ | correction |
|---|---|---|
| 0° | 0.69 | ×1.45 |
| 60° (routine for SEVIRI over Iberia/Greece) | 0.48 | **×2.1** |
| ~70-75° (disk edge) | ~0.33 | **×3** |

**This is by far the largest single systematic in geostationary FRP, and it is
deterministic and cheap to compute.**

### What `vhagar.physics.atmosphere` actually is

A *physically structured parametric surrogate*. Beer-Lambert in air mass,
curve-of-growth (√W) in water vapour, pinned to two anchors (0.69 at TCWV 20 /
nadir, ~0.92 dry). It is **meant to be replaced** in Stage 4 by a surrogate
fitted to ~10⁵ RTTOV v14 runs.

### RTM options, assessed

| Model | MIR/TIR? | Licence | Differentiable? | Verdict |
|---|---|---|---|---|
| **RTTOV v14.1** | Yes; 3-5 µm treated as mixed thermal+solar | Free to registered users | **Tangent-linear, adjoint, K (Jacobian)** routines | **The right choice.** Coefficients for MODIS, VIIRS, SEVIRI, SLSTR, ABI, FCI; native CAMEL v3 emissivity interface |
| libRadtran | Yes (`source thermal`, DISORT) | GPL | No | Best free full-physics option for building the training set. Slow. |
| MODTRAN 6 | Yes, excellent | Commercial per-seat | No | Reference standard; licensing kills it for open pipelines |
| ARTS 2.6 | Yes, Jacobians | Free | Jacobians, no autodiff | Overkill, MW/IR-sounding oriented |
| **6S / Py6S** | **NO** | Free | No | ⚠️ **Do not use.** Solar code, 0.25-4.0 µm, **no thermal emission source term**. It will run at 3.9 µm and return plausible-looking, physically meaningless numbers. |

**There is no off-the-shelf JAX/PyTorch MIR-TIR fire-band RTM you can drop in
as a layer.** The pragmatic answer: τ_MIR is a smooth function of ~5 variables
(TCWV, sec θ_v, sec θ_s, AOD, surface pressure). Fit a 3-layer MLP or a GP to
~10⁵ RTTOV runs and you hit <0.5 % error, trivially differentiable. That is a
weekend, not a research programme. GPs additionally give closed-form Jacobians
and calibrated uncertainty; NNs are faster at scale.

### What the contextual window already does for you

Upwelling path radiance and reflected downwelling are near-identical for the
fire pixel and its background window, so differencing cancels most of them.
**The contextual background subtraction is already a first-order atmospheric and
emissivity correction.** For 80 % of SEVIRI cases the background-window mean BT
is within 1 K of the central pixel.

It fails exactly where background homogeneity fails: coastlines, desert margins,
topography (pressure and column-water-vapour gradients within the window),
view-angle gradients across a scan, and cloud edges. And it is fragile. 
degrading background characterisation by **only 10 K inflated simulated GOES FRP
by 82 %**.

### Smoke: an honest gap

**There is no published quantitative MIR attenuation coefficient for wildfire
smoke.** Dark sooty smoke measurably affects both 4 µm and 2 µm observations
(2 µm worse), and global FRP inventories may under-estimate in heavy-smoke
regimes. "MIR sees through smoke" is only partly true. VHAGAR treats this as an
unquantified systematic, applies no correction, and records the caveat in
product provenance via `atmosphere.smoke_attenuation_warning()`.

---

## 3. Emissivity: why it belongs in the feature vector

**⚠️ ASTER GED will not do.** It is 100 m and excellent, but **8-12 µm only. 
no MIR coverage at all.** This is the most common gap in fire pipelines.

Use **CAMEL** (Combined ASTER-MODIS Emissivity over Land): 13 hinge points
including **3.6 µm**, extended by PC regression to 417 wavenumbers, 0.05°,
monthly, 2000-2021, algorithm uncertainty 0.01-0.03 emissivity units. RTTOV v14
interfaces to CAMEL v3 natively.

Error propagation, `dBT ≈ (T/b)·(dε/ε)`, at 300 K with dε = 0.01:

| λ | b | dBT |
|---|---|---|
| 3.9 µm | 12.3 | **0.26 K** |
| 11 µm | 4.4 | **0.72 K** |

Uncorrelated across channels (which they are over quartz-rich and mixed
surfaces), that is ~0.8 K of ΔT error per 0.01. MODIS C6's contextual threshold
is `ΔT > mean + 6 K`, so **a 0.05 emissivity error at 11 µm alone consumes
~60 % of the detection margin.** That is the mechanism behind desert-margin
commission errors.

Note that MIR emissivity is genuinely poorly constrained, it cannot be
retrieved by daytime TES because of solar contamination. Pull values from CAMEL
and the ECOSTRESS spectral library directly; do not trust secondary tables.

---

## 4. False-alarm physics: what the model should learn

| Class | Physical signature | Feature that captures it |
|---|---|---|
| **Gas flares** | Colour temperature **~1750 K median** vs biomass **~1000 K**. Wien peak at 1750 K is 1.66 µm → flares are bright at 1.6 and 2.2 µm where a wildfire is orders of magnitude weaker. Compact, persistent, no diurnal phase, no plume. | `swir16_mir_ratio`, `swir22_mir_ratio`, fitted colour temperature, persistence |
| **Industrial heat** | Persistent, geometrically fixed, weekly/shift periodicity, no seasonal fuel dependence | `persistence_count`, `persistence_days`, day-of-week Fourier |
| **Solar farms / specular reflectors** | **Not thermal at all**, reflected solar. Peaks at glint geometry, MIR rises with *no* corresponding 11 µm rise, vanishes at night | **`glint_angle_deg`**, `is_night`, `mir_tir_excess_ratio` |
| **Sun glint over water** | Same geometric signature. MODIS C6 rejects at glint < 2°, and at < 10° with high VIS/SWIR reflectance | **`glint_angle_deg`** |
| **Hot bare desert soil** | Real, whole-pixel, therefore **b-consistent**: raises MIR and TIR *together*, so ΔT stays modest. Strong diurnal phase locked to solar noon plus thermal-inertia lag | `mir_tir_excess_ratio`, `local_solar_time_h`, `ndvi`, `emissivity_*` |
| **Volcanic / geothermal** | Persistent, fixed, often hotter than wildfire with a broad temperature distribution | Static gazetteer + persistence |
| **Agricultural burning** | Genuine fire, but <1 h, low FRP, post-harvest window, field-shaped | Land cover, FRP magnitude, duration |
| **Newly commissioned sources** | **The hardest class.** FIRMS' Static Thermal Anomalies mask flags 2023 cells with ≥5 detections, anything built after the mask epoch reads as wildfire | **Online** persistence estimation, never a frozen mask |
| **Superheated smoke plumes** | Elevated plume radiating at altitude, decoupled from the surface; causes nighttime artefacts | Plume geometry, displacement from parent hotspot |

**Baseline to beat:** MODIS C6 global daytime commission error is **1.2 %**
(down from 2.4 % in C5), with no commission errors found in the analysed
nighttime sample.

---

## 5. The staged plan

### Stage 1: Physics as features + leakage-controlled evaluation
**Weeks. Expected impact: large. Status: implemented.**

`vhagar.features.physics_features` builds 34 features: radiometric contrast,
Planck sensitivity, geometry (solar/view zenith, **glint angle**, relative
azimuth, air mass, pixel area growth), diurnal and seasonal phase, atmospheric
state and transmittance, emissivity, SWIR ratios, causal temporal context.

**Raw coordinates are excluded by construction.** `FORBIDDEN_FEATURES` is
enforced by `assert_no_forbidden_features`, called inside `build_features`, and
covered by a parametrised test.

The justification is the single most decision-relevant published result for
VHAGAR: in a FIRMS wildfire/non-wildfire classification, **raw lat/lon
accounted for 88.9 % of LightGBM split-gain**, and F1 fell **0.985 (random
split) → 0.767 (event-aware) → 0.627 (5° spatial block)**. Dropping coordinates
*raised* spatial-block F1 to **0.818**. Coordinates are a memorisation
shortcut; physics is the transferable substitute.

Glint angle alone addresses solar farms, specular reflectors and water glint.
Emissivity plus NDVI addresses desert margins.

### Stage 2: Temporal and persistence context
**Weeks. Expected impact: large. Status: schema in place, archive needed.**

Per-pixel multi-year detection statistics computed **online**, not from a frozen
mask; diurnal phase; day-of-week periodicity; causal short-window density.

This is the highest-yield discriminant for flares, industrial sources,
volcanoes *and* newly commissioned sources, precisely the class a static
gazetteer cannot cover.

Adopt the normalised temporal anomaly `z = (V − µ_V(x,y,T)) / σ_V(x,y,T)` as an
input feature (`temporal_anomaly_z`): it implicitly absorbs local atmospheric
and emissivity biases. On SEVIRI rapid-scan, the RST-FIRES approach raised
detected events **145 %** (27 vs 11 of 189 incidents) and roughly doubled mean
lead time, **35 → 65 min** ahead of official reporting.

Requires ~80 images/month over 3-5 years of history. **Plan the archive now**. 
this is the long-lead item.

### Stage 3: Multi-band spectral shape
**Months. Expected impact: moderate-to-large for the flare/industrial class.**

SWIR ratios and fitted colour temperature wherever the sensor provides them
(SLSTR S5/S6, ABI C06, FCI). Caveat: SWIR is more smoke-degraded than MIR, so
treat SWIR features as **missing-not-at-random** and give the model an explicit
missingness indicator rather than imputing. 
`physics_features.missingness_indicators` does this.

### Stage 4: Explicit atmospheric correction with a differentiable surrogate
**Months. Small impact on false alarms; large on FRP accuracy. Do not conflate.**

Fit `τ̂_MIR(TCWV, sec θ_v, sec θ_s, AOD, p_s)` to ~10⁵ RTTOV v14 runs, validate
against libRadtran, register it via `TransmittanceModel(backend="table")`.

Then restructure the FRP head as a **constrained correction**, implemented as
`ConstrainedFRPHead`:

```
FRP̂ = A_pix·σ / (a·τ̂) · (L_MIR − L̄_MIR) · g_θ(z),    g_θ > 0, bounded
```

Physical structure is free: monotone in radiance contrast and in 1/τ, positive,
correct area and transmittance scaling. The network learns only what the physics
leaves out, and is initialised at `g = 1` so it starts as pure physics and must
earn every departure. `max_log_correction = ln 3` caps the learned term at a
factor of 3.

### Stage 5: Differentiable forward model in the loop
**6-12 months. Research risk. Implemented as a prototype.**

Encoder → `(p, T_f, T_b)` → **fixed differentiable** Planck-mixture decoder →
radiance reconstruction loss. This is the encoder + physics-decoder autoencoder
that works well in EO retrieval elsewhere (the PROSAIL inversion literature
reports LAI RMSE 0.99 / R² 0.83 vs operational SNAP's 1.24 / 0.71, **trained on
synthetic data only**).

It fits fire *better* than vegetation because the forward model is analytically
simple, a two-component Planck mixture with a transmittance factor. You do not
need a neural emulator to make it differentiable; `PlanckMixtureDecoder` is
about thirty lines.

**Honest expected value:** for *detection and false alarms* this is probably
**operationally marginal**, a well-featured GBM with Stages 1-3 will be close.
The real payoff is sub-pixel characterisation and calibrated uncertainty, and
even there the two-channel information limit does not go away. It becomes
genuinely worthwhile with **≥4 usable thermal/SWIR bands**, and compelling with
hyperspectral (EMIT/EnMAP/PRISMA) or 70 m ECOSTRESS-class data.

---

## 6. Sub-pixel retrieval: read this before promising anything

The Dozier system is two equations in three unknowns once T_b is uncertain, and
even with T_b fixed the Jacobian is near-singular: `p` and `T_f` trade off along
`p·T_f^b ≈ const`. **That is precisely why FRP, which measures that product. 
is well-conditioned while (p, T_f) is not.**

The empirical record is blunt. GOES Dozier fire-area estimates correlated with
ASTER/ETM+ reference at **r = −0.22** (no skill), with fire temperatures biased
low against a 688-1153 K field range. The published conclusion, use FRP, not
(A, T), has been the operational consensus for 15 years. LSA-SAF, MODIS C6,
VIIRS and SLSTR all report FRP and not (p, T_f).

**Has ML been shown to improve the Dozier inversion? No published result
demonstrates it.** ML can supply a better prior and better uncertainty
quantification; it cannot manufacture information two noisy channels do not
contain.

VHAGAR therefore ships the retrieval **with its condition number attached**.
`dozier.retrieve` returns `DozierResult.condition`, and `is_trustworthy()`
gates on convergence, the physical prior (600-1400 K) and conditioning.
Unphysical solutions are returned as NaN rather than as numbers. Do not strip
this, an answer that looks like a number is worse than an admitted failure.

**The real path forward is more bands, not better inversion.** VIIRS Nightfire
already does 9-band Planck-simplex fitting. ECOSTRESS at 70 m reduces `p` by ~2
orders of magnitude, which improves conditioning enormously. Hyperspectral VSWIR
is the frontier.

---

## 7. Explicitly deprioritised

- **PINNs that solve the RTE as a PDE.** No operational advantage over
  DISORT/RTTOV; that literature is a numerical-methods contribution, not a
  retrieval one.
- **6S/Py6S at 3.9 µm.** No thermal source term. Physically meaningless output.
- **MODTRAN**, unless a licence already exists.
- **Chasing (p, T_f) from two channels.** Report FRP; report (p, T_f) only with
  honest posteriors and only from ≥4 bands.

---

## 8. Open items

1. **No quantitative MIR smoke attenuation coefficient exists.** If VHAGAR
   operates in heavy-smoke regimes this is an unquantified systematic, worth a
   dedicated libRadtran + soot-optical-property study.
2. **Per-surface MIR (3.6-4.0 µm) emissivity values**, pull from CAMEL and the
   ECOSTRESS spectral library directly, not from any secondary table.
3. **No prior art for physics-informed ML in fire detection.** Budget genuine
   research risk in Stage 5.
4. The `a` constant for VIIRS, SLSTR, FCI and METimage is **not published**. 
   `frp.wooster_a` warns and falls back to the MODIS C6 value. Re-derive per
   instrument before using FRP quantitatively; it is worth ~10 %.
