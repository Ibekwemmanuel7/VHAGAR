# CMIP decoder plan (Tier B radiance)

Status: draft for review, not yet implemented. This is the shape of the work,
so we agree on the interfaces and the open decisions before writing code.

## Why this is the next big thing

Everything on the radiance side is blocked on it. Tier B climatology (the
per-pixel diurnal and persistence baselines), the high-cadence tier, and the
early-detection anomaly model all need calibrated thermal imagery, and none of
it can be built until VHAGAR can read ABI L2 CMIP. It is also the one cost in the
whole archive plan whose wall clock is still completely unmeasured, because
`measure_granule(product="CMIP")` currently times a bare byte fetch with no
decoder attached. So this unblocks the roadmap and closes the last guessed
number in `plan.py`.

## What CMIP is, and the one fact that shapes the decoder

ABI L2 CMIP (Cloud and Moisture Imagery Product) is calibrated, navigated,
per-channel imagery. For the emissive bands VHAGAR uses (C07, C11, C13, C14,
C15, all 2 km), the `CMI` variable is **already brightness temperature in
kelvin**. It is not radiance. That matters:

- We do not need L1b `Rad` plus a radiance-to-BT calibration step. CMIP hands us
  BT directly.
- Where a physics stage wants radiance (Wooster FRP uses MIR radiance), we invert
  with `vhagar.physics.planck.planck_radiance` at the band centre. The planck
  module already warns that monochromatic central-wavelength conversions differ
  from band-integrated radiances by a few kelvin, so anything quantitative should
  eventually use the band spectral response, not the centre. For features and
  climatology, centre-wavelength BT is the right starting point.

The second shaping fact: CMIP for the 2 km thermal set rides the **same ABI
fixed grid as FDC**. So `_fixed_grid_navigation` (the cache from the section 10
work) applies unchanged, and lat/lon/view-zenith/pixel-area are computed once and
shared across every channel and every timestep. This is the single biggest reason
the radiance tier is affordable in wall clock.

The third fact, already in `plan.py`: CMIP ships **one file per channel**. Five
bands is five S3 reads per timestep. FDC was one multiband file. The decoder must
assemble a multi-band stack from N separate channel files that share a timestamp.

## Channels and their centres

| band | centre um | role |
|------|-----------|------|
| C07  | 3.9  | MIR fire channel, the whole reason MIR exists (planck b = 12.3 at 300 K) |
| C11  | 8.4  | cloud phase, part of the split-window context |
| C13  | 10.3 | clean longwave window |
| C14  | 11.2 | longwave window, MIR-minus-TIR partner for C07 |
| C15  | 12.3 | split-window water-vapour correction |

`ABI_BAND_RESOLUTION_M` in `plan.py` already confirms all five are 2 km. C07
minus C14 is the classic contextual fire signal; the split window C13/C14/C15
carries the atmospheric and cloud context.

## Module shape, mirroring the FDC reader

New module `src/vhagar/io/cmip_reader.py`, built to look like
`io/goes_reader.py` so the two are learnable together:

- `CMIPChannel` dataclass: one decoded channel on the grid (band id, centre
  wavelength, scan_start, bt array, dqf array, plus the shared geometry).
- `CMIPStack` dataclass: several channels on one grid at one timestamp, plus the
  shared lat/lon/view-zenith/pixel-area (held once, not per channel).
- `list_cmip_granules(satellite, start, end, channels, domain)`: S3 keys grouped
  by timestamp and channel. Reuses the `ABI-L2-CMIPC` prefix logic already
  sketched in `plan.measure_granule`.
- `open_cmip(key, satellite, bbox)`: fetch and decode one channel file. Crops
  before decoding, exactly like `open_fdc`.
- `open_cmip_stack(keys_by_channel, satellite, bbox)`: read the N channel files
  for one timestamp and stack them on the shared grid.
- `decode_cmip(ds, satellite, bbox)`: the offline-testable core, mirroring
  `decode_fdc`. Reuses `_fixed_grid_navigation`.

Design rules carried over from the FDC reader, unchanged:

- Crop before decode. Convert the bbox to scan-angle limits once and slice.
- Attach geometry at read time (view zenith and true pixel area), because FRP is
  proportional to pixel area and to 1/transmittance and both depend on the angle.
- **Saturation is censoring, not noise.** ABI Ch7 saturates near 400 K. A
  saturated MIR pixel is a real hot source with an unusable value, so it is
  flagged and its BT is not passed downstream as a number, the same way FDC
  handles saturated FRP.
- Fill and bad-DQF pixels become NaN, never a number. `CMI` non-values and DQF
  flags map to NaN so nothing downstream averages nodata into "cold ground".
- Reuse the navigation cache. No new navigation code.

## What it feeds, and therefore how it is stored

Unlike FDC, radiance is **dense**: every pixel carries signal, so it is stored as
int16 rasters, which is what `ArchivePlan(storage="dense")` already models. The
Tier B climatology does not keep every frame. It keeps per-pixel-per-hour
statistics, `mu(pixel, hour)` and `sigma(pixel, hour)`, at 15-minute sampling.
So the pipeline is:

1. decode CMIP stacks over the climatology window,
2. reduce them online into running mean and variance per pixel and per local
   hour (Welford, so we never hold the whole cube),
3. store the reduced statistics, not the frames.

The high-cadence tier keeps true 5-minute frames, but only over a handful of
tiles for one fire season, so its size is small.

## Wall clock, the number we actually close

Once `decode_cmip` exists, update `plan.measure_granule(product="CMIP")` to time
the real decode (fetch, parse, navigate via the warm cache, stack) instead of a
bare byte read, then set `DEFAULT_GRANULE_MB` and a CMIP `seconds_per_granule`
from that measurement on this machine. Until then the radiance wall clock stays
labelled unmeasured, per the honesty rule. The navigation cache should make the
per-granule decode dominated by fetch and parse rather than geometry, the same
way it did for FDC.

## Testing, all offline

Mirror `tests/test_goes_reader.py`. Add a synthetic CMIP fixture next to
`_synthetic_fdc` (a small grid with a `CMI` BT field and a `DQF` field), then:

- decode produces finite geolocation and geometry on the shared grid,
- the navigation cache is reused across channels and timesteps (one miss, many
  hits), which is the whole efficiency argument,
- BT values round-trip through planck within tolerance,
- a saturated Ch7 pixel is censored, not passed through,
- fill and bad-DQF pixels become NaN,
- a multi-channel stack aligns channels that share a timestamp and rejects a
  mismatched grid.

## Staged delivery

1. Single-channel decoder plus fixture and tests. No network.
2. Multi-channel stack aligner and its tests.
3. Update `measure_granule` to decode, measure on this machine, set the CMIP
   granule size and seconds-per-granule in `plan.py` from real numbers.
4. Climatology reducer: Welford mean and variance per pixel and per local hour.
5. Wire a Tier B backfill on top, reusing the manifest and coverage machinery
   from Tier A so radiance coverage is recorded the same way.

Steps 1 to 3 are the core decoder and are the natural first PR. Steps 4 and 5 are
the archive build and can follow.

## Open decisions, need a call before coding

1. **Source product: CMIP CMI (brightness temperature) or L1b Rad (radiance)?**
   **DECIDED: CMIP CMI.** Calibrated and gap-filled, gives BT directly, matches
   what `plan.py` assumes, least work. Radiance is derived from BT via planck
   where FRP needs it. L1b Rad was rejected: a calibration step for no gain on
   the 2 km thermal set.
2. **Monochromatic centre wavelength or band-integrated response?** Start
   monochromatic (centre wavelength) for features and climatology, and revisit
   band-integrated radiances only if a quantitative FRP path needs the few-kelvin
   accuracy. The planck docstring already flags this.
3. **Channel-time alignment tolerance.** The five channel files for one nominal
   timestep can carry slightly different scan-start times. Need a small tolerance
   for grouping them into one stack, analogous to the coverage gap tolerance.
4. **Climatology storage layout.** Where the reduced mu/sigma rasters live and
   how they are partitioned. Likely per tile per local hour, but worth deciding
   alongside step 4 rather than now.

Decision 1 is the pivotal one and blocks the module skeleton. The rest can be
settled as we reach their step.
