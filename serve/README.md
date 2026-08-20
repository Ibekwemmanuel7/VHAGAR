# VHAGAR fire API + console (self-hosted)

Serves VHAGAR's real GOES FDC detections and clustered fire events as GeoJSON,
and hosts `vhagar_console.html` off the same process. No external services, no
Vercel, no Mapbox token.

## Run

```
pip install -r serve/requirements.txt
# the console renders on Mapbox GL; set a free token (mapbox.com/account/access-tokens):
$env:VHAGAR_MAPBOX_TOKEN = "pk...."     # PowerShell; bash: export VHAGAR_MAPBOX_TOKEN=pk....
uvicorn serve.vhagar_api:app --host 127.0.0.1 --port 8000
# then open http://127.0.0.1:8000/console
```

The `/console` route injects `VHAGAR_MAPBOX_TOKEN` into the page. Without it the
console shows a "token needed" panel (the KPIs, feed and danger strip still work).
Basemaps: Satellite (satellite-streets), Dark, Terrain. A per-satellite Sensor
tab row (All / GOES-18 / GOES-19) filters the map and feed.

First request triggers a one-time load: it reads the FDC parquet under
`data/detections/detections` (about 188k detections for the cached Aug 2026
CONUS week) and clusters them into events with VHAGAR's own parallax-aware
fusion. Expect roughly 45 seconds on first hit, instant afterwards (cached for
the process lifetime).

## Endpoints

- `GET /console` the operations console.
- `GET /api/events?region=&days=` clustered fire events as a GeoJSON polygon
  FeatureCollection (convex hull of each cluster's detection pixels).
- `GET /api/detections?region=&days=` FDC detection pixels as GeoJSON points.
- `GET /api/export/geojson?region=&days=` events as a download.
- `GET /api/export/kmz?region=&days=` agency-ready KMZ (Google Earth / GIS) with
  risk-styled event perimeters.
- `GET /api/health` counts and the data window.
- `GET /v1/danger?dryness=&fuel=&wind=&slope=&temp=&rh=&rainfall=&month=` the three
  T3 fire-danger quantities kept separate: FWI (+ class), ignition probability, and
  expected burned area. Models train lazily on VHAGAR's synthetic danger scenarios
  and cache; the console shows this as a "Fire danger, T3 (demo)" strip. Wire real
  fuels / weather / occurrence for operational values.

`region` is one of `california`, `us_west`, `conus`. `days` is 1..14, counted
back from the newest detection in the dataset.

## Deploy free on Render

The repo ships a self-contained snapshot under `serve/demo/`
(`detections.parquet` + `events.pkl`, about 3 MB), so a hosted service starts
instantly: no raw GOES parquet, no 45 s clustering. `VHAGAR_FROZEN=1` forces
that path (it is also used automatically when the raw dataset is absent, the
usual case on a fresh clone or a slim image).

The `render.yaml` at the repo root uses Render's native Python runtime (the
recommended path for a plain FastAPI app: no image build, faster deploys):

1. Push to GitHub (the `serve/demo` snapshot must be committed; it is exempted
   in `.gitignore`).
2. In Render: New > Blueprint, point it at the repo. Render reads `render.yaml`
   and provisions a free web service (`pip install -r serve/requirements.txt`,
   `uvicorn serve.vhagar_api:app`) with a `/api/health` health check.
3. When prompted, paste your Mapbox public token (`pk...`) for
   `VHAGAR_MAPBOX_TOKEN`. It is marked `sync: false`, so it is never stored in
   the repo.
4. After the first deploy, copy the service URL
   (`https://<name>.onrender.com`), add it to that token's allowed URLs in your
   Mapbox account, then open `https://<name>.onrender.com/console`.

Caveats for the free plan: the service sleeps after about 15 minutes idle, so
the first request after a nap has a cold start of roughly a minute (the app
itself starts fast; the delay is Render waking the instance). The snapshot is
the cached August 2026 CONUS week, a real-data demo, not a live feed. To make a
hosted instance live, run the ingester somewhere with internet and point the
service at a rolling store (see below), or redeploy with a refreshed snapshot.

A `Dockerfile` is also committed for anyone who prefers a container (it sets
`VHAGAR_FROZEN=1` and runs the same uvicorn command). It is optional; the
Render blueprint does not use it.

```
docker build -t vhagar .
docker run -p 8000:8000 -e VHAGAR_MAPBOX_TOKEN=pk.... vhagar
# open http://127.0.0.1:8000/console
```

The `.github/workflows/deploy-check.yml` workflow reproduces the native deploy
(install `serve/requirements.txt`, start uvicorn in frozen mode) and asserts the
endpoints serve, so a broken deploy path fails CI before it reaches Render.

## Sensors (multi-sensor fusion)

The live build fuses every configured source in one parallax-aware clustering
pass (`serve/build_snapshot.py`), each detection carrying its own matching
tolerance (GOES parallax-corrected, VIIRS ~1.1 km, MODIS ~3 km):

- GOES-18 and GOES-19 ABI FDC, geostationary, off the public NOAA S3 buckets
  (anonymous, no key). Each satellite is pulled into its own store and merged.
- VIIRS (S-NPP, NOAA-20, NOAA-21) and MODIS, polar-orbiting, off NASA FIRMS.
  This needs a free key from https://firms.modaps.eosdis.nasa.gov/api/ (instant).
  Set it as a repo secret named `FIRMS_MAP_KEY` (used by the live-snapshot
  workflow). Without the key the feed is GOES-18 + GOES-19 only.

Every source is mapped into the shared CONUS analysis grid (x/y + tile_id), so
the fusion treats a VIIRS pixel and a GOES pixel identically. The console's
Sensor dropdown lists whatever sensors are present in the current snapshot.

## Hosted live, free (scheduled snapshot)

The free Render tier cannot run an always-on ingester (it sleeps when idle and
has too little memory for the GOES decode stack), so the live feed is driven
from outside it. The `.github/workflows/live-snapshot.yml` workflow runs every
30 minutes on GitHub's runners, where the internet and memory exist: it pulls
the newest GOES-18/19 FDC granules from NOAA's public S3, clusters them with
VHAGAR's fusion (`serve/build_snapshot.py`), and publishes the result as a
rolling GitHub Release asset (tag `live-snapshot`).

The deployed API serves that asset. `render.yaml` sets:

- `VHAGAR_SNAPSHOT_URL` to the release asset, pulled on load and on every
  refresh. Until the first snapshot is published it falls back to the committed
  `serve/demo` bundle, so the site works immediately.
- `VHAGAR_REFRESH_SEC=1800` so an awake instance re-pulls every 30 min; a
  sleeping free instance re-pulls on its next cold-start wake.
- `VHAGAR_WEATHER=1` so each event is enriched with current Open-Meteo
  conditions and the spread-risk score, per request, labelled as current
  conditions (the fire data and the weather stay separate measurements).

So the satellite feed is never more than about 30 minutes behind real time, and
the weather is live. GitHub Actions runs unlimited minutes on public repos; on a
private repo a 30-minute cadence is well within the free monthly allowance. No
snapshot data is committed to git (it lives as a release asset), so history does
not grow.

## Live (near real time), self-hosted

By default the API serves the cached August window (a real-data demo, not a live
feed). To make it live without the scheduled snapshot, run the ingester and
point the API at a rolling store.

The ingester (`serve/ingest.py`) reuses VHAGAR's own resumable archive builder
(`vhagar.archive.backfill`) to pull the newest GOES ABI L2 FDC granules from the
public NOAA S3 bucket (anonymous, no credentials), decode them, append to a
rolling store, and prune anything older than the retention window. It needs
internet and the decode stack (`s3fs`, `xarray`, `h5netcdf`, `pyproj`), so it
runs on your machine, not in a sandbox.

```
# terminal 1: poll every 5 minutes, keep 3 days (GOES-18 = US West)
python -m serve.ingest --out data/detections_nrt --sat 18 --interval 300 --retention-days 3

# terminal 2: serve that store and rebuild the snapshot every 5 minutes
#   PowerShell:
$env:VHAGAR_DET_DIR = "$PWD\data\detections_nrt\detections"
$env:VHAGAR_REFRESH_SEC = "300"
uvicorn serve.vhagar_api:app
# open http://127.0.0.1:8000/console
```

`VHAGAR_REFRESH_SEC>0` starts a background thread that rebuilds the in-memory
snapshot on that cadence, so new granules appear without a restart and requests
never block on the clustering (they read the last-good snapshot until the new
one is ready). `/api/health` reports `refreshes` and `last_refresh`. The console
auto-refreshes every 5 minutes, so it tracks the feed on its own. GOES-18 covers
the US West; run a second ingester with `--sat 19` into its own store for the
East, or use `--sat 19` if your fires are eastern.

## Honesty

GOES FDC gives position, FRP, brightness temperature, confidence, view zenith
and time. It does not give spread risk, fire weather, or a validated burned
area, so those fields are absent rather than invented. The event polygon is the
convex hull of a cluster's detection pixels: a detection footprint, not a
burned-area measurement. The API sets `schema="fdc"` and the console labels
every number for what it actually is (footprint, peak FRP, sensors), hides the
spread-risk and weather panels, and colors perimeters by peak FRP.

To point the console at a different host, set `window.VHAGAR_API` before its
script runs; otherwise it uses the serving origin, with a bundled sample as an
offline fallback.
