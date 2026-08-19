"""VHAGAR command line interface.

    vhagar grid info --region conus
    vhagar splits build --records units.json --scheme leave_year_out --out splits/
    vhagar splits verify splits/leave_year_out.json
    vhagar fwi demo
    vhagar area-estimate --confusion 97,3,10,90 --areas 200000,20000
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, timedelta
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from vhagar import __version__

app = typer.Typer(add_completion=False, help="VHAGAR, multi-sensor wildfire intelligence")
grid_app = typer.Typer(help="Analysis grid utilities")
splits_app = typer.Typer(help="Leakage-proof cross-validation splits")
labels_app = typer.Typer(help="Fire event label registry")
app.add_typer(grid_app, name="grid")
app.add_typer(splits_app, name="splits")
app.add_typer(labels_app, name="labels")

console = Console()


@app.callback(invoke_without_command=True)
def _root(
    version: bool = typer.Option(False, "--version", help="Print version and exit"),
) -> None:
    if version:
        console.print(f"vhagar {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------- grid ----


@grid_app.command("info")
def grid_info(region: str = typer.Option("conus", help="conus | canada | europe")) -> None:
    """Summarise a region's analysis grid."""
    from vhagar.grid import RESOLUTION_M, TILE_CELLS, AnalysisGrid

    g = AnalysisGrid(region)
    t = Table(title=f"VHAGAR analysis grid. {region}")
    t.add_column("property")
    t.add_column("value", justify="right")
    t.add_row("CRS", g.crs)
    t.add_row("resolution", f"{RESOLUTION_M:.0f} m")
    t.add_row("tile size", f"{TILE_CELLS} cells ({g.tile_size_m / 1000:.0f} km)")
    t.add_row("tiles", f"{g.n_x} x {g.n_y} = {g.n_tiles:,}")
    t.add_row("origin", f"({g.origin_x:,.0f}, {g.origin_y:,.0f})")
    console.print(t)


@grid_app.command("tile")
def grid_tile(
    region: str = typer.Option("conus"),
    ix: int = typer.Argument(...),
    iy: int = typer.Argument(...),
) -> None:
    """Print one tile's geometry."""
    from vhagar.grid import AnalysisGrid

    tile = AnalysisGrid(region).tile(ix, iy)
    console.print_json(
        json.dumps(
            {
                "tile_id": tile.tile_id,
                "crs": tile.crs,
                "bounds": tile.bounds,
                "haloed_bounds": tile.haloed_bounds,
                "shape": tile.shape,
            }
        )
    )


# -------------------------------------------------------------- splits ----


@splits_app.command("build")
def splits_build(
    records: Path = typer.Option(None, help="JSON list of split-unit records"),
    registry: Path = typer.Option(None, help="registry Parquet from 'vhagar labels build'"),
    scheme: str = typer.Option("leave_year_out", help="spatial_block | leave_year_out | leave_one_<key>_out"),
    out: Path = typer.Option(Path("splits"), help="Output directory"),
    n_folds: int = typer.Option(5),
    block_degrees: float = typer.Option(5.0),
) -> None:
    """Build and persist a split manifest from a records JSON or the registry."""
    from vhagar.eval import splits as S

    if (records is None) == (registry is None):
        console.print("[red]pass exactly one of --records or --registry[/red]")
        raise typer.Exit(1)
    if registry is not None:
        from vhagar.labels.registry import EventRegistry

        units = EventRegistry.from_parquet(registry).to_split_units()
    else:
        units = S.units_from_records(json.loads(records.read_text()))

    if scheme == "spatial_block":
        manifest = S.spatial_block_split(units, n_folds=n_folds, block_degrees=block_degrees)
    elif scheme == "leave_year_out":
        manifest = S.leave_year_out(units)
    elif scheme.startswith("leave_one_") and scheme.endswith("_out"):
        key = scheme[len("leave_one_") : -len("_out")]
        manifest = S.leave_one_group_out(units, by=key)
    else:
        raise typer.BadParameter(f"unknown scheme {scheme!r}")

    S.verify_no_overlap(manifest)
    path = manifest.to_json(out / f"{manifest.scheme}.json")
    console.print(S.summarise(manifest))
    console.print(f"\n[green]wrote[/green] {path}")


@labels_app.command("build")
def labels_build(
    source: str = typer.Option("mtbs", help="label source (currently: mtbs)"),
    path: Path = typer.Option(..., exists=True, help="source file, e.g. an MTBS shapefile"),
    out: Path = typer.Option(Path("registry.parquet"), help="output registry Parquet"),
    region: str = typer.Option("conus"),
    geometry_dir: str = typer.Option("", help="prefix for per-fire geometry files"),
    severity_dir: str = typer.Option("", help="prefix for per-fire dNBR severity rasters"),
) -> None:
    """Ingest a label source into the fire event registry.

    Normalises records, assigns analysis-grid tiles, writes the versioned
    registry Parquet, and prints counts by region and source. MTBS carries the
    dNBR severity raster, so with ``--severity-dir`` its records are trainable.
    """
    from vhagar.grid import AnalysisGrid
    from vhagar.labels.registry import EventRegistry
    from vhagar.labels.tiles import assign_tiles

    if source != "mtbs":
        console.print(f"[red]unknown source {source!r}; currently only 'mtbs'[/red]")
        raise typer.Exit(1)

    from vhagar.labels.ingest import read_mtbs

    try:
        records = read_mtbs(
            path, region=region,
            geometry_dir=geometry_dir or None, severity_dir=severity_dir or None,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]failed to read {source}[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from None

    grid = AnalysisGrid(region)
    for r in records:
        r.tile_ids = assign_tiles(r, grid)
    reg = EventRegistry(records)
    reg.to_parquet(out)

    t = Table(title=f"Registry, {len(reg)} events")
    t.add_column("region/source")
    t.add_column("count", justify="right")
    for k, v in reg.summary().items():
        t.add_row(k, str(v))
    console.print(t)
    console.print(f"[green]wrote[/green] {out}")


@splits_app.command("verify")
def splits_verify(manifest: Path = typer.Argument(..., exists=True)) -> None:
    """Assert train/val/test disjointness. Exit code 1 on failure, use in CI."""
    from vhagar.eval.splits import SplitManifest, summarise, verify_no_overlap

    m = SplitManifest.from_json(manifest)
    try:
        verify_no_overlap(m)
    except AssertionError as exc:
        console.print(f"[red]LEAKAGE[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(summarise(m))
    console.print("[green]OK[/green] no train/val/test overlap")


# ----------------------------------------------------------------- fwi ----


@app.command("fwi")
def fwi_demo(
    temp: float = typer.Option(17.0, help="noon temperature, degC"),
    rh: float = typer.Option(42.0, help="noon relative humidity, %"),
    wind: float = typer.Option(25.0, help="noon 10 m wind, km/h"),
    rain: float = typer.Option(0.0, help="24 h precipitation, mm"),
    month: int = typer.Option(4),
    days: int = typer.Option(5, help="repeat the same forcing for N days"),
) -> None:
    """Run the FWI System forward from season-start values."""
    from vhagar.features.fwi import FWIState, fwi_system

    state = FWIState.season_start()
    t = Table(title="Canadian FWI System (1987)")
    for col in ("day", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "DSR"):
        t.add_column(col, justify="right")
    for d in range(1, days + 1):
        out, state = fwi_system(temp, rh, wind, rain, state, month=month)
        t.add_row(
            str(d),
            *[f"{float(np.atleast_1d(out[k])[0]):.2f}" for k in
              ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr")],
        )
    console.print(t)
    console.print("[dim]Average DSR, never raw FWI.[/dim]")


# ------------------------------------------------------- area estimate ----


@app.command("area-estimate")
def area_estimate(
    confusion: str = typer.Option(..., help="Row-major square matrix, e.g. '97,3,10,90'"),
    areas: str = typer.Option(..., help="Mapped area per class, e.g. '200000,20000'"),
    names: str = typer.Option("", help="Optional comma-separated class names"),
) -> None:
    """Olofsson error-adjusted area with 95% confidence intervals."""
    from vhagar.eval.area_estimation import estimate_areas

    vals = [float(v) for v in confusion.split(",")]
    n = int(round(len(vals) ** 0.5))
    if n * n != len(vals):
        raise typer.BadParameter("confusion must be a square matrix")
    conf = np.array(vals).reshape(n, n)
    a = np.array([float(v) for v in areas.split(",")])
    class_names = [s.strip() for s in names.split(",")] if names else None

    t = Table(title="Error-adjusted area (Olofsson et al.)")
    for col in ("class", "mapped", "adjusted", "95% CI", "UA", "PA"):
        t.add_column(col, justify="right")
    for e in estimate_areas(conf, a, class_names):
        t.add_row(
            e.class_name,
            f"{e.mapped_area:,.0f}",
            f"{e.adjusted_area:,.0f}",
            f"±{e.margin_of_error:,.0f}",
            f"{e.users_accuracy:.3f}",
            f"{e.producers_accuracy:.3f}",
        )
    console.print(t)
    console.print("[dim]Never report a pixel count as an area.[/dim]")


# ------------------------------------------------------- constellation ----


@app.command("sensors")
def sensors_cmd(
    on: str = typer.Option("", help="ISO date; defaults to today"),
) -> None:
    """Show which platforms are delivering data, and the caveats that bite."""
    from datetime import date as _date

    from vhagar.io.sensors import PLATFORMS, SENSORS, coverage_report

    when = _date.fromisoformat(on) if on else _date.today()
    console.print(coverage_report(when))

    t = Table(title="Platforms")
    for col in ("platform", "instrument", "status", "local time", "data ends", "FRP bias"):
        t.add_column(col)
    for p in sorted(PLATFORMS.values(), key=lambda q: (q.status, q.name)):
        style = {"ending": "yellow", "planned": "dim"}.get(p.status, "")
        t.add_row(
            p.name,
            SENSORS[p.sensor].name,
            p.status,
            p.local_time,
            p.data_end.isoformat() if p.data_end else ". ",
            f"x{p.frp_bias_factor:.2f}" if p.frp_bias_factor != 1.0 else ". ",
            style=style,
        )
    console.print(t)

    console.print("\n[bold]Caveats[/bold]")
    for p in PLATFORMS.values():
        for c in p.caveats:
            console.print(f"  [yellow]![/yellow] {p.name}: {c}")


# ----------------------------------------------------------------- frp ----


@app.command("frp")
def frp_cmd(
    bt_mir: float = typer.Option(360.0, help="fire-pixel MIR brightness temperature, K"),
    bt_bg: float = typer.Option(305.0, help="background MIR brightness temperature, K"),
    pixel_m: float = typer.Option(375.0, help="pixel side length, m"),
    tcwv: float = typer.Option(20.0, help="total column water vapour, kg/m2"),
    view_zenith: float = typer.Option(0.0, help="view zenith angle, deg"),
    sensor: str = typer.Option("modis_c6"),
) -> None:
    """Fire Radiative Power, showing what atmospheric correction is worth."""
    from vhagar.physics.atmosphere import transmittance_mir
    from vhagar.physics.frp import frp_from_brightness_temperature
    from vhagar.physics.planck import brightness_temperature, planck_radiance

    tau = float(transmittance_mir(tcwv, view_zenith))
    area = pixel_m**2
    kw = {"pixel_area_m2": area, "sensor": sensor}
    uncorrected = float(
        frp_from_brightness_temperature(bt_mir, bt_bg, transmittance=1.0, **kw)
    )
    corrected = float(
        frp_from_brightness_temperature(bt_mir, bt_bg, transmittance=tau, **kw)
    )

    t = Table(title="Wooster MIR-radiance FRP")
    t.add_column("quantity")
    t.add_column("value", justify="right")
    t.add_row("MIR transmittance", f"{tau:.3f}")
    t.add_row("correction factor", f"x{1 / tau:.2f}")
    t.add_row("FRP, tau = 1 (uncorrected)", f"{uncorrected:,.1f} MW")
    t.add_row("[bold]FRP, corrected[/bold]", f"[bold]{corrected:,.1f} MW[/bold]")
    t.add_row("low bias if uncorrected", f"{100 * (1 - tau):.0f}%")
    console.print(t)

    l_mir = float(planck_radiance(3.9, bt_mir))
    bt_check = float(brightness_temperature(3.9, l_mir))
    console.print(f"[dim]Planck round-trip check: {bt_check:.6f} K[/dim]")
    console.print(
        "[dim]FRP errors are multiplicative, a, tau, emissivity and the "
        "background all enter as products. Model in log space.[/dim]"
    )


# ------------------------------------------------------------ archive ----


@app.command("archive-plan")
def archive_plan(
    disk_gb: float = typer.Option(100.0, help="disk budget in GB"),
    measure: bool = typer.Option(False, help="measure real granule size/time (needs network)"),
) -> None:
    """Size the Step-2 tile archive before committing to weeks of downloading.

    Three budgets, not one: disk (small), download traffic (usually the binding
    constraint), and wall clock.
    """
    from vhagar.archive.plan import STANDARD_PLANS, recommend_plan

    t = Table(title="Reference plans")
    t.add_column("plan", justify="left", no_wrap=True)
    for col in ("tiles", "yr", "cad", "bd", "store", "disk GB", "wire TB", "hrs"):
        t.add_column(col, justify="right", no_wrap=True)
    for plan in STANDARD_PLANS:
        c = plan.cost()
        style = "yellow" if "naive" in plan.name else ""
        t.add_row(
            plan.name, str(plan.n_tiles), f"{plan.years:g}", f"{plan.cadence_min:g}m",
            str(plan.n_bands), plan.storage, f"{c.disk_gb:,.2f}", f"{c.download_tb:,.2f}",
            f"{c.wall_clock_hours:,.1f}", style=style,
        )
    console.print(t)
    console.print()
    console.print(recommend_plan(disk_gb))

    if measure:
        from vhagar.archive.plan import measure_granule

        console.print("\n[bold]Measuring real granules on this connection...[/bold]")
        measured: dict[str, dict] = {}
        for product in ("FDC", "CMIP"):
            try:
                m = measure_granule(product=product)
                measured[product] = m
                console.print(
                    f"  {product:5s} {int(m['n_sampled'])} granules, "
                    f"{m['granule_mb']:>7.2f} MB, {m['seconds_per_granule']:.2f} s each, "
                    f"{m['mb_per_second']:>6.2f} MB/s"
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"  {product:5s} [red]failed[/red] {type(exc).__name__}: {exc}")

        if len(measured) == 2:
            fdc, cmip = measured["FDC"], measured["CMIP"]
            ratio = cmip["granule_mb"] / max(fdc["granule_mb"], 1e-9)
            time_ratio = cmip["seconds_per_granule"] / max(fdc["seconds_per_granule"], 1e-9)
            console.print(
                f"\n  CMIP radiance granules are [bold]{ratio:.0f}x[/bold] the size of FDC. "
                "Size the detection\n  tier from FDC and the radiance tier from CMIP; one "
                "figure for both is\n  wrong by that factor."
            )
            console.print(
                f"\n  Both figures are full decodes now, so they are comparable: a CMIP "
                f"granule\n  takes about [bold]{time_ratio:.1f}x[/bold] as long to decode as "
                "an FDC one. Remember a CMIP\n  timestep is one file per channel, so a five-band "
                "read is five of these.\n  Use [bold]vhagar probe-workers[/bold] to size "
                "concurrency."
            )
        console.print(
            "[dim]  Feed these into ArchivePlan(granule_mb=..., seconds_per_granule=...) "
            "for a plan calibrated to your connection. Both times are full-domain\n  decodes "
            "with a warm navigation cache.[/dim]"
        )


@app.command("probe-workers")
def probe_workers_cmd(
    candidates: str = typer.Option("1,4,8,16,32", help="worker counts to try"),
    satellite: int = typer.Option(18, help="GOES satellite number"),
    n_granules: int = typer.Option(48, help="granules read at each setting"),
    repeats: int = typer.Option(3, help="passes per setting; the median is reported"),
    mode: str = typer.Option("full", help="full = fetch+decode (use this) | fetch = bytes only"),
) -> None:
    """Find where concurrency stops helping, before committing to a long backfill.

    The planner assumes throughput scales linearly with workers and it does
    not: it scales until something saturates. Guessing 12 when the knee is at
    48 wastes most of a day.

    Defaults to timing the *full* operation the backfill performs, not a bare
    S3 read. On an FDC granule the bare read is about a sixth of the total, so
    a fetch-only probe finds where the network saturates and then that number
    gets applied to a workload bottlenecked elsewhere.
    """
    from vhagar.archive.backfill import probe_workers, recommend_workers

    try:
        wanted = [int(c) for c in candidates.split(",") if c.strip()]
    except ValueError:
        console.print("[red]candidates must be a comma separated list of integers[/red]")
        raise typer.Exit(1) from None

    console.print(
        f"[bold]Probing {wanted} workers, {n_granules} granules x {repeats} passes, "
        f"mode={mode}[/bold]\nThis is meant to take a few minutes. A probe that "
        "finishes instantly has measured nothing.\n"
    )
    try:
        rows = probe_workers(
            wanted, satellite=satellite, n_granules=n_granules,
            repeats=repeats, mode=mode,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]failed[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from None

    t = Table(title=f"Throughput against worker count (mode={mode}, median of {repeats})")
    for col in ("workers", "granules/s", "s/granule", "seconds", "spread", "speedup"):
        t.add_column(col, justify="right")
    base = rows[0]["granules_per_second"]
    for r in rows:
        style = "yellow" if r["too_fast"] or r["spread"] > 0.25 else ""
        t.add_row(
            f"{int(r['workers'])}",
            f"{r['granules_per_second']:.2f}",
            f"{r['seconds_per_granule']:.3f}",
            f"{r['seconds']:.1f}",
            f"{r['spread'] * 100:.0f}%",
            f"{r['granules_per_second'] / base:.1f}x",
            style=style,
        )
    console.print(t)
    console.print(
        "[dim]  spread is (slowest - fastest) / median across passes. Yellow rows ran "
        "too\n  briefly or varied too much to rank.[/dim]"
    )

    verdict = recommend_workers(rows)
    if verdict["workers"] is None:
        console.print(
            f"\n  [yellow]No reliable knee.[/yellow] {verdict['reason']}.\n"
            f"  Fastest was {verdict.get('best')} workers, but treat that as noise.\n"
            f"  Try: vhagar probe-workers --n-granules {n_granules * 4} --repeats 5"
        )
    else:
        console.print(
            f"\n  Fastest at {verdict['best']} workers. Within 10% of that from "
            f"[bold]{verdict['workers']}[/bold] workers\n  onward, so anything above "
            "that is buying very little. Use the smaller number:\n  it is gentler on a "
            "public bucket and fails less."
        )
    if mode == "fetch":
        console.print(
            "\n[yellow]  mode=fetch measured bare byte reads, not the work the backfill\n"
            "  does. Do not size --workers from this. Re-run with --mode full.[/yellow]"
        )


@app.command("backfill")
def backfill_cmd(
    out: Path = typer.Argument(..., help="output directory for the detection archive"),
    start: str = typer.Option(..., help="start date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="end date, YYYY-MM-DD, inclusive"),
    satellite: int = typer.Option(18, help="GOES satellite number"),
    domain: str = typer.Option("C", help="ABI domain: C, F, M1, M2"),
    region: str = typer.Option("conus", help="VHAGAR analysis region"),
    bbox: str = typer.Option("", help="west,south,east,north in degrees. Empty reads all."),
    workers: int = typer.Option(12, help="concurrent granule reads, see probe-workers"),
    min_confidence: float = typer.Option(0.0, help="drop detections below this confidence"),
    drop_filtered: bool = typer.Option(False, help="keep only the 10-15 mask series"),
) -> None:
    """Tier A: build the FDC detection history. Resumable, safe to interrupt.

    Re-run the same command after any interruption. Granules already read are
    skipped, failures are retried, and coverage is recorded so that a tile with
    no detections stays distinguishable from a tile that was never read.
    """
    from datetime import datetime as _dt

    from vhagar.archive.backfill import BackfillConfig, backfill, coverage_intervals, load_manifest

    def parse_day(s: str, end_of_day: bool) -> _dt:
        d = _dt.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)
        return d.replace(hour=23, minute=59, second=59) if end_of_day else d

    try:
        t0, t1 = parse_day(start, False), parse_day(end, True)
    except ValueError:
        console.print("[red]dates must be YYYY-MM-DD[/red]")
        raise typer.Exit(1) from None
    if t1 < t0:
        console.print("[red]end is before start[/red]")
        raise typer.Exit(1)

    box = None
    if bbox:
        parts = [float(p) for p in bbox.split(",")]
        if len(parts) != 4:
            console.print("[red]bbox needs four numbers: west,south,east,north[/red]")
            raise typer.Exit(1)
        box = (parts[0], parts[1], parts[2], parts[3])

    cfg = BackfillConfig(
        out_dir=out, start=t0, end=t1, satellite=satellite, domain=domain,
        region=region, bbox=box, workers=workers,
        include_filtered=not drop_filtered, min_confidence=min_confidence,
    )

    n_days = (t1.date() - t0.date()).days + 1
    console.print(
        f"[bold]Backfilling {n_days} days[/bold] from GOES-{satellite} {domain} "
        f"into {out} at {workers} workers.\n"
        "Safe to interrupt. Re-run the same command to resume.\n"
    )

    def on_day(day, day_result):
        if day_result is None:
            console.print(f"  {day:%Y-%m-%d}  already complete")
        else:
            console.print(f"  {day:%Y-%m-%d}  {day_result}")

    try:
        result = backfill(cfg, progress=on_day)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted. Progress is on disk; re-run to resume.[/yellow]")
        raise typer.Exit(130) from None

    console.print(f"\n[bold]{result}[/bold]")
    if result.errors:
        console.print(f"  errors by type: {result.errors}")

    covered = coverage_intervals(load_manifest(out).values())
    total = sum((b - a for a, b in covered), timedelta())
    console.print(
        f"  coverage: {len(covered)} interval(s), {total.total_seconds() / 3600:.1f} h observed"
    )
    if len(covered) > 1:
        console.print(
            "[dim]  More than one interval means there are holes. That is recorded, not\n"
            "  hidden: a loader mining negatives will skip them rather than treat them\n"
            "  as quiet.[/dim]"
        )


@app.command("climatology-backfill")
def climatology_backfill_cmd(
    out: Path = typer.Argument(..., help="output directory for the climatology checkpoint"),
    start: str = typer.Option(..., help="start date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="end date, YYYY-MM-DD, inclusive"),
    bbox: str = typer.Option(..., help="west,south,east,north in degrees. Required."),
    satellite: int = typer.Option(18, help="GOES satellite number"),
    channels: str = typer.Option("C07,C11,C13,C14,C15", help="thermal channels to reduce"),
    cadence_min: int = typer.Option(15, help="diurnal sampling in minutes"),
    n_bins: int = typer.Option(24, help="diurnal bins across the day; must divide 1440"),
    workers: int = typer.Option(8, help="concurrent stack reads, see probe-workers"),
) -> None:
    """Tier B: reduce CMIP stacks into a diurnal climatology. Resumable.

    Reads the thermal channels over the window, folds each timestep into a
    per-pixel, per-hour running mean and variance on the native ABI grid, and
    checkpoints atomically so an interrupted run resumes without re-folding.
    Re-run the same command to resume.
    """
    from datetime import datetime as _dt

    from vhagar.archive.climatology_backfill import (
        ClimatologyBackfillConfig,
        backfill_climatology,
        climatology_coverage,
    )

    try:
        t0 = _dt.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        t1 = _dt.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError:
        console.print("[red]dates must be YYYY-MM-DD[/red]")
        raise typer.Exit(1) from None
    parts = [p for p in bbox.split(",") if p.strip()]
    if len(parts) != 4:
        console.print("[red]bbox needs four numbers: west,south,east,north[/red]")
        raise typer.Exit(1)
    box = tuple(float(p) for p in parts)
    chans = tuple(c.strip() for c in channels.split(",") if c.strip())

    cfg = ClimatologyBackfillConfig(
        out_dir=out, start=t0, end=t1, bbox=box, satellite=satellite,
        channels=chans, cadence_min=cadence_min, n_bins=n_bins, workers=workers,
    )
    console.print(
        f"[bold]Reducing CMIP climatology[/bold] from GOES-{satellite} into {out}\n"
        f"  channels {', '.join(chans)}, {cadence_min}-min cadence, {n_bins} bins, "
        f"{workers} workers.\n  Safe to interrupt. Re-run the same command to resume.\n"
    )

    def on_day(day, day_ok):
        console.print(f"  {day:%Y-%m-%d}  {day_ok} frames folded")

    try:
        result = backfill_climatology(cfg, progress=on_day)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted. Checkpoint is on disk; re-run to resume.[/yellow]")
        raise typer.Exit(130) from None

    console.print(f"\n[bold]{result}[/bold]")
    if result.errors:
        console.print(f"  errors by type: {result.errors}")
    covered = climatology_coverage(out)
    total = sum((b - a for a, b in covered), timedelta())
    console.print(
        f"  coverage: {len(covered)} interval(s), {total.total_seconds() / 3600:.1f} h observed"
    )


@app.command("t2-prithvi-build")
def t2_prithvi_build_cmd(
    registry: Path = typer.Option(..., exists=True, help="registry Parquet from 'labels build'"),
    mosaic: Path = typer.Option(..., exists=True, help="MTBS thematic severity GeoTIFF (reference)"),
    region: str = typer.Option("conus"),
    year: int = typer.Option(2021, help="fire year"),
    min_area_ha: float = typer.Option(2000.0, help="only fires at least this large"),
    max_fires: int = typer.Option(20, help="cap the number of fires (imagery is the cost)"),
    select: str = typer.Option("size", help="fire selection: largest | size (size-stratified)"),
    max_cloud: float = typer.Option(60.0, help="max scene cloud cover percent"),
    max_scenes: int = typer.Option(12, help="least-cloudy post-fire scenes per window"),
    res_m: float = typer.Option(30.0, help="analysis resolution; 30 m is Prithvi's native"),
    cache_dir: Path = typer.Option(Path("data/t2_prithvi"), help="cache six-band samples here"),
) -> None:
    """Build the six-band Prithvi dataset: the multi-band re-pull for the T2 fine-tune.

    Selects fires and pulls the six-band post-fire Sentinel-2 surface-reflectance composite
    (Blue/Green/Red/narrow-NIR/SWIR1/SWIR2, Prithvi's band order) at 30 m, paired with the
    MTBS burned mask on the identical grid, and caches each as a T2Sample ``.npz``. This is
    the input the Prithvi-EO-2.0 fine-tune needs; see docs/13 for the terratorch runbook.
    Needs an open network. The fold split and skill-vs-RBR scoring reuse the same leakage-
    proof machinery as t2-unet, so the fine-tune is a fair head-to-head.
    """
    from vhagar.datasets.t2_optical import build_prithvi_sample, select_fires
    from vhagar.labels.registry import EventRegistry

    reg = EventRegistry.from_parquet(registry)
    candidates = [
        r for r in reg
        if r.region == region and r.ignition_date and r.ignition_date.year == year
        and (r.area_ha or 0) >= min_area_ha and r.tile_ids
    ]
    fires = select_fires(candidates, max_fires, strategy=select)
    if len(fires) < 3:
        console.print(f"[red]only {len(fires)} fires match; need at least 3[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]Building six-band Prithvi samples for {len(fires)} {region} {year} "
                  f"fires[/bold] (>= {min_area_ha:g} ha) at {res_m:g} m.\n")

    built = 0
    for i, rec in enumerate(fires, 1):
        console.print(f"  [{i}/{len(fires)}] {rec.event_id} ({(rec.area_ha or 0):,.0f} ha)...",
                      end=" ")
        try:
            s = build_prithvi_sample(
                rec, mosaic, max_cloud=max_cloud, max_scenes=max_scenes, res_m=res_m,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]skip[/yellow] ({type(exc).__name__})")
            continue
        bf = f"{100 * s.burned_fraction:.0f}% burned" if s.n_valid else "no valid px"
        console.print(f"[green]ok[/green] ({s.features.shape[0]}-band, {s.n_valid:,} valid px, {bf})")
        built += 1
    console.print(f"\n[green]{built} six-band samples cached to {cache_dir}[/green]. "
                  "Next: export terratorch chips + fine-tune Prithvi-EO-2.0 (GPU). See docs/13.")


def _load_prithvi_cache(cache_dir: Path) -> dict:
    """Load cached six-band Prithvi samples (``*p6.npz``) keyed by event id."""
    import glob as _glob

    from vhagar.datasets.burned_area import T2Sample

    out = {}
    for p in sorted(_glob.glob(f"{cache_dir}/*p6.npz")):
        s = T2Sample.load(p)
        out[s.event_id] = s
    return out


@app.command("t2-prithvi-export")
def t2_prithvi_export_cmd(
    cache_dir: Path = typer.Option(Path("data/t2_prithvi"), exists=True, help="six-band sample cache"),
    out_dir: Path = typer.Option(Path("data/t2_prithvi_chips"), help="terratorch chip dataset out"),
    chip: int = typer.Option(224, help="chip size (Prithvi img size)"),
    val_frac: float = typer.Option(0.15), test_frac: float = typer.Option(0.15),
    seed: int = typer.Option(0),
    burn_balance: bool = typer.Option(
        False, "--burn-balance",
        help="rebalance TRAIN chips toward burned areas (fixes 'predicts all unburned')",
    ),
    max_bg_ratio: float = typer.Option(1.0, help="burn-balance: max background chips per burn chip"),
) -> None:
    """Export a terratorch-ready chip dataset from the six-band samples, split by fire.

    Partitions whole fires into train/val/test (leakage-proof), tiles each into paired
    six-band image / signed-label (0/1/-1) GeoTIFF chips, and writes them under
    ``out_dir/{split}/{images,labels}`` with a ``_split.json``. Point the terratorch
    datamodule here. Needs rasterio. See docs/13.
    """
    from vhagar.eval.t2_prithvi import export_prithvi_chips

    samples = _load_prithvi_cache(cache_dir)
    if len(samples) < 3:
        console.print(f"[red]only {len(samples)} cached six-band samples; run t2-prithvi-build[/red]")
        raise typer.Exit(1)
    counts = export_prithvi_chips(
        samples, out_dir, chip=chip, val_frac=val_frac, test_frac=test_frac, seed=seed,
        burn_balance=burn_balance, max_bg_ratio=max_bg_ratio,
    )
    console.print(f"[green]exported chips[/green] to {out_dir}: "
                  + ", ".join(f"{k}={v}" for k, v in counts.items()))
    console.print("Next: terratorch fit -c prithvi_burnscars_vhagar.yaml (GPU). See docs/13.")


@app.command("t2-prithvi-score")
def t2_prithvi_score_cmd(
    cache_dir: Path = typer.Option(Path("data/t2_prithvi"), exists=True, help="six-band sample cache"),
    pred_dir: Path = typer.Option(..., exists=True, help="predicted-mask GeoTIFFs"),
    chips_manifest: Path = typer.Option(
        None, help="chips _chips.json to stitch PER-CHIP preds ({stem}_*.tif) into per-fire masks"
    ),
) -> None:
    """Score Prithvi predictions with the SAME skill-over-naive metric as RBR / U-Net.

    Two input modes. Default: one predicted mask per fire as ``{event_id}.tif`` (``:``/``/``
    replaced by ``_``). With ``--chips-manifest`` pointing at the export's ``_chips.json``:
    ``pred_dir`` holds terratorch's per-chip predictions named by chip stem, and they are
    stitched back into per-fire masks first. Either way, F1/IoU and skill over the predict-
    all-burned baseline are scored on the identical valid pixels and averaged. Compare the
    mean to t2-unet / t2-stage0 on the same fires: the honest head-to-head. See docs/13.
    """
    import glob as _glob
    import json as _json

    import rasterio

    from vhagar.eval.t2_prithvi import score_masks, stitch_chip_predictions, summarise_scores

    samples = _load_prithvi_cache(cache_dir)
    if chips_manifest is not None:
        manifest = _json.loads(chips_manifest.read_text(encoding="utf-8"))
        pred_by_stem = {}
        for p in sorted(_glob.glob(f"{pred_dir}/*.tif")):
            stem = Path(p).stem
            # tolerate terratorch suffixes like "{stem}_pred" / "{stem}_merged"
            key = stem if stem in manifest else stem.rsplit("_", 1)[0]
            if key not in manifest:
                continue
            with rasterio.open(p) as src:
                pred_by_stem[key] = src.read(1)
        matched = stitch_chip_predictions(pred_by_stem, manifest)
    else:
        # one mask per fire, keyed by the underscored event id
        stems = {e.replace(":", "_").replace("/", "_"): e for e in samples}
        matched = {}
        for p in sorted(_glob.glob(f"{pred_dir}/*.tif")):
            eid = stems.get(Path(p).stem)
            if eid is None:
                continue
            with rasterio.open(p) as src:
                matched[eid] = src.read(1)
    scores = score_masks(matched, samples)
    if not scores:
        console.print("[yellow]no predictions matched cached fires; check pred filenames "
                      "({event_id}.tif with ':' and '/' replaced by '_').[/yellow]")
        raise typer.Exit(1)
    summ = summarise_scores(scores)
    t = Table(title="T2 Prithvi predictions vs MTBS (skill over naive, same metric as RBR/U-Net)")
    for col in ("held out", "F1", "IoU", "naive F1", "skill"):
        t.add_column(col, justify="right")
    for s in scores:
        sk = f"{s.skill_f1:+.3f}"
        t.add_row(s.event_id[:22], f"{s.f1:.3f}", f"{s.iou:.3f}", f"{s.naive_f1:.3f}",
                  f"[green]{sk}[/green]" if s.skill_f1 > 0 else f"[red]{sk}[/red]")
    console.print(t)
    console.print(f"[bold]mean skill {summ['skill_mean']:+.3f}[/bold] over {summ['fires']} fires "
                  f"({summ['fires_positive_skill']} positive). Compare to U-Net +0.54 on the same fires.")


@app.command("t2-prithvi-build-emsr")
def t2_prithvi_build_emsr_cmd(
    emsr_manifest: Path = typer.Option(Path("emsr.csv"), exists=True,
                                       help="CSV: activation_id,delineation_path,event_date"),
    max_cloud: float = typer.Option(60.0), max_scenes: int = typer.Option(12),
    res_m: float = typer.Option(30.0, help="30 m is Prithvi's native resolution"),
    cache_dir: Path = typer.Option(Path("data/t2_prithvi_emsr"), help="cache European six-band samples"),
) -> None:
    """Build six-band Prithvi samples for the European (Copernicus EMS) fires, for the
    leave-one-continent-out transfer test. Same as t2-prithvi-build but the reference is each
    fire's EMS burnt-area delineation instead of MTBS. Needs the open Sentinel-2 network. See docs/13.
    """
    import csv as _csv

    from vhagar.datasets.t2_optical import build_prithvi_sample, read_emsr_reference_on_grid
    from vhagar.labels.ingest import read_emsr

    rows = list(_csv.DictReader(emsr_manifest.open(encoding="utf-8")))
    console.print(f"[bold]Building six-band Prithvi samples for {len(rows)} EMS fires[/bold] at {res_m:g} m.\n")
    built = 0
    for row in rows:
        path = row["delineation_path"]
        aid = row.get("activation_id") or Path(path).stem
        console.print(f"  {aid}...", end=" ")
        try:
            rec = read_emsr(path, row["event_date"], activation_id=aid)
            ref = lambda grid, p=path: read_emsr_reference_on_grid(p, grid)  # noqa: E731
            s = build_prithvi_sample(rec, ref, max_cloud=max_cloud, max_scenes=max_scenes,
                                     res_m=res_m, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]skip[/yellow] ({type(exc).__name__})")
            continue
        bf = f"{100 * s.burned_fraction:.0f}% burned" if s.n_valid else "no valid px"
        console.print(f"[green]ok[/green] ({s.features.shape[0]}-band, {s.n_valid:,} px, {bf})")
        built += 1
    console.print(f"\n[green]{built} European six-band samples cached to {cache_dir}[/green]. "
                  "Next: t2-prithvi-export-infer, predict with the CONUS checkpoint, t2-prithvi-transfer.")


@app.command("t2-prithvi-export-infer")
def t2_prithvi_export_infer_cmd(
    cache_dir: Path = typer.Option(..., exists=True, help="six-band sample cache to chip (e.g. EMSR)"),
    out_dir: Path = typer.Option(..., help="terratorch chip dataset out (all fires, no split)"),
    chip: int = typer.Option(224),
) -> None:
    """Chip every sample into one flat inference dataset (no train/val/test split).

    For predicting a held-out cohort (the European fires) with a model trained elsewhere.
    Writes ``out_dir/data`` + ``out_dir/splits/all.txt`` + ``out_dir/_chips.json``. See docs/13.
    """
    from vhagar.eval.t2_prithvi import export_inference_chips

    samples = _load_prithvi_cache(cache_dir)
    if not samples:
        console.print(f"[red]no cached samples in {cache_dir}[/red]")
        raise typer.Exit(1)
    n = export_inference_chips(samples, out_dir, chip=chip)
    console.print(f"[green]exported {n} inference chips[/green] to {out_dir} "
                  f"({len(samples)} fires). Predict these with the CONUS checkpoint in Colab.")


@app.command("t2-prithvi-transfer")
def t2_prithvi_transfer_cmd(
    emsr_cache: Path = typer.Option(Path("data/t2_prithvi_emsr"), exists=True, help="European six-band samples"),
    pred_dir: Path = typer.Option(..., exists=True, help="European per-chip predictions from Colab"),
    chips_manifest: Path = typer.Option(..., exists=True, help="_chips.json from t2-prithvi-export-infer"),
    conus_cache: Path = typer.Option(Path("data/t2_prithvi"), exists=True, help="CONUS samples (NBR baseline train)"),
) -> None:
    """Leave-one-continent-out transfer: CONUS-trained Prithvi vs an NBR threshold, on European fires.

    Stitches the European per-chip Prithvi predictions into per-fire masks, scores them against
    the EMS delineations (skill over naive), and compares to a post-fire NBR threshold *tuned on
    CONUS and applied to Europe*, the spectral-cut analogue of the same transfer. Does the
    foundation model's pretraining generalise across continents better than a threshold? See docs/13.
    """
    import glob as _glob
    import json as _json

    import rasterio

    from vhagar.eval.t2_prithvi import (
        nbr_threshold_transfer,
        score_masks,
        stitch_chip_predictions,
        summarise_scores,
    )

    eu = _load_prithvi_cache(emsr_cache)
    conus = _load_prithvi_cache(conus_cache)
    manifest = _json.loads(chips_manifest.read_text(encoding="utf-8"))
    pred_by_stem = {}
    for p in sorted(_glob.glob(f"{pred_dir}/*.tif")):
        stem = Path(p).stem
        key = stem if stem in manifest else stem.rsplit("_", 1)[0]
        if key not in manifest:
            continue
        with rasterio.open(p) as src:
            pred_by_stem[key] = src.read(1)
    prithvi = score_masks(stitch_chip_predictions(pred_by_stem, manifest), eu)
    nbr, thr = nbr_threshold_transfer(list(conus.values()), list(eu.values()))
    nbr_by = {s.event_id: s for s in nbr}

    t = Table(title="T2 leave-one-continent-out: CONUS-trained Prithvi vs NBR threshold on EU fires")
    for col in ("EU fire", "Prithvi skill", "NBR skill", "Prithvi - NBR"):
        t.add_column(col, justify="right")
    for s in prithvi:
        n = nbr_by.get(s.event_id)
        d = s.skill_f1 - (n.skill_f1 if n else 0.0)
        t.add_row(s.event_id[:22], f"{s.skill_f1:+.3f}", f"{n.skill_f1:+.3f}" if n else "n/a",
                  f"[green]{d:+.3f}[/green]" if d > 0 else f"[red]{d:+.3f}[/red]")
    console.print(t)
    wins = sum(1 for s in prithvi if s.skill_f1 > (nbr_by[s.event_id].skill_f1 if s.event_id in nbr_by else 0.0))
    ps, ns = summarise_scores(prithvi), summarise_scores(nbr)
    console.print(f"[bold]Prithvi mean skill {ps['skill_mean']:+.3f}[/bold] vs "
                  f"NBR-threshold {ns['skill_mean']:+.3f} over {ps['fires']} European fires "
                  f"(Prithvi wins {wins}/{len(prithvi)}; NBR cut {thr:.3f} tuned on CONUS).")


@app.command("t2-prithvi-baseline")
def t2_prithvi_baseline_cmd(
    cache_dir: Path = typer.Option(Path("data/t2_prithvi"), exists=True, help="six-band sample cache"),
    seed: int = typer.Option(0, help="split seed (must match the export used for Prithvi)"),
) -> None:
    """Same-fire baseline: a post-fire NBR threshold scored like Prithvi, on the same test fires.

    Fits one NBR cut on the train fires and scores the identical test fires with the same
    skill-over-naive metric, so the number is directly comparable to `t2-prithvi-score`: does
    the foundation model beat a pointwise spectral threshold on these exact fires? Pure numpy,
    no GPU. See docs/13.
    """
    from vhagar.eval.t2_prithvi import nbr_threshold_baseline, summarise_scores

    samples = _load_prithvi_cache(cache_dir)
    if len(samples) < 3:
        console.print(f"[red]only {len(samples)} cached six-band samples; run t2-prithvi-build[/red]")
        raise typer.Exit(1)
    scores, thr = nbr_threshold_baseline(samples, seed=seed)
    t = Table(title=f"T2 post-NBR threshold baseline vs MTBS (same test fires, thr={thr:.3f})")
    for col in ("held out", "F1", "IoU", "naive F1", "skill"):
        t.add_column(col, justify="right")
    for s in scores:
        sk = f"{s.skill_f1:+.3f}"
        t.add_row(s.event_id[:22], f"{s.f1:.3f}", f"{s.iou:.3f}", f"{s.naive_f1:.3f}",
                  f"[green]{sk}[/green]" if s.skill_f1 > 0 else f"[red]{sk}[/red]")
    console.print(t)
    su = summarise_scores(scores)
    console.print(f"[bold]NBR-threshold mean skill {su['skill_mean']:+.3f}[/bold] over {su['fires']} "
                  f"fires. Compare to t2-prithvi-score (the deep model) on the same fires.")


@app.command("t2-stage0")
def t2_stage0_cmd(
    registry: Path = typer.Option(..., exists=True, help="registry Parquet from 'labels build'"),
    mosaic: Path = typer.Option(..., exists=True, help="MTBS thematic severity GeoTIFF (reference)"),
    region: str = typer.Option("conus"),
    year: int = typer.Option(2021, help="fire year to evaluate"),
    min_area_ha: float = typer.Option(2000.0, help="only fires at least this large"),
    max_fires: int = typer.Option(15, help="cap the number of fires (imagery is the cost)"),
    select: str = typer.Option("largest", help="fire selection: largest | size (size-stratified)"),
    max_cloud: float = typer.Option(60.0, help="max scene cloud cover percent"),
    max_scenes: int = typer.Option(6, help="least-cloudy scenes per window (fewer = faster)"),
    res_m: float = typer.Option(100.0, help="analysis resolution in metres (coarser = faster)"),
    cache_dir: Path = typer.Option(Path("data/t2_cache"), help="cache built samples here"),
    method: str = typer.Option("global", help="threshold: global | otsu | perstratum"),
    objective: str = typer.Option(
        "f1", help="threshold objective: f1 | iou | youden (balanced, robust to burn-heavy windows)"
    ),
    stratify_raster: Path = typer.Option(
        None, help="global class raster (e.g. Koppen); enables per-stratum thresholds"
    ),
    with_stack: bool = typer.Option(
        False, help="also cache the pre/post NBR stack for deep models (t2-deep); tag _w15bgs"
    ),
    n_reference: int = typer.Option(500, help="Olofsson reference-sample size per fold"),
    seed: int = typer.Option(0),
) -> None:
    """T2 Stage-0 with an INDEPENDENT Sentinel-2 RBR predictor vs MTBS severity.

    Calibrates a burn-severity threshold per leave-one-fire-out fold and reports
    F1/IoU and the Olofsson error-adjusted burned area with a 95% CI. The
    Sentinel-2 pull needs an open network; run this where that is available.
    """
    from vhagar.datasets.t2_optical import build_optical_samples
    from vhagar.eval.splits import leave_one_group_out
    from vhagar.eval.t2_stage0 import run_stage0, summarise_stage0
    from vhagar.labels.registry import EventRegistry

    reg = EventRegistry.from_parquet(registry)
    from vhagar.datasets.t2_optical import select_fires

    candidates = [
        r for r in reg
        if r.region == region and r.ignition_date and r.ignition_date.year == year
        and (r.area_ha or 0) >= min_area_ha and r.tile_ids
    ]
    fires = select_fires(candidates, max_fires, strategy=select)
    if len(fires) < 3:
        console.print(f"[red]only {len(fires)} fires match; need at least 3[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold]Building Sentinel-2 RBR for {len(fires)} {region} {year} fires[/bold] "
        f"(>= {min_area_ha:g} ha), {res_m:g} m, up to {max_scenes} scenes/window.\n"
    )

    import time as _time
    clock = {"t": 0.0, "i": 0}

    def on_start(rec):
        clock["t"] = _time.perf_counter()
        clock["i"] += 1
        console.print(
            f"  [{clock['i']}/{len(fires)}] {rec.event_id} "
            f"({(rec.area_ha or 0):,.0f} ha)...", end=" "
        )

    def on_done(rec, sample):
        bf_s = f"{100 * sample.burned_fraction:.0f}% burned" if sample.n_valid else "no valid px"
        console.print(
            f"[green]ok[/green] ({_time.perf_counter() - clock['t']:.0f}s, "
            f"{sample.n_valid:,} valid px, {bf_s})"
        )

    def on_error(rec, exc):
        console.print(f"[yellow]skip[/yellow] ({type(exc).__name__})")

    samples = build_optical_samples(
        fires, mosaic, on_start=on_start, on_done=on_done, on_error=on_error,
        max_cloud=max_cloud, max_scenes=max_scenes, res_m=res_m, cache_dir=cache_dir,
        with_stack=with_stack,
    )
    # Drop fires that cannot calibrate a threshold: all-cloud windows (no valid
    # predictor) or windows entirely inside or outside the burn.
    usable = {k: s for k, s in samples.items() if s.is_usable}
    dropped = set(samples) - set(usable)
    for d in dropped:
        console.print(f"  [yellow]drop[/yellow] {d}: not calibratable (empty or single-class)")
    if len(usable) < 3:
        console.print(f"[red]only {len(usable)} usable fires; need 3[/red]")
        raise typer.Exit(1)
    samples = usable
    console.print(f"\n[green]{len(samples)} usable fires[/green]. Calibrating per fold...\n")

    units = [u for u in reg.to_split_units() if u.uid in samples]
    manifest = leave_one_group_out(units, by="group")
    pixel_area_ha = (res_m ** 2) / 1e4  # 100 m pixel = 1 ha; 30 m = 0.09 ha

    strata = None
    if stratify_raster is not None:
        from vhagar.datasets.strata import assign_strata

        strata = assign_strata([r for r in fires if r.event_id in samples], stratify_raster)
        method = "perstratum"
        console.print(f"[dim]  strata: {len(set(strata.values()))} classes over {len(strata)} fires[/dim]")

    results = run_stage0(
        samples, manifest, pixel_area_ha=pixel_area_ha, n_reference=n_reference,
        method=method, strata=strata, objective=objective, seed=seed,
    )

    t = Table(title=f"T2 Stage-0, independent RBR vs MTBS ({region} {year}, leave-one-fire-out)")
    for col in ("held out", "thresh", "F1", "naive F1", "skill", "IoU", "adjusted ha", "95% CI"):
        t.add_column(col, justify="right")
    for r in results:
        skill_str = f"{r.skill_f1:+.3f}"
        t.add_row(
            r.held_out[:22], f"{r.threshold:.3f}", f"{r.f1:.3f}",
            f"{r.naive_f1:.3f}",
            f"[red]{skill_str}[/red]" if r.skill_f1 <= 0 else f"[green]{skill_str}[/green]",
            f"{r.iou:.3f}",
            f"{r.adjusted_burned_ha:,.0f}" if r.adjusted_burned_ha is not None else "-",
            f"±{r.ci95_ha:,.0f}" if r.ci95_ha is not None else "-",
        )
    console.print(t)
    s = summarise_stage0(results)
    if s.get("folds"):
        console.print(
            f"\n  [bold]{s['folds']} folds[/bold]: F1 {s['f1_mean']:.3f} ± {s['f1_std']:.3f}, "
            f"IoU {s['iou_mean']:.3f} ± {s['iou_std']:.3f}"
        )
        console.print(
            f"  [bold]skill over naive[/bold] (predict-all-burned F1 {s['naive_f1_mean']:.3f}): "
            f"{s['skill_f1_mean']:+.3f}, beating naive on "
            f"{s['folds_beating_naive']}/{s['folds']} folds"
        )
    console.print(
        "[dim]  Read F1 against the naive predict-all-burned baseline: on burn-heavy\n"
        "  windows a high F1 can be a window artefact with zero or negative skill. Only\n"
        "  a positive skill margin is an accuracy claim. See docs/11.[/dim]"
    )


@app.command("t2-continent-out")
def t2_continent_out_cmd(
    registry: Path = typer.Option(..., exists=True, help="registry Parquet (MTBS training)"),
    mosaic: Path = typer.Option(..., exists=True, help="MTBS thematic mosaic (US reference)"),
    emsr_manifest: Path = typer.Option(
        ..., exists=True, help="CSV: activation_id,delineation_path,event_date"
    ),
    min_area_ha: float = typer.Option(10000.0, help="US training fires at least this large"),
    max_fires: int = typer.Option(6, help="cap US training fires"),
    select: str = typer.Option(
        "size", help="US fire selection: size (climate-diverse, needed for per-stratum) | largest"
    ),
    max_scenes: int = typer.Option(4),
    res_m: float = typer.Option(100.0),
    cache_dir: Path = typer.Option(Path("data/t2_cache")),
    method: str = typer.Option("global", help="threshold: global | otsu | perstratum"),
    objective: str = typer.Option(
        "f1", help="threshold objective: f1 | iou | youden (balanced, robust to burn-heavy windows)"
    ),
    stratify_raster: Path = typer.Option(
        None, help="global class raster (e.g. Koppen); matches US to EU climate strata"
    ),
    n_reference: int = typer.Option(500),
    seed: int = typer.Option(0),
) -> None:
    """Leave-one-continent-out: train the RBR threshold on US MTBS fires, test on
    European Copernicus EMS fires. The architecture's headline generalisation
    number. The US samples are reused from the t2-stage0 cache; the European
    ones are pulled fresh (Sentinel-2 over Europe) and cached too.
    """
    import csv as _csv
    import time as _time

    from vhagar.datasets.t2_optical import (
        build_optical_sample,
        build_optical_samples,
        read_emsr_reference_on_grid,
        select_fires,
    )
    from vhagar.eval.t2_stage0 import evaluate_fold
    from vhagar.labels.ingest import read_emsr
    from vhagar.labels.registry import EventRegistry

    pixel_area_ha = (res_m ** 2) / 1e4

    # --- US training side (cached from t2-stage0) ---
    # Default to size-stratified selection, not largest-N. Per-stratum transfer
    # needs a climate-diverse training set: the largest CONUS fires cluster in a
    # few western zones (Dsb, BSk), so a "largest" pick has no Cfa/Csa fire to match
    # a European Cfa/Csa fire against, and per-stratum silently falls back to the
    # global threshold. Size stratification pulls in the small fires that carry the
    # other climate zones. See docs/11.
    reg = EventRegistry.from_parquet(registry)
    us = [
        r for r in reg
        if r.region == "conus" and r.ignition_date and r.ignition_date.year == 2021
        and (r.area_ha or 0) >= min_area_ha and r.tile_ids
    ]
    us = select_fires(us, max_fires, strategy=select)
    console.print(f"[bold]US training: {len(us)} MTBS fires[/bold] (reusing cache where present).")
    train = build_optical_samples(
        us, mosaic, cache_dir=cache_dir, max_scenes=max_scenes, res_m=res_m,
        on_error=lambda r, e: console.print(f"  [yellow]skip US[/yellow] {r.event_id}"),
    )
    train = {k: s for k, s in train.items() if s.is_usable}
    if len(train) < 3:
        console.print(f"[red]only {len(train)} usable US fires[/red]")
        raise typer.Exit(1)

    # --- European test side (EMS delineations) ---
    console.print("\n[bold]European test: Copernicus EMS fires[/bold] (pulling Sentinel-2).")
    test = {}
    eu_records = []
    with emsr_manifest.open() as fh:
        rows = list(_csv.DictReader(fh))
    for row in rows:
        path = row["delineation_path"]
        aid = row.get("activation_id") or Path(path).stem
        console.print(f"  {aid}...", end=" ")
        t0 = _time.perf_counter()
        try:
            rec = read_emsr(path, row["event_date"], activation_id=aid)
            ref = lambda grid, p=path: read_emsr_reference_on_grid(p, grid)  # noqa: E731
            s = build_optical_sample(
                rec, ref, cache_dir=cache_dir, max_scenes=max_scenes, res_m=res_m,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]skip[/yellow] ({type(exc).__name__})")
            continue
        if s.is_usable:
            test[rec.event_id] = s
            eu_records.append(rec)
            console.print(
                f"[green]ok[/green] ({_time.perf_counter() - t0:.0f}s, "
                f"{s.n_valid:,} px, {100 * s.burned_fraction:.0f}% burned)"
            )
        else:
            console.print("[yellow]drop[/yellow] (not calibratable)")
    if len(test) < 1:
        console.print("[red]no usable European fires[/red]")
        raise typer.Exit(1)

    # --- one fold: train US, test Europe ---
    strata = None
    if stratify_raster is not None:
        from vhagar.datasets.strata import assign_strata

        recs = [r for r in us if r.event_id in train] + eu_records
        strata = assign_strata(recs, stratify_raster)
        method = "perstratum"
        eu_classes = {strata.get(r.event_id) for r in eu_records}
        console.print(
            f"[dim]  strata: {len(set(strata.values()))} classes; EU fires in {eu_classes}[/dim]"
        )
    # Per-fire breakdown first. The pooled row below concatenates every EU fire's
    # pixels into one F1, so a few near-degenerate fires (very low burn fraction)
    # can mask clean per-zone transfer. The honest read is per fire, per climate
    # zone (docs/11, "Seven-fire EU generalisation").
    from vhagar.labels.emsr_fetch import koppen_name

    train_list = list(train.values())
    per = Table(title="Per-fire cross-continent skill (train US, test each EU fire)")
    for col in ("EU fire", "zone", "burn%", "F1", "naive", "skill"):
        per.add_column(col, justify="right")
    for eid, s in test.items():
        zone = koppen_name(strata.get(eid)) if strata else "-"
        try:
            rf = evaluate_fold(
                train_list, [s], pixel_area_ha=pixel_area_ha, n_reference=n_reference,
                method=method, strata=strata, objective=objective, seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            per.add_row(eid.split(":")[-1][:20], zone,
                        f"{100 * s.burned_fraction:.1f}", "-", "-", f"[dim]{type(exc).__name__}[/dim]")
            continue
        sk = f"{rf.skill_f1:+.3f}"
        per.add_row(
            eid.split(":")[-1][:20], zone, f"{100 * s.burned_fraction:.1f}",
            f"{rf.f1:.3f}", f"{rf.naive_f1:.3f}",
            f"[red]{sk}[/red]" if rf.skill_f1 <= 0 else f"[green]{sk}[/green]",
        )
    console.print(per)
    console.print(
        "[dim]  A low-burn-fraction fire (cloud-thinned or tiny) can be near single-\n"
        "  class and unmeasurable; read its skill next to its burn %. See docs/11.[/dim]\n"
    )

    r = evaluate_fold(
        list(train.values()), list(test.values()), held_out="EMSR (Europe)",
        pixel_area_ha=pixel_area_ha, n_reference=n_reference,
        method=method, strata=strata, objective=objective, seed=seed,
    )
    t = Table(title="T2 leave-one-continent-out (pooled): train US MTBS, test EU EMS")
    for col in ("test", "US fires", "EU fires", "thresh", "F1", "naive F1", "skill", "IoU", "adjusted ha", "95% CI"):
        t.add_column(col, justify="right")
    skill_str = f"{r.skill_f1:+.3f}"
    t.add_row(
        "EU EMS", str(len(train)), str(len(test)), f"{r.threshold:.3f}",
        f"{r.f1:.3f}", f"{r.naive_f1:.3f}",
        f"[red]{skill_str}[/red]" if r.skill_f1 <= 0 else f"[green]{skill_str}[/green]",
        f"{r.iou:.3f}",
        f"{r.adjusted_burned_ha:,.0f}" if r.adjusted_burned_ha is not None else "-",
        f"±{r.ci95_ha:,.0f}" if r.ci95_ha is not None else "-",
    )
    console.print(t)
    console.print(
        "[dim]  The threshold is calibrated only on US fires and never sees Europe.\n"
        "  Skill = F1 minus the predict-all-burned baseline; only a positive skill\n"
        "  margin on these balanced EU windows is a real accuracy claim. See docs/11.[/dim]"
    )


@app.command("t1-temporal")
def t1_temporal_cmd(
    n_days: int = typer.Option(4, help="length of the synthetic BT record"),
    fire_ramp_k_per_h: float = typer.Option(20.0, help="fire brightness-rise rate (K/h)"),
    fars: str = typer.Option("0.05,0.01,0.002", help="false-alarm rates to compare at"),
    seed: int = typer.Option(1),
    climatology: Path = typer.Option(
        None, help="optional real DiurnalClimatology .npz: report real 3.9um amplitude"
    ),
) -> None:
    """T1 differentiator: temporal-anomaly early detection vs an absolute-BT threshold.

    Demonstrates, on a synthetic 3.9 um series with a night fire injected, that flagging
    residual excursions against a per-pixel diurnal forecast detects the fire earlier
    than an absolute contextual threshold, at the same false-alarm rate. This is the
    numpy demonstration of the mechanism; the production forecaster is
    ``TemporalAnomalyNet`` trained on real clear-sky 3.9 um cubes (train_temporal_net,
    needs torch + a CMIP band-7 pull). See docs/12.
    """
    from vhagar.eval.t1_temporal import (
        DiurnalForecaster,
        climatology_diurnal_amplitude,
        early_detection_experiment,
        synthetic_bt_series,
    )

    if climatology is not None:
        a = climatology_diurnal_amplitude(climatology, channel="C07")
        console.print(
            f"[bold]Real 3.9um (C07) diurnal amplitude[/bold] over {a['n_pixels']:,} "
            f"GOES pixels:\n"
            f"  median [green]{a['amplitude_k_median']:.1f} K[/green] "
            f"(p25 {a['amplitude_k_p25']:.1f}, p90 {a['amplitude_k_p90']:.1f}); "
            f"per-hour sigma ~{a['sigma_k_median']:.2f} K.\n"
            f"[dim]  That amplitude is the night sensitivity an absolute contextual\n"
            f"  threshold sacrifices (it must sit ~one amplitude above the night baseline\n"
            f"  to avoid daytime false alarms); the residual detector recovers it. Real\n"
            f"  climatology grounding the synthetic lead-time demo below. docs/12.[/dim]"
        )

    hours, bt, fp, onset = synthetic_bt_series(
        n_days=n_days, fire_ramp_k_per_h=fire_ramp_k_per_h, seed=seed,
    )
    fc = DiurnalForecaster.fit(hours[:onset], bt[:, :onset], n_harmonics=3)   # clear-sky
    console.print(
        f"[bold]Synthetic 3.9um series[/bold]: {bt.shape[1]} steps @5min, "
        f"night fire injected at hour {hours[onset] % 24:.0f}."
    )
    t = Table(title="T1 temporal anomaly vs absolute-BT threshold (equal FAR)")
    for col in ("target FAR", "residual (min after onset)", "absolute (min)", "lead (min)"):
        t.add_column(col, justify="right")
    for far in [float(x) for x in fars.split(",") if x.strip()]:
        r = early_detection_experiment(hours, bt, fp, onset, fc, target_far=far)
        t.add_row(f"{far:.3f}", f"{r.residual_detect_min_after_onset:.0f}",
                  f"{r.absolute_detect_min_after_onset:.0f}",
                  f"[green]+{r.lead_minutes:.0f}[/green]" if r.lead_minutes > 0
                  else f"{r.lead_minutes:.0f}")
    console.print(t)
    console.print(
        "[dim]  Residual-against-diurnal-baseline catches the fire as soon as it lifts BT\n"
        "  above the pixel's own night baseline; the absolute cut must wait for the global\n"
        "  threshold. Same mechanism on real 3.9um cubes with TemporalAnomalyNet. docs/12.[/dim]"
    )


@app.command("t1-pull-cube")
def t1_pull_cube_cmd(
    out: Path = typer.Argument(..., help="output .npz for the BT cube"),
    start: str = typer.Option(..., help="start datetime, YYYY-MM-DDTHH:MM (UTC)"),
    end: str = typer.Option(..., help="end datetime, YYYY-MM-DDTHH:MM (UTC), inclusive"),
    bbox: str = typer.Option(..., help="west,south,east,north in degrees. Keep it small."),
    satellite: int = typer.Option(18, help="GOES satellite number"),
    channel: str = typer.Option("C07", help="ABI emissive channel (C07 is the 3.9um fire band)"),
    cadence_min: int = typer.Option(5, help="frame spacing; 5 is native CONUS cadence"),
    workers: int = typer.Option(8, help="concurrent frame reads"),
) -> None:
    """Pull a time-ordered 3.9um BT cube [T,H,W] over a small region for the temporal detector.

    Reads GOES ABI L2 CMIP from the public S3 archive, crops each 5-minute frame to the
    bbox, and stacks them on the one stationary ABI grid into an .npz carrying its own UTC
    timestamps and geometry. This is the real input for `t1-temporal-real`. Needs s3fs +
    xarray + network. Keep the bbox small (a fire-prone box, not CONUS): the cube is dense.
    See docs/12.
    """
    from datetime import datetime as _dt

    from vhagar.archive.temporal_cube import TemporalCubeConfig, pull_bt_cube

    def _parse(s: str) -> _dt:
        for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
            with contextlib.suppress(ValueError):
                return _dt.strptime(s, fmt).replace(tzinfo=UTC)
        console.print("[red]datetimes must be YYYY-MM-DDTHH:MM or YYYY-MM-DD[/red]")
        raise typer.Exit(1)

    t0, t1 = _parse(start), _parse(end)
    parts = [p for p in bbox.split(",") if p.strip()]
    if len(parts) != 4:
        console.print("[red]bbox needs four numbers: west,south,east,north[/red]")
        raise typer.Exit(1)
    box = tuple(float(p) for p in parts)

    cfg = TemporalCubeConfig(
        out_path=out, start=t0, end=t1, bbox=box, satellite=satellite,
        channel=channel, cadence_min=cadence_min, workers=workers,
    )
    console.print(f"[bold]Pulling {channel} cube[/bold] {t0:%Y-%m-%d %H:%M}..{t1:%H:%M} "
                  f"over {box} from GOES-{satellite}.")
    with console.status("reading frames from S3..."):
        cube = pull_bt_cube(cfg)
    valid = float(np.isfinite(cube.bt).mean())
    console.print(
        f"[green]cube {cube.shape}[/green] ({len(cube.times)} frames @ {cadence_min}min, "
        f"{100 * valid:.0f}% valid pixels) saved to {out}."
    )


@app.command("t1-temporal-real")
def t1_temporal_real_cmd(
    cube_path: Path = typer.Argument(..., exists=True, help="BT cube .npz from t1-pull-cube"),
    detections: Path = typer.Option(Path("data/detections/detections"), help="FDC parquet root"),
    fars: str = typer.Option("0.05,0.01,0.002", help="false-alarm rates to compare at"),
    clear_frac: float = typer.Option(0.6, help="leading fraction of frames used as clear-sky baseline"),
    n_bins: int = typer.Option(24, help="diurnal bins for the baseline"),
    far_bins: int = typer.Option(
        1, help="time-of-day bins for the FAR threshold; >1 calibrates night separately"
    ),
    min_consec: int = typer.Option(
        3, help="consecutive exceedances to confirm a detection (filters pre-fire blips)"
    ),
    learned: bool = typer.Option(
        False, "--learned/--baseline",
        help="use the learned TemporalAnomalyNet forecaster (needs torch) vs the hourly mean",
    ),
    window: int = typer.Option(6, help="learned: past frames the forecaster sees (6 = 30 min)"),
    epochs: int = typer.Option(10, help="learned: training epochs on the clear-sky span"),
    no_solar: bool = typer.Option(False, "--no-solar", help="learned: drop the solar-zenith covariate"),
) -> None:
    """Real lead time: residual-vs-forecast early detection timed against GOES FDC.

    Loads a pulled 3.9um cube, builds an expected-BT forecast on the leading clear-sky
    fraction, and for every pixel FDC eventually flags, measures how many minutes earlier
    the residual persistently crossed a matched-FAR threshold than FDC's first detection.
    The threshold is calibrated on fire-free pixels (equal false-alarm rate); ``--far-bins``
    calibrates night separately; ``--min-consec`` requires confirmation. Default forecaster
    is the NaN-safe hourly diurnal mean; ``--learned`` trains ``TemporalAnomalyNet`` (with a
    solar-zenith covariate) instead, the state-of-the-art path. See docs/12.
    """
    from vhagar.archive.temporal_cube import fdc_first_detection_grid, load_bt_cube
    from vhagar.eval.t1_temporal import (
        HourlyBaselineForecaster,
        baseline_contamination,
        learned_residuals,
        real_lead_experiment,
    )

    cube = load_bt_cube(cube_path)
    T, H, W = cube.shape
    hours = cube.hours_of_day()
    bbox = (float(cube.lon.min()), float(cube.lat.min()),
            float(cube.lon.max()), float(cube.lat.max()))
    console.print(f"[bold]Cube[/bold] {cube.shape} ({cube.channel}), "
                  f"{cube.times[0]:%Y-%m-%d %H:%M}..{cube.times[-1]:%H:%M} UTC.")

    bt2d = cube.bt.reshape(T, H * W).T                       # [n_pixels, T]
    clear = np.zeros(T, dtype=bool)
    clear_end = max(1, int(clear_frac * T))
    clear[:clear_end] = True
    if learned:
        from vhagar.archive.temporal_cube import solar_zenith_cube
        cov = None
        if not no_solar:
            zen = solar_zenith_cube(cube.lat, cube.lon, cube.times)   # [T,H,W] degrees
            cov = np.cos(np.radians(zen))[:, None]                    # [T,1,H,W] insolation factor
        forecaster_label = f"learned TemporalAnomalyNet (window {window}, {epochs} epochs" \
                           f"{', +solar' if not no_solar else ''})"
        with console.status(f"training {forecaster_label} on {clear_end} clear frames..."):
            resid = learned_residuals(cube.bt, clear_end, window=window, epochs=epochs,
                                      covariates=cov)
    else:
        forecaster_label = f"hourly diurnal mean ({n_bins} bins)"
        fc = HourlyBaselineForecaster.fit(hours, bt2d, n_bins=n_bins, clear_mask=clear)
        resid = fc.residual(hours, bt2d)

    first_idx = fdc_first_detection_grid(detections, bbox, cube.times, cube.lat, cube.lon)
    n_fire = int((first_idx >= 0).sum())
    console.print(f"FDC flags {n_fire} of {H * W} cube pixels in window. "
                  f"Forecaster: [bold]{forecaster_label}[/bold].")
    if n_fire == 0:
        console.print("[yellow]No FDC detections landed in this cube; pick a box/window with "
                      "a known fire to measure lead time.[/yellow]")
        return

    # Honesty guard: the diurnal baseline is only a clean reference if it is fit on
    # fire-free frames. If a fire ignites inside the clear window, its hot BT contaminates
    # its own baseline and the lead-time table is meaningless (inflated at loose FAR,
    # negative at strict). Refuse to present the table in that case.
    contam = baseline_contamination(first_idx.ravel(), clear)
    if contam > 0.2:
        last_clear = cube.times[int(np.flatnonzero(clear)[-1])]
        console.print(
            f"[red]Baseline contaminated:[/red] {contam:.0%} of fire pixels are already "
            f"flagged by FDC within the clear-sky window (ends {last_clear:%m-%d %H:%M} UTC).\n"
            "[yellow]The diurnal baseline is fit on the fire's own hot BT, so the lead-time\n"
            "numbers below would be an artefact, not a real lead. Re-pull a window whose\n"
            "leading span is genuinely pre-ignition, or lower --clear-frac so the baseline\n"
            "ends before the fire starts.[/yellow]"
        )

    thr_kind = f"{far_bins} time-of-day bins" if far_bins > 1 else "one global threshold"
    fc_kind = "learned" if learned else "hourly-mean"
    t = Table(title=f"T1 {fc_kind} residual vs GOES FDC first detection "
                    f"(real cube, equal FAR, {thr_kind}, {min_consec}-frame confirm)")
    for col in ("target FAR", "fire pixels", "residual led", "median lead (min)", "IQR"):
        t.add_column(col, justify="right")
    for far in [float(x) for x in fars.split(",") if x.strip()]:
        r = real_lead_experiment(resid, first_idx.ravel(), target_far=far,
                                 hours=hours, far_bins=far_bins, min_consec=min_consec,
                                 eval_start=clear_end)
        t.add_row(
            f"{far:.3f}", f"{r.n_fire_pixels}",
            f"{r.frac_residual_led:.0%}",
            f"[green]+{r.median_lead_min:.0f}[/green]" if r.median_lead_min > 0
            else f"{r.median_lead_min:.0f}",
            f"{r.p25_lead_min:.0f}..{r.p75_lead_min:.0f}",
        )
    console.print(t)
    console.print(
        "[dim]  Positive lead: the residual persistently crossed its matched-FAR threshold\n"
        "  before FDC's first detection on that pixel. --learned trains TemporalAnomalyNet\n"
        "  with a solar covariate in place of the hourly mean. docs/12.[/dim]"
    )
    if far_bins <= 1:
        console.print(
            "[dim]  A single global threshold is set by daytime residual variance and can\n"
            "  desensitise a night fire; try --far-bins 6 to calibrate night separately.[/dim]"
        )


@app.command("t1-cohort-select")
def t1_cohort_select_cmd(
    detections: Path = typer.Option(Path("data/detections/detections"), help="FDC parquet root"),
    out: Path = typer.Option(Path("cohort/cohort.json"), help="cohort spec JSON to write"),
    per_stratum: int = typer.Option(3, help="fires to keep per stratum"),
    box_deg: float = typer.Option(0.35, help="cube box size in degrees"),
) -> None:
    """Select a stratified cohort of fires to test the temporal detector properly (n>1).

    A single fire cannot answer whether the residual detector beats FDC; its edge is
    specifically the cold-start night fire an absolute threshold is slowest on. This
    clusters the FDC parquet into fires, classifies each by ignition local-solar-time and
    early ramp, and picks ``night_coldstart`` fires (the edge) plus ``day`` controls,
    writing a spec of ready-to-pull cube windows and printing the pull commands. See docs/12.
    """
    import json as _json

    from vhagar.archive.temporal_cube import select_fire_cohort

    specs = select_fire_cohort(detections, box_deg=box_deg, per_stratum=per_stratum)
    if not specs:
        console.print("[yellow]No fires met the cohort criteria (baseline + detections).[/yellow]")
        raise typer.Exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps([s.to_json() for s in specs], indent=2), encoding="utf-8")

    tbl = Table(title="Stratified fire cohort for the temporal lead-time test")
    for col in ("name", "stratum", "lat", "lon", "local hr", "ramp MW/h", "dets", "clear_frac"):
        tbl.add_column(col, justify="right")
    for s in specs:
        tbl.add_row(s.name, s.stratum, f"{s.lat}", f"{s.lon}", f"{s.local_solar_hour}",
                    f"{s.ramp_slope_mw_per_h:.0f}", f"{s.n_detections}", f"{s.clear_frac}")
    console.print(tbl)
    console.print(f"[green]Wrote {len(specs)} specs to {out}.[/green] Pull each cube (needs S3):")
    for s in specs:
        console.print(f"  {s.pull_command(cube_dir=str(out.parent))}")
    console.print(
        "\nThen score the whole cohort:\n"
        f"  vhagar t1-temporal-cohort --spec {out} "
        "--detections data\\detections\\detections --far 0.01 --far-bins 6 --min-consec 3"
    )


@app.command("t1-cohort-pull")
def t1_cohort_pull_cmd(
    spec: Path = typer.Option(Path("cohort/cohort.json"), exists=True, help="cohort spec JSON"),
    workers: int = typer.Option(8, help="concurrent frame reads per cube"),
    refetch: bool = typer.Option(False, "--refetch", help="re-pull cubes already on disk"),
) -> None:
    """Pull every cube in a cohort spec in one command (resumable). Needs S3.

    Runs the per-fire ``t1-pull-cube`` for you over the whole spec, skipping cubes already on
    disk unless ``--refetch``. Then score with ``t1-temporal-cohort``. See docs/12.
    """
    from vhagar.archive.temporal_cube import cohort_pull

    def _prog(i, n, name, status):
        colour = {"ok": "green", "skip": "dim", "fail": "red"}.get(status, "white")
        console.print(f"  [{i}/{n}] [{colour}]{status}[/{colour}] {name}")

    with console.status("pulling cohort cubes from S3..."):
        res = cohort_pull(spec, workers=workers, only_missing=not refetch, progress=_prog)
    console.print(f"[green]pulled {len(res['pulled'])}[/green], skipped {len(res['skipped'])}"
                  f", failed {len(res['failed'])}.")
    for f in res["failed"]:
        console.print(f"  [red]{f}[/red]")
    console.print(
        f"\nScore the cohort:\n  vhagar t1-temporal-cohort --spec {spec} "
        "--detections data\\detections\\detections --far 0.01 --far-bins 6 --min-consec 3"
    )


@app.command("t1-temporal-cohort")
def t1_temporal_cohort_cmd(
    spec: Path = typer.Option(..., exists=True, help="cohort spec JSON from t1-cohort-select"),
    detections: Path = typer.Option(Path("data/detections/detections"), help="FDC parquet root"),
    far: float = typer.Option(0.01, help="single false-alarm rate to compare at"),
    far_bins: int = typer.Option(6, help="time-of-day bins for the FAR threshold"),
    min_consec: int = typer.Option(3, help="consecutive exceedances to confirm a detection"),
    learned: bool = typer.Option(False, "--learned/--baseline", help="learned forecaster (torch)"),
    epochs: int = typer.Option(15, help="learned: training epochs"),
    window: int = typer.Option(6, help="learned: forecaster window"),
) -> None:
    """Score the whole fire cohort and aggregate lead over FDC per stratum.

    For each fire in the spec (whose cube must already be pulled), builds the residual (hourly
    mean or ``--learned`` TemporalAnomalyNet), runs the matched-FAR / far-bins / persistence
    lead-time eval vs FDC, then aggregates by stratum. The scientific read: does the residual
    detector lead FDC on ``night_coldstart`` fires (its theoretical edge) more than on ``day``
    controls? A single fire cannot say; a stratified cohort can. See docs/12.
    """
    import json as _json

    from vhagar.archive.temporal_cube import (
        fdc_first_detection_grid,
        load_bt_cube,
        solar_zenith_cube,
    )
    from vhagar.eval.t1_temporal import (
        HourlyBaselineForecaster,
        baseline_contamination,
        cohort_lead_summary,
        learned_residuals,
        real_lead_experiment,
    )

    specs = _json.loads(spec.read_text(encoding="utf-8"))
    per_fire = []
    skipped = []
    for s in specs:
        cube_path = spec.parent / f"{s['name']}.npz"
        if not cube_path.exists():
            skipped.append(s["name"])
            continue
        cube = load_bt_cube(cube_path)
        T, H, W = cube.shape
        hours = cube.hours_of_day()
        bbox = (float(cube.lon.min()), float(cube.lat.min()),
                float(cube.lon.max()), float(cube.lat.max()))
        clear = np.zeros(T, dtype=bool)
        clear_end = max(1, int(float(s["clear_frac"]) * T))
        clear[:clear_end] = True
        bt2d = cube.bt.reshape(T, H * W).T
        if learned:
            zen = solar_zenith_cube(cube.lat, cube.lon, cube.times)
            cov = np.cos(np.radians(zen))[:, None]
            resid = learned_residuals(cube.bt, clear_end, window=window, epochs=epochs,
                                      covariates=cov)
        else:
            fc = HourlyBaselineForecaster.fit(hours, bt2d, clear_mask=clear)
            resid = fc.residual(hours, bt2d)
        first_idx = fdc_first_detection_grid(detections, bbox, cube.times, cube.lat, cube.lon)
        if int((first_idx >= 0).sum()) == 0:
            skipped.append(s["name"] + " (no FDC in box)")
            continue
        contam = baseline_contamination(first_idx.ravel(), clear)
        if contam > 0.2:
            skipped.append(s["name"] + f" (baseline {contam:.0%} contaminated)")
            continue
        r = real_lead_experiment(resid, first_idx.ravel(), target_far=far, hours=hours,
                                 far_bins=far_bins, min_consec=min_consec,
                                 eval_start=clear_end)
        per_fire.append((s["stratum"], r))
        if r.n_fire_pixels == 0:
            console.print(f"  {s['name']}: [red]no detection[/red] "
                          f"(0/{r.n_fire_pixels_total} fire px flagged in held-out window)")
        else:
            console.print(f"  {s['name']}: detected {r.n_fire_pixels}/{r.n_fire_pixels_total} px, "
                          f"median lead {r.median_lead_min:+.0f} min")

    if not per_fire:
        console.print("[yellow]No cube scored. Pull the cohort cubes first "
                      f"(t1-cohort-select). Skipped: {skipped}[/yellow]")
        raise typer.Exit(1)

    summary = cohort_lead_summary(per_fire)
    fc_kind = "learned" if learned else "hourly-mean"
    tbl = Table(title=f"T1 {fc_kind} residual vs FDC by stratum "
                      f"(FAR {far}, {far_bins} bins, {min_consec}-confirm)")
    for col in ("stratum", "fires", "detection rate", "fires led", "median lead (det)",
                "pooled px lead", "px led"):
        tbl.add_column(col, justify="right")

    def _lead(x):
        if x != x:                                    # NaN: nothing detected
            return "[dim]n/a[/dim]"
        return f"[green]+{x:.0f}[/green]" if x > 0 else f"{x:.0f}"

    for stratum in ("night_coldstart", "day"):
        if stratum not in summary:
            continue
        v = summary[stratum]
        tbl.add_row(
            stratum, f"{v['n_fires']:.0f}",
            f"{v['detection_rate']:.0%} px ({v['frac_fires_detected']:.0%} fires)",
            f"{v['frac_fires_led']:.0%}",
            _lead(v["median_fire_lead_min"]), _lead(v["pooled_pixel_median_lead_min"]),
            f"{v['pooled_pixel_frac_led']:.0%}",
        )
    console.print(tbl)
    if skipped:
        console.print(f"[dim]  skipped: {', '.join(skipped)}[/dim]")
    console.print(
        "[dim]  The scientific read: a residual detector should lead FDC on night_coldstart\n"
        "  fires (an absolute threshold is slowest there) more than on day controls. docs/12.[/dim]"
    )


@app.command("t1-classify")
def t1_classify_cmd(
    detections: Path = typer.Option(Path("data/detections/detections"), help="FDC parquet root"),
    firms_csv: Path = typer.Option(..., exists=True, help="VIIRS FIRMS CSV (labels)"),
    thin_deg: float = typer.Option(0.02, help="spatial thinning cell for speed"),
    thin_minutes: int = typer.Option(60, help="temporal thinning bin"),
    folds: int = typer.Option(5),
) -> None:
    """T1 Stage-2 preview: does raw lat/lon leak in a GOES fire-event classifier?

    Labels each GOES detection by VIIRS coincidence, trains a gradient-boosted
    classifier with and without raw lon/lat, and reports F1 under random, cell-grouped
    (event-aware), and 5-degree spatial-block splits. If lon/lat lifts the random-split
    F1 but that lift turns negative on the spatial block, the coordinates were
    memorising geography, not fire physics, the architecture's warning, on our data.
    Needs scikit-learn. See docs/12.
    """
    import glob as _glob

    import numpy as np
    import pandas as pd

    from vhagar.eval.t1_classifier import build_samples, evaluate_leakage
    from vhagar.io.firms import parse_firms_csv

    files = sorted(_glob.glob(f"{detections}/**/*.parquet", recursive=True))
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["t"] = pd.to_datetime(df["t"])
    kl = (df["lon"] / thin_deg).round().astype("int64")
    ka = (df["lat"] / thin_deg).round().astype("int64")
    kt = df["t"].astype("int64") // (thin_minutes * 60 * 1_000_000_000)
    df = df.loc[~pd.DataFrame({"a": kl, "b": ka, "c": kt}).duplicated()].reset_index(drop=True)
    recs = parse_firms_csv(firms_csv.read_text())
    vll = np.array([[r.longitude, r.latitude] for r in recs])
    vt = np.array([r.acq_datetime.timestamp() for r in recs])
    console.print(f"[bold]Labelling[/bold] {len(df):,} thinned GOES detections by VIIRS coincidence...")
    s = build_samples(df, vll, vt)
    console.print(f"  {len(s.y):,} samples, VIIRS-confirmed rate {s.y.mean():.3f}")
    try:
        r = evaluate_leakage(s, n_folds=folds)
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    t = Table(title="T1 lat/lon leakage: classifier F1 across leakage-proof splits")
    for col in ("split", "physical", "+ lat/lon", "lat/lon gain"):
        t.add_column(col, justify="right")
    order = ["random", "cell_grouped", "spatial_block_5deg"]
    for k in order:
        vv = r[k]
        g = vv["latlon_gain"]
        t.add_row(k, f"{vv['physical']:.3f}", f"{vv['with_latlon']:.3f}",
                  f"[red]{g:+.3f}[/red]" if g < 0 else f"[green]{g:+.3f}[/green]")
    console.print(t)
    console.print(
        "[dim]  F1 falling from random -> spatial-block is the honest generalisation gap;\n"
        "  a lat/lon gain that is positive in-region and negative out-of-region is the\n"
        "  leak. Production T1 features exclude raw coordinates by construction. docs/12.[/dim]"
    )


@app.command("t3-ignition")
def t3_ignition_cmd(
    synthetic: bool = typer.Option(True, help="run the synthetic reporting-bias demo"),
    occurrence: Path = typer.Option(None, help="real: presence cell-days parquet"),
    candidates: Path = typer.Option(None, help="real: target-group candidate cell-days parquet"),
    features: str = typer.Option("", help="real: comma-separated covariate columns"),
    tau: float = typer.Option(0.0, help="real: true base rate (0 = infer presences/candidates)"),
    n_cells: int = typer.Option(3500, help="synthetic cells per year"),
    neg_per_pos: float = typer.Option(3.0, help="background pseudo-absences per presence"),
    folds: int = typer.Option(4, help="spatial-block CV folds"),
    seed: int = typer.Option(0),
) -> None:
    """T3 Layer 2: cause-stratified, blocked, properly-scored ignition probability.

    The state of the art here is gradient boosting, not a deep net (ECMWF's
    operational Probability-of-Fire), so this fits a gradient-boosted classifier
    per cause under spatial-block CV and scores it only with proper scores
    (AUPRC, Brier + Murphy decomposition, ECE, Brier skill vs a base-rate
    climatology), with King-Zeng rare-event correction and lon/lat excluded.

    It also demonstrates the sampling trap (docs/00 5.6): naive random background
    inflates apparent skill and leans on the human-footprint covariate (an
    observation artefact); target-group background drawn from the same reporting
    process collapses that reliance and reveals the honest skill. docs/14.
    """
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        console.print("[red]t3-ignition needs scikit-learn[/red]")
        raise typer.Exit(1) from exc

    from vhagar.datasets import danger as dg
    from vhagar.eval.danger import evaluate_ignition, top_features

    if not synthetic:
        if occurrence is None or candidates is None or not features:
            console.print("[red]real mode needs --occurrence, --candidates and --features[/red]")
            raise typer.Exit(1)
        import pandas as pd
        fcols = [c.strip() for c in features.split(",") if c.strip()]
        pres_df = pd.read_parquet(occurrence)
        cand_df = pd.read_parquet(candidates)
        pres, cand, fn = dg.frames_to_records(pres_df, cand_df, fcols)
        base_rate = tau if tau > 0 else float(len(pres_df) / max(len(cand_df), 1))
        s = dg.assemble_ignition_samples(pres, cand, fn, np.random.default_rng(seed + 7),
                                         tau=base_rate, neg_per_pos=neg_per_pos,
                                         use_target_group=True, stratify=False)
        r = evaluate_ignition(s, n_folds=folds, seed=seed, prior_correct=True)
        tf = top_features(s, folds, seed, k=5)
        p = r["pooled"]
        console.print(f"[bold]Real ignition run[/bold]: {len(pres_df):,} presences, "
                      f"{len(cand_df):,} candidates, base rate {base_rate:.4f}\n")
        rt = Table(title="T3 ignition (real data, target-group background, prior-corrected)")
        for c in ("AUPRC", "Brier", "reliability", "resolution", "ECE", "BSS vs climo"):
            rt.add_column(c, justify="right")
        rt.add_row(f"{p['auprc']:.3f}", f"{p['brier']:.4f}", f"{p['reliability']:.4f}",
                   f"{p['resolution']:.4f}", f"{p['ece']:.4f}", f"{p['bss_vs_climatology']:+.3f}")
        console.print(rt)
        h = r.get("human", {}).get("auprc", float("nan"))
        ltg = r.get("lightning", {}).get("auprc", float("nan"))
        console.print(f"[dim]  cause-stratified AUPRC  human {h:.3f}  lightning {ltg:.3f}[/dim]")
        console.print("[dim]  top features: " + ", ".join(f"{n}({v:+.3f})" for n, v in tf) + "[/dim]")
        return

    rng = np.random.default_rng(seed)
    pres, cand, fn, tau = dg.synthetic_reporting_scenario(rng, n_cells=n_cells)
    bias = float(pres["people"].mean() - cand["people"].mean())
    console.print(
        f"[bold]Synthetic reporting-bias world[/bold]: {len(pres['id']):,} reported ignitions, "
        f"true base rate {tau:.3f}, reporting bias in human footprint +{bias:.3f}\n")

    rows = []
    for tag, kw in [("naive random", dict(use_target_group=False, stratify=False)),
                    ("target-group", dict(use_target_group=True, stratify=False))]:
        s = dg.assemble_ignition_samples(pres, cand, fn, np.random.default_rng(seed + 7),
                                         tau=tau, neg_per_pos=neg_per_pos, **kw)
        r = evaluate_ignition(s, n_folds=folds, seed=seed, prior_correct=True)
        tf = dict(top_features(s, folds, seed, k=5))
        rows.append((tag, r, tf.get("people", 0.0) + tf.get("roads", 0.0)))

    t = Table(title="T3 ignition: proper scores under blocked spatial CV (prior-corrected)")
    for c in ("sampling", "AUPRC", "Brier", "reliability", "resolution",
              "ECE", "BSS vs climo", "footprint imp"):
        t.add_column(c, justify="right")
    for tag, r, ppl in rows:
        p = r["pooled"]
        t.add_row(tag, f"{p['auprc']:.3f}", f"{p['brier']:.4f}", f"{p['reliability']:.4f}",
                  f"{p['resolution']:.4f}", f"{p['ece']:.4f}", f"{p['bss_vs_climatology']:+.3f}",
                  f"{ppl:+.3f}")
    console.print(t)
    for tag, r, _ in rows:
        h = r.get("human", {}).get("auprc", float("nan"))
        ltg = r.get("lightning", {}).get("auprc", float("nan"))
        console.print(f"[dim]  {tag}: cause-stratified AUPRC  human {h:.3f}  lightning {ltg:.3f}[/dim]")
    console.print(
        "[dim]  The trap: naive background inflates AUPRC and leans on the human-footprint\n"
        "  covariate (an observation artefact); target-group background collapses that\n"
        "  reliance and reveals the honest, lower skill. Probabilities are King-Zeng\n"
        "  prior-corrected to the true base rate. docs/14.[/dim]")


@app.command("t3-expected-ba")
def t3_expected_ba_cmd(
    synthetic: bool = typer.Option(True, help="run the synthetic heavy-tailed demo"),
    fires: Path = typer.Option(None, help="real: per-fire parquet with area_ha + features + lon/lat/year"),
    features: str = typer.Option("", help="real: comma-separated covariate columns"),
    p_ignition: float = typer.Option(0.02, help="illustrative P(ignition) for the E[BA] combination"),
    n: int = typer.Option(4000, help="synthetic fires"),
    folds: int = typer.Option(4),
    seed: int = typer.Option(0),
    no_tail: bool = typer.Option(False, "--no-tail", help="disable the GPD extreme-value tail"),
) -> None:
    """T3: expected burned area E[BA] = P(ignition) x E[BA | ignition].

    The heavy-tailed quantity. Fits a log-space quantile-boosting distribution
    with a generalised-Pareto tail for the extremes, and scores it with CRPS and
    pinball loss under spatial-block CV, never RMSE (RMSE is reported only to
    show its tail-driven instability). lon/lat are excluded from the model. docs/14.
    """
    try:
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        console.print("[red]t3-expected-ba needs scikit-learn and scipy[/red]")
        raise typer.Exit(1) from exc

    from vhagar.eval import burned_area as ba

    if synthetic:
        X, area, _yr, lon, lat, _fn = ba.synthetic_burned_area_scenario(
            np.random.default_rng(seed), n=n)
        src = f"synthetic heavy-tailed world, {len(area):,} fires"
    else:
        if fires is None or not features:
            console.print("[red]real mode needs --fires and --features (parquet with area_ha, lon, lat, year)[/red]")
            raise typer.Exit(1)
        import pandas as pd
        df = pd.read_parquet(fires)
        fcols = [c.strip() for c in features.split(",") if c.strip()]
        X = df[fcols].to_numpy(dtype=float)
        area = df["area_ha"].to_numpy(dtype=float)
        lon, lat = df["lon"].to_numpy(dtype=float), df["lat"].to_numpy(dtype=float)
        src = f"{fires.name}, {len(area):,} fires"

    r = ba.evaluate_expected_ba(X, area, lon, lat, n_folds=folds, seed=seed, use_tail=not no_tail)
    console.print(f"[bold]E[BA | ignition][/bold]: {src}; median {np.median(area):.0f} ha, "
                  f"p99 {np.quantile(area, 0.99):.0f} ha\n")
    t = Table(title="T3 expected burned area: proper scoring under blocked spatial CV")
    for c in ("CRPS", "CRPS climatology", "CRPS skill", "RMSE fold mean", "RMSE fold std"):
        t.add_column(c, justify="right")
    t.add_row(f"{r['crps']:.1f}", f"{r['crps_climatology']:.1f}",
              f"{r['crps_skill_vs_climatology']:+.3f}", f"{r['rmse_mean']:.0f}", f"{r['rmse_std']:.0f}")
    console.print(t)
    pin = ", ".join(f"q{int(k * 100)} {v:.0f}" for k, v in r["pinball"].items())
    console.print(f"[dim]  pinball loss by quantile: {pin}[/dim]")
    m = ba.BurnedAreaModel(seed=seed, use_tail=not no_tail).fit(X, area)
    eba = ba.expected_burned_area(np.full(X.shape[0], p_ignition), m.predict_quantiles(X))
    console.print(f"[dim]  E[BA] = P(ig) x E[BA|ig]: at P(ig)={p_ignition}, mean E[BA] {eba.mean():.2f} ha/cell "
                  f"(E[BA|ig] mean {eba.mean() / p_ignition:.0f} ha).[/dim]")
    console.print(
        "[dim]  Scored with CRPS + pinball, never RMSE: RMSE's fold-to-fold std is a large fraction\n"
        "  of its mean (tail-driven), while CRPS gives a stable skill over climatology. docs/14.[/dim]")


@app.command("t3-challenger")
def t3_challenger_cmd(
    obs_noise: float = typer.Option(0.15, help="per-cell observation noise (where spatial context helps)"),
    intercept: float = typer.Option(-9.0, help="danger intercept; more negative = lower base rate"),
    folds: int = typer.Option(4, help="leave-time-block-out folds"),
    seed: int = typer.Option(0),
    torch_net: bool = typer.Option(False, "--torch", help="also train the deep U-Net challenger (needs pytorch)"),
) -> None:
    """T3 Layer 3: the deep challenger, in shadow mode.

    Gridded ignition danger, verified spatially. Fits a pointwise gradient-boosting
    baseline and a spatial challenger (the booster on neighborhood-pooled features)
    under leave-time-block-out CV, scores both with Fractions Skill Score at
    40/80/120 km plus base-rate-preserving AUPRC and Brier, and applies the
    promotion gate: the challenger is promoted only if it beats the baseline on
    AUPRC AND Brier. The torch U-Net (models/ignition_conv.py) trains with a
    soft-FSS loss; --torch runs it (GPU box). docs/14.
    """
    try:
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        console.print("[red]t3-challenger needs scikit-learn and scipy[/red]")
        raise typer.Exit(1) from exc

    from vhagar.eval.danger_grid import shadow_evaluate, synthetic_ignition_grid

    X, ev, _fn, ckm = synthetic_ignition_grid(np.random.default_rng(seed),
                                              intercept=intercept, obs_noise=obs_noise)
    r = shadow_evaluate(X, ev, cell_km=ckm, n_folds=folds, seed=seed)
    b, c = r["baseline"], r["challenger"]
    console.print(f"[bold]Gridded ignition danger[/bold]: {X.shape[0]} days x {X.shape[2]}x{X.shape[3]} cells "
                  f"@ {ckm:.0f} km, base rate {r['base_rate']:.3f}, obs noise {obs_noise}\n")
    t = Table(title="T3 deep challenger, shadow mode: spatial verification (FSS) + pixel gate (AUPRC/Brier)")
    for col in ("model", "AUPRC", "Brier", "FSS 40km", "FSS 80km", "FSS 120km"):
        t.add_column(col, justify="right")
    for name, s in (("pointwise baseline (GBDT)", b), ("spatial challenger", c)):
        t.add_row(name, f"{s['auprc']:.3f}", f"{s['brier']:.4f}",
                  f"{s['fss'][40]:.3f}", f"{s['fss'][80]:.3f}", f"{s['fss'][120]:.3f}")
    console.print(t)
    colour = "green" if r["promote"] else "yellow"
    console.print(f"[{colour}]  {r['verdict']}[/{colour}]")

    if torch_net:
        try:
            from vhagar.models.ignition_conv import predict_spatial, train_spatial
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        from vhagar.eval.metrics import average_precision, brier_score
        cut = int(X.shape[0] * 0.7)
        net = train_spatial(X[:cut], ev[:cut], seed=seed)
        pred = predict_spatial(net, X[cut:])
        yt = ev[cut:].reshape(-1)
        console.print(f"[dim]  deep U-Net (temporal holdout): AUPRC {average_precision(yt, pred.reshape(-1)):.3f} "
                      f"Brier {brier_score(yt, pred.reshape(-1)):.4f}. Promote only on a full blocked run. docs/14.[/dim]")


@app.command("t4-spread")
def t4_spread_cmd(
    n_fires: int = typer.Option(12, help="synthetic fires per regime"),
    ros_err: float = typer.Option(0.7, help="spatially-correlated ROS estimate error (the ceiling)"),
    seed: int = typer.Option(0),
) -> None:
    """T4 spread: physics level-set forecast vs the mandatory persistence baselines.

    Grows synthetic fires to truth with the Fast Marching solver (with hidden
    suppression, spotting and fine-scale fuel heterogeneity), then forecasts from
    the perimeter at t0 with a spatially-biased ROS estimate and scores on the
    INCREMENTAL new-burn region only (never cumulative). Reports AP, IoU, Dice,
    burned-area ratio and arrival-time MAE against persistence and
    persistence+buffer. docs/15.
    """
    try:
        import scipy  # noqa: F401
    except ImportError as exc:
        console.print("[red]t4-spread needs scipy[/red]")
        raise typer.Exit(1) from exc

    from vhagar.eval.spread import evaluate_spread

    r = evaluate_spread(n_fires=n_fires, seed=seed, ros_err=ros_err)
    for regime, ag in r.items():
        t = Table(title=f"T4 spread, {regime}-regime: incremental next-step skill (base rate {ag['new_burn_rate']:.3f})")
        for col in ("model", "AP", "IoU", "Dice", "burned-area ratio", "arrival MAE"):
            t.add_column(col, justify="right")
        for name, key in (("physics level-set", "physics"),
                          ("persistence + buffer", "persistence_buffer"),
                          ("persistence", "persistence")):
            s = ag[key]
            t.add_row(name, f"{s['ap']:.3f}", f"{s['iou']:.3f}", f"{s['dice']:.3f}",
                      f"{s['ba_ratio']:.2f}", f"{s.get('arrival_mae', float('nan')):.2f}"
                      if key == "physics" else "-")
        console.print(t)
    console.print(
        "[dim]  The physics forecast beats persistence and persistence+buffer, and IoU sits in the\n"
        "  cited wind-driven band. Absolute AP is optimistic here because the synthetic truth is a\n"
        "  perturbed level set, close to the forecaster's model class; the real next-day ceiling is\n"
        "  AP 0.35-0.45 (model-form error, fuel maps, wind, suppression). burned-area ratio > 1 is the\n"
        "  honest over-prediction from unmodelled suppression. Real numbers need real perimeters\n"
        "  (NIROPS / VIIRS). docs/15.[/dim]")


@app.command("t4-assimilate")
def t4_assimilate_cmd(
    n_fires: int = typer.Option(12, help="synthetic fires"),
    n_passes: int = typer.Option(6, help="satellite passes assimilated per fire"),
    prior_bias: float = typer.Option(0.6, help="true multiplicative ROS error the loop must recover"),
    regime: str = typer.Option("wind", help="wind | plume"),
    seed: int = typer.Option(0),
) -> None:
    """T4 state estimation + assimilation: online per-fire ROS calibration.

    The highest-return spread piece (docs/00 6.2). Sparse timed detections are
    assimilated pass by pass into a continuous arrival-time analysis by calibrating
    the per-fire ROS scale, then forecasting to the next pass. Scored on the
    incremental new burn (where naive persistence has no skill) with Sorensen and
    false-alarm ratio, against naive persistence and the uncalibrated prior, plus
    the full-perimeter Sorensen. docs/15.
    """
    try:
        import scipy  # noqa: F401
    except ImportError as exc:
        console.print("[red]t4-assimilate needs scipy[/red]")
        raise typer.Exit(1) from exc

    from vhagar.eval.assimilation import assimilation_experiment

    r = assimilation_experiment(n_fires=n_fires, n_passes=n_passes, regime=regime,
                                prior_bias=prior_bias, seed=seed)
    o = r["overall"]
    t = Table(title=f"T4 assimilation, {regime}-regime: per-fire ROS calibration from timed detections")
    for col in ("pass", "cal. scale k", "incr Sorensen (analysis)", "incr (uncal. prior)",
                "incr (naive)", "new-burn FAR", "full-perim Sorensen"):
        t.add_column(col, justify="right")
    for s in r["steps"]:
        t.add_row(str(s["step"] + 1), f"{s['k']:.2f}", f"{s['soren_analysis']:.3f}",
                  f"{s['soren_prior']:.3f}", f"{s['soren_naive']:.3f}",
                  f"{s['far_analysis']:.3f}", f"{s['soren_full']:.3f}")
    console.print(t)
    console.print(
        f"[dim]  Online calibration recovers the ROS bias: k -> {o['k_final']:.2f} (ideal "
        f"{o['k_ideal']:.2f}). The calibrated analysis reconstructs the perimeter at Sorensen "
        f"~{o['soren_full']:.2f} (published conditional-GAN ~0.81) and forecasts the between-pass\n"
        f"  new burn far better than naive persistence ({o['soren_analysis']:.2f} vs "
        f"{o['soren_naive']:.2f}) or the uncalibrated prior ({o['soren_prior']:.2f}). The high new-burn\n"
        "  FAR is the honest over-prediction from unmodelled suppression; the generative / diffusion\n"
        "  score-filter upgrades (torch, GPU) are what reduce it. docs/15.[/dim]")


@app.command("t4-aniso")
def t4_aniso_cmd(
    winds: str = typer.Option("0,0.3,0.6,0.9", help="comma-separated normalised wind speeds"),
    grid: int = typer.Option(141, help="grid size"),
    lb_max: float = typer.Option(4.0, help="length-to-breadth at full wind"),
) -> None:
    """T4 anisotropic wind-driven spread: elliptical fire growth.

    Grows a fire from a point under each wind speed with the elliptical
    arrival-time solver and reports the length-to-breadth ratio (long axis / short
    axis), which should match the prescribed value and be ~1 at zero wind. docs/15.
    """
    from vhagar.models.spread import anisotropic_arrival, front_length_breadth, length_to_breadth

    ws = [float(w) for w in winds.split(",") if w.strip()]
    c = grid // 2
    head = np.ones((grid, grid))
    seed = np.zeros((grid, grid), dtype=bool)
    seed[c, c] = True
    t = Table(title="T4 anisotropic spread: fire length-to-breadth vs wind")
    for col in ("wind", "prescribed LB", "measured LB", "downwind ext", "upwind ext", "crosswind ext"):
        t.add_column(col, justify="right")
    for w in ws:
        T = anisotropic_arrival(head, wind_speed=w, wind_dir=0.0, seeds=seed)
        fin = np.isfinite(T)
        tau = float(np.quantile(T[fin], 0.06))
        m = tau >= T
        ys, xs = np.where(m)
        t.add_row(f"{w:.2f}", f"{float(length_to_breadth(w, lb_max)):.2f}",
                  f"{front_length_breadth(m):.2f}", str(int(xs.max() - c)),
                  str(int(c - xs.min())), str(int(ys.max() - ys.min())))
    console.print(t)
    console.print(
        "[dim]  Zero wind is a circle (LB ~ 1); wind stretches the fire into a downwind ellipse whose\n"
        "  head outruns the back. This is the 8-connected elliptical solver; the rigorous continuous\n"
        "  counterpart is the Ordered Upwind Method. Plug in a calibrated FBP/Alexander LB. docs/15.[/dim]")


@app.command("firms-fetch")
def firms_fetch_cmd(
    detections: Path = typer.Option(
        Path("data/detections/detections"), help="FDC parquet root (defines the window)"
    ),
    out: Path = typer.Option(Path("viirs_truth.csv"), help="write the combined FIRMS CSV here"),
    sources: str = typer.Option(
        "viirs_noaa20_nrt,viirs_snpp_nrt", help="comma-separated FIRMS sources"
    ),
    map_key: str = typer.Option(None, help="FIRMS map key (or set FIRMS_MAP_KEY)"),
    pad_deg: float = typer.Option(0.25, help="bbox padding in degrees"),
) -> None:
    """Pull the VIIRS reference truth for the GOES FDC window (the T1 Stage-0 truth).

    Reads the dates and bbox spanned by the FDC parquet and fetches the matching VIIRS
    active-fire detections from the FIRMS area API, in <=10-day chunks, concatenated to
    one CSV that ``t1-stage0 --firms-csv`` consumes. Needs the network and a free FIRMS
    map key (https://firms.modaps.eosdis.nasa.gov/api/map_key/). See docs/12.
    """
    from datetime import date, timedelta

    from vhagar.eval.t1_stage0 import fdc_window
    from vhagar.io.firms import FirmsClient

    win = fdc_window(detections, pad_deg=pad_deg)
    console.print(
        f"[bold]FDC window[/bold]: {win['start_date']} to {win['end_date']} "
        f"({win['n_days']} d), bbox {tuple(round(b, 2) for b in win['bbox'])}"
    )
    client = FirmsClient(map_key=map_key)
    start = date.fromisoformat(win["start_date"])
    end = date.fromisoformat(win["end_date"])
    header, rows = None, []
    for src in [s.strip() for s in sources.split(",") if s.strip()]:
        cur = start
        while cur <= end:
            # FIRMS area API caps the day range at 5 per request.
            span = min(5, (end - cur).days + 1)
            console.print(f"  {src} {cur.isoformat()} +{span}d...", end=" ")
            try:
                text = client.area_csv(src, win["bbox"], day_range=span, start=cur)
            except Exception as exc:  # noqa: BLE001
                import contextlib
                import urllib.error

                detail = str(exc)
                if isinstance(exc, urllib.error.HTTPError):
                    body = ""
                    with contextlib.suppress(Exception):
                        body = exc.read().decode("utf-8", "replace")[:200].replace("\n", " ")
                    detail = f"HTTP {exc.code} {exc.reason}: {body}"
                console.print(f"[yellow]skip[/yellow] {detail}")
                cur += timedelta(days=span)
                continue
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if lines:
                if header is None:
                    header = lines[0]
                rows.extend(lines[1:] if lines[0] == header else lines)
            console.print(f"[green]{max(0, len(lines) - 1)} rows[/green]")
            cur += timedelta(days=span)
    if header is None:
        console.print("[red]no FIRMS rows returned[/red]")
        raise typer.Exit(1)
    out.write_text("\n".join([header, *rows]) + "\n")
    console.print(f"[green]wrote {out}[/green] ({len(rows):,} VIIRS detections). "
                  "Now: vhagar t1-stage0 --firms-csv " + str(out))


@app.command("t1-stage0")
def t1_stage0_cmd(
    detections: Path = typer.Option(
        Path("data/detections/detections"), help="FDC detections parquet root"
    ),
    firms_csv: Path = typer.Option(
        None, help="FIRMS/VIIRS CSV (the reference truth); omit to see the GOES side only"
    ),
    region_crs: str = typer.Option("EPSG:5070", help="equal-area CRS for planar matching"),
    max_gap_hours: float = typer.Option(12.0, help="temporal gap that separates events"),
) -> None:
    """T1 Stage-0: GOES FDC active-fire events vs VIIRS truth (POD, FAR, latency).

    Clusters the GOES FDC detections into fire events (per tile), and, if a FIRMS
    CSV of VIIRS detections is supplied, matches them with a parallax-aware GEO/LEO
    tolerance and reports probability of detection, false-alarm rate, the naive-2km
    vs parallax FAR difference (geometry, not model error), and detection latency.
    Without a FIRMS CSV it summarises the GOES side only. See docs/12.
    """
    from vhagar.eval.t1_stage0 import (
        coincidence_scores,
        firms_to_detections,
        load_fdc_detections,
        precision_far_scores,
    )
    from vhagar.io.firms import parse_firms_csv

    console.print(f"[bold]Loading GOES FDC detections[/bold] under {detections}...")
    goes = load_fdc_detections(detections, region_crs=region_crs)
    console.print(f"  {len(goes):,} GOES detections")
    if firms_csv is None:
        console.print(
            "[yellow]No FIRMS truth given[/yellow]: pass --firms-csv <viirs.csv> to score POD. "
            "Pull it with `vhagar firms-fetch` (FIRMS area API, free map key)."
        )
        return

    truth_recs = parse_firms_csv(firms_csv.read_text())
    viirs = firms_to_detections(truth_recs, region_crs=region_crs)
    console.print(f"  {len(viirs):,} VIIRS detections from {len(truth_recs):,} FIRMS records")

    # Detection-level coincidence (space cell + time window, restricted to the GOES
    # domain). Naive 2 km vs parallax-scale 4 km isolates the GEO/LEO geometry effect.
    naive = coincidence_scores(goes, viirs, cell_m=2_000.0, window_min=30.0)
    par = coincidence_scores(goes, viirs, cell_m=4_000.0, window_min=30.0)
    t = Table(title="T1 Stage-0: GOES FDC vs VIIRS POD (detection coincidence, +/-30 min)")
    for col in ("matching", "cell", "POD", "TP", "VIIRS in domain", "median gap (min)"):
        t.add_column(col, justify="right")
    for name, s in (("naive", naive), ("parallax-aware", par)):
        t.add_row(name, f"{s['cell_m'] / 1000:.0f} km", f"{s['pod']:.3f}", str(s["tp"]),
                  str(s["n_viirs"]), f"{s['median_gap_min']:.0f}")
    console.print(t)
    console.print(
        f"  [bold]POD geometry gain[/bold]: {naive['pod']:.3f} (2 km) -> {par['pod']:.3f} "
        f"(4 km), +{par['pod'] - naive['pod']:.3f}. GOES sees the fire a median "
        f"{par['median_gap_min']:.0f} min from the VIIRS overpass."
    )

    # Precision / FAR, conditioned on VIIRS overpass coincidence (a GOES detection is
    # only judged when VIIRS was actually observing its area then, so a fire between
    # overpasses is not miscounted as a false alarm).
    np_ = precision_far_scores(goes, viirs, cell_m=2_000.0, window_min=30.0)
    pp = precision_far_scores(goes, viirs, cell_m=4_000.0, window_min=30.0)
    t2 = Table(title="T1 Stage-0: GOES FDC precision / FAR (VIIRS-coincident detections)")
    for col in ("matching", "cell", "precision", "FAR", "TP", "FP", "evaluable"):
        t2.add_column(col, justify="right")
    for name, s in (("naive", np_), ("parallax-aware", pp)):
        t2.add_row(name, f"{s['cell_m'] / 1000:.0f} km", f"{s['precision']:.3f}",
                   f"{s['far']:.3f}", str(s["tp"]), str(s["fp"]), str(s["n_evaluable"]))
    console.print(t2)
    console.print(
        f"  [bold]FAR from geometry[/bold]: {np_['far']:.3f} (2 km) -> {pp['far']:.3f} (4 km), "
        f"a {pp['far'] - np_['far']:+.3f} change that is footprint quantisation + terrain\n"
        "  parallax, not model error, the published 26-36% -> 7-15% result on our data. See docs/12."
    )


@app.command("t2-unet")
def t2_unet_cmd(
    cache_dir: Path = typer.Option(Path("data/t2_cache"), help="cached T2 samples"),
    pattern: str = typer.Option("mtbs_*_w15bg.npz", help="glob for samples to use"),
    folds: int = typer.Option(5, help="grouped k-fold (leakage-proof, by fire)"),
    epochs: int = typer.Option(20, help="training epochs per fold"),
    crop: int = typer.Option(128, help="training tile size"),
    method: str = typer.Option("global", help="threshold baseline: global | perstratum"),
    objective: str = typer.Option("youden", help="threshold objective for the baseline"),
    seed: int = typer.Option(0),
) -> None:
    """Companion baseline: a U-Net over the RBR field vs the RBR threshold.

    Trains a single-channel U-Net on the same RBR windows the threshold sees, in
    leakage-proof grouped folds, and reports each held-out fire's skill over the
    predict-all-burned baseline next to the threshold's skill on the identical fold.
    The question is narrow and fair: does a spatial model beat a pointwise cut on the
    same input? Needs ``vhagar[torch]``; run where torch is installed.
    """
    import glob as _glob

    from vhagar.datasets.burned_area import T2Sample
    from vhagar.eval.t2_unet import run_unet_cv, summarise_unet_cv

    paths = sorted(_glob.glob(str(cache_dir / pattern)))
    samples = {}
    for p in paths:
        s = T2Sample.load(p)
        if s.is_usable:
            samples[s.event_id] = s
    if len(samples) < folds:
        console.print(f"[red]only {len(samples)} usable samples for {folds} folds[/red]")
        raise typer.Exit(1)
    console.print(
        f"[bold]U-Net companion baseline[/bold]: {len(samples)} fires, {folds}-fold, "
        f"{epochs} epochs. Training on the RBR field (torch)..."
    )
    try:
        results = run_unet_cv(
            samples, k=folds, method=method, objective=objective,
            epochs=epochs, crop=crop, seed=seed,
        )
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    t = Table(title="T2 U-Net vs RBR threshold (grouped k-fold, per held-out fire)")
    for col in ("held-out fire", "U-Net F1", "U-Net skill", "thr skill", "U-Net - thr"):
        t.add_column(col, justify="right")
    for r in results:
        diff = r.skill_f1 - r.thr_skill_f1
        t.add_row(
            r.held_out.split(":")[-1][:20], f"{r.f1:.3f}",
            f"{r.skill_f1:+.3f}", f"{r.thr_skill_f1:+.3f}",
            f"[green]{diff:+.3f}[/green]" if diff > 0 else f"[red]{diff:+.3f}[/red]",
        )
    console.print(t)
    s = summarise_unet_cv(results)
    if s.get("fires"):
        console.print(
            f"\n  [bold]{s['fires']} fires[/bold]: U-Net mean skill {s['unet_skill_mean']:+.3f}, "
            f"threshold mean skill {s['thr_skill_mean']:+.3f}, "
            f"U-Net - threshold {s['unet_minus_thr']:+.3f} "
            f"(U-Net wins {s['unet_beats_thr']}/{s['fires']})"
        )
    console.print(
        "[dim]  Same RBR input, same leakage-proof folds, same naive baseline. If the\n"
        "  U-Net does not clear the threshold here, a spatial model adds nothing on this\n"
        "  input, which is worth knowing before anything fancier. See docs/11.[/dim]"
    )


@app.command("t2-deep")
def t2_deep_cmd(
    cache_dir: Path = typer.Option(Path("data/t2_cache"), help="cached T2 samples with stack"),
    pattern: str = typer.Option("mtbs_*_w15bgs.npz", help="glob for stack samples (_w15bgs)"),
    model: str = typer.Option("siamese", help="deep model: siamese | unet"),
    folds: int = typer.Option(5, help="grouped k-fold (leakage-proof, by fire)"),
    epochs: int = typer.Option(20, help="training epochs per fold"),
    crop: int = typer.Option(128, help="training tile size"),
    method: str = typer.Option("global", help="threshold baseline: global | perstratum"),
    objective: str = typer.Option("youden", help="threshold objective for the baseline"),
    seed: int = typer.Option(0),
) -> None:
    """Deep models on the pre/post NBR stack vs the RBR threshold.

    ``--model siamese`` (default) gives a shared-weight change model the pre- and
    post-fire NBR as separate inputs; ``--model unet`` runs a multi-channel U-Net over
    the full stack. Same leakage-proof folds and skill-over-naive protocol as t2-unet.
    Needs samples built with ``with_stack=True`` (cache tag ``_w15bgs``) and torch.
    """
    import glob as _glob

    from vhagar.datasets.burned_area import T2Sample
    from vhagar.eval.t2_deep import run_deep_cv, summarise_deep_cv

    paths = sorted(_glob.glob(str(cache_dir / pattern)))
    samples = {}
    for p in paths:
        s = T2Sample.load(p)
        if s.is_usable:
            samples[s.event_id] = s
    if len(samples) < folds:
        console.print(
            f"[red]only {len(samples)} usable stack samples matching {pattern}[/red]\n"
            "Build them first with a stack pull (build_optical_sample(..., with_stack=True))."
        )
        raise typer.Exit(1)
    has_stack = sum(1 for s in samples.values() if s.stack is not None)
    console.print(
        f"[bold]T2 deep baseline[/bold] ({model}): {len(samples)} fires "
        f"({has_stack} with a stack), {folds}-fold, {epochs} epochs (torch)..."
    )
    try:
        results = run_deep_cv(
            samples, model_kind=model, k=folds, method=method, objective=objective,
            epochs=epochs, crop=crop, seed=seed,
        )
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    t = Table(title=f"T2 {model} vs RBR threshold (grouped k-fold, per held-out fire)")
    for col in ("held-out fire", f"{model} F1", f"{model} skill", "thr skill", "model - thr"):
        t.add_column(col, justify="right")
    for r in results:
        diff = r.skill_f1 - r.thr_skill_f1
        t.add_row(
            r.held_out.split(":")[-1][:20], f"{r.f1:.3f}",
            f"{r.skill_f1:+.3f}", f"{r.thr_skill_f1:+.3f}",
            f"[green]{diff:+.3f}[/green]" if diff > 0 else f"[red]{diff:+.3f}[/red]",
        )
    console.print(t)
    s = summarise_deep_cv(results)
    if s.get("fires"):
        console.print(
            f"\n  [bold]{s['fires']} fires[/bold]: {model} mean skill {s['deep_skill_mean']:+.3f}, "
            f"threshold mean skill {s['thr_skill_mean']:+.3f}, "
            f"{model} - threshold {s['deep_minus_thr']:+.3f} "
            f"({model} wins {s['deep_beats_thr']}/{s['fires']})"
        )
    console.print(
        "[dim]  Pre/post NBR as separate inputs, same leakage-proof folds and naive\n"
        "  baseline. Compare to t2-unet (RBR, one channel) to see whether richer inputs\n"
        "  and the change formulation help beyond a single-channel segmenter. docs/11.[/dim]"
    )


@app.command("emsr-candidates")
def emsr_candidates_cmd(
    koppen: Path = typer.Option(
        None, help="Koppen class raster; tags each fire with its climate zone"
    ),
    out: Path = typer.Option(Path("emsr_candidates.csv"), help="write candidate table here"),
    year_min: int = typer.Option(2017, help="earliest event year (Sentinel-2 starts ~2017)"),
    year_max: int = typer.Option(2100),
    min_products: int = typer.Option(1, help="require at least this many products"),
    limit: int = typer.Option(0, help="cap printed rows (0 = all)"),
) -> None:
    """List CEMS wildfire activations (public API) tagged by Koppen climate zone.

    Use this to pick a climate-diverse set of European fires to generalise the
    cross-continent transfer result: matching US strata to more than just Greek
    Csa. Writes a CSV; download the chosen activations' vector packages from the
    portal, then wire them up with ``emsr-ingest``. Needs the network.
    """
    from vhagar.labels.emsr_fetch import list_wildfire_candidates, write_candidates_csv

    console.print("[bold]Querying CEMS Rapid Mapping public API for wildfires...[/bold]")
    acts = list_wildfire_candidates(
        koppen_raster=koppen, year_min=year_min, year_max=year_max,
        min_products=min_products,
    )
    acts.sort(key=lambda a: (a.koppen or 999, a.event_date))
    write_candidates_csv(acts, out)

    from collections import Counter
    by_zone = Counter(a.koppen_label for a in acts)
    t = Table(title=f"CEMS wildfire candidates ({len(acts)} activations, {year_min}-{year_max})")
    for col in ("code", "date", "Koppen", "country", "prod", "name"):
        t.add_column(col, justify="left", overflow="ellipsis")
    for a in (acts if limit == 0 else acts[:limit]):
        t.add_row(a.code, a.event_date, a.koppen_label,
                  ",".join(a.countries)[:16], str(a.n_products), a.name[:44])
    console.print(t)
    console.print(f"[dim]  climate spread: {dict(by_zone)}[/dim]")
    console.print(f"[green]wrote {out}[/green] (edit down to your picks, then emsr-ingest)")


@app.command("emsr-ingest")
def emsr_ingest_cmd(
    root: Path = typer.Argument(..., exists=True, help="folder of downloaded EMS vector packages"),
    dates: Path = typer.Option(
        None, help="candidates CSV (from emsr-candidates) to supply event dates"
    ),
    out: Path = typer.Option(Path("emsr.csv"), help="write the t2-continent-out manifest here"),
) -> None:
    """Build the EMSR manifest from a folder of downloaded EMS delineations.

    Finds each AOI's burnt-area observedEventA layer (latest monitoring step) and
    writes ``emsr.csv`` for ``t2-continent-out``. No network. Extract the portal
    zips into ``root`` first (any nesting is fine, it recurses).
    """
    from vhagar.labels.emsr_fetch import (
        ingest_delineations,
        load_dates_from_candidates,
        write_manifest_csv,
    )

    date_map = load_dates_from_candidates(dates) if dates else {}
    rows = ingest_delineations(root, dates=date_map)
    if not rows:
        console.print(f"[red]no observedEventA delineations found under {root}[/red]")
        raise typer.Exit(1)
    write_manifest_csv(rows, out)
    t = Table(title=f"EMSR manifest: {len(rows)} delineations")
    for col in ("activation_id", "event_date", "delineation"):
        t.add_column(col, justify="left", overflow="ellipsis")
    missing = 0
    for r in rows:
        if not r["event_date"]:
            missing += 1
        t.add_row(r["activation_id"], r["event_date"] or "[red]?[/red]",
                  Path(r["delineation_path"]).name[:48])
    console.print(t)
    if missing:
        console.print(f"[yellow]{missing} rows have no event date[/yellow]; "
                      "pass --dates or fill them in before t2-continent-out.")
    console.print(f"[green]wrote {out}[/green]")


@app.command("t2-perimeter")
def t2_perimeter_cmd(
    mosaic: Path = typer.Argument(..., exists=True, help="MTBS thematic burn-severity GeoTIFF"),
    burned_classes: str = typer.Option("2,3,4", help="severity classes counted as burned"),
    pixel_area_ha: float = typer.Option(0.09, help="pixel area in hectares (30 m = 0.09)"),
) -> None:
    """T2 first number: how much a rasterised perimeter overstates burned area.

    Streams the MTBS thematic mosaic, then compares the mapped perimeter interior
    (the 'all burned' claim) against the per-pixel severity classes. The gap is
    the unburned-islands commission the architecture warns about.
    """
    from vhagar.eval.t2_perimeter import class_histogram, perimeter_vs_severity

    burned = tuple(int(c) for c in burned_classes.split(",") if c.strip())
    console.print(f"[bold]Streaming {mosaic.name}...[/bold]")
    hist = class_histogram(mosaic)
    res = perimeter_vs_severity(hist, pixel_area_ha=pixel_area_ha, burned_classes=burned)

    t = Table(title="Perimeter vs per-pixel severity (census)")
    t.add_column("quantity")
    t.add_column("value", justify="right")
    t.add_row("rasterised-perimeter burned", f"{res.perimeter_ha:,.0f} ha")
    t.add_row("severity-classified burned", f"{res.severity_burned_ha:,.0f} ha")
    t.add_row("unburned within perimeter", f"{res.unburned_within_ha:,.0f} ha")
    t.add_row("[bold]commission (overstatement)[/bold]", f"[bold]{100 * res.commission_fraction:.1f}%[/bold]")
    console.print(t)
    console.print(
        f"[dim]  burned classes {burned}; class histogram 0-6: "
        f"{[int(hist[i]) for i in range(7)]}\n"
        "  Census over MTBS severity, exact w.r.t. that product and lineage-shared "
        "with the perimeter.[/dim]"
    )


@app.command("compact")
def compact_cmd(
    directory: Path = typer.Argument(..., exists=True, help="a backfill output directory"),
    min_files: int = typer.Option(2, help="smallest file count per tile worth compacting"),
    dry_run: bool = typer.Option(False, help="report what would happen without touching disk"),
) -> None:
    """Merge each tile's per-day Parquet files into one file per year.

    Safe and idempotent: the merged file is written and its row count verified
    before any original is deleted. Run it periodically once an archive has
    accumulated many day files.
    """
    from vhagar.archive.compaction import compact_detections

    report = compact_detections(directory, min_files=min_files, dry_run=dry_run)
    if dry_run:
        console.print("[bold]dry run, nothing written[/bold]")
    console.print(str(report))
    if report.tiles_compacted == 0:
        console.print("[dim]  nothing to compact[/dim]")


@app.command("coverage")
def coverage_cmd(
    directory: Path = typer.Argument(..., exists=True, help="a backfill output directory"),
    max_gap_min: float = typer.Option(
        20.0, help="minutes between observations before a hole is declared"
    ),
) -> None:
    """Report observed coverage and every hole in a backfill directory.

    Reads ``_manifest.jsonl`` and prints the observed intervals, each gap with
    its start, end and duration, and the count of failed granules. This is what
    explains a multi-interval coverage line: a single dropped granule never
    splits an interval, so a second interval means a real hole, and this shows
    where it is rather than leaving you to guess.
    """
    from vhagar.archive.backfill import (
        coverage_gaps,
        coverage_intervals,
        failed_records,
        load_manifest,
    )

    records = list(load_manifest(directory).values())
    if not records:
        console.print(f"[yellow]no manifest found in {directory}[/yellow]")
        raise typer.Exit(1)

    gap = timedelta(minutes=max_gap_min)
    intervals = coverage_intervals(records, max_gap=gap)
    gaps = coverage_gaps(records, max_gap=gap)
    failed = failed_records(records)
    observed = sum((b - a for a, b in intervals), timedelta())

    console.print(
        f"[bold]{len(records)} granules in manifest[/bold], "
        f"{sum(1 for r in records if r.ok)} ok, {len(failed)} failed"
    )
    console.print(
        f"  {len(intervals)} observed interval(s), "
        f"{observed.total_seconds() / 3600:.1f} h observed"
    )

    if intervals:
        t = Table(title="Observed intervals (UTC)")
        for col in ("start", "end", "duration"):
            t.add_column(col, justify="left")
        for a, b in intervals:
            t.add_row(a.isoformat(), b.isoformat(), f"{(b - a).total_seconds() / 3600:.2f} h")
        console.print(t)

    if gaps:
        t = Table(title=f"Holes longer than {max_gap_min:g} min")
        for col in ("gap start", "gap end", "duration"):
            t.add_column(col, justify="left")
        for a, b, d in gaps:
            t.add_row(a.isoformat(), b.isoformat(), f"{d.total_seconds() / 60:.1f} min")
        console.print(t)
        console.print(
            "[dim]  Each hole is a period no granule was successfully read. A loader\n"
            "  mining negatives skips these rather than treating them as quiet.[/dim]"
        )
    else:
        console.print("[green]  no holes: one continuous observed interval[/green]")

    if failed:
        t = Table(title=f"Failed granules ({len(failed)})")
        for col in ("granule", "error"):
            t.add_column(col, justify="left", no_wrap=False)
        for r in failed[:20]:
            t.add_row(r.key, r.error or "unknown")
        console.print(t)
        if len(failed) > 20:
            console.print(f"[dim]  ...and {len(failed) - 20} more[/dim]")


if __name__ == "__main__":  # pragma: no cover
    app()
