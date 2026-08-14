"""VHAGAR command line interface.

    vhagar grid info --region conus
    vhagar splits build --records units.json --scheme leave_year_out --out splits/
    vhagar splits verify splits/leave_year_out.json
    vhagar fwi demo
    vhagar area-estimate --confusion 97,3,10,90 --areas 200000,20000
"""

from __future__ import annotations

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
