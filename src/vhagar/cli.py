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
