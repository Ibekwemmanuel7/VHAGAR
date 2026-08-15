"""Batch discovery and ingest of Copernicus EMS (EMSR) wildfire activations.

Two jobs, kept separate because one needs the network and one does not.

``list_wildfire_candidates`` queries the public CEMS Rapid Mapping activation API
(no login) and returns wildfire activations tagged with their Koppen climate class
(sampled from a global raster at the activation centroid). This is how you pick a
*climate-diverse* set of European fires to generalise the cross-continent transfer
result in docs/11, matching US strata to more than just Greek Csa.

``ingest_delineations`` scans a folder of already-downloaded EMS vector packages,
finds each AOI's burnt-area ``observedEventA`` delineation (preferring the latest
monitoring step), and writes the ``emsr.csv`` manifest that ``t2-continent-out``
consumes. No network: you download the packages from the portal, this wires them up.

The split matters because the per-activation product download needs an authenticated
API call, but the *listing* is public and the *ingest* is purely local file work.
"""

from __future__ import annotations

import csv
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "KOPPEN_NAMES",
    "EMSActivation",
    "list_wildfire_candidates",
    "ingest_delineations",
    "parse_centroid",
    "koppen_name",
]

#: Beck et al. (2018) Koppen-Geiger integer legend, class -> label.
KOPPEN_NAMES = {
    1: "Af", 2: "Am", 3: "Aw", 4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc", 11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc", 17: "Dsa", 18: "Dsb", 19: "Dsc",
    20: "Dsd", 21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd", 25: "Dfa",
    26: "Dfb", 27: "Dfc", 28: "Dfd", 29: "ET", 30: "EF",
}

PUBLIC_API = (
    "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/"
    "public-activations-info/"
)


def koppen_name(class_int: int | None) -> str:
    """Human label for a Koppen integer class, or ``?`` if unknown/nodata."""
    return KOPPEN_NAMES.get(int(class_int), "?") if class_int else "?"


def parse_centroid(wkt: str) -> tuple[float, float]:
    """Return ``(lon, lat)`` from a ``POINT (lon lat)`` WKT string."""
    nums = re.findall(r"-?\d+\.?\d*", wkt or "")
    if len(nums) < 2:
        raise ValueError(f"cannot parse centroid: {wkt!r}")
    return float(nums[0]), float(nums[1])


@dataclass(slots=True)
class EMSActivation:
    """One CEMS activation's public metadata, plus an assigned Koppen class."""

    code: str
    name: str
    countries: tuple[str, ...]
    event_date: str
    lon: float
    lat: float
    n_products: int
    closed: bool
    koppen: int | None = None

    @property
    def koppen_label(self) -> str:
        return koppen_name(self.koppen)


def _fetch_page(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (public gov API)
        return json.loads(resp.read().decode("utf-8"))


def list_wildfire_candidates(
    koppen_raster: str | Path | None = None,
    year_min: int = 2017,
    year_max: int = 2100,
    min_products: int = 1,
    closed_only: bool = True,
    page_size: int = 100,
    max_pages: int = 40,
    timeout: float = 60.0,
    api_base: str = PUBLIC_API,
) -> list[EMSActivation]:
    """Query the public CEMS Rapid Mapping API for wildfire activations.

    Filters to ``category == "Wildfire"`` in the year window, with at least
    ``min_products`` products (so there is a delineation to download), optionally
    closed activations only (finished mapping). If ``koppen_raster`` is given, each
    activation is tagged with the Koppen class sampled at its centroid, so you can
    pick a climate-diverse set. Sorted newest first. Needs the network.
    """
    out: list[EMSActivation] = []
    for page in range(max_pages):
        url = f"{api_base}?limit={page_size}&offset={page * page_size}"
        payload = _fetch_page(url, timeout)
        results = payload.get("results", [])
        if not results:
            break
        for r in results:
            if r.get("category") != "Wildfire":
                continue
            if int(r.get("n_products", 0)) < min_products:
                continue
            if closed_only and not r.get("closed", False):
                continue
            event_date = (r.get("eventTime") or "")[:10]
            year = int(event_date[:4]) if event_date[:4].isdigit() else 0
            if not (year_min <= year <= year_max):
                continue
            try:
                lon, lat = parse_centroid(r.get("centroid", ""))
            except ValueError:
                continue
            out.append(
                EMSActivation(
                    code=r["code"],
                    name=r.get("name", ""),
                    countries=tuple(r.get("countries", [])),
                    event_date=event_date,
                    lon=lon,
                    lat=lat,
                    n_products=int(r.get("n_products", 0)),
                    closed=bool(r.get("closed", False)),
                )
            )
        if payload.get("next") in (None, ""):
            break

    if koppen_raster is not None and out:
        from vhagar.datasets.strata import sample_class_raster

        classes = sample_class_raster(
            [a.lon for a in out], [a.lat for a in out], koppen_raster
        )
        for a, c in zip(out, classes, strict=True):
            a.koppen = int(c) if int(c) != 0 else None
    return out


def write_candidates_csv(activations: list[EMSActivation], out_path: str | Path) -> Path:
    """Write the candidate table for ``emsr-ingest --dates`` to read back."""
    out_path = Path(out_path)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "event_date", "koppen", "koppen_label",
                    "countries", "lon", "lat", "n_products", "name"])
        for a in activations:
            w.writerow([a.code, a.event_date, a.koppen if a.koppen else "",
                        a.koppen_label, ";".join(a.countries),
                        f"{a.lon:.4f}", f"{a.lat:.4f}", a.n_products, a.name])
    return out_path


_CODE_RE = re.compile(r"(EMSR\d+)")
_AOI_RE = re.compile(r"(AOI\d+)")
_MONIT_RE = re.compile(r"MONIT(\d+)")


def _monit_rank(path: Path) -> int:
    """Higher = later monitoring step (more complete burn). No MONIT -> 0."""
    m = _MONIT_RE.search(path.as_posix())
    return int(m.group(1)) if m else 0


def ingest_delineations(
    root: str | Path,
    dates: dict[str, str] | None = None,
    pattern: str = "*observedEventA*.shp",
) -> list[dict]:
    """Scan ``root`` for EMS burnt-area delineations and build manifest rows.

    Finds every ``observedEventA`` shapefile (the burnt-area layer), groups by
    ``(EMSR code, AOI)``, and keeps the **latest monitoring step** per group so a
    fire mapped several times contributes its most complete delineation once.
    Returns rows ``{activation_id, delineation_path, event_date}`` ready for
    ``emsr.csv``. ``dates`` maps an ``EMSR`` code to its event date; a code with no
    date is still emitted with an empty date so you can fill it in. Pure file work.
    """
    root = Path(root)
    dates = dates or {}
    best: dict[tuple[str, str], Path] = {}
    for shp in root.rglob(pattern):
        cm = _CODE_RE.search(shp.as_posix())
        if not cm:
            continue
        code = cm.group(1)
        am = _AOI_RE.search(shp.as_posix())
        aoi = am.group(1) if am else "AOI01"
        key = (code, aoi)
        if key not in best or _monit_rank(shp) > _monit_rank(best[key]):
            best[key] = shp

    rows: list[dict] = []
    for (code, aoi), shp in sorted(best.items()):
        rows.append({
            "activation_id": f"{code}_{aoi}",
            "delineation_path": str(shp),
            "event_date": dates.get(code, ""),
        })
    return rows


def write_manifest_csv(rows: list[dict], out_path: str | Path) -> Path:
    """Write ingest rows to the ``emsr.csv`` manifest format."""
    out_path = Path(out_path)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["activation_id", "delineation_path", "event_date"])
        w.writeheader()
        w.writerows(rows)
    return out_path


def load_dates_from_candidates(csv_path: str | Path) -> dict[str, str]:
    """Read a ``code -> event_date`` map from a candidates CSV."""
    out: dict[str, str] = {}
    with Path(csv_path).open() as fh:
        for row in csv.DictReader(fh):
            if row.get("code") and row.get("event_date"):
                out[row["code"]] = row["event_date"]
    return out
