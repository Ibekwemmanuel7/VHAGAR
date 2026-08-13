"""Atmospheric transmittance in the mid-infrared fire window.

Why this module is not optional
-------------------------------
Published LSA-SAF figures for SEVIRI: at nadir, TCWV 20 kg/m2, mid-latitude
summer, the radiance-difference-effective MIR transmittance is **0.69**. Using
the naive band-averaged value (0.74) instead makes bottom-of-atmosphere
radiance ~10% too low.

So a pipeline that ignores atmospheric correction entirely -- as MODIS
Collection 6 L2 does, setting tau = 1 -- carries a **~31% low bias in FRP at
nadir in a moist mid-latitude atmosphere**, i.e. it needs a multiplicative
correction of ~1.45.

View angle makes it worse, deterministically:

    tau(theta_v) ~= tau(0) ** sec(theta_v)

At theta_v = 60 deg (routine for SEVIRI over Iberia or Greece) tau ~ 0.48, a
correction factor of ~2.1x. Near the geostationary disk edge it approaches 3x.

**This is the largest single systematic in geostationary FRP and it is cheap
and deterministic to compute.** There is no excuse for a model not to have it.

What is and is not in here
--------------------------
:func:`transmittance_mir` is a *physically structured parametric surrogate*,
not a radiative transfer model. It has the right functional form (Beer-Lambert
in air mass, curve-of-growth in water vapour) and is pinned to two anchors --
0.69 at TCWV 20 / nadir, and ~0.92 in a dry atmosphere. It is meant to be
**replaced** by a surrogate fitted to ~1e5 RTTOV v14 runs, which is a weekend
of work and is the stage-4 item in ``docs/03_PHYSICS.md``.

Do **not** substitute 6S/Py6S here. 6S spans 0.25-4.0 um as a *solar* code with
no thermal emission source term. It will run happily at 3.9 um and return
plausible-looking numbers that are physically meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AtmosphereState",
    "TransmittanceModel",
    "air_mass_factor",
    "frp_atmospheric_correction_factor",
    "transmittance_mir",
]

#: Reference anchor: LSA-SAF SEVIRI effective MIR transmittance at nadir,
#: TCWV = 20 kg/m2, mid-latitude summer.
TAU_REF = 0.69
TCWV_REF = 20.0
#: Dry-atmosphere MIR window transmittance (residual continuum + CO2/N2O/O3).
TAU_DRY = 0.92

_OD_DRY = -np.log(TAU_DRY)
_K_WATER = (-np.log(TAU_REF) - _OD_DRY) / np.sqrt(TCWV_REF)
#: MIR aerosol optical depth per unit 550 nm AOD. Small: MIR is far less
#: aerosol-sensitive than the visible. Order-of-magnitude placeholder.
_K_AEROSOL = 0.06


def air_mass_factor(view_zenith_deg, max_zenith_deg: float = 80.0) -> np.ndarray:
    """Slant-path multiplier ``sec(theta_v)``, clipped for numerical sanity.

    The plane-parallel approximation degrades badly beyond ~80 deg; geostationary
    fire products cut off processing beyond 80 deg view zenith for exactly this
    reason, so clipping there is consistent with operational practice rather
    than an arbitrary numerical guard.
    """
    z = np.clip(np.asarray(view_zenith_deg, dtype=np.float64), 0.0, max_zenith_deg)
    return 1.0 / np.cos(np.radians(z))


def transmittance_mir(
    tcwv_kg_m2,
    view_zenith_deg=0.0,
    aod_550=0.0,
    tau_dry: float = TAU_DRY,
    k_water: float = _K_WATER,
    k_aerosol: float = _K_AEROSOL,
) -> np.ndarray:
    """MIR (3.7-4.0 um) atmospheric transmittance.

    ``tau = exp( -m * (od_dry + k_w * sqrt(W) + k_a * AOD) )`` with
    ``m = sec(theta_v)``.

    The ``sqrt(W)`` dependence is the standard curve-of-growth behaviour for a
    partially saturated absorber band; the coefficient is pinned so that
    ``transmittance_mir(20, 0) == 0.69``.

    >>> float(round(transmittance_mir(20.0, 0.0), 3))
    0.69
    >>> float(round(transmittance_mir(20.0, 60.0), 3))   # sec 60 = 2
    0.476
    """
    w = np.maximum(np.asarray(tcwv_kg_m2, dtype=np.float64), 0.0)
    m = air_mass_factor(view_zenith_deg)
    od = -np.log(tau_dry) + k_water * np.sqrt(w) + k_aerosol * np.asarray(aod_550, dtype=np.float64)
    return np.exp(-m * od)


def frp_atmospheric_correction_factor(tcwv_kg_m2, view_zenith_deg=0.0, aod_550=0.0) -> np.ndarray:
    """``1 / tau``, what an uncorrected FRP must be multiplied by.

    >>> float(round(frp_atmospheric_correction_factor(20.0, 0.0), 2))
    1.45
    >>> float(round(frp_atmospheric_correction_factor(20.0, 60.0), 2))
    2.1
    """
    return 1.0 / transmittance_mir(tcwv_kg_m2, view_zenith_deg, aod_550)


@dataclass(slots=True)
class AtmosphereState:
    """Per-pixel atmospheric state used by the forward model."""

    tcwv_kg_m2: np.ndarray
    view_zenith_deg: np.ndarray
    aod_550: np.ndarray | float = 0.0
    surface_pressure_hpa: np.ndarray | float = 1013.25

    def transmittance_mir(self) -> np.ndarray:
        return transmittance_mir(self.tcwv_kg_m2, self.view_zenith_deg, self.aod_550)

    def air_mass(self) -> np.ndarray:
        return air_mass_factor(self.view_zenith_deg)


class TransmittanceModel:
    """Swappable transmittance backend.

    The parametric surrogate is the default so the pipeline runs with no extra
    dependencies. Fit a table or a small MLP to RTTOV/libRadtran output and
    register it here when you get to stage 4; the call signature does not change,
    which is the point.
    """

    def __init__(self, backend: str = "parametric", table=None) -> None:
        if backend not in {"parametric", "table"}:
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "table" and table is None:
            raise ValueError("backend='table' requires a fitted lookup callable")
        self.backend = backend
        self._table = table

    def __call__(self, tcwv_kg_m2, view_zenith_deg=0.0, aod_550=0.0) -> np.ndarray:
        if self.backend == "table":
            return np.asarray(self._table(tcwv_kg_m2, view_zenith_deg, aod_550), dtype=np.float64)
        return transmittance_mir(tcwv_kg_m2, view_zenith_deg, aod_550)

    def __repr__(self) -> str:  # pragma: no cover
        return f"TransmittanceModel(backend={self.backend!r})"


def smoke_attenuation_warning() -> str:
    """The honest state of MIR-through-smoke attenuation.

    Returned as a string so it can be logged into a product's provenance rather
    than living only in a docstring nobody reads.
    """
    return (
        "MIR attenuation by wildfire smoke is NOT quantified in the published "
        "literature. Dark sooty smoke measurably affects both 4 um and 2 um "
        "observations (2 um worse), and global FRP inventories may therefore "
        "underestimate in heavy-smoke regimes. 'MIR sees through smoke' is only "
        "partly true. VHAGAR treats this as an unquantified systematic and does "
        "not apply a smoke correction."
    )
