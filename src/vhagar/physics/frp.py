"""Fire Radiative Power, the Wooster MIR-radiance method, done properly.

The method
----------
Because the Planck sensitivity exponent ``b`` is ~4 over the flaming range
(650-1350 K), the fire's MIR radiance can be written ``L_f ~= eps_f * a * T_f^4``
and combined with Stefan-Boltzmann ``M = eps_f * sigma * T_f^4`` to eliminate
both the fire temperature and the fire fraction:

    FRP = (A_pix * sigma) / (a * tau_MIR) * (L_MIR - L_MIR_background)

Accuracy is **+-12% for fire temperatures between 665 K and 1365 K** and
degrades outside that band because ``b`` departs from 4. Fire radiative energy
converts to fuel consumed at **0.368 +- 0.015 kg/MJ**.

Three things this module insists on
-----------------------------------
1. **Atmospheric transmittance is applied.** Leaving ``tau = 1`` costs ~31% at
   nadir in a moist mid-latitude atmosphere and >50% at 60 degrees view zenith.
2. **Saturation is censoring, not noise.** Every sensor has a hard MIR
   saturation temperature. A model trained on saturated FRP with an ordinary
   loss systematically under-predicts on the largest fires -- the ones that
   matter. :func:`saturation_mask` flags them; use a censored likelihood.
3. **Errors are multiplicative.** ``a``, ``tau``, ``eps`` and the background all
   enter FRP as products or quotients, so uncertainty is naturally lognormal.
   Predict FRP in log space, or predict a *correction factor*, never an
   additive offset. See :class:`vhagar.models.physics_heads.ConstrainedFRPHead`.
"""

from __future__ import annotations

import numpy as np

from vhagar.physics.atmosphere import transmittance_mir
from vhagar.physics.planck import STEFAN_BOLTZMANN, planck_radiance

__all__ = [
    "FRE_TO_FUEL_KG_PER_MJ",
    "WOOSTER_A",
    "frp_from_radiance",
    "frp_uncertainty",
    "fuel_consumed_kg",
    "saturation_mask",
    "wooster_a",
]

#: Sensor-specific Wooster constant ``a`` [W m^-2 sr^-1 um^-1 K^-4], fitted by
#: least squares over 650-1300 K. Values are as published.
WOOSTER_A: dict[str, float] = {
    "modis": 2.96e-9,        # Terra MODIS, Wooster et al. 2005
    "modis_c6": 3.0e-9,      # value MODIS Collection 6 uses operationally
    "seviri": 3.06e-9,       # Meteosat-8 SEVIRI
    "goes_imager": 3.07e-9,  # GOES-8 Imager
    "bird_hsrs": 3.33e-9,
}
#: Fallback for sensors with no published fit. Flagged loudly by :func:`wooster_a`.
_DEFAULT_A = 3.0e-9

#: Fire radiative energy -> dry matter consumed.
FRE_TO_FUEL_KG_PER_MJ = 0.368
FRE_TO_FUEL_SIGMA = 0.015

#: The temperature range within which the b~=4 approximation holds to ~+-12%.
WOOSTER_VALID_RANGE_K = (665.0, 1365.0)


def wooster_a(sensor: str, strict: bool = False) -> float:
    """Look up the Wooster constant for a sensor.

    Unknown sensors fall back to 3.0e-9 with a warning, because a wrong-by-10%
    constant is far better than a crash in an operational pipeline -- but
    ``strict=True`` turns it into an error for offline science work where the
    10% matters.
    """
    key = sensor.lower()
    if key in WOOSTER_A:
        return WOOSTER_A[key]
    if strict:
        raise KeyError(
            f"no published Wooster constant for {sensor!r}; "
            f"known: {sorted(WOOSTER_A)}. Re-derive per instrument before "
            "using FRP quantitatively."
        )
    import warnings

    warnings.warn(
        f"no published Wooster constant for {sensor!r}; using the MODIS C6 value "
        f"{_DEFAULT_A:.2e}. This is sensor-specific and worth up to ~10% in FRP.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _DEFAULT_A


def frp_from_radiance(
    l_mir,
    l_mir_background,
    pixel_area_m2,
    sensor: str = "modis_c6",
    transmittance=None,
    tcwv_kg_m2=None,
    view_zenith_deg=0.0,
    a_constant: float | None = None,
) -> np.ndarray:
    """Fire Radiative Power in megawatts.

    Parameters
    ----------
    l_mir, l_mir_background
        Top-of-atmosphere MIR spectral radiance of the fire pixel and of its
        contextual background window, in W m^-2 sr^-1 um^-1.
    pixel_area_m2
        Ground area of the pixel. **This grows strongly off nadir** -- ~8x from
        nadir to scan edge for MODIS -- so pass the geometry-corrected value,
        not the nominal one.
    transmittance
        MIR transmittance. If ``None``, computed from ``tcwv_kg_m2`` and
        ``view_zenith_deg``; if those are also ``None``, defaults to 1.0 **and
        warns**, because an uncorrected FRP is biased ~31% low at nadir.

    Returns
    -------
    FRP in MW. Negative values (background hotter than the pixel) are clipped
    to 0 -- they are noise, not negative power.

    >>> frp = frp_from_radiance(0.35, 0.02, 375.0**2, "modis_c6", transmittance=0.69)
    >>> bool(frp > 0)
    True
    """
    lf = np.asarray(l_mir, dtype=np.float64)
    lb = np.asarray(l_mir_background, dtype=np.float64)
    area = np.asarray(pixel_area_m2, dtype=np.float64)

    if transmittance is None:
        if tcwv_kg_m2 is None:
            import warnings

            warnings.warn(
                "FRP computed with transmittance=1.0. Uncorrected FRP is biased "
                "~31% low at nadir in a moist mid-latitude atmosphere and >50% "
                "low at 60 deg view zenith. Pass tcwv_kg_m2 or transmittance.",
                RuntimeWarning,
                stacklevel=2,
            )
            tau = 1.0
        else:
            tau = transmittance_mir(tcwv_kg_m2, view_zenith_deg)
    else:
        tau = np.asarray(transmittance, dtype=np.float64)

    a = a_constant if a_constant is not None else wooster_a(sensor)
    frp_w = (area * STEFAN_BOLTZMANN / (a * tau)) * (lf - lb)
    return np.maximum(frp_w, 0.0) / 1e6


def frp_from_brightness_temperature(
    bt_mir_k,
    bt_mir_background_k,
    pixel_area_m2,
    wavelength_um: float = 3.9,
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper for products that ship brightness temperature.

    Converts BT to radiance through the Planck function first. Note this is an
    approximation: the true conversion uses the sensor's spectral response
    function, not a monochromatic wavelength. Fine for triage; use radiance for
    quantitative work.
    """
    return frp_from_radiance(
        planck_radiance(wavelength_um, bt_mir_k),
        planck_radiance(wavelength_um, bt_mir_background_k),
        pixel_area_m2,
        **kwargs,
    )


def frp_uncertainty(
    frp_mw,
    l_fire,
    l_background,
    sigma_l_fire,
    sigma_l_background,
    rel_sigma_a: float = 0.10,
    rel_sigma_tau: float = 0.084,
) -> np.ndarray:
    """Propagated 1-sigma FRP uncertainty (the LSA-SAF budget).

        sigma_FRP / FRP = sqrt( (s_a/a)^2 + (s_tau/tau)^2
                                + (s_Lb / (Lf - Lb))^2 + (s_Lf / (Lf - Lb))^2 )

    Defaults: 10% on the Wooster constant across 650-1350 K, and 8.4% for the
    level-1.0 -> 1.5 pre-processing term.

    Note the structure: as ``Lf -> Lb`` (a marginal detection) the last two terms
    blow up. Small-FRP detections are intrinsically far more uncertain than
    their point estimate suggests, which is why FRP-weighted metrics matter.
    """
    frp = np.asarray(frp_mw, dtype=np.float64)
    contrast = np.asarray(l_fire, dtype=np.float64) - np.asarray(l_background, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.sqrt(
            rel_sigma_a**2
            + rel_sigma_tau**2
            + (np.asarray(sigma_l_background, dtype=np.float64) / contrast) ** 2
            + (np.asarray(sigma_l_fire, dtype=np.float64) / contrast) ** 2
        )
    return frp * rel


def saturation_mask(bt_mir_k, saturation_k: float) -> np.ndarray:
    """Flag pixels at or above the sensor's MIR saturation temperature.

    Saturation is a **censoring** process: the true value is known only to be
    ``>= observed``. Train with a censored (Tobit-style) likelihood on these
    pixels rather than treating the clipped value as truth, or the model will
    systematically under-predict the largest fires.
    """
    return np.asarray(bt_mir_k, dtype=np.float64) >= saturation_k


def fuel_consumed_kg(fre_mj, rate_kg_per_mj: float = FRE_TO_FUEL_KG_PER_MJ) -> np.ndarray:
    """Dry matter consumed from fire radiative energy. 0.368 +- 0.015 kg/MJ."""
    return np.asarray(fre_mj, dtype=np.float64) * rate_kg_per_mj


def wooster_validity(t_fire_k) -> np.ndarray:
    """Whether the b~=4 approximation holds at this fire temperature."""
    t = np.asarray(t_fire_k, dtype=np.float64)
    lo, hi = WOOSTER_VALID_RANGE_K
    return (t >= lo) & (t <= hi)
