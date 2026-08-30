"""Planck radiometry, the foundation every other physics module stands on.

Everything here is analytic and differentiable, which is the whole point: it can
be used as a NumPy utility *and*, via :mod:`vhagar.physics.torch_ops`, as a
non-trainable layer inside a neural network.

Why the mid-infrared channel exists
-----------------------------------
The local logarithmic sensitivity of the Planck function,

    b(lambda, T) = d ln B / d ln T = x * e^x / (e^x - 1),   x = c2 / (lambda * T)

is what makes fire detection possible at all:

    lambda    T        x       b
    3.9 um    300 K    12.3    12.3
    11  um    300 K     4.36    4.4
    3.9 um   1000 K     3.69    3.8

At a 300 K background the 3.9 um channel is ~2.8x more sensitive per unit
relative temperature change than 11 um, and because the contrast is exponential
the monochromatic radiance ratio of a 1000 K flame to a 300 K background is
**5.6e3 at 3.9 um versus 28.6 at 11 um** -- a factor of ~196 advantage.

Concretely: a pixel 0.5% covered by a 1000 K fire over a 300 K background reads
**413 K** at 3.9 um and only ~307 K at 11 um.

(Published summaries often quote ~394 K for that example. That figure comes from
the linearised form ``T_b * (1 + p * ratio)**(1/b)`` evaluated with a *constant*
b = 12.3. Since b falls as temperature rises, the linearisation under-reads by
~19 K. The functions here are exact; the tests assert the exact values and this
note exists so nobody "fixes" them back to the approximation. Real sensors
integrate over a spectral response function, which shifts things again by a few
kelvin -- use band-integrated radiances for quantitative work.)

The second fact that falls out of ``b``: it passes through 4 near 950 K and
averages ~4 over 650-1350 K. That is the entire justification for the Wooster
power-law FRP method in :mod:`vhagar.physics.frp` -- and also the reason that
method is only accurate (~+-12%) inside that temperature range.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "C2_UM_K",
    "STEFAN_BOLTZMANN",
    "brightness_temperature",
    "dozier_contrast_ratio",
    "emissivity_error_to_bt_error",
    "mixed_pixel_radiance",
    "planck_radiance",
    "planck_sensitivity_exponent",
]

#: First radiation constant for spectral radiance, W um^4 m^-2 sr^-1.
C1_L = 1.191042953e8
#: Second radiation constant, um K.
C2_UM_K = 1.4387769e4
#: Stefan-Boltzmann constant, W m^-2 K^-4.
STEFAN_BOLTZMANN = 5.670374419e-8


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def planck_radiance(wavelength_um, temperature_k) -> np.ndarray:
    """Spectral radiance of a blackbody.

    Returns W m^-2 sr^-1 um^-1.

    >>> round(float(planck_radiance(3.9, 300.0)), 4)
    0.6025
    >>> round(float(planck_radiance(11.0, 300.0)), 4)
    9.5732
    """
    lam = _f(wavelength_um)
    t = _f(temperature_k)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = C2_UM_K / (lam * t)
        out = C1_L / (lam**5 * np.expm1(x))
    return np.where(t > 0, out, 0.0)


def brightness_temperature(wavelength_um, radiance) -> np.ndarray:
    """Invert the Planck function: radiance -> equivalent blackbody temperature.

    Non-positive radiance returns NaN rather than a complex or negative
    temperature -- a silently negative BT is how nodata becomes "cold ground".

    >>> round(float(brightness_temperature(3.9, planck_radiance(3.9, 412.0))), 6)
    412.0
    """
    lam = _f(wavelength_um)
    rad = _f(radiance)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = C2_UM_K / (lam * np.log1p(C1_L / (lam**5 * rad)))
    return np.where(rad > 0, out, np.nan)


def planck_sensitivity_exponent(wavelength_um, temperature_k) -> np.ndarray:
    """Local power-law exponent ``b`` such that ``B ~ T**b``.

    ``b = x e^x / (e^x - 1)`` with ``x = c2 / (lambda T)``. This single number
    explains why MIR is the fire channel and why the Wooster method works.

    >>> float(round(planck_sensitivity_exponent(3.9, 300.0), 1))
    12.3
    >>> float(round(planck_sensitivity_exponent(11.0, 300.0), 1))
    4.4
    """
    lam = _f(wavelength_um)
    t = _f(temperature_k)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = C2_UM_K / (lam * t)
        out = x * np.exp(x) / np.expm1(x)
    return out


def dozier_contrast_ratio(wavelength_um, t_fire_k, t_background_k) -> np.ndarray:
    """Radiance ratio ``B(T_fire) / B(T_background)`` at one wavelength.

    The quantity that decides whether a sub-pixel fire is detectable at all.

    >>> r39 = dozier_contrast_ratio(3.9, 1000.0, 300.0)
    >>> r11 = dozier_contrast_ratio(11.0, 1000.0, 300.0)
    >>> bool(190 < r39 / r11 < 200)
    True
    """
    return planck_radiance(wavelength_um, t_fire_k) / planck_radiance(
        wavelength_um, t_background_k
    )


def mixed_pixel_radiance(
    wavelength_um,
    fire_fraction,
    t_fire_k,
    t_background_k,
    emissivity_fire: float = 1.0,
    emissivity_background: float = 1.0,
) -> np.ndarray:
    """Two-component (Dozier) surface-leaving radiance of a mixed pixel.

    ``L = p * eps_f * B(T_f) + (1 - p) * eps_b * B(T_b)``

    This is the *surface* term only. Atmospheric transmittance and path radiance
    are applied separately in :mod:`vhagar.physics.atmosphere`, deliberately,
    so that the forward model composes cleanly and each stage is testable.
    """
    p = np.clip(_f(fire_fraction), 0.0, 1.0)
    return p * emissivity_fire * planck_radiance(wavelength_um, t_fire_k) + (
        1.0 - p
    ) * emissivity_background * planck_radiance(wavelength_um, t_background_k)


def emissivity_error_to_bt_error(temperature_k, wavelength_um, delta_emissivity) -> np.ndarray:
    """Brightness-temperature error induced by an emissivity error.

    ``dBT ~= (T / b) * (d_eps / eps)``.

    At 300 K a 0.01 emissivity error is worth ~0.26 K at 3.9 um but ~0.72 K at
    11 um. If the two channels' emissivity errors are uncorrelated -- which they
    are over quartz-rich and mixed surfaces -- the induced error in
    dT(MIR-TIR) is ~0.8 K per 0.01.

    MODIS Collection 6's contextual threshold is ``dT > mean + 6 K``, so a 0.05
    emissivity error at 11 um alone consumes ~60% of the detection margin.
    That is the mechanism behind desert-margin commission errors, and the reason
    emissivity belongs in the feature vector.
    """
    t = _f(temperature_k)
    b = planck_sensitivity_exponent(wavelength_um, t)
    return (t / b) * _f(delta_emissivity)
