# T4 Rothermel surface fire spread (physics ROS)

Until now the T4 front used a fuel-and-wind **surrogate** rate of spread
(`models/fuels.py`): a relative fuel factor times a base rate. This adds the canonical
**Rothermel (1972)** surface fire spread model (`models/rothermel.py`), the equation
that underlies BehavePlus, FlamMap, and FARSITE, so the ROS field can be real physics
instead of a surrogate.

## What it computes

Head rate of spread in m/min from fuel-model parameters, dead-fuel moisture, midflame
wind, and slope, following Rothermel (1972) and Andrews (2018, RMRS-GTR-371):

    R = I_R * xi * (1 + phi_w + phi_s) / (rho_b * epsilon * Q_ig)

with reaction intensity, propagating flux ratio, wind and slope factors, bulk density,
effective heating number, and heat of preignition all as published. Verified behaviors
(tests): ROS rises with wind and slope, falls with moisture, and is zero at or above the
moisture of extinction and on non-burnable fuel.

## Scope and honesty

This is the standard **single characteristic surface-area-to-volume** formulation (one
weighted dead-fuel class). It does **not** yet do the full multi-size-class dead-and-live
fuel weighting of the 40 Scott and Burgan models; that is the documented remaining
refinement. The per-group fuel parameters (`STANDARD_FUELS`) are representative surrogates
for the full per-model tables, not the exact published constants for all 40 models. So
this closes the physics ROS core, honestly, and states what is left.

Wind is the **midflame** wind. Open (10 m) winds must be reduced by a midflame wind
adjustment factor first; `rothermel_head_ros_field(..., wind_adj=0.3)` does that.

## Coupling to the anisotropic front

Rothermel gives the head (maximum, downwind) ROS **magnitude**, including wind and slope.
The anisotropic arrival-time solver then distributes that magnitude **directionally**
through the Richards ellipse. So Rothermel sets the rate and the ellipse sets the shape,
and wind is not double-counted (the ellipse uses wind only for eccentricity):

    head = rothermel_head_ros_field(codes, m_f, wind_ms, wind_adj, slope_tan)
    T = anisotropic_arrival(head, wind_speed, wind_dir, seeds, dx, lb_max)

## Run

    python -m vhagar.cli t4-rothermel --moisture 0.06 --wind 5 --slope 0.2

Representative output (moisture 6%, midflame wind 5 m/s): grass ~57, grass-shrub ~78,
shrub ~122, timber-understory ~40, timber-litter ~7 m/min, rising with slope. The batch
catalogue can use it per event with `EventSpec(use_rothermel=True, dead_moisture=...,
wind_adj=..., slope_tan=...)`.
