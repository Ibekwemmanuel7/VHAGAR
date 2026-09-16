"""T4 WUI extension: ember spotting and structure-to-structure spread.

The wildland spread core (:mod:`vhagar.models.spread`) tracks a fire front as the
level set of an Eikonal arrival-time field. Two behaviours dominate loss in the
**wildland-urban interface (WUI)** that a pure wildland front misses, and both
layer on top of that core rather than replacing it:

* **Ember spotting.** Firebrands lofted from the flaming front land downwind and
  start new fires *ahead* of the perimeter, often the mechanism that carries fire
  into a community across a road or firebreak. Modelled here as a wind-oriented
  landing kernel convolved with the front's brand emission, then Poisson-thinned
  by fuel/structure receptivity into a spot-ignition probability field.

* **Structure-to-structure spread.** Once a structure ignites, radiant and
  convective heat plus its own embers ignite neighbours, a conflagration that
  propagates through the building network, not the vegetation. Modelled as
  probabilistic propagation on the structure graph (distance- and wind-weighted),
  seeded by the structures the wildland front (or a spot fire) reaches.

Design, consistent with the rest of VHAGAR: the transport and propagation are
explicit, physically-motivated surrogates (any calibrated spotting-distance
distribution, e.g. Albini or Sardoy, or a fitted structure-ignition curve, drops
in); ML belongs at the boundary (fitting those curves to observed loss). Ground
truth for validation is CAL FIRE **DINS** structure damage inspections
(:mod:`vhagar.eval.wui`). Nothing here is calibrated to real loss yet; treat the
outputs as a physically-reasonable prototype pending that fit.

All functions are pure ``numpy`` (+ ``scipy`` for FFT convolution and the
neighbour tree, both already project dependencies). No global RNG state: the
stochastic pieces return *probabilities*, so results are deterministic and
scorable; draw samples explicitly with a passed generator only where noted.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rasterize_structures",
    "spotting_kernel",
    "spot_ignition_probability",
    "structures_reached_by_front",
    "structure_to_structure_spread",
    "wui_spread",
]


def rasterize_structures(rows, cols, shape, weights=None) -> np.ndarray:
    """Accumulate point structures onto a grid as per-cell exposure weight.

    ``rows``/``cols`` are integer grid indices of each structure (e.g. building
    centroids snapped to the analysis grid); ``weights`` is an optional per-
    structure value (replacement cost, footprint area), default 1 = a count.
    Out-of-bounds structures are dropped. Returns a ``[H, W]`` float grid whose
    sum equals the total in-bounds weight, the exposure surface T5 loss modelling
    consumes.

    >>> import numpy as np
    >>> g = rasterize_structures([0, 0, 2], [0, 0, 1], (3, 3))
    >>> float(g[0, 0]), float(g[2, 1]), float(g.sum())
    (2.0, 1.0, 3.0)
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    H, W = int(shape[0]), int(shape[1])
    grid = np.zeros((H, W), dtype=np.float64)
    if rows.size == 0:
        return grid
    w = np.ones(rows.shape, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    m = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    np.add.at(grid, (rows[m], cols[m]), w[m])
    return grid


def spotting_kernel(max_dist_cells: float, wind_speed: float, wind_dir: float,
                    focus: float = 3.0, floor: float = 0.05) -> np.ndarray:
    """Wind-oriented firebrand landing kernel (sums to 1).

    Returns a square ``(2R+1, 2R+1)`` kernel, ``R = ceil(max_dist_cells)``, giving
    the fraction of brands emitted at the centre cell that land at each offset.
    Two factors multiply:

    * distance decay ``exp(-d / scale)`` with ``scale`` growing with wind, so
      stronger wind throws brands farther;
    * a directional weight biased **downwind**: with ``wind_dir`` the downwind
      heading (grid ``atan2(dy, dx)``), offsets aligned downwind are favoured by
      ``clip(cos(theta), 0, 1) ** focus``, blended toward isotropic as wind falls
      (at ``wind_speed = 0`` the kernel is radially symmetric). ``floor`` keeps a
      small non-downwind probability so flanking/backing spotting is not zero.

    The centre cell is set to 0 (a brand is not a self-spot) and the kernel is
    normalised, so convolving an emission field with it conserves total brands.

    >>> import numpy as np
    >>> k = spotting_kernel(4, wind_speed=1.0, wind_dir=0.0)   # downwind = +x
    >>> R = k.shape[1] // 2
    >>> bool(k[R, R] == 0.0 and abs(k.sum() - 1.0) < 1e-9)
    True
    >>> bool(k[:, R + 1:].sum() > k[:, :R].sum())   # more mass downwind (+x) than up
    True
    >>> k0 = spotting_kernel(4, wind_speed=0.0, wind_dir=0.0)
    >>> bool(abs(k0[:, R + 1:].sum() - k0[:, :R].sum()) < 1e-9)   # isotropic
    True
    """
    R = int(np.ceil(max(max_dist_cells, 1.0)))
    ys, xs = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    d = np.hypot(xs, ys)
    s = float(np.clip(wind_speed, 0.0, 1.0))
    scale = max(max_dist_cells, 1e-6) * (0.35 + 0.65 * s) / 3.0
    dist_w = np.exp(-d / max(scale, 1e-6))
    ang = np.arctan2(ys, xs)
    align = np.clip(np.cos(ang - wind_dir), 0.0, 1.0) ** focus
    dir_w = (1.0 - s) + s * (floor + (1.0 - floor) * align)
    k = dist_w * dir_w
    k[R, R] = 0.0
    total = k.sum()
    return k / total if total > 0 else k


def spot_ignition_probability(emission, receptivity, wind_speed, wind_dir,
                              max_dist_cells: float = 6.0, intensity: float = 1.0,
                              focus: float = 3.0) -> np.ndarray:
    """Per-cell probability of receiving a spot ignition from the front.

    ``emission`` is a ``[H, W]`` field of relative brand production at actively
    burning cells (e.g. fireline intensity of the current front, 0 elsewhere).
    It is convolved with :func:`spotting_kernel` to get expected brand landings
    per cell, scaled by ``intensity`` (a global loft/production factor), then
    thinned by ``receptivity`` in ``[0, 1]`` (fuel presence, or structure
    presence for WUI ignition) via a Poisson law ``p = 1 - exp(-landings *
    receptivity)``. Uniform ``wind_speed``/``wind_dir`` (scalars) for this cut.

    Returns a probability field in ``[0, 1]``. Cells already emitting are left for
    the caller to treat as burned.

    >>> import numpy as np
    >>> em = np.zeros((21, 21)); em[10, 10] = 1.0            # one hot front cell
    >>> rec = np.ones((21, 21))
    >>> p = spot_ignition_probability(em, rec, wind_speed=1.0, wind_dir=0.0,
    ...                               max_dist_cells=6, intensity=5.0)
    >>> bool(p[10, 13] > p[10, 7])       # ignition more likely downwind (+x)
    True
    >>> bool(p.max() <= 1.0 and p.min() >= 0.0)
    True
    """
    from scipy.signal import fftconvolve

    emission = np.asarray(emission, dtype=np.float64)
    receptivity = np.clip(np.asarray(receptivity, dtype=np.float64), 0.0, 1.0)
    k = spotting_kernel(max_dist_cells, wind_speed, wind_dir, focus=focus)
    landings = fftconvolve(emission, k, mode="same") * float(intensity)
    landings = np.clip(landings, 0.0, None)
    return 1.0 - np.exp(-landings * receptivity)


def structures_reached_by_front(struct_rows, struct_cols, arrival, horizon: float) -> np.ndarray:
    """Boolean per-structure: did the wildland front reach it within ``horizon``?

    ``arrival`` is the arrival-time field from the wildland solver
    (:func:`vhagar.models.spread.fast_marching_arrival`). A structure is seeded as
    ignited by the vegetation fire when the front reaches its cell by ``horizon``.
    Out-of-bounds structures are treated as not reached.

    >>> import numpy as np
    >>> arr = np.full((3, 3), np.inf); arr[0, 0] = 2.0; arr[2, 2] = 20.0
    >>> structures_reached_by_front([0, 2], [0, 2], arr, horizon=10.0).tolist()
    [True, False]
    """
    rows = np.asarray(struct_rows, dtype=np.int64)
    cols = np.asarray(struct_cols, dtype=np.int64)
    arrival = np.asarray(arrival, dtype=np.float64)
    H, W = arrival.shape
    out = np.zeros(rows.shape, dtype=bool)
    m = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    idx = np.where(m)[0]
    out[idx] = arrival[rows[m], cols[m]] <= horizon
    return out


def structure_to_structure_spread(
    struct_rows, struct_cols, ignited0, wind_speed: float = 0.0, wind_dir: float = 0.0,
    radius_cells: float = 3.0, base_p: float = 0.6, focus: float = 2.0,
    threshold: float = 0.5, max_rounds: int = 50,
) -> np.ndarray:
    """Propagate ignition through the structure network from a seed set.

    Structures within ``radius_cells`` of a burning structure can ignite; the
    pairwise ignition probability decays with separation ``exp(-d / (radius/2))``
    and is boosted downwind (``clip(cos(theta - wind_dir), 0, 1) ** focus`` blended
    to isotropic as wind falls), scaled by ``base_p``. Each non-ignited structure
    accumulates an independent-OR probability from all currently-burning neighbours;
    when that exceeds ``threshold`` it ignites and can propagate in the next round.
    Iterates to a fixed point (or ``max_rounds``). Deterministic, so the output is
    directly scorable; for stochastic realisations, sample against the per-round
    probabilities with an explicit RNG instead.

    Returns the final boolean ignited/destroyed mask over structures.

    >>> import numpy as np
    >>> # four structures in a downwind (+x) line, 1 cell apart; ignite the first
    >>> r = [0, 0, 0, 0]; c = [0, 1, 2, 3]
    >>> seed = np.array([True, False, False, False])
    >>> out = structure_to_structure_spread(r, c, seed, wind_speed=1.0, wind_dir=0.0,
    ...                                      radius_cells=3, base_p=0.9)
    >>> out.tolist()
    [True, True, True, True]
    >>> # a structure far upwind and out of radius should not ignite
    >>> r2 = [0, 0]; c2 = [0, 20]
    >>> structure_to_structure_spread(r2, c2, np.array([True, False]),
    ...                               radius_cells=3).tolist()
    [True, False]
    """
    from scipy.spatial import cKDTree

    rows = np.asarray(struct_rows, dtype=np.float64)
    cols = np.asarray(struct_cols, dtype=np.float64)
    ignited = np.asarray(ignited0, dtype=bool).copy()
    n = ignited.size
    if n == 0 or not ignited.any():
        return ignited
    pts = np.column_stack([cols, rows])          # (x, y) so angle uses atan2(dy, dx)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=radius_cells, output_type="ndarray")
    if pairs.size == 0:
        return ignited
    # Precompute per-pair ignition probability (symmetric in distance, but the
    # directional boost depends on which end is burning, so keep both directed edges).
    # Vectorised so it scales to tens of thousands of structures.
    s = float(np.clip(wind_speed, 0.0, 1.0))
    decay_scale = max(radius_cells, 1e-6)
    a = np.concatenate([pairs[:, 0], pairs[:, 1]])
    b = np.concatenate([pairs[:, 1], pairs[:, 0]])
    dx = cols[b] - cols[a]
    dy = rows[b] - rows[a]
    dist = np.hypot(dx, dy)
    align = np.clip(np.cos(np.arctan2(dy, dx) - wind_dir), 0.0, 1.0) ** focus
    dirw = (1.0 - s) + s * align
    pab = base_p * np.exp(-dist / decay_scale) * dirw

    for _ in range(max_rounds):
        src_burning = ignited[a]
        # for each target b, combine (1 - p) over burning sources: independent OR
        surv = np.ones(n, dtype=np.float64)
        active = src_burning & ~ignited[b]
        np.multiply.at(surv, b[active], 1.0 - pab[active])
        p_new = 1.0 - surv
        newly = (p_new >= threshold) & ~ignited
        if not newly.any():
            break
        ignited |= newly
    return ignited


def wui_spread(
    burned_now, ros, struct_rows, struct_cols, horizon: float,
    wind_speed: float = 0.0, wind_dir: float = 0.0, dx: float = 1.0,
    fireline_intensity=None, spotting_max_dist: float = 6.0, spotting_intensity: float = 3.0,
    struct_radius_cells: float = 3.0, struct_base_p: float = 0.6,
    anisotropic: bool = False, arrival=None, struct_edge_cells: float = 0.0,
) -> dict:
    """End-to-end WUI spread: wildland front -> spot ignitions -> structure loss.

    Runs the wildland arrival-time solver from ``burned_now`` over ``ros``, seeds
    the structures the front reaches by ``horizon``, adds spot ignitions from
    embers landing on structures ahead of the front, then propagates
    structure-to-structure. Returns a dict with the wildland ``arrival`` field, the
    per-structure ``reached`` / ``spot_ignited`` / ``destroyed`` boolean masks, and
    the ``spot_prob`` field. ``fireline_intensity`` (``[H, W]``, defaults to the
    ROS field) drives brand emission from the current front ring.

    With ``anisotropic=True`` the wildland front is solved with the wind-driven
    elliptical solver (:func:`vhagar.models.spread.anisotropic_arrival`) using
    ``wind_speed``/``wind_dir``, so the front elongates downwind, essential for real
    fires whose destroyed footprint is set by wind, not radial distance. The default
    (isotropic fast marching) is kept for the no-wind case and existing callers.
    """
    from vhagar.models.spread import anisotropic_arrival, fast_marching_arrival

    burned_now = np.asarray(burned_now, dtype=bool)
    ros = np.asarray(ros, dtype=np.float64)
    if arrival is not None:
        # caller-supplied precomputed field: the arrival solve is invariant to the
        # spotting/structure parameters, so a calibration sweep computes it once.
        arrival = np.asarray(arrival, dtype=np.float64)
    elif anisotropic:
        arrival = anisotropic_arrival(ros, wind_speed, wind_dir, burned_now, dx=dx)
    else:
        arrival = fast_marching_arrival(ros, burned_now, dx=dx)
    rows = np.asarray(struct_rows, dtype=np.int64)
    cols = np.asarray(struct_cols, dtype=np.int64)

    if struct_edge_cells > 0:
        # Seed structures within struct_edge_cells of the burned wildland, not only those
        # on a burned cell. Essential with real fuels: structures sit in non-burnable urban
        # cells the front cannot enter, so the front reaches the town edge and ignites the
        # adjacent structures, then the structure graph carries the conflagration inward.
        from scipy.ndimage import distance_transform_edt
        burned = burned_now | (arrival <= horizon)
        reach_field = distance_transform_edt(~burned) <= struct_edge_cells
        H0, W0 = arrival.shape
        rr, cc = rows, cols
        reached = np.zeros(rr.shape, dtype=bool)
        inbf = (rr >= 0) & (rr < H0) & (cc >= 0) & (cc < W0)
        reached[inbf] = reach_field[rr[inbf], cc[inbf]]
    else:
        reached = structures_reached_by_front(rows, cols, arrival, horizon)

    # Brand emission: cells the front is passing through within the horizon window.
    intensity = ros if fireline_intensity is None else np.asarray(fireline_intensity, dtype=np.float64)
    front = (arrival <= horizon) & ~burned_now
    emission = np.where(front, intensity, 0.0)
    structure_grid = rasterize_structures(rows, cols, arrival.shape)
    receptivity = np.clip(structure_grid, 0.0, 1.0)
    spot_prob = spot_ignition_probability(
        emission, receptivity, wind_speed, wind_dir,
        max_dist_cells=spotting_max_dist, intensity=spotting_intensity,
    )
    H, W = arrival.shape
    inb = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    spot_at_struct = np.zeros(rows.shape, dtype=np.float64)
    spot_at_struct[inb] = spot_prob[rows[inb], cols[inb]]
    spot_ignited = (~reached) & (spot_at_struct >= 0.5)

    seed = reached | spot_ignited
    destroyed = structure_to_structure_spread(
        rows, cols, seed, wind_speed=wind_speed, wind_dir=wind_dir,
        radius_cells=struct_radius_cells, base_p=struct_base_p,
    )
    return {
        "arrival": arrival,
        "reached": reached,
        "spot_ignited": spot_ignited,
        "destroyed": destroyed,
        "spot_prob": spot_prob,
        "n_structures": int(rows.size),
        "n_destroyed": int(destroyed.sum()),
    }
