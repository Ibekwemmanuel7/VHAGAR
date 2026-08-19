"""T4 spread: physics propagation core (level-set fire arrival time).

The architecture (`docs/00` section 6) keeps a level-set propagation model as a
first-class component and puts ML at the boundaries (state estimation,
assimilation, residual correction). Fire spread is a front-tracking problem: the
fire perimeter is the level set ``T(x) = t`` of an arrival-time field ``T`` that
satisfies the Eikonal equation

    |grad T(x)| = 1 / ROS(x),

i.e. the front advances normal to itself at the local rate of spread. This
module solves it with the Fast Marching Method (Sethian): a single monotone
sweep, O(N log N), exact for the upwind scheme, no time stepping and no
self-intersecting polygons. ROS is a physical field from fuel, wind and slope.

``skfmm`` is not a dependency; the solver is hand-rolled so the physics core has
no heavy optional install. Anisotropy (wind-driven elliptical spread) is a known
extension noted where it bites; this first cut uses an isotropic-per-cell speed
with a wind/slope-scaled magnitude.
"""

from __future__ import annotations

import heapq

import numpy as np

__all__ = [
    "rate_of_spread",
    "fast_marching_arrival",
    "spread_forecast",
    "persistence_buffer",
    "length_to_breadth",
    "eccentricity_from_lb",
    "anisotropic_arrival",
    "front_length_breadth",
]


def rate_of_spread(fuel, wind, slope, base: float = 0.2, wind_k: float = 1.8,
                   slope_k: float = 1.2) -> np.ndarray:
    """A monotone rate-of-spread field (m/min-ish, arbitrary units).

    ROS rises with fuel load, wind and upslope, the three first-order drivers.
    This is a deliberately simple, documented surrogate for the Rothermel / FBP
    behaviour equations, not a replacement for them; the point of this module is
    the front-tracking and the honest validation, and any calibrated ROS field
    can be dropped in. Values are floored so the front always advances.
    """
    fuel = np.clip(np.asarray(fuel, dtype=np.float64), 0, 1)
    wind = np.clip(np.asarray(wind, dtype=np.float64), 0, 1)
    slope = np.clip(np.asarray(slope, dtype=np.float64), 0, 1)
    ros = base * (0.15 + fuel) * (1.0 + wind_k * wind) * np.exp(slope_k * slope)
    return np.maximum(ros, 1e-3)


def fast_marching_arrival(speed, seeds, dx: float = 1.0) -> np.ndarray:
    """Solve the Eikonal equation for arrival time by Fast Marching.

    ``speed`` is the per-cell rate of spread (>0); ``seeds`` is a boolean mask of
    already-burning cells (arrival time 0). Returns the arrival-time field; cells
    unreachable stay ``inf``. Uses the Godunov upwind quadratic update over
    4-neighbours.

    >>> import numpy as np
    >>> s = np.ones((21, 21)); seed = np.zeros((21, 21), bool); seed[10, 10] = True
    >>> T = fast_marching_arrival(s, seed)
    >>> bool(abs(T[10, 0] - 10.0) < 0.6)   # ~ Euclidean distance / speed
    True
    """
    speed = np.asarray(speed, dtype=np.float64)
    H, W = speed.shape
    F = np.maximum(speed, 1e-9)
    T = np.full((H, W), np.inf)
    accepted = np.zeros((H, W), dtype=bool)
    heap: list = []
    for y, x in zip(*np.where(np.asarray(seeds, dtype=bool)), strict=True):
        T[y, x] = 0.0
        heapq.heappush(heap, (0.0, int(y), int(x)))

    def update(y, x) -> float:
        cx = []
        if x > 0 and accepted[y, x - 1]:
            cx.append(T[y, x - 1])
        if x < W - 1 and accepted[y, x + 1]:
            cx.append(T[y, x + 1])
        cy = []
        if y > 0 and accepted[y - 1, x]:
            cy.append(T[y - 1, x])
        if y < H - 1 and accepted[y + 1, x]:
            cy.append(T[y + 1, x])
        tx = min(cx) if cx else np.inf
        ty = min(cy) if cy else np.inf
        f = dx / F[y, x]
        if np.isinf(tx) and np.isinf(ty):
            return np.inf
        if np.isinf(ty):
            return tx + f
        if np.isinf(tx):
            return ty + f
        if abs(tx - ty) >= f:
            return min(tx, ty) + f
        # 2T^2 - 2(tx+ty)T + (tx^2 + ty^2 - f^2) = 0
        b = -2.0 * (tx + ty)
        c = tx * tx + ty * ty - f * f
        disc = b * b - 8.0 * c
        return (-b + np.sqrt(max(disc, 0.0))) / 4.0

    while heap:
        t, y, x = heapq.heappop(heap)
        if accepted[y, x]:
            continue
        accepted[y, x] = True
        T[y, x] = t
        for dy, dxi in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dxi
            if 0 <= ny < H and 0 <= nx < W and not accepted[ny, nx]:
                nt = update(ny, nx)
                if nt < T[ny, nx]:
                    T[ny, nx] = nt
                    heapq.heappush(heap, (nt, ny, nx))
    return T


def spread_forecast(burned_now, ros, horizon: float, dx: float = 1.0,
                    soft_scale: float | None = None):
    """Forecast the burned area a time ``horizon`` after the current front.

    Runs the arrival-time solver outward from the current burned mask and returns
    ``(mask, prob, arrival)``: the hard burned mask at ``t0 + horizon``, a graded
    burn probability (a logistic of how far inside the horizon each cell's
    arrival time falls, so AP/AUPRC are meaningful), and the arrival-time field
    measured from the current front. Cells already burned stay burned.
    """
    burned_now = np.asarray(burned_now, dtype=bool)
    arrival = fast_marching_arrival(ros, burned_now, dx=dx)
    mask = burned_now | (arrival <= horizon)
    scale = soft_scale if soft_scale is not None else max(horizon * 0.25, 1e-6)
    with np.errstate(over="ignore"):
        prob = 1.0 / (1.0 + np.exp((arrival - horizon) / scale))
    prob = np.where(burned_now, 1.0, prob)
    return mask, prob, arrival


def length_to_breadth(wind, lb_max: float = 4.0):
    """Fire length-to-breadth ratio from normalised wind (>= 1).

    Wind stretches the fire into an ellipse elongated downwind; ``LB`` is the
    ratio of its long axis to its short axis. A simple bounded, monotone surrogate
    ``LB = 1 + (lb_max - 1) * wind`` (any calibrated form, e.g. Alexander 1985 or
    the FBP length-to-breadth, drops in). ``wind=0`` gives ``LB=1`` (a circle).
    """
    return 1.0 + (lb_max - 1.0) * np.clip(np.asarray(wind, dtype=np.float64), 0.0, 1.0)


def eccentricity_from_lb(lb):
    """Ellipse eccentricity from length-to-breadth ratio."""
    lb = np.maximum(np.asarray(lb, dtype=np.float64), 1.0)
    return np.sqrt(lb * lb - 1.0) / lb


# 8-neighbour offsets: (dy, dx, distance, direction angle in grid coords)
_NEI8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]
_NEI8 = [(dy, dx, dd, float(np.arctan2(dy, dx))) for dy, dx, dd in _NEI8]


def anisotropic_arrival(head_ros, wind_speed, wind_dir, seeds, dx: float = 1.0,
                        lb_max: float = 4.0) -> np.ndarray:
    """Wind-driven (elliptical) fire arrival time by 8-connected shortest path.

    The front advances fastest downwind and slowest into wind, tracing an ellipse
    (Richards' elliptical wavelet). In direction ``psi`` from the head the rate is

        ROS(psi) = head_ros * (1 - e) / (1 - e * cos(psi)),

    with eccentricity ``e`` set by the local wind. Arrival time is the
    least-cost path on the 8-neighbour grid where each step costs
    ``distance / ROS(step direction)``, a discrete anisotropic solver (the
    continuous rigorous counterpart is the Ordered Upwind Method; 8-connectivity
    is an adequate, simple approximation that renders the elongation).

    ``wind_dir`` is the downwind (head) direction per cell in radians, grid
    convention ``atan2(dy, dx)``. ``head_ros``, ``wind_speed``, ``wind_dir`` are
    ``[H, W]`` fields (scalars broadcast).
    """
    head_ros = np.asarray(head_ros, dtype=np.float64)
    H, W = head_ros.shape
    wind_speed = np.broadcast_to(np.asarray(wind_speed, dtype=np.float64), (H, W))
    wind_dir = np.broadcast_to(np.asarray(wind_dir, dtype=np.float64), (H, W))
    ecc = eccentricity_from_lb(length_to_breadth(wind_speed, lb_max))

    T = np.full((H, W), np.inf)
    done = np.zeros((H, W), dtype=bool)
    heap: list = []
    for y, x in zip(*np.where(np.asarray(seeds, dtype=bool)), strict=True):
        T[y, x] = 0.0
        heapq.heappush(heap, (0.0, int(y), int(x)))

    while heap:
        t, y, x = heapq.heappop(heap)
        if done[y, x]:
            continue
        done[y, x] = True
        e = ecc[y, x]
        r0 = max(head_ros[y, x], 1e-9)
        wd = wind_dir[y, x]
        for dy, dxi, dd, ang in _NEI8:
            ny, nx = y + dy, x + dxi
            if 0 <= ny < H and 0 <= nx < W and not done[ny, nx]:
                psi = ang - wd
                ros = r0 * (1.0 - e) / (1.0 - e * np.cos(psi))
                nt = t + dd * dx / max(ros, 1e-9)
                if nt < T[ny, nx]:
                    T[ny, nx] = nt
                    heapq.heappush(heap, (nt, ny, nx))
    return T


def front_length_breadth(mask) -> float:
    """Measured length-to-breadth of a burned mask, via its principal moments.

    Returns the ratio of the long to the short principal-axis extent (~1 for a
    circle, larger for a wind-elongated ellipse). Used to check the solver.
    """
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if ys.size < 3:
        return 1.0
    cov = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-9, None)
    return float(np.sqrt(ev[-1] / ev[0]))


def persistence_buffer(burned_now, radius_cells: float):
    """Mandatory baseline: persistence plus an isotropic buffer.

    Predicts the current burned area dilated by a fixed radius, the naive
    "it will grow a ring outward" forecast that any real model must beat. Returns
    ``(mask, prob)`` with a distance-decayed probability in the buffer.
    """
    from scipy.ndimage import distance_transform_edt

    burned_now = np.asarray(burned_now, dtype=bool)
    if burned_now.any():
        dist = distance_transform_edt(~burned_now)
    else:
        dist = np.full(burned_now.shape, np.inf)
    mask = burned_now | (dist <= radius_cells)
    r = max(radius_cells, 1e-6)
    prob = np.clip(1.0 - dist / r, 0.0, 1.0)
    prob = np.where(burned_now, 1.0, prob)
    return mask, prob
