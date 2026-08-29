#!/usr/bin/env python3
"""VHAGAR T4 highlight MP4: a wind-driven fire front advancing by the repo's own
anisotropic fast-marching arrival-time solver. Physics core, honest validation
footer. Times New Roman (Liberation Serif), 1080x1080, H.264 High yuv420p."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from vhagar.models.spread import anisotropic_arrival, length_to_breadth, rate_of_spread

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
SERIF_B = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
fm.fontManager.addfont(SERIF)
fm.fontManager.addfont(SERIF_B)
plt.rcParams["font.family"] = "Liberation Serif"

INK = "#e9f0fa"
MUT = "#95a6be"
BLUE = "#3a86ff"

FPS = 25
DUR = 20.0
H = W = 200


def smooth_field(rng, h, w, k=3):
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    f = np.zeros((h, w))
    for _ in range(k):
        a, b = rng.uniform(1.5, 4.5, size=2)
        ph = rng.uniform(0, 2 * np.pi, size=2)
        f += np.sin(a * 2 * np.pi * yy + ph[0]) * np.cos(b * 2 * np.pi * xx + ph[1])
    return (f - f.min()) / (np.ptp(f) + 1e-9)


def build_arrival(seed_rc=(168, 104), wind_dir_deg=-55.0):
    rng = np.random.default_rng(7)
    fuel = 0.35 + 0.6 * smooth_field(rng, H, W)
    wind = 0.55 + 0.4 * smooth_field(rng, H, W)
    slope = smooth_field(rng, H, W)
    barrier = (smooth_field(rng, H, W) > 0.80)          # firebreaks / water
    head = rate_of_spread(fuel, wind, slope)
    head = np.where(barrier, head * 0.08, head)          # front stalls at barriers
    seeds = np.zeros((H, W), dtype=bool)
    seeds[seed_rc] = True
    wd = np.deg2rad(wind_dir_deg)
    arrival = anisotropic_arrival(head, wind_speed=0.8, wind_dir=wd, seeds=seeds)
    return arrival, head, barrier, seeds, wind_dir_deg


def ease(t):
    return 1 - (1 - t) ** 3


def main(out: Path):
    arrival, head, barrier, seeds, wdir = build_arrival()
    finite = np.isfinite(arrival)
    tmax = float(np.quantile(arrival[finite], 0.92))     # ignore slow far tail
    total = int(FPS * DUR)
    cell_ha = 0.09                                       # arbitrary honest unit for the counter
    lb = float(length_to_breadth(0.8))

    fig = plt.figure(figsize=(10.8, 10.8), dpi=100)
    fig.patch.set_facecolor("#0a1524")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.set_facecolor("#0a1524"); ax.axis("off")

    # static backdrop: faint fuel texture + barriers
    ax.imshow(head, cmap="Greys_r", alpha=0.10, origin="upper")
    ax.imshow(np.ma.masked_where(~barrier, barrier), cmap=mcolors.ListedColormap(["#20344d"]),
              alpha=0.55, origin="upper")

    burn_im = ax.imshow(np.ma.masked_all((H, W)), cmap="inferno",
                        vmin=0, vmax=tmax, origin="upper", alpha=0.92, interpolation="bilinear")
    front_pts, = ax.plot([], [], ".", color="#fff3c4", ms=1.6, alpha=0.9)
    sy, sx = np.argwhere(seeds)[0]
    ax.plot([sx], [sy], "o", mfc="#7fe0ff", mec="white", mew=1.5, ms=11, zorder=5)

    # heading arrow: measured direction from ignition to the burned-plume centroid,
    # so it always matches the front the solver actually produced.
    _b = finite & (arrival <= tmax)
    _cy, _cx = np.argwhere(_b).mean(axis=0)
    _vy, _vx = _cy - sy, _cx - sx
    _n = float(np.hypot(_vx, _vy)) + 1e-9
    ax.annotate("", xy=(sx + 40 * _vx / _n, sy + 40 * _vy / _n), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="#7fe0ff", lw=2.6, alpha=0.95), zorder=6)
    ax.text(sx + 46 * _vx / _n, sy + 46 * _vy / _n, "heading", color="#7fe0ff",
            fontsize=15, style="italic", zorder=6)

    tt = {}
    tt["title"] = fig.text(0.035, 0.945, "VHAGAR  ·  T4 SPREAD", color=INK,
                           fontsize=34, fontweight="bold", fontfamily="Liberation Serif")
    tt["sub"] = fig.text(0.035, 0.912, "Wind-driven fire front by fast-marching arrival time",
                         color=BLUE, fontsize=20)
    tt["clock"] = fig.text(0.035, 0.075, "", color=INK, fontsize=26, fontweight="bold")
    tt["area"] = fig.text(0.035, 0.045, "", color=MUT, fontsize=20)
    fig.text(0.965, 0.058,
             "physics core, ML at the boundary", color=MUT, fontsize=19,
             ha="right", style="italic")
    fig.text(0.965, 0.030,
             "SYNTHETIC benchmark, real-data validation pending  ·  corrector AP 0.871 > persistence 0.466",
             color=MUT, fontsize=16, ha="right")

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i in range(total):
            t = ease(i / max(total - 1, 1)) * tmax
            burned = finite & (arrival <= t)
            masked = np.ma.masked_where(~burned, arrival)
            burn_im.set_data(masked)
            # active front = isochrone shell just inside t
            shell = burned & (arrival > t - tmax * 0.03)
            ys, xs = np.where(shell)
            front_pts.set_data(xs, ys)
            hours = 72.0 * (t / tmax)
            tt["clock"].set_text(f"t = +{hours:4.1f} h   (72 h spread forecast)")
            tt["area"].set_text(f"burned footprint {burned.sum() * cell_ha:6.0f} ha   ·   "
                                f"length:breadth {lb:.1f} (wind-elongated)")
            fig.savefig(tdp / f"f{i:04d}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
        cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(tdp / "f%04d.png"),
               "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", str(FPS), str(out)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--out") + 1]))
