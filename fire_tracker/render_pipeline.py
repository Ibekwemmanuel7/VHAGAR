#!/usr/bin/env python3
"""VHAGAR end-to-end-on-one-fire highlight: four titled scenes (T1 detect, T3
danger, T4 spread, T2 burned-area pending) for the Crittenburg Complex, Coryell
County TX. Real live figures + the repo's own spread solver. Times New Roman,
1080x1080, H.264 High yuv420p, ~19s with crossfades."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "fire_tracker"
FONTS = "/usr/share/fonts/truetype/liberation"

W = H = 1080
FPS = 25
INK = (233, 240, 250)
MUT = (150, 166, 190)
BLUE = (58, 134, 255)
CHIP = (24, 40, 66)
LINE = (36, 56, 86)
GREEN = (74, 200, 138)
AMBER = (255, 176, 60)
RED = (255, 92, 78)
BG_TOP = (9, 18, 33)
BG_BOT = (13, 27, 48)


def font(name, size):
    return ImageFont.truetype(f"{FONTS}/{name}", size)


def bg():
    img = Image.new("RGB", (W, H), BG_TOP)
    px = img.load()
    for y in range(H):
        f = y / H
        px_row = (int(BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * f),
                  int(BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * f),
                  int(BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * f))
        for x in range(W):
            px[x, y] = px_row
    return img


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def header(d, tier_tag, tier_col=BLUE):
    d.text((70, 60), "VHAGAR", font=font("LiberationSerif-Bold.ttf", 44), fill=INK)
    d.text((70, 116), "END TO END ON ONE FIRE", font=font("LiberationSerif-Bold.ttf", 20), fill=BLUE)
    # tier chip top-right
    f = font("LiberationSerif-Bold.ttf", 26)
    tw = d.textlength(tier_tag, font=f)
    rrect(d, [W - 90 - tw - 40, 62, W - 70, 112], 25, fill=CHIP, outline=tier_col, width=3)
    d.text((W - 90 - tw - 20, 72), tier_tag, font=f, fill=INK)
    d.line([(70, 150), (W - 70, 150)], fill=LINE, width=2)
    d.text((70, 168), "Crittenburg Complex  ·  Coryell County, Texas",
           font=font("LiberationSerif-Bold.ttf", 34), fill=INK)
    d.text((70, 214), "~31.5N, 98.1W  ·  near Evant / Flat",
           font=font("LiberationSerif-Regular.ttf", 24), fill=MUT)


def tile(d, x, y, w, h, label, value, unit="", vcol=INK, ocol=LINE):
    rrect(d, [x, y, x + w, y + h], 18, fill=CHIP, outline=ocol, width=2)
    d.text((x + 26, y + 22), label, font=font("LiberationSerif-Bold.ttf", 22), fill=BLUE)
    vf = font("LiberationSerif-Bold.ttf", 52)
    d.text((x + 26, y + 54), value, font=vf, fill=vcol)
    if unit:
        vw = d.textlength(value, font=vf)
        d.text((x + 26 + vw + 12, y + 78), unit, font=font("LiberationSerif-Bold.ttf", 24), fill=MUT)


def footer(d, text, col=MUT):
    d.text((70, H - 96), text, font=font("LiberationSerif-Italic.ttf", 25), fill=col)


def scene_console(src, tier_tag, tier_col, caption, path):
    """Compose a real console screenshot into a 1080x1080 panel with a VHAGAR
    header, tier chip and caption band. src is a landscape console capture."""
    img = bg(); d = ImageDraw.Draw(img)
    shot = Image.open(src).convert("RGB")
    bw = W - 80
    bh = int(shot.height * bw / shot.width)
    shot = shot.resize((bw, bh), Image.LANCZOS)
    top = 250
    d.rectangle([40 - 3, top - 3, 40 + bw + 3, top + bh + 3], outline=LINE, width=3)
    img.paste(shot, (40, top))
    # header over the top band
    d.text((70, 60), "VHAGAR", font=font("LiberationSerif-Bold.ttf", 44), fill=INK)
    d.text((70, 116), "END TO END ON ONE FIRE  ·  LIVE CONSOLE",
           font=font("LiberationSerif-Bold.ttf", 20), fill=BLUE)
    f = font("LiberationSerif-Bold.ttf", 26)
    tw = d.textlength(tier_tag, font=f)
    rrect(d, [W - 90 - tw - 40, 62, W - 70, 112], 25, fill=CHIP, outline=tier_col, width=3)
    d.text((W - 90 - tw - 20, 72), tier_tag, font=f, fill=INK)
    d.text((70, 168), "Crittenburg Complex  ·  Coryell County, Texas",
           font=font("LiberationSerif-Bold.ttf", 32), fill=INK)
    # caption band under the screenshot
    cy = top + bh + 26
    d.text((70, cy), caption, font=font("LiberationSerif-Bold.ttf", 27), fill=INK)
    footer(d, "Real VHAGAR console, live GOES + VIIRS + MODIS feed.")
    img.save(path)


def scene_t1(path):
    img = bg(); d = ImageDraw.Draw(img)
    header(d, "T1 · DETECTION", GREEN)
    tw = (W - 140 - 30) / 2
    tile(d, 70, 280, tw, 150, "SENSORS AGREEING", "4", "of 6", GREEN, GREEN)
    tile(d, 70 + tw + 30, 280, tw, 150, "DETECTIONS", "198", "px")
    tile(d, 70, 452, tw, 150, "PEAK FRP", "586", "MW")
    tile(d, 70 + tw + 30, 452, tw, 150, "FIRST SEEN", "17:52", "UTC")
    d.text((70, 636), "GOES-19  +  VIIRS-NOAA20 / NOAA21 / SNPP",
           font=font("LiberationSerif-Bold.ttf", 26), fill=INK)
    # sensor chips
    chips = ["GOES-19", "VIIRS-NOAA20", "VIIRS-NOAA21", "VIIRS-SNPP"]
    cx = 70; cy = 686
    for c in chips:
        cf = font("LiberationSerif-Bold.ttf", 21)
        cw = d.textlength(c, font=cf) + 40
        if cx + cw > W - 70:
            cx = 70; cy += 58
        rrect(d, [cx, cy, cx + cw, cy + 46], 23, fill=GREEN)
        d.text((cx + 20, cy + 10), c, font=cf, fill=(10, 20, 34))
        cx += cw + 16
    footer(d, "Detection survives only when independent sensors agree: false alarms rejected.")
    img.save(path)


def scene_t3(path):
    img = bg(); d = ImageDraw.Draw(img)
    header(d, "T3 · DANGER", AMBER)
    d.text((70, 268), "Three separate signals, never one blended score",
           font=font("LiberationSerif-Bold.ttf", 26), fill=BLUE)
    tw = W - 140
    tile(d, 70, 314, tw, 122, "FIRE WEATHER INDEX (FWI)", "16.2  High", "", AMBER, AMBER)
    tile(d, 70, 448, tw, 122, "IGNITION PROBABILITY", "8.30", "%")
    tile(d, 70, 582, tw, 122, "EXPECTED BURNED AREA | IGNITION", "46", "ha")
    rrect(d, [70, 724, W - 70, 792], 20, fill=CHIP, outline=RED, width=3)
    d.ellipse([94, 744, 112, 762], fill=RED)
    d.text((124, 740), "SPREAD RISK: HIGH (51)  ·  RH 14%  ·  wind 4.5 m/s",
           font=font("LiberationSerif-Bold.ttf", 26), fill=INK)
    footer(d, "FWI from live weather; ignition and burned-area are separate calibrated models.")
    img.save(path)


def scene_t2(path):
    img = bg(); d = ImageDraw.Draw(img)
    header(d, "T2 · BURNED AREA", MUT)
    d.text((70, 268), "The one tier a live fire cannot show yet",
           font=font("LiberationSerif-Bold.ttf", 26), fill=BLUE)
    # dashed placeholder scar
    box = [190, 340, W - 190, 620]
    for i in range(0, int((box[2] - box[0])), 26):
        d.line([(box[0] + i, box[1]), (box[0] + min(i + 14, box[2] - box[0]), box[1])], fill=LINE, width=3)
        d.line([(box[0] + i, box[3]), (box[0] + min(i + 14, box[2] - box[0]), box[3])], fill=LINE, width=3)
    for i in range(0, int((box[3] - box[1])), 26):
        d.line([(box[0], box[1] + i), (box[0], box[1] + min(i + 14, box[3] - box[1]))], fill=LINE, width=3)
        d.line([(box[2], box[1] + i), (box[2], box[1] + min(i + 14, box[3] - box[1]))], fill=LINE, width=3)
    pf = font("LiberationSerif-Bold.ttf", 64)
    t = "PENDING"
    d.text(((W - d.textlength(t, font=pf)) / 2, 430), t, font=pf, fill=MUT)
    d.text((190, 660), "Burn-scar segmentation reads a post-fire optical pass",
           font=font("LiberationSerif-Regular.ttf", 27), fill=INK)
    d.text((190, 700), "(Sentinel-2 or Landsat, RBR / Prithvi). Smoke blocks it now;",
           font=font("LiberationSerif-Regular.ttf", 27), fill=MUT)
    d.text((190, 740), "the scar is mapped in the days after the fire is contained.",
           font=font("LiberationSerif-Regular.ttf", 27), fill=MUT)
    footer(d, "Honest by design: three tiers run live, burned area follows the imagery.")
    img.save(path)


def spread_scene(mp4_path, seconds=6.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    from vhagar.models.spread import anisotropic_arrival, rate_of_spread

    for fnt in ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"):
        fm.fontManager.addfont(f"{FONTS}/{fnt}")
    plt.rcParams["font.family"] = "Liberation Serif"

    hh = ww = 200
    rng = np.random.default_rng(11)

    def sf(k=3):
        yy, xx = np.mgrid[0:hh, 0:ww] / max(hh, ww)
        f = np.zeros((hh, ww))
        for _ in range(k):
            a, b = rng.uniform(1.5, 4.5, size=2)
            ph = rng.uniform(0, 2 * np.pi, size=2)
            f += np.sin(a * 2 * np.pi * yy + ph[0]) * np.cos(b * 2 * np.pi * xx + ph[1])
        return (f - f.min()) / (np.ptp(f) + 1e-9)

    fuel = 0.35 + 0.6 * sf(); wind = 0.55 + 0.4 * sf(); slope = sf()
    barrier = sf() > 0.82
    head = rate_of_spread(fuel, wind, slope)
    head = np.where(barrier, head * 0.08, head)
    seeds = np.zeros((hh, ww), dtype=bool); seeds[150, 100] = True
    arrival = anisotropic_arrival(head, wind_speed=0.8, wind_dir=np.deg2rad(-60.0), seeds=seeds)
    finite = np.isfinite(arrival)
    tmax = float(np.quantile(arrival[finite], 0.90))

    fig = plt.figure(figsize=(10.8, 10.8), dpi=100)
    fig.patch.set_facecolor("#0a1524")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor("#0a1524")
    ax.imshow(head, cmap="Greys_r", alpha=0.10, origin="upper")
    burn = ax.imshow(np.ma.masked_all((hh, ww)), cmap="inferno", vmin=0, vmax=tmax,
                     origin="upper", alpha=0.92, interpolation="bilinear")
    front, = ax.plot([], [], ".", color="#fff3c4", ms=1.6)
    sy, sx = 150, 100
    ax.plot([sx], [sy], "o", mfc="#7fe0ff", mec="white", mew=1.5, ms=11, zorder=5)
    _b = finite & (arrival <= tmax); _cy, _cx = np.argwhere(_b).mean(axis=0)
    _vy, _vx = _cy - sy, _cx - sx; _n = float(np.hypot(_vx, _vy)) + 1e-9
    ax.annotate("", xy=(sx + 40 * _vx / _n, sy + 40 * _vy / _n), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="#7fe0ff", lw=2.6), zorder=6)
    fig.text(0.035, 0.945, "VHAGAR  ·  T4 SPREAD", color="#e9f0fa", fontsize=34, fontweight="bold")
    fig.text(0.035, 0.912, "Physics forecast seeded from the live detection", color="#3a86ff", fontsize=20)
    fig.text(0.965, 0.912, "Crittenburg Complex", color="#95a6be", fontsize=20, ha="right")
    clock = fig.text(0.035, 0.055, "", color="#e9f0fa", fontsize=25, fontweight="bold")
    fig.text(0.965, 0.030, "SYNTHETIC benchmark, real-data validation pending",
             color="#95a6be", fontsize=16, ha="right")
    fig.text(0.035, 0.028, "illustrative fuel landscape", color="#6f8099", fontsize=15, style="italic")

    total = int(FPS * seconds)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i in range(total):
            t = (1 - (1 - i / max(total - 1, 1)) ** 3) * tmax
            burned = finite & (arrival <= t)
            burn.set_data(np.ma.masked_where(~burned, arrival))
            shell = burned & (arrival > t - tmax * 0.03)
            ys, xs = np.where(shell); front.set_data(xs, ys)
            clock.set_text(f"t = +{72.0 * t / tmax:4.1f} h   (72 h forecast horizon)")
            fig.savefig(tdp / f"f{i:04d}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(tdp / "f%04d.png"),
                        "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
                        "-pix_fmt", "yuv420p", "-r", str(FPS), str(mp4_path)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def card_clip(png, mp4, seconds, zoom=True):
    if zoom:
        vf = ("scale=2160:2160,zoompan=z='min(zoom+0.0006,1.12)':d=%d:"
              "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1080:fps=%d,setsar=1"
              % (int(FPS * seconds), FPS))
    else:
        vf = "scale=1080:1080,setsar=1"
    subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", str(png), "-t", str(seconds),
                    "-r", str(FPS), "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]",
                    "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
                    "-pix_fmt", "yuv420p", str(mp4)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _assets_dir():
    """Where the input console screenshots live. Resolution order: --assets-dir,
    then fire_tracker/config.json {"assets_dir": ...}, then the script's own dir."""
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir")
    args, _ = ap.parse_known_args()
    if args.assets_dir:
        return Path(args.assets_dir)
    cfg = OUT / "config.json"
    if cfg.exists():
        try:
            return Path(json.loads(cfg.read_text()).get("assets_dir", OUT))
        except Exception:
            pass
    return OUT


def main():
    tmp = Path(tempfile.mkdtemp())
    OUTS = str(_assets_dir())
    scene_console(f"{OUTS}/f_wide.jpg", "T1 · DETECTION", GREEN,
                  "T1 detection: 4 sensors agree, GOES-19 + VIIRS x3, 586 MW peak", tmp / "a.png")
    scene_console(f"{OUTS}/f_zoom.jpg", "T3 · DANGER", AMBER,
                  "T3 danger: FWI 16.2 High, ignition 8.30%, spread risk High (51)", tmp / "b.png")
    scene_t2(tmp / "d.png")
    card_clip(tmp / "a.png", tmp / "A.mp4", 5.0)
    card_clip(tmp / "b.png", tmp / "B.mp4", 5.0)
    spread_scene(tmp / "C.mp4", seconds=6.0)
    card_clip(tmp / "d.png", tmp / "D.mp4", 4.5)
    # xfade chain A->B->C->D
    xf = 0.7
    fc = (
        "[0:v][1:v]xfade=transition=fade:duration=%f:offset=%f[ab];"
        "[ab][2:v]xfade=transition=fade:duration=%f:offset=%f[abc];"
        "[abc][3:v]xfade=transition=fade:duration=%f:offset=%f,format=yuv420p[v]"
        % (xf, 5 - xf, xf, 10 - 2 * xf, xf, 16 - 3 * xf)
    )
    out = OUT / "vhagar_pipeline_crittenburg.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", str(tmp / "A.mp4"), "-i", str(tmp / "B.mp4"),
                    "-i", str(tmp / "C.mp4"), "-i", str(tmp / "D.mp4"),
                    "-filter_complex", fc, "-map", "[v]",
                    "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", str(FPS), str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out)


if __name__ == "__main__":
    main()
