#!/usr/bin/env python3
"""Render a branded VHAGAR "live fire tracker" MP4 card from a fire reading.

Pure PIL + ffmpeg, no browser. Two modes:
  kickoff  -> the "one clock, one fire" concept animation (thread opener)
  update   -> an animated stat card for the latest reading in state.json

Output is 1080x1080 H.264 High/4.0 yuv420p +faststart (LinkedIn-safe).

Usage:
  render_fire_card.py kickoff --out kickoff.mp4
  render_fire_card.py update --state state.json --n 1 --out update1.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W = H = 1080
FPS = 30
FONTS = "/usr/share/fonts/truetype/liberation"  # Liberation Serif = Times New Roman metric-compatible

# palette
BG_TOP = (9, 18, 33)
BG_BOT = (13, 27, 48)
INK = (233, 240, 250)
MUT = (150, 166, 190)
BLUE = (56, 132, 255)
BLUE_D = (30, 74, 150)
RED = (255, 86, 74)
AMBER = (255, 176, 60)
GREEN = (74, 200, 138)
CHIP = (24, 40, 66)
LINE = (34, 54, 84)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(f"{FONTS}/{name}", size)


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def gradient_bg() -> Image.Image:
    img = Image.new("RGB", (W, H), BG_TOP)
    px = img.load()
    for y in range(H):
        f = y / H
        r = int(BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * f)
        g = int(BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * f)
        b = int(BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * f)
        for x in range(W):
            px[x, y] = (r, g, b)
    return img


def draw_header(d: ImageDraw.ImageDraw, pulse: float) -> None:
    d.text((70, 66), "VHAGAR", font=font("LiberationSerif-Bold.ttf", 46), fill=INK)
    d.text((70, 122), "LIVE FIRE TRACKER", font=font("LiberationSerif-Bold.ttf", 22), fill=BLUE)
    # pulsing live dot
    r = 9 + 5 * pulse
    cx, cy = W - 150, 92
    d.ellipse([cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6],
              fill=(RED[0], RED[1], RED[2]))
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=INK)
    d.text((W - 128, 80), "LIVE", font=font("LiberationSerif-Bold.ttf", 22), fill=INK)


def rrect(d, box, radius, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def center_text(d, cx, y, text, fnt, fill):
    w = d.textlength(text, font=fnt)
    d.text((cx - w / 2, y), text, font=fnt, fill=fill)


def encode(frames_dir: Path, out: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(frames_dir / "f%04d.png"),
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-r", str(FPS), str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------- update
def render_update(reading: dict, n: int, out: Path, anchor: dict, dur: float = 13.0) -> None:
    import tempfile

    total = int(dur * FPS)
    hold = int(1.6 * FPS)          # count-up window
    name = anchor.get("name", "Tracked fire")
    loc = f"{reading.get('last_seen','')[11:16]}Z  |  update {n}"
    foot = float(reading.get("footprint_ha") or 0)
    frp = reading.get("max_frp_mw")
    dets = int(reading.get("n_detections") or 0)
    sensors = [s.strip() for s in str(reading.get("sensors", "")).split(",") if s.strip()]
    risk = str(reading.get("risk_class", "n/a"))
    rs = reading.get("risk_score")
    wind = reading.get("wind_ms")
    rh = reading.get("rh_pct")
    risk_col = {"Low": GREEN, "Moderate": AMBER, "High": RED,
                "Very High": RED, "Extreme": RED}.get(risk, MUT)

    f_big = font("LiberationSerif-Bold.ttf", 128)
    f_unit = font("LiberationSerif-Bold.ttf", 40)
    f_lbl = font("LiberationSerif-Bold.ttf", 24)
    f_name = font("LiberationSerif-Bold.ttf", 40)
    f_sub = font("LiberationSerif-Regular.ttf", 26)
    f_stat = font("LiberationSerif-Bold.ttf", 54)
    f_chip = font("LiberationSerif-Bold.ttf", 21)
    f_tag = font("LiberationSerif-Italic.ttf", 26)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        base = gradient_bg()
        for i in range(total):
            img = base.copy()
            d = ImageDraw.Draw(img)
            t = i / max(total - 1, 1)
            pulse = 0.5 + 0.5 * math.sin(i / FPS * 3.2)
            draw_header(d, pulse)

            d.line([(70, 168), (W - 70, 168)], fill=LINE, width=2)

            # fire name + meta (shrink to fit width)
            fn = f_name
            for sz in range(40, 21, -2):
                fn = font("LiberationSerif-Bold.ttf", sz)
                if d.textlength(name, font=fn) <= W - 140:
                    break
            d.text((70, 196), name, font=fn, fill=INK)
            d.text((70, 250), f"~{anchor.get('lat')}N, {abs(anchor.get('lon',0))}W   |   {loc}",
                   font=f_sub, fill=MUT)

            # footprint big counter
            cu = ease_out_cubic(min(i / hold, 1.0))
            val = int(foot * cu)
            d.text((70, 322), "FOOTPRINT", font=f_lbl, fill=BLUE)
            d.text((70, 352), f"{val:,}", font=f_big, fill=INK)
            vw = d.textlength(f"{val:,}", font=f_big)
            d.text((70 + vw + 20, 430), "ha", font=f_unit, fill=MUT)

            # growing bar under footprint
            bar_w = int((W - 140) * cu)
            rrect(d, [70, 512, W - 70, 524], 6, fill=LINE)
            rrect(d, [70, 512, 70 + bar_w, 524], 6, fill=BLUE)

            # two stat tiles
            frp_txt = "n/a" if frp is None else f"{int(round(float(frp))):,}"
            tiles = [("PEAK FRP", frp_txt, "MW"),
                     ("DETECTIONS", f"{int(dets * cu):,}", "px")]
            tx = 70
            tw = (W - 140 - 30) / 2
            for lbl, v, u in tiles:
                rrect(d, [tx, 560, tx + tw, 690], 18, fill=CHIP, outline=LINE, width=2)
                d.text((tx + 26, 582), lbl, font=f_lbl, fill=MUT)
                d.text((tx + 26, 612), v, font=f_stat, fill=INK)
                vw2 = d.textlength(v, font=f_stat)
                d.text((tx + 26 + vw2 + 12, 632), u, font=f_lbl, fill=MUT)
                tx += tw + 30

            # sensor chips light up sequentially
            d.text((70, 720), "SENSORS CORROBORATING", font=f_lbl, fill=BLUE)
            chip_names = sensors if sensors else ["GOES-19"]
            per = 0.10
            cx = 70
            cy = 758
            maxx = W - 70
            for si, sname in enumerate(chip_names):
                on = t > 0.15 + si * per
                cw = d.textlength(sname, font=f_chip) + 40
                if cx + cw > maxx:
                    cx = 70
                    cy += 58
                col = GREEN if on else CHIP
                tcol = (10, 20, 34) if on else MUT
                rrect(d, [cx, cy, cx + cw, cy + 46], 23,
                      fill=col if on else CHIP, outline=LINE if not on else col, width=2)
                d.text((cx + 20, cy + 10), sname, font=f_chip, fill=tcol)
                cx += cw + 16

            # risk badge (bottom)
            by = 928
            rs_txt = "" if rs is None else f" ({rs})"
            rlabel = f"SPREAD RISK: {risk.upper()}{rs_txt}"
            rrect(d, [70, by, 70 + d.textlength(rlabel, font=f_lbl) + 52, by + 50], 25,
                  fill=CHIP, outline=risk_col, width=3)
            d.ellipse([94, by + 17, 110, by + 33], fill=risk_col)
            d.text((122, by + 12), rlabel, font=f_lbl, fill=INK)
            wx = []
            if wind is not None:
                wx.append(f"wind {wind} m/s")
            if rh is not None:
                wx.append(f"RH {rh}%")
            if wx:
                d.text((70, by + 66), "  |  ".join(wx), font=f_sub, fill=MUT)

            center_text(d, W / 2, H - 52, "one clock, one fire  |  fused GOES + VIIRS + MODIS",
                        f_tag, MUT)
            img.save(tdp / f"f{i:04d}.png")
        encode(tdp, out)


# -------------------------------------------------------------------- kickoff
def render_kickoff(out: Path) -> None:
    import tempfile

    dur = 12.0
    total = int(dur * FPS)
    f_h1 = font("LiberationSerif-Bold.ttf", 64)
    f_h2 = font("LiberationSerif-Bold.ttf", 40)
    f_sub = font("LiberationSerif-Regular.ttf", 30)
    f_tag = font("LiberationSerif-Italic.ttf", 30)
    f_small = font("LiberationSerif-Bold.ttf", 22)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        base = gradient_bg()
        n_marks = 12
        for i in range(total):
            img = base.copy()
            d = ImageDraw.Draw(img)
            t = i / max(total - 1, 1)
            pulse = 0.5 + 0.5 * math.sin(i / FPS * 3.2)
            draw_header(d, pulse)

            d.text((70, 250), "One image is a", font=f_h2, fill=MUT)
            d.text((70, 300), "snapshot.", font=f_h1, fill=INK)
            d.text((70, 400), "A clock is", font=f_h2, fill=MUT)
            d.text((70, 450), "intelligence.", font=f_h1, fill=BLUE)

            # timeline that fills with pulsing markers
            y = 640
            x0, x1 = 90, W - 90
            prog = ease_out_cubic(min(t / 0.8, 1.0))
            d.line([(x0, y), (x0 + (x1 - x0) * prog, y)], fill=BLUE, width=6)
            d.line([(x0 + (x1 - x0) * prog, y), (x1, y)], fill=LINE, width=6)
            for m in range(n_marks):
                mx = x0 + (x1 - x0) * m / (n_marks - 1)
                lit = (mx - x0) <= (x1 - x0) * prog
                r = 12 if lit else 7
                if lit and m == int((n_marks - 1) * prog):
                    r = 12 + 6 * pulse
                d.ellipse([mx - r, y - r, mx + r, y + r],
                          fill=BLUE if lit else CHIP,
                          outline=BLUE if lit else LINE, width=2)
            d.text((x0, y + 40), "every 30 minutes, same fire", font=f_small, fill=MUT)

            d.text((70, 790), "Cadence over resolution.", font=f_h2, fill=INK)
            d.text((70, 848), "Follow one wildfire become a time series.",
                   font=f_sub, fill=MUT)

            center_text(d, W / 2, H - 60,
                        "VHAGAR  |  fused GOES + VIIRS + MODIS, every number labelled",
                        f_tag, MUT)
            img.save(tdp / f"f{i:04d}.png")
        encode(tdp, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kickoff", "update"])
    ap.add_argument("--state")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--dur", type=float, default=13.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if a.mode == "kickoff":
        render_kickoff(out)
    else:
        st = json.loads(Path(a.state).read_text())
        reading = st["readings"][-1]
        render_update(reading, a.n or len(st["readings"]), out, st["anchor"], dur=a.dur)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
