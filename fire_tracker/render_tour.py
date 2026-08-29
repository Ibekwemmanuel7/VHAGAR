#!/usr/bin/env python3
"""Guided console tour MP4: US West -> Continental US -> Texas -> Cluster 8 (Ross
Fire) -> detail. Per-scene caption, gentle push-in, crossfades. 1280x720 H.264."""
import subprocess
from pathlib import Path

_FT = Path(__file__).resolve().parent


def _assets_dir():
    """Input frame directory: --assets-dir, then config.json, then script dir."""
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir")
    args, _ = ap.parse_known_args()
    if args.assets_dir:
        return Path(args.assets_dir)
    cfg = _FT / "config.json"
    if cfg.exists():
        try:
            return Path(json.loads(cfg.read_text()).get("assets_dir", _FT))
        except Exception:
            pass
    return _FT


O = _assets_dir()
OUT = _FT / "vhagar_ross_tour.mp4"
FB = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
FPS = 25
CR = "crop=1397:786:115:0"

SCENES = [
    ("tourA.jpg", 4.0, 0.0006, "US West  ·  179 live fire events"),
    ("tourB.jpg", 4.0, 0.0006, "Continental US  ·  363 live fire events"),
    ("tourC.jpg", 4.5, 0.0007, "Texas  ·  Ross and Crittenburg complexes"),
    ("tourD.jpg", 4.5, 0.0007, "Cluster 8  ·  Ross Fire, Palo Pinto County, TX"),
    ("tourE.jpg", 6.0, 0.0006,
     "28,533 ha  ·  2,666 MW peak  ·  wind 4.2 m/s @ 165°  ·  FWI 16.2 High"),
]


def esc(t):
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def seg(img, dur, zr, cap, out):
    d = int(FPS * (dur + 1))
    vf = (f"[0:v]{CR},scale=2794:1572,"
          f"zoompan=z='min(zoom+{zr},1.14)':d={d}:x='iw/2-(iw/zoom/2)':"
          f"y='ih/2-(ih/zoom/2)':s=1280x720:fps={FPS},setsar=1,"
          f"drawbox=x=0:y=650:w=1280:h=70:color=black@0.58:t=fill,"
          f"drawtext=fontfile={FB}:text='{esc(cap)}':fontcolor=white:fontsize=26:x=40:y=670[v]")
    subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", str(O / img), "-t", str(dur),
                    "-r", str(FPS), "-filter_complex", vf, "-map", "[v]",
                    "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
                    "-pix_fmt", "yuv420p", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    segs = []
    for i, (img, dur, zr, cap) in enumerate(SCENES):
        p = O / f"tseg{i}.mp4"
        seg(img, dur, zr, cap, p)
        segs.append((p, dur))
    # xfade chain
    x = 0.6
    inputs = []
    for p, _ in segs:
        inputs += ["-i", str(p)]
    fc = ""
    prev = "0:v"
    acc = segs[0][1]
    for i in range(1, len(segs)):
        off = acc - x
        lab = "v" if i == len(segs) - 1 else f"x{i}"
        fc += f"[{prev}][{i}:v]xfade=transition=fade:duration={x}:offset={off:.3f}[{lab}];"
        acc = acc + segs[i][1] - x
        prev = lab
    fc = fc.rstrip(";")
    if not fc.endswith("[v]"):
        fc += ",format=yuv420p[v]"  # single-scene guard (unused here)
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[v]",
                    "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", str(FPS), str(OUT)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
