"""
Frame renderer + encoder.

  python3 render.py --probe 0.5 2.1 9.0      # write single PNG stills
  python3 render.py --sheet                  # contact sheet of the whole film
  python3 render.py --video out/film.mp4     # full parallel render + encode

Frames are rendered at 1920x1080 and encoded to H.264 at a true 60 fps.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

import engine as E
import scenes as S
import widgets as Wg
from engine import H, W, FPS, clamp, mix, span, in_quad, out_cubic

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
def _scene_at(t):
    for i, (a, b, fn, hud_on) in enumerate(S.TIMELINE):
        if t < b or i == len(S.TIMELINE) - 1:
            return i, a, b, fn, hud_on
    return 0, 0.0, 1.0, S.TIMELINE[0][2], True


def zoom_shake(canvas, z, sx, sy):
    if abs(z - 1.0) < 1e-4 and abs(sx) < 0.2 and abs(sy) < 0.2:
        return canvas
    z = max(z, 1.0 + (abs(sx) + abs(sy)) / 420.0)
    img = Image.fromarray(E.to_u8(canvas))
    nw, nh = int(W * z), int(H * z)
    img = img.resize((nw, nh), Image.BICUBIC)
    ox = int((nw - W) / 2 + sx)
    oy = int((nh - H) / 2 + sy)
    ox = max(0, min(nw - W, ox))
    oy = max(0, min(nh - H, oy))
    return np.asarray(img.crop((ox, oy, ox + W, oy + H)), np.float32) / 255.0


def make_frame(fi: int) -> np.ndarray:
    t = fi / FPS
    rng = np.random.default_rng((fi * 7919 + 104729) & 0xFFFFFFFF)

    idx, a, b, fn, hud_on = _scene_at(t)
    lt = t - a

    canvas = np.zeros((H, W, 3), np.float32)
    fx = {"chroma": 0.0, "flash": 0.0, "glitch": 0.0, "zoom": 1.0, "shake": 0.0}
    fn(canvas, lt, t, fx, rng)

    # automatic cut impact at every scene boundary
    if idx > 0 and lt < 0.14:
        k = 1.0 - clamp(lt / 0.14)
        fx["flash"] = max(fx["flash"], k * 0.18)
        fx["chroma"] = max(fx["chroma"], k * 10.0)
        fx["glitch"] = max(fx["glitch"], k * 0.40)

    S.glitch(canvas, fx["glitch"], rng)

    sx, sy = E.shake(idx + 1, fx["shake"], t)
    canvas = zoom_shake(canvas, fx["zoom"], sx, sy)

    canvas = E.chroma(canvas, fx["chroma"])
    canvas = E.bloom(canvas, thresh=0.60, strength=0.50, radius=9.0)
    canvas = E.lens(canvas, vig=0.60, scan=0.035)

    if hud_on:
        ha = span(t, a + 0.25, a + 0.9, out_cubic) * (1.0 - span(t, b - 0.45, b, in_quad))
        if ha > 0.01:
            E.over(canvas, Wg.hud(t, S.TOTAL, S.ACC, ha))

    if fx["flash"] > 0.002:
        canvas += fx["flash"]

    canvas = E.grain(canvas, 0.030, rng)
    canvas = E.tonemap(canvas, 0.95)
    return E.to_u8(canvas)


# --------------------------------------------------------------------------
def _encode_segment(args):
    i, f0, f1, tmpdir, crf = args
    out = os.path.join(tmpdir, "seg_%03d.mp4" % i)
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-x264-params", "keyint=120:min-keyint=120:scenecut=0:ref=4:bframes=3",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        out,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        p.stdin.write(make_frame(fi).tobytes())
    p.stdin.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"segment {i} failed rc={rc}")
    return out


def render_video(path, crf=16, workers=2, segments=8, audio=None, total=None):
    total_t = total if total else S.TOTAL
    nframes = int(round(total_t * FPS))
    tmpdir = os.path.join(ROOT, ".rendertmp")
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir, exist_ok=True)

    bounds = [int(round(k * nframes / segments)) for k in range(segments + 1)]
    jobs = [(i, bounds[i], bounds[i + 1], tmpdir, crf) for i in range(segments)]

    t0 = time.time()
    print(f"rendering {nframes} frames @ {W}x{H} {FPS}fps "
          f"({segments} segments, {workers} workers)", flush=True)
    done = []
    with Pool(workers) as pool:
        for k, out in enumerate(pool.imap_unordered(_encode_segment, jobs)):
            done.append(out)
            el = time.time() - t0
            print(f"  segment {k+1}/{segments} done  [{el:6.1f}s elapsed]", flush=True)

    listfile = os.path.join(tmpdir, "list.txt")
    with open(listfile, "w") as f:
        for i in range(segments):
            f.write("file '%s'\n" % os.path.join(tmpdir, "seg_%03d.mp4" % i))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", listfile]
    if audio and os.path.exists(audio):
        cmd += ["-i", audio, "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
                "-shortest"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", path]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB) "
          f"in {time.time()-t0:.1f}s", flush=True)


# --------------------------------------------------------------------------
def probe(times, outdir):
    os.makedirs(outdir, exist_ok=True)
    for tt in times:
        fi = int(round(float(tt) * FPS))
        t0 = time.time()
        img = Image.fromarray(make_frame(fi))
        p = os.path.join(outdir, "t%07.2f.png" % float(tt))
        img.save(p)
        print(f"{p}   {time.time()-t0:.2f}s")


def sheet(outpath, cols=5, rows=5, total=None):
    total_t = total if total else S.TOTAL
    n = cols * rows
    tw, th = W // 5, H // 5
    sheet_img = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    for i in range(n):
        tt = (i + 0.5) * total_t / n
        fi = int(round(tt * FPS))
        im = Image.fromarray(make_frame(fi)).resize((tw, th), Image.LANCZOS)
        sheet_img.paste(im, ((i % cols) * tw, (i // cols) * th))
        print("  sheet frame", i + 1, "/", n, "t=%.2f" % tt, flush=True)
    sheet_img.save(outpath)
    print("wrote", outpath)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs="*", type=float)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "preview"))
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--sheetpath", default=os.path.join(ROOT, "preview", "sheet.jpg"))
    ap.add_argument("--video")
    ap.add_argument("--audio")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--segments", type=int, default=8)
    args = ap.parse_args()

    if args.probe:
        probe(args.probe, args.outdir)
    elif args.sheet:
        os.makedirs(os.path.dirname(args.sheetpath), exist_ok=True)
        sheet(args.sheetpath)
    elif args.video:
        render_video(args.video, args.crf, args.workers, args.segments, args.audio)
    else:
        ap.print_help()
