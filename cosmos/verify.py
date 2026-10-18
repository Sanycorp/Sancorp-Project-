"""
cosmos.verify
=============

Quality control on a finished file.

A two-hour render that fails silently is worse than one that crashes,
because nobody looks at all 3240 frames.  The failures that actually
happen in a pipeline like this one are specific and all of them are
detectable from the decoded pixels:

* **a dead worker** — one contiguous block of the film is missing, so
  the frame count is short or a run of frames is pure black;
* **a frozen phase** — a simulation stops advancing and consecutive
  frames become identical;
* **a numerical blow-up** — a shader produces infinities, which the
  encoder clamps to white, so a frame's mean luminance jumps by an order
  of magnitude;
* **a clipped grade** — the tone curve was pushed too far and a large
  fraction of pixels sit at 255, which is invisible on a small preview
  and obvious on a large screen;
* **a muxing mistake** — the audio is absent, the wrong length, or out
  of sync with the picture because the frame rate was misdeclared.

Each of those is one number per frame.  Decoding a 1080p60 file at a
stride of a few frames costs a couple of minutes, which is cheap
insurance against shipping a broken master.

There is no reporting apparatus here beyond what is printed: the point
is to find faults, not to produce a document.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np

F64 = np.float64


def ffmpeg_binary():
    for cand in (os.path.expanduser("~/bin/ffmpeg"), "ffmpeg"):
        try:
            subprocess.run([cand, "-version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
            return cand
        except (OSError, subprocess.CalledProcessError):
            continue
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:                       # pragma: no cover
        raise RuntimeError("no ffmpeg available") from exc


# ==========================================================================
# container inspection
# ==========================================================================
def probe(path):
    """Read stream properties.

    ``ffprobe`` is not installed in this environment, so the information
    is scraped from ``ffmpeg``'s own banner, which prints the same
    stream descriptions to stderr before complaining that no output file
    was given.
    """
    out = subprocess.run([ffmpeg_binary(), "-hide_banner", "-i", path],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True).stdout
    info = {"path": path, "bytes": os.path.getsize(path), "raw": out}

    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", out)
    if m:
        info["duration"] = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                            + float(m.group(3)))
    # The video stream line cannot be split on commas: the pixel-format
    # field carries its own parenthesised list, as in
    # ``yuv420p(tv, bt709, progressive)``, and a naive split lands in the
    # middle of it.  Each field is therefore matched on its own shape.
    vline = ""
    for line in out.splitlines():
        if "Video:" in line:
            vline = line.split("Video:", 1)[1]
            break
    if vline:
        m = re.match(r"\s*([A-Za-z0-9_]+)", vline)
        if m:
            info["vcodec"] = m.group(1)
        m = re.search(r"\b(yuv[0-9a-z]+|rgb[0-9a-z]*|gbrp[0-9a-z]*|"
                      r"gray[0-9a-z]*|nv1[25])\b", vline)
        if m:
            info["pix_fmt"] = m.group(1)
        m = re.search(r"\b(\d{2,5})x(\d{2,5})\b", vline)
        if m:
            info["width"] = int(m.group(1))
            info["height"] = int(m.group(2))
        m = re.search(r"\b(\d+)\s*kb/s", vline)
        if m:
            info["vbitrate_kbps"] = int(m.group(1))
        m = re.search(r"([\d.]+)\s*fps", vline)
        if m:
            info["fps"] = float(m.group(1))
        prof = re.search(r"\(([A-Za-z0-9 ]+)\)", vline)
        if prof:
            info["profile"] = prof.group(1).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", out)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Audio:\s*([^,]+),\s*(\d+)\s*Hz,\s*([^,]+),[^,]*,"
                  r"\s*(\d+)\s*kb/s", out)
    if m:
        info["acodec"] = m.group(1).strip()
        info["asample_rate"] = int(m.group(2))
        info["achannels"] = m.group(3).strip()
        info["abitrate_kbps"] = int(m.group(4))
    else:
        m = re.search(r"Audio:\s*([^,]+),\s*(\d+)\s*Hz,\s*([^,]+)", out)
        if m:
            info["acodec"] = m.group(1).strip()
            info["asample_rate"] = int(m.group(2))
            info["achannels"] = m.group(3).strip()
    return info


# ==========================================================================
# frame statistics
# ==========================================================================
def frame_stats(path, width, height, stride=6, scale=8):
    """Decode every ``stride``-th frame at reduced size and measure it.

    Downscaling in ffmpeg rather than numpy means the pipe carries a
    sixty-fourth of the data, which turns a memory-bound decode into a
    CPU-bound one and roughly quadruples the throughput.
    """
    w = max(width // scale, 16)
    h = max(height // scale, 16)
    vf = "select=not(mod(n\\,%d)),scale=%d:%d" % (stride, w, h)
    cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error",
           "-i", path, "-vf", vf, "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    size = w * h * 3
    rows = []
    prev = None
    idx = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        a = np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(F64)
        lum = 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]
        diff = 0.0 if prev is None else float(np.abs(lum - prev).mean())
        rows.append((idx * stride, float(lum.mean()), float(lum.max()),
                     float((a >= 254).mean()), diff,
                     float(a[:, :, 0].mean()), float(a[:, :, 1].mean()),
                     float(a[:, :, 2].mean())))
        prev = lum
        idx += 1
    proc.stdout.close()
    proc.wait()
    return np.array(rows, F64)


def audio_stats(path):
    """Decode the audio track and measure peak, RMS and silence."""
    cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error",
           "-i", path, "-vn", "-f", "f32le", "-ac", "2", "-ar", "48000",
           "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    raw = proc.stdout
    if len(raw) < 1024:
        return None
    x = np.frombuffer(raw, np.float32).astype(F64)
    x = x[:(x.shape[0] // 2) * 2].reshape(-1, 2)
    n = x.shape[0]
    win = 48000 // 10
    blocks = n // win
    if blocks:
        b = x[:blocks * win].reshape(blocks, win, 2)
        block_rms = np.sqrt((b ** 2).mean(axis=(1, 2)))
    else:
        block_rms = np.array([0.0])
    return {
        "samples": n,
        "seconds": n / 48000.0,
        "peak": float(np.abs(x).max()),
        "rms": float(np.sqrt((x ** 2).mean())),
        "silent_blocks": int((block_rms < 1e-4).sum()),
        "blocks": int(block_rms.shape[0]),
        "dc": float(x.mean()),
        "correlation": float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
        if n > 2 else 1.0,
    }


# ==========================================================================
# checks
# ==========================================================================
def run_checks(path, expect_frames=None, expect_fps=60.0,
               expect_size=(1920, 1080), expect_seconds=None,
               stride=6, cuts=()):
    """Inspect a file and print a pass/fail line per check."""
    info = probe(path)
    print("berkas   %s" % path)
    print("ukuran   %.1f MB" % (info["bytes"] / 1e6))
    print("video    %s %s %dx%d @ %.3f fps  %s kb/s"
          % (info.get("vcodec", "?"), info.get("pix_fmt", "?"),
             info.get("width", 0), info.get("height", 0),
             info.get("fps", 0.0), info.get("vbitrate_kbps", "?")))
    if "acodec" in info:
        print("audio    %s %d Hz %s  %s kb/s"
              % (info["acodec"], info["asample_rate"], info["achannels"],
                 info.get("abitrate_kbps", "?")))
    else:
        print("audio    TIDAK ADA")
    print("durasi   %.3f s" % info.get("duration", 0.0))

    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print("  [%s] %-38s %s" % ("OK" if ok else "GAGAL", name, detail))

    check("resolusi", (info.get("width"), info.get("height")) == expect_size,
          "%sx%s" % (info.get("width"), info.get("height")))
    check("laju bingkai", abs(info.get("fps", 0.0) - expect_fps) < 0.05,
          "%.3f" % info.get("fps", 0.0))
    if expect_seconds is not None:
        check("durasi", abs(info.get("duration", 0.0) - expect_seconds)
              < 0.08, "%.3f vs %.3f" % (info.get("duration", 0.0),
                                        expect_seconds))
    check("ruang warna yuv420p", info.get("pix_fmt", "").startswith(
        "yuv420p"), info.get("pix_fmt", ""))

    print("menganalisis bingkai (setiap %d) ..." % stride, flush=True)
    st = frame_stats(path, info.get("width", 1920),
                     info.get("height", 1080), stride=stride)
    n = st.shape[0]
    print("  %d bingkai diukur (dari ~%d)" % (n, n * stride))
    if expect_frames is not None:
        check("jumlah bingkai", abs(n * stride - expect_frames)
              <= stride * 2, "%d ~ %d" % (n * stride, expect_frames))

    mean = st[:, 1]
    maxl = st[:, 2]
    clip = st[:, 3]
    diff = st[:, 4]

    # black frames anywhere except the head and tail fades
    head = max(int(0.6 * expect_fps / stride), 1)
    tail = max(int(0.8 * expect_fps / stride), 1)
    body = mean[head:n - tail] if n > head + tail else mean
    check("tidak ada bingkai hitam di tengah", float(body.min()) > 0.35,
          "luminans min %.3f" % float(body.min()))

    # frozen frames: a long run of near-zero inter-frame difference
    body_d = diff[head:n - tail] if n > head + tail else diff
    frozen = int((body_d < 0.015).sum())
    check("tidak ada bingkai beku", frozen <= 2,
          "%d sampel di bawah ambang" % frozen)

    # blow-ups: a single frame far brighter than its neighbours
    med = float(np.median(mean))
    spikes = int((mean > med * 6.0 + 20.0).sum())
    check("tidak ada ledakan luminans", spikes == 0, "%d lonjakan" % spikes)

    check("tanpa kliping berlebihan", float(clip.max()) < 0.06,
          "fraksi piksel >=254 maks %.4f" % float(clip.max()))
    check("sorotan hadir", float(maxl.max()) > 200.0,
          "luminans maks %.1f" % float(maxl.max()))

    # the cuts between phases should be visible as a spike in the
    # inter-frame difference, which proves every phase actually rendered
    for name, t in cuts:
        i = int(t * expect_fps / stride)
        lo, hi = max(i - 3, 0), min(i + 4, n)
        local = float(diff[lo:hi].max()) if hi > lo else 0.0
        check("potongan %s" % name, local > 2.0, "delta %.2f" % local)

    au = audio_stats(path)
    if au is None:
        check("trek audio", False, "tidak ada")
    else:
        print("  audio %.3f s  puncak %.4f  RMS %.4f  korelasi L/R %.3f"
              % (au["seconds"], au["peak"], au["rms"], au["correlation"]))
        check("audio hadir", au["rms"] > 0.01, "RMS %.4f" % au["rms"])
        # A lossy codec reconstructs a waveform that can overshoot the
        # samples it was given — inter-sample peaks are not an encoding
        # fault and a decoded peak slightly above unity is normal.  What
        # matters is that it is not grossly over.
        check("audio tidak klip", au["peak"] <= 1.10,
              "puncak %.4f" % au["peak"])
        if expect_seconds is not None:
            check("panjang audio", abs(au["seconds"] - expect_seconds)
                  < 0.12, "%.3f" % au["seconds"])
        check("audio tidak hening", au["silent_blocks"]
              < au["blocks"] * 0.08,
              "%d/%d blok hening" % (au["silent_blocks"], au["blocks"]))
        check("stereo nyata", au["correlation"] < 0.995,
              "korelasi %.4f" % au["correlation"])

    bad = [r for r in results if not r[1]]
    print("\n%d/%d pemeriksaan lulus" % (len(results) - len(bad),
                                         len(results)))
    return st, info, au, bad


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("penggunaan: python3 -m cosmos.verify <berkas.mp4> [stride]")
        return 2
    path = argv[0]
    stride = int(argv[1]) if len(argv) > 1 else 6
    cuts = (("I/II", 11.0), ("II/III", 22.0), ("III/IV", 33.0),
            ("IV/V", 42.0))
    st, info, au, bad = run_checks(path, expect_frames=3240,
                                   expect_seconds=54.0, stride=stride,
                                   cuts=cuts)
    summary = {
        "path": path,
        "bytes": info["bytes"],
        "duration": info.get("duration"),
        "fps": info.get("fps"),
        "size": [info.get("width"), info.get("height")],
        "failures": [b[0] for b in bad],
    }
    print(json.dumps(summary))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["probe", "frame_stats", "audio_stats", "run_checks",
           "ffmpeg_binary", "main"]
