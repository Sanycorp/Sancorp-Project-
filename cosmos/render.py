"""
cosmos.render
=============

The render driver: turns the timeline into an H.264 file.

Strategy
--------
The machine has two cores and three gigabytes of memory, and the most
expensive phase costs several seconds a frame, so the only thing that
matters is keeping both cores saturated with as little duplicated setup as
possible.

The film is cut into contiguous **segments**, one worker per segment, each
worker encoding its own range straight to an elementary stream.  The
segments are then joined with the concat demuxer and stream-copied, so no
frame is ever encoded twice and the final mux costs seconds rather than
minutes.

Each worker builds the phases it needs *once* and loads only the volumes
those phases reference, which matters because the baked library is eighty
megabytes and loading all of it in both workers at once would put real
pressure on memory.

Frames are handed to ffmpeg as raw RGB24 on a pipe.  Writing PNGs to disk
and reading them back would triple the I/O for no benefit.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from multiprocessing import Process, Queue

import numpy as np

F32 = np.float32

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def ffmpeg_binary():
    """Locate a usable ffmpeg.

    The sandbox has no system ffmpeg; imageio-ffmpeg ships a static build,
    and a symlink to it is kept in ~/bin.  Both are checked so the module
    works either way.
    """
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
    except Exception as exc:          # pragma: no cover
        raise RuntimeError("no ffmpeg available") from exc


# ==========================================================================
# per-phase render quality
# ==========================================================================
#: March resolution and step count per phase, as a fraction of the output.
#:
#: These are not uniform because the phases are not alike.  Phase I is
#: entirely soft volumetric gas, so a half-resolution march upsampled is
#: indistinguishable from a full one.  Phase V has a hard-edged shadow and
#: a thin photon ring, so it is marched much closer to full resolution even
#: though its rays are by far the most expensive.
QUALITY = {
    "I · GALAKSI":     dict(march=0.3125, steps=58),
    "II · TATA SURYA": dict(march=1.0, steps=1),
    "III · MATAHARI":  dict(march=0.3334, steps=48),
    "IV · TRANSISI":   dict(march=0.3334, steps=38),
    "V · LUBANG HITAM": dict(march=0.3959, steps=1),
}


def context_for(phase, width, height, lib):
    from .phases.base import RenderContext
    q = QUALITY.get(phase.name, dict(march=0.5, steps=48))
    mw = max(64, int(round(width * q["march"] / 2) * 2))
    mh = max(36, int(round(height * q["march"] / 2) * 2))
    return RenderContext(width=width, height=height,
                         march_width=mw, march_height=mh,
                         lib=lib, steps=q["steps"])


# ==========================================================================
# worker
# ==========================================================================
def render_range(width, height, fps, first, last, out_path, crf, preset,
                 progress: Queue, worker_id: int):
    """Render frames ``[first, last)`` and encode them to ``out_path``."""
    from .core import vcache
    from .post import PostChain
    from .timeline import Timeline, Overlay, SLATES, build_phases, volumes_for

    phases = build_phases()
    lib = vcache.QuantLibrary()
    lib.load_all(volumes_for(phases))

    tl = Timeline(phases)
    contexts = {}
    for ph in phases:
        ctx = context_for(ph, width, height, lib)
        ph.setup(ctx)
        contexts[ph.name] = ctx

    post = PostChain(width, height)
    overlay = Overlay(width, height)

    cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (width, height), "-r", str(fps),
           "-i", "pipe:0",
           "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2",
           "-x264-params",
           "keyint=%d:min-keyint=%d:scenecut=0:ref=4:bframes=3:"
           "aq-mode=3:aq-strength=0.9:psy-rd=1.0,0.15:deblock=-1,-1"
           % (fps * 2, fps * 2),
           "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709",
           "-f", "mpegts", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    t_start = time.perf_counter()
    try:
        for idx in range(first, last):
            film_t = idx / float(fps)
            seg, local, fade = tl.at(film_t)
            ph = seg.phase
            ctx = contexts[ph.name]
            ctx.frame = idx
            ctx.rng = np.random.default_rng(idx * 7919 + 101)

            img = ph.render(ctx, local)
            exposure_glare = 0.0
            rng = np.random.default_rng(idx * 2654435761 % (2 ** 31))
            u8 = post(img, rng, fade=fade, extra_glare=exposure_glare)
            u8 = overlay.draw(u8, SLATES[ph.name], local, seg.duration,
                              film_t, tl.total, idx)
            proc.stdin.write(u8.tobytes())

            if (idx - first) % 10 == 0:
                done = idx - first + 1
                el = time.perf_counter() - t_start
                progress.put((worker_id, done, last - first,
                              el / max(done, 1)))
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    progress.put((worker_id, last - first, last - first,
                  (time.perf_counter() - t_start) / max(last - first, 1)))


# ==========================================================================
# joining the segments
# ==========================================================================
#: PID the mpegts muxer assigns to the first video stream
VIDEO_PID = 0x0100


def join_transport_streams(parts, out_path):
    """Concatenate segments by unwrapping the transport stream in Python.

    The obvious way to do this is ffmpeg's concat demuxer, and it does not
    work here: the static ffmpeg build available in this environment
    segmentation-faults the instant it is asked to *read* an MPEG transport
    stream, although it writes them perfectly well.  That failure arrives
    after the entire film has been rendered, which is the worst possible
    moment, so the join no longer depends on it.

    Transport stream framing is simple enough to undo directly.  The file
    is a sequence of 188-byte packets, each beginning with the sync byte
    0x47 and carrying a 13-bit PID.  Packets on the video PID hold the
    H.264 elementary stream, wrapped in PES packets whose header is nine
    bytes plus a declared extension.  Stripping the transport and PES
    layers leaves the raw Annex-B bitstream, and because each segment was
    encoded independently it begins with its own parameter sets and an IDR
    picture — so the segments can simply be written one after another.

    Returns the number of access units in the joined stream, counted from
    the access unit delimiters, which is a direct check that no frame was
    lost at a junction.
    """
    frames = 0
    with open(out_path, "wb") as out:
        for part in parts:
            with open(part, "rb") as fh:
                while True:
                    buf = fh.read(188 * 8192)
                    if not buf:
                        break
                    for i in range(0, len(buf) - 187, 188):
                        pkt = buf[i:i + 188]
                        if pkt[0] != 0x47:
                            continue
                        if (((pkt[1] & 0x1F) << 8) | pkt[2]) != VIDEO_PID:
                            continue
                        afc = (pkt[3] >> 4) & 0x3
                        j = 4
                        if afc in (2, 3):
                            # adaptation field: one length byte then that
                            # many bytes of padding, PCR and flags
                            j = 5 + pkt[4]
                            if afc == 2 or j >= 188:
                                continue
                        if (pkt[1] >> 6) & 1:
                            # payload unit start: a PES header follows
                            if pkt[j:j + 3] != b"\x00\x00\x01":
                                continue
                            j = j + 9 + pkt[j + 8]
                            if j >= 188:
                                continue
                        chunk = pkt[j:188]
                        frames += chunk.count(b"\x00\x00\x00\x01\x09")
                        out.write(chunk)
    return frames


# ==========================================================================
# driver
# ==========================================================================
#: Measured seconds per frame at 1920x1080, used only to balance the
#: workers.  The phases differ by a factor of thirty, so an even split by
#: frame count leaves one core idle for twenty minutes while the other
#: grinds through the geodesics.
COST = {
    "I · GALAKSI": 4.10,
    "II · TATA SURYA": 0.16,
    "III · MATAHARI": 3.05,
    "IV · TRANSISI": 1.15,
    "V · LUBANG HITAM": 5.00,
}


def plan_segments(timeline, fps, first, last, workers):
    """Split the film into contiguous blocks of equal estimated cost.

    Contiguous rather than interleaved, so each worker sees one phase for
    most of its range and the phase's caches stay warm; cost-weighted
    rather than frame-weighted, because a frame of Phase II is thirty
    times cheaper than a frame of Phase V and an even split by count would
    leave one core finished while the other still had half an hour to go.
    """
    costs = np.empty(last - first, np.float64)
    for k in range(first, last):
        seg, _, _ = timeline.at(k / float(fps))
        costs[k - first] = COST.get(seg.phase.name, 2.0)
    cum = np.cumsum(costs)
    total = float(cum[-1]) if len(cum) else 1.0
    out = []
    prev = 0
    for w in range(1, workers):
        target = total * w / workers
        cut = int(np.searchsorted(cum, target))
        cut = max(prev + 1, min(cut, last - first - (workers - w)))
        out.append((prev, cut))
        prev = cut
    out.append((prev, last - first))
    return out


def build_score(wav_path, total, seed=20261018, force=False):
    """Render the score to ``wav_path`` unless it is already there.

    Kept separate from the picture so a re-render of one does not force a
    re-render of the other; the score takes under a minute and the film
    takes two hours.
    """
    if (not force) and os.path.exists(wav_path) \
            and os.path.getsize(wav_path) > 1000:
        return wav_path
    from .audio import score as SC
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    mix = SC.build(seed=seed, total=total)
    SC.write_wav(wav_path, mix)
    return wav_path


def mux_audio(video_path, wav_path, out_path, bitrate="320k"):
    """Attach the score to the picture without re-encoding either stream.

    The video is stream-copied — it has already been through one lossy
    encode and a second would cost quality for nothing.  The audio is
    encoded once from the float mix straight to AAC at a high bit rate.
    """
    cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
           "-i", video_path, "-i", wav_path,
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy",
           "-c:a", "aac", "-b:a", bitrate, "-ar", "48000", "-ac", "2",
           "-movflags", "+faststart", "-shortest", out_path]
    subprocess.run(cmd, check=True)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="render the cosmic sequence")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--out", default=os.path.join(ROOT, "out",
                                                  "cosmos_1080p60.mp4"))
    ap.add_argument("--tmp", default=os.path.join(ROOT, ".rendertmp"))
    ap.add_argument("--start", type=float, default=None,
                    help="first film second to render (for previews)")
    ap.add_argument("--stop", type=float, default=None)
    ap.add_argument("--audio", default="auto",
                    help="'auto' builds the score and muxes it, 'none' "
                         "leaves the film silent, or a path to a WAV")
    ap.add_argument("--audio-bitrate", default="320k")
    args = ap.parse_args(argv)

    from .timeline import Timeline, build_phases
    tl = Timeline(build_phases())
    total = tl.frames(args.fps)
    first = 0 if args.start is None else int(args.start * args.fps)
    last = total if args.stop is None else min(total,
                                               int(args.stop * args.fps))
    count = last - first

    os.makedirs(args.tmp, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("durasi %.2f s · %d frame · %dx%d @ %d fps · %d pekerja"
          % (tl.total, count, args.width, args.height, args.fps,
             args.workers), flush=True)
    for seg in tl.segments:
        print("   %-18s %6.2f - %6.2f s" % (seg.phase.name, seg.start,
                                            seg.end), flush=True)

    blocks = plan_segments(tl, args.fps, first, last, args.workers)
    q = Queue()
    procs = []
    parts = []
    for wid, (a, b) in enumerate(blocks):
        part = os.path.join(args.tmp, "seg_%02d.ts" % wid)
        parts.append(part)
        est = sum(COST.get(tl.at((first + k) / float(args.fps))[0]
                           .phase.name, 2.0) for k in range(a, b))
        print("   pekerja %d: frame %5d-%5d  perkiraan %5.1f menit"
              % (wid, first + a, first + b, est / 60.0), flush=True)
        p = Process(target=render_range,
                    args=(args.width, args.height, args.fps,
                          first + a, first + b, part, args.crf,
                          args.preset, q, wid))
        p.start()
        procs.append(p)

    state = {}
    t0 = time.perf_counter()
    alive = True
    while alive:
        try:
            wid, done, tot, per = q.get(timeout=5.0)
            state[wid] = (done, tot, per)
            tot_done = sum(v[0] for v in state.values())
            tot_all = sum(v[1] for v in state.values()) or 1
            rate = max(sum(1.0 / v[2] for v in state.values() if v[2] > 0),
                       1e-6)
            eta = (count - tot_done) / rate
            print("  %5d/%d  %5.1f%%  %.2f s/frame/pekerja  ETA %5.1f menit"
                  % (tot_done, count, 100.0 * tot_done / count,
                     sum(v[2] for v in state.values()) / len(state),
                     eta / 60.0), flush=True)
        except Exception:
            pass
        alive = any(p.is_alive() for p in procs)
    for p in procs:
        p.join()

    print("penyandian selesai dalam %.1f s · menggabungkan"
          % (time.perf_counter() - t0), flush=True)

    joined = os.path.join(args.tmp, "joined.h264")
    frames = join_transport_streams(parts, joined)
    print("aliran gabungan: %d frame" % frames, flush=True)

    cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
           "-r", str(args.fps), "-f", "h264", "-i", joined,
           "-c:v", "copy", "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True)

    if args.audio and args.audio != "none":
        wav = args.audio
        if wav == "auto":
            wav = os.path.join(ROOT, "out", "kosmik_score.wav")
            print("membangun skor ...", flush=True)
            build_score(wav, tl.total)
        silent = args.out[:-4] + "_silent.mp4"
        os.replace(args.out, silent)
        print("memasang skor ...", flush=True)
        mux_audio(silent, wav, args.out, args.audio_bitrate)
        os.remove(silent)

    size = os.path.getsize(args.out)
    print("selesai: %s · %.1f MB · %.1f menit total"
          % (args.out, size / 1e6, (time.perf_counter() - t0) / 60.0),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
