"""
Procedural soundtrack, synthesised with numpy and locked to the cut list.

120 BPM -> beat = 0.5 s, bar = 2.0 s.  Everything is additive synthesis plus
noise shaped in the frequency domain, so no external DSP library is needed.

    python3 audio.py out/score.wav
"""
from __future__ import annotations

import math
import os
import struct
import sys
import wave

import numpy as np

SR = 48000
BPM = 120.0
BEAT = 60.0 / BPM          # 0.5 s
BAR = BEAT * 4             # 2.0 s
DUR = 56.0

RNG = np.random.default_rng(20251016)

# --------------------------------------------------------------------------
# note table (equal temperament, A4 = 440)
# --------------------------------------------------------------------------
_NAMES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
          "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
          "Bb": 10, "B": 11}


def hz(name: str) -> float:
    i = 1 if name[1].isdigit() else 2
    pc, octv = _NAMES[name[:i]], int(name[i:])
    return 440.0 * 2 ** ((pc + (octv - 4) * 12 - 9) / 12.0)


# F minor progression, one chord every two bars
PROG = [
    ("Fm", ["F3", "Ab3", "C4"], "F2"),
    ("Db", ["Db3", "F3", "Ab3"], "Db2"),
    ("Ab", ["Ab3", "C4", "Eb4"], "Ab2"),
    ("Eb", ["Eb3", "G3", "Bb3"], "Eb2"),
]


def chord_at(bar):
    return PROG[int(bar // 2) % len(PROG)]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def add(buf, sig, t, gain=1.0, pan=0.0):
    """Mix a mono signal into the stereo buffer at time `t`."""
    i = int(t * SR)
    if i >= buf.shape[0] or len(sig) == 0:
        return
    j = min(buf.shape[0], i + len(sig))
    s = sig[: j - i] * gain
    l = math.sqrt(0.5 * (1.0 - pan))
    r = math.sqrt(0.5 * (1.0 + pan))
    buf[i:j, 0] += s * l
    buf[i:j, 1] += s * r


def noise(n):
    return RNG.standard_normal(n).astype(np.float32)


def spectral(x, lo=None, hi=None, slope=0.0):
    """Shape a signal in the frequency domain (brick-ish band + tilt)."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    m = np.ones_like(f)
    if lo:
        m *= 1.0 / (1.0 + (lo / np.maximum(f, 1.0)) ** 4)
    if hi:
        m *= 1.0 / (1.0 + (np.maximum(f, 1.0) / hi) ** 4)
    if slope:
        m *= (np.maximum(f, 20.0) / 1000.0) ** slope
    return np.fft.irfft(X * m, n).astype(np.float32)


def adsr(n, a=0.005, d=0.1, s=0.0, r=0.1):
    e = np.zeros(n, np.float32)
    ai, di = int(a * SR), int(d * SR)
    ri = int(r * SR)
    si = max(0, n - ai - di - ri)
    k = 0
    if ai:
        e[:ai] = np.linspace(0, 1, ai)
        k = ai
    if di:
        e[k:k + di] = np.linspace(1, s, di)
        k += di
    if si:
        e[k:k + si] = s
        k += si
    if ri and k < n:
        e[k:] = np.linspace(e[k - 1] if k else s, 0, n - k)
    return e


# --------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------
def kick(dur=0.5, punch=1.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = 46.0 + 140.0 * np.exp(-t / 0.026)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.155)
    click = spectral(noise(n), lo=900) * np.exp(-t / 0.0035) * 0.5 * punch
    sub = np.sin(2 * np.pi * 44 * t) * np.exp(-t / 0.085) * 0.45
    x = body + click + sub
    return np.tanh(x * 1.5).astype(np.float32)


def clap(dur=0.40):
    n = int(SR * dur)
    t = np.arange(n) / SR
    nz = spectral(noise(n), lo=1100, hi=7600)
    x = np.zeros(n, np.float32)
    for i, dly in enumerate((0.0, 0.008, 0.017, 0.027)):
        s = int(dly * SR)
        seg = nz[: n - s] * np.exp(-np.arange(n - s) / SR / 0.012)
        x[s:] += seg * (0.78 ** i)
    x += nz * np.exp(-t / 0.115) * 0.42          # body / tail
    return (x * 0.55).astype(np.float32)


def hat(dur=0.07, open_=False):
    d = 0.34 if open_ else dur
    n = int(SR * d)
    t = np.arange(n) / SR
    nz = spectral(noise(n), lo=6500, slope=0.25)
    e = np.exp(-t / (0.085 if open_ else 0.014))
    return (nz * e * 0.30).astype(np.float32)


def pluck(f, dur=0.5, harm=14, bright=1.0, detune=0.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    x = np.zeros(n, np.float32)
    for k in range(1, harm + 1):
        dk = 0.32 / (k ** (0.85 / max(bright, 0.2)))
        ph = RNG.uniform(0, 2 * np.pi)
        x += (np.sin(2 * np.pi * f * k * t + ph) / k) * np.exp(-t / dk)
        if detune:
            x += (np.sin(2 * np.pi * f * (1 + detune) * k * t + ph) / k) \
                * np.exp(-t / dk) * 0.6
    x *= adsr(n, 0.002, 0.0, 1.0, dur * 0.5)
    return (x * 0.38).astype(np.float32)


def bass(f, dur=0.45):
    n = int(SR * dur)
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * f * t) * 1.0
         + np.sin(2 * np.pi * 2 * f * t) * 0.34
         + np.sin(2 * np.pi * 3 * f * t) * 0.14)
    x *= adsr(n, 0.006, 0.07, 0.72, dur * 0.42)
    return (np.tanh(x * 1.2) * 0.42).astype(np.float32)


def pad(freqs, dur, gain=1.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    x = np.zeros(n, np.float32)
    for f in freqs:
        for det in (-0.004, 0.0, 0.005):
            for k, a in ((1, 1.0), (2, 0.30), (3, 0.13), (4, 0.06)):
                x += np.sin(2 * np.pi * f * k * (1 + det) * t
                            + RNG.uniform(0, 6.28)) * a
    x /= max(1, len(freqs) * 9)
    x *= adsr(n, dur * 0.22, 0.0, 1.0, dur * 0.42)
    # slow shimmer
    x *= 1.0 + 0.10 * np.sin(2 * np.pi * 0.17 * t)
    return (x * 0.34 * gain).astype(np.float32)


def riser(dur=2.0, f0=200, f1=4200):
    n = int(SR * dur)
    t = np.arange(n) / SR
    u = t / dur
    nz = noise(n)
    sweep = spectral(nz, lo=f0, hi=f0 * 1.9)
    # a second, brighter band crossfaded in
    sweep2 = spectral(nz, lo=f1 * 0.45, hi=f1)
    x = sweep * (1 - u) + sweep2 * u
    tone = np.sin(2 * np.pi * np.cumsum(np.linspace(220, 1500, n)) / SR) * 0.18
    x = x * (u ** 1.6) + tone * (u ** 2.4)
    return (x * 0.5).astype(np.float32)


def impact(dur=2.4, bright=1.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = 120 * np.exp(-t / 0.09) + 34
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.45)
    crash = spectral(noise(n), lo=2200, slope=-0.3) * np.exp(-t / 0.55) * 0.5 * bright
    tail = spectral(noise(n), lo=300, hi=2600) * np.exp(-t / 0.9) * 0.22
    return (np.tanh((boom * 1.1 + crash + tail) * 1.1) * 0.75).astype(np.float32)


def reverse_swell(dur=1.6):
    n = int(SR * dur)
    t = np.arange(n) / SR
    x = spectral(noise(n), lo=700, hi=9000)
    return (x * (t / dur) ** 2.6 * 0.42).astype(np.float32)


def tick(dur=0.05):
    n = int(SR * dur)
    t = np.arange(n) / SR
    x = spectral(noise(n), lo=2800, hi=11000) * np.exp(-t / 0.006)
    return (x * 0.30).astype(np.float32)


def sub_drone(dur, f=43.65):
    n = int(SR * dur)
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2 * t))
    x *= 1.0 + 0.2 * np.sin(2 * np.pi * 0.25 * t)
    return (x * 0.20).astype(np.float32)


# --------------------------------------------------------------------------
# echo / space
# --------------------------------------------------------------------------
def echo(buf, time=BEAT * 0.75, fb=0.34, n=4, mix=0.3):
    out = buf.copy()
    for i in range(1, n + 1):
        d = int(time * i * SR)
        if d >= len(buf):
            break
        g = (fb ** i) * mix
        out[d:, 0] += buf[:-d, 1] * g
        out[d:, 1] += buf[:-d, 0] * g
    return out


# --------------------------------------------------------------------------
# arrangement
# --------------------------------------------------------------------------
def reverb_ir(dur=2.4, decay=4.2, seed=5, hi=5200):
    """Exponentially decaying, high-damped stereo impulse response."""
    n = int(dur * SR)
    r = np.random.default_rng(seed)
    ir = r.standard_normal((n, 2)).astype(np.float32)
    env = np.exp(-np.arange(n, dtype=np.float32) / SR * decay)
    env[:int(0.012 * SR)] = 0.0                 # pre-delay
    ir *= env[:, None]
    for ch in range(2):
        ir[:, ch] -= spectral(ir[:, ch], lo=hi) * 0.72   # damp the top end
    ir /= np.sqrt((ir ** 2).sum(axis=0)).max() + 1e-9
    return ir


def convolve(x, ir):
    """Fast FFT convolution, length-preserving."""
    n, m = len(x), len(ir)
    N = 1
    while N < n + m:
        N <<= 1
    out = np.empty_like(x)
    for ch in range(2):
        out[:, ch] = np.fft.irfft(
            np.fft.rfft(x[:, ch], N) * np.fft.rfft(ir[:, ch], N), N)[:n]
    return out.astype(np.float32)


def widen(x, amount=1.0):
    """Mid/side stereo widening."""
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5 * amount
    out = np.empty_like(x)
    out[:, 0] = mid + side
    out[:, 1] = mid - side
    return out


def limit(x, ceiling=0.96, look=0.004, smooth=0.010):
    """Lookahead peak limiter — transparent where tanh would have dulled
    every transient in the mix."""
    env = np.abs(x).max(axis=1)
    g = np.minimum(1.0, ceiling / np.maximum(env, 1e-9)).astype(np.float32)
    k = max(1, int(look * SR))
    g = np.minimum.reduce([np.roll(g, i) for i in range(0, k, 12)])
    w = np.hanning(max(3, int(smooth * SR) | 1)).astype(np.float32)
    w /= w.sum()
    g = np.convolve(g, w, mode="same").astype(np.float32)
    return x * g[:, None]


def build():
    n = int(DUR * SR)
    drums = np.zeros((n, 2), np.float32)
    music = np.zeros((n, 2), np.float32)
    fx = np.zeros((n, 2), np.float32)

    KICK = kick()
    KICK_S = kick(0.34, 0.7)
    CLAP = clap()
    HAT = hat()
    HATO = hat(open_=True)
    TICK = tick()

    def bar_t(b):
        return b * BAR

    # ---------------- sections ------------------------------------------
    # 0-4 s   intro stabs | 4-8 drop | 8-18 groove | 18-26 break
    # 26-34 peak | 34-42 tension+hit | 42-48 montage | 48-56 outro
    full_kick = [(2, 4), (4, 9), (11, 13), (13, 17), (19.3, 21), (21, 24), (24, 27)]
    kick_bars = set()
    for a, b in full_kick:
        bb = a
        while bb < b:
            kick_bars.add(round(bb, 3))
            bb += 1

    # bars that drop to a half-time pulse instead of going silent
    half_time = [(9, 11), (17, 19.3)]

    for b in range(28):
        t0 = bar_t(b)
        active = any(a <= b < bb for a, bb in full_kick)
        half = (not active) and any(a <= b < bb for a, bb in half_time)

        if active:
            for q in range(4):
                add(drums, KICK, t0 + q * BEAT, 1.0)
            # ghost / off-beat kick for drive
            if b % 2 == 1:
                add(drums, KICK_S, t0 + 3.5 * BEAT, 0.5)
            add(drums, CLAP, t0 + BEAT, 0.85)
            add(drums, CLAP, t0 + 3 * BEAT, 0.85)
        elif half:
            # keeps the floor from dropping out under the quiet scenes
            add(drums, KICK, t0, 0.85)
            add(drums, KICK, t0 + 2 * BEAT, 0.70)
            add(drums, CLAP, t0 + 2 * BEAT, 0.45)
            _, _, rt = chord_at(b)
            add(music, bass(hz(rt), BEAT * 1.7), t0, 0.75)
            add(music, bass(hz(rt), BEAT * 1.7), t0 + 2 * BEAT, 0.65)

        # hats — 8ths, with 16th flourishes in the busier bars
        if b >= 2:
            hg = 1.0 if active else (0.42 if half else 0.6)
            for e in range(8):
                tt = t0 + e * BEAT / 2
                g = 0.9 if e % 2 == 0 else 0.55
                add(drums, HAT, tt, g * hg,
                    pan=0.18 if e % 2 else -0.18)
            if b % 4 == 3:
                for s in range(4):
                    add(drums, HAT, t0 + 3 * BEAT + s * BEAT / 4, 0.5, pan=0.3)
            if b % 8 == 7:
                add(drums, HATO, t0 + 3.5 * BEAT, 0.7)

        # ---- harmony
        name, notes, root = chord_at(b)
        fr = [hz(x) for x in notes]

        if b >= 1:
            g = 1.0
            if 9 <= b < 11:
                g = 1.05
            if 17 <= b < 19.3:
                g = 0.95
            if b >= 24:
                g = 1.15
            add(music, pad(fr, BAR * 1.02, g), t0, 1.0)

        # ---- bass: 8th-note gate
        if active:
            bf = hz(root)
            patt = [1, 0, 1, 1, 0, 1, 1, 0]
            for e, on in enumerate(patt):
                if on:
                    add(music, bass(bf, BEAT * 0.52), t0 + e * BEAT / 2, 0.95)

        # ---- arp: 16ths in the high mids
        arp_on = (4 <= b < 9) or (13 <= b < 17) or (21 <= b < 24) or (24 <= b < 27)
        if arp_on:
            seq = [fr[0] * 2, fr[1] * 2, fr[2] * 2, fr[1] * 4,
                   fr[2] * 2, fr[0] * 4, fr[1] * 2, fr[2] * 2]
            for e in range(8):
                tt = t0 + e * BEAT / 2
                f = seq[e % len(seq)]
                add(music, pluck(f, 0.42, 12, 1.1), tt, 0.52,
                    pan=-0.35 if e % 2 else 0.35)

    # ---------------- hook: four word slams -----------------------------
    for i, tt in enumerate((0.5, 1.0, 1.5, 2.0)):
        add(fx, impact(1.6, 0.6 + i * 0.18), tt, 0.55 + i * 0.12)
        add(drums, KICK, tt, 1.0)
    add(music, sub_drone(4.4), 0.0, 1.0)
    add(fx, reverse_swell(1.4), 2.6, 0.5)

    # scene-cut accents
    for tt, g in ((4.0, 0.95), (8.0, 0.8), (18.0, 0.7), (26.0, 0.95),
                  (34.0, 0.7), (42.0, 0.85), (48.0, 1.0)):
        add(fx, impact(2.6, 1.0), tt, g)

    # risers into each cut
    for tt, d in ((4.0, 1.6), (26.0, 2.0), (42.0, 1.6), (48.0, 2.0)):
        add(fx, riser(d, 180, 5200), tt - d, 0.55)

    # ---------------- break: typewriter ticks (18 - 24 s) ---------------
    t = 18.55
    while t < 20.95:
        add(fx, TICK, t, 0.55 + 0.25 * RNG.random(), pan=RNG.uniform(-0.4, 0.4))
        t += BEAT / 4
    for i in range(4):                      # result rows "ping"
        add(fx, pluck(hz("Ab4") * (1 + 0.0 + i * 0.12), 0.6, 10, 1.4),
            21.7 + i * 0.5, 0.42, pan=0.2 * (-1) ** i)

    # ---------------- render scene: countdown + the big hit -------------
    t = 34.6
    while t < 38.4:
        add(fx, TICK, t, 0.52, pan=0.0)
        t += BEAT / 2
    add(fx, riser(2.2, 150, 6000), 36.4, 0.95)
    add(music, sub_drone(4.2, hz('F2')), 34.4, 0.9)
    add(fx, impact(3.4, 1.3), 38.6, 1.25)
    add(drums, KICK, 38.6, 1.2)
    add(fx, reverse_swell(1.2), 37.4, 0.45)

    # ---------------- montage stabs (panels pop on 16ths) ---------------
    for i in range(6):
        tt = 42.25 + i * 0.25
        add(fx, pluck(hz("F4") * (1 + 0.0), 0.3, 9, 1.5), tt, 0.45,
            pan=(-0.4 + i * 0.16))
        add(drums, KICK_S, tt, 0.45)
    add(fx, impact(2.2, 1.1), 47.3, 0.8)

    # ---------------- outro ---------------------------------------------
    add(music, pad([hz("F3"), hz("Ab3"), hz("C4"), hz("F4")], 7.4, 1.25), 48.3, 1.0)
    add(fx, reverse_swell(1.6), 46.6, 0.5)
    for i, nm in enumerate(["F4", "Ab4", "C5", "F4"]):
        add(fx, pluck(hz(nm), 1.2, 12, 1.2), 48.5 + i * 0.5, 0.4,
            pan=-0.3 + i * 0.2)
    add(fx, impact(3.0, 0.8), 52.0, 0.5)

    # ---------------- bassline ------------------------------------------
    # roots the harmony an octave below the pads: the mix had a hole between
    # the sub drone and the chords, which is what made it feel thin
    for b in sorted(kick_bars):
        f = hz(chord_at(b)[2])          # PROG rows are (name, voicing, root)
        while f > 110.0:
            f *= 0.5
        for q, (off, dur, g) in enumerate(((0.0, 0.46, 1.0), (1.5, 0.30, 0.62),
                                           (2.0, 0.46, 0.86), (3.5, 0.26, 0.55))):
            add(music, bass(f, dur), b * BAR + off * BEAT, 0.50 * g)

    # ---------------- glue ----------------------------------------------
    fx = echo(fx, BEAT * 0.75, 0.32, 4, 0.26)
    music = echo(music, BEAT * 1.5, 0.22, 3, 0.16)

    # side-chain pump driven by the kick grid
    t_axis = np.arange(n) / SR
    duck = np.ones(n, np.float32)
    for b in sorted(kick_bars):
        for q in range(4):
            tt = b * BAR + q * BEAT
            i0 = int(tt * SR)
            i1 = min(n, i0 + int(0.34 * SR))
            if i0 >= n:
                continue
            seg = np.linspace(0.0, 1.0, i1 - i0)
            duck[i0:i1] = np.minimum(duck[i0:i1], 0.42 + 0.58 * seg ** 0.55)
    music *= duck[:, None]
    fx *= (0.55 + 0.45 * duck)[:, None]

    # reverb sends: the dry buses alone had no room around them
    IR = reverb_ir(2.4, 4.2)
    IR_S = reverb_ir(1.1, 9.0, seed=9, hi=6400)
    music = music + convolve(music, IR) * 0.26
    fx = fx + convolve(fx, IR_S) * 0.22
    drums = drums + convolve(drums, IR_S) * 0.07

    music = widen(music, 1.30)
    fx = widen(fx, 1.18)

    # keep the bottom octave for kick + sub only, so the low end stays tight
    for ch in range(2):
        music[:, ch] = spectral(music[:, ch], lo=38)
        fx[:, ch] = spectral(fx[:, ch], lo=70)

    mix = drums * 0.95 + music * 0.92 + fx * 0.82

    # ---------------- master --------------------------------------------
    # gentle high shelf for air
    for ch in range(2):
        mix[:, ch] += spectral(mix[:, ch], lo=5200) * 0.18

    # fades
    fi = int(0.04 * SR)
    mix[:fi] *= np.linspace(0, 1, fi)[:, None]
    f0, f1 = int(54.6 * SR), int(55.95 * SR)
    mix[f0:f1] *= np.linspace(1, 0, f1 - f0)[:, None]
    mix[f1:] = 0.0

    # master: drive into a lookahead limiter, then a soft safety clip
    peak = np.abs(mix).max()
    mix = mix / max(peak, 1e-6) * 1.45
    mix = limit(mix, 0.96)
    mix = np.tanh(mix * 1.02)
    mix *= 0.95 / max(np.abs(mix).max(), 1e-6)
    return mix.astype(np.float32)


def write_wav(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"wrote {path}  {len(data)/SR:.2f}s  peak={np.abs(data).max():.3f}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out/score.wav"
    write_wav(out, build())
