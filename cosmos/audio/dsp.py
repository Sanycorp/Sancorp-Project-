"""
cosmos.audio.dsp
================

Signal processing primitives for the score.

Everything here is written against numpy array operations rather than
per-sample loops.  That is not a stylistic preference: the score is
fifty-four seconds of stereo at forty-eight kilohertz, which is five
million samples, and a Python loop over five million samples costs
several seconds *per filter instance*.  A score with sixty voices and a
reverb on each bus would take an hour.

The two techniques that make this possible are worth stating plainly,
because they shape every function below.

**Infinite-impulse-response filters become convolutions.**  A one-pole
lowpass, ``y[n] = a y[n-1] + (1-a) x[n]``, has the closed-form solution
``y[n] = (1-a) sum_k a^(n-k) x[k]`` — that is, a convolution with a
decaying exponential.  Any stable IIR filter is a convolution with its
impulse response, and a convolution of a five-million-sample signal is
an FFT away.  So rather than iterate, the impulse response is built
once and applied in the frequency domain.

**Time-varying filters become block-wise static ones.**  A filter whose
cutoff sweeps cannot be a single convolution, but over a window of a
couple of thousand samples — forty milliseconds — the cutoff barely
moves.  The signal is cut into overlapping blocks, each block is
filtered with the response appropriate to its midpoint, and the blocks
are summed back with a window that sums to unity.  This is the standard
overlap-add construction and it is indistinguishable from a true
time-varying filter for the sweep rates music uses.
"""
from __future__ import annotations

import math

import numpy as np

#: Sample rate.  Forty-eight kilohertz because that is what the video
#: muxer expects and resampling at the end would be a pointless loss.
SR = 48000

F32 = np.float32
F64 = np.float64
TAU = 2.0 * math.pi


# ==========================================================================
# small helpers
# ==========================================================================
def n_samples(seconds):
    """Number of samples in a duration, rounded to the nearest sample."""
    return int(round(float(seconds) * SR))


def seconds(n):
    return float(n) / SR


def db(x):
    """Decibels to linear amplitude."""
    return 10.0 ** (float(x) / 20.0)


def silence(dur, channels=1):
    n = n_samples(dur)
    return np.zeros(n if channels == 1 else (n, channels), F64)


def time_axis(n):
    return np.arange(n, dtype=F64) / SR


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-12), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def normalize(x, peak=1.0):
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-12:
        return x
    return x * (peak / m)


def mono_to_stereo(x):
    return np.stack([x, x], axis=1)


# ==========================================================================
# noise
# ==========================================================================
def white(n, rng):
    return rng.standard_normal(n).astype(F64)


def pink(n, rng):
    """Pink noise by spectral shaping.

    The Voss-McCartney method is the usual time-domain recipe, but it is
    a loop; in the frequency domain pink noise is simply white noise with
    each bin scaled by ``1/sqrt(f)``, which is one transform and one
    multiply.  The result is exact rather than the staircase
    approximation Voss-McCartney gives.
    """
    x = white(n, rng)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1] if n > 1 else 1.0
    X /= np.sqrt(f)
    y = np.fft.irfft(X, n)
    return normalize(y, 1.0)


def brown(n, rng):
    """Brownian (``1/f^2``) noise — the rumble under everything."""
    x = white(n, rng)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1] if n > 1 else 1.0
    X /= f
    y = np.fft.irfft(X, n)
    return normalize(y, 1.0)


def velvet(n, rng, density=900.0):
    """Velvet noise: sparse unit impulses of random sign.

    Perceptually it is noise, but it has far fewer non-zero samples than
    white noise, which makes it the right excitation for a reverb tail —
    the tail sounds smooth instead of grainy and costs a fraction as much
    to convolve.
    """
    out = np.zeros(n, F64)
    count = max(1, int(n / SR * density))
    idx = rng.integers(0, max(n, 1), count)
    out[idx] = rng.choice(np.array([-1.0, 1.0]), count)
    return out


# ==========================================================================
# spectral (static) filtering
# ==========================================================================
def spectral_filter(x, response):
    """Apply an arbitrary magnitude response, given as a function of Hz."""
    n = x.shape[0]
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    X *= response(f)
    return np.fft.irfft(X, n)


def _butter_mag(f, fc, order, kind):
    """Butterworth magnitude response.

    Zero phase, because it is applied as a pure magnitude mask.  For a
    musical filter that is an advantage: there is no phase smearing of
    transients, which is exactly what a linear-phase mastering filter is
    for.
    """
    fc = max(float(fc), 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(f > 0.0, f / fc, 0.0)
    if kind == "low":
        return 1.0 / np.sqrt(1.0 + ratio ** (2 * order))
    if kind == "high":
        inv = np.where(ratio > 0.0, fc / np.maximum(f, 1e-9), 1e9)
        return 1.0 / np.sqrt(1.0 + inv ** (2 * order))
    raise ValueError(kind)


def lowpass(x, fc, order=4):
    return spectral_filter(x, lambda f: _butter_mag(f, fc, order, "low"))


def highpass(x, fc, order=4):
    return spectral_filter(x, lambda f: _butter_mag(f, fc, order, "high"))


def bandpass(x, lo, hi, order=4):
    return spectral_filter(
        x, lambda f: (_butter_mag(f, lo, order, "high")
                      * _butter_mag(f, hi, order, "low")))


def tilt(x, slope_db_per_oct, pivot=700.0):
    """Shelve the whole spectrum about a pivot frequency."""
    def resp(f):
        ff = np.maximum(f, 1e-6)
        octs = np.log2(ff / pivot)
        return 10.0 ** (octs * slope_db_per_oct / 20.0)
    return spectral_filter(x, resp)


def resonant_peak(x, fc, q=6.0, gain_db=9.0):
    """A single resonant bell, for giving a drone a formant."""
    a = db(gain_db) - 1.0

    def resp(f):
        ff = np.maximum(f, 1e-6)
        w = (ff / fc - fc / ff) * q
        return 1.0 + a / (1.0 + w * w)
    return spectral_filter(x, resp)


# ==========================================================================
# time-varying filtering
# ==========================================================================
def sweep_lowpass(x, cutoff, order=4, block=2048):
    """Lowpass whose cutoff follows a per-sample envelope.

    Overlap-add with a Hann window and fifty per cent hop, so the window
    pairs sum to one.  Each block is filtered at the cutoff its centre
    sample calls for.
    """
    n = x.shape[0]
    cutoff = np.broadcast_to(np.asarray(cutoff, F64), (n,))
    hop = block // 2
    win = np.hanning(block + 1)[:block]
    out = np.zeros(n + block, F64)
    padded = np.concatenate([x, np.zeros(block, F64)])
    f = np.fft.rfftfreq(block, 1.0 / SR)
    for start in range(0, n, hop):
        seg = padded[start:start + block]
        if seg.shape[0] < block:
            seg = np.concatenate([seg, np.zeros(block - seg.shape[0], F64)])
        c = float(cutoff[min(start + hop, n - 1)])
        mag = _butter_mag(f, c, order, "low")
        y = np.fft.irfft(np.fft.rfft(seg * win) * mag, block)
        out[start:start + block] += y
    return out[:n]


def sweep_bandpass(x, centre, q=3.0, block=2048):
    """Band-pass whose centre frequency follows an envelope."""
    n = x.shape[0]
    centre = np.broadcast_to(np.asarray(centre, F64), (n,))
    hop = block // 2
    win = np.hanning(block + 1)[:block]
    out = np.zeros(n + block, F64)
    padded = np.concatenate([x, np.zeros(block, F64)])
    f = np.fft.rfftfreq(block, 1.0 / SR)
    ff = np.maximum(f, 1e-6)
    for start in range(0, n, hop):
        seg = padded[start:start + block]
        if seg.shape[0] < block:
            seg = np.concatenate([seg, np.zeros(block - seg.shape[0], F64)])
        c = max(float(centre[min(start + hop, n - 1)]), 20.0)
        w = (ff / c - c / ff) * q
        mag = 1.0 / np.sqrt(1.0 + w * w)
        y = np.fft.irfft(np.fft.rfft(seg * win) * mag, block)
        out[start:start + block] += y
    return out[:n]


# ==========================================================================
# envelopes
# ==========================================================================
def adsr(n, a=0.01, d=0.2, s=0.6, r=0.4, curve=2.0):
    """Attack-decay-sustain-release, with exponential segments.

    Linear envelope segments sound wrong because loudness is roughly
    logarithmic in amplitude: a linear fade appears to hang and then drop
    away at the end.  Raising the ramp to a power corrects it.
    """
    na = min(n, n_samples(a))
    nd = min(max(n - na, 0), n_samples(d))
    nr = min(max(n - na - nd, 0), n_samples(r))
    ns = max(n - na - nd - nr, 0)
    out = np.empty(n, F64)
    i = 0
    if na:
        out[i:i + na] = np.linspace(0.0, 1.0, na) ** (1.0 / curve)
        i += na
    if nd:
        out[i:i + nd] = 1.0 + (s - 1.0) * np.linspace(0.0, 1.0, nd) ** curve
        i += nd
    if ns:
        out[i:i + ns] = s
        i += ns
    if nr:
        out[i:i + nr] = s * (1.0 - np.linspace(0.0, 1.0, nr)) ** curve
        i += nr
    if i < n:
        out[i:] = 0.0
    return out


def exp_decay(n, tau):
    """A single falling exponential of time constant ``tau`` seconds."""
    return np.exp(-time_axis(n) / max(tau, 1e-6))


def percussive(n, attack=0.002, tau=0.8):
    na = min(n, n_samples(attack))
    env = exp_decay(n, tau)
    if na:
        env[:na] *= np.linspace(0.0, 1.0, na) ** 0.5
    return env


def swell(n, peak=0.5, curve=2.0):
    """Rise to a peak at fractional position ``peak``, then fall."""
    u = np.linspace(0.0, 1.0, n)
    up = np.clip(u / max(peak, 1e-6), 0.0, 1.0) ** curve
    dn = np.clip((1.0 - u) / max(1.0 - peak, 1e-6), 0.0, 1.0) ** curve
    return up * dn / max(float((up * dn).max()), 1e-9)


def fade_in(x, dur):
    k = min(x.shape[0], n_samples(dur))
    if k > 0:
        ramp = np.linspace(0.0, 1.0, k) ** 1.6
        if x.ndim == 1:
            x[:k] *= ramp
        else:
            x[:k] *= ramp[:, None]
    return x


def fade_out(x, dur):
    k = min(x.shape[0], n_samples(dur))
    if k > 0:
        ramp = np.linspace(1.0, 0.0, k) ** 1.6
        if x.ndim == 1:
            x[-k:] *= ramp
        else:
            x[-k:] *= ramp[:, None]
    return x


# ==========================================================================
# oscillators
# ==========================================================================
def phase_from_freq(freq, n):
    """Integrate a frequency envelope into a phase, in radians.

    For anything whose pitch moves — a glissando, a Doppler shift, a
    gravitational-wave chirp — the oscillator argument must be the
    *integral* of frequency, not frequency times time.  Writing
    ``sin(2 pi f(t) t)`` instead is the classic chirp bug: it produces an
    instantaneous frequency of ``f + t df/dt``, which is not the
    requested sweep and is audibly wrong at the ends.
    """
    freq = np.broadcast_to(np.asarray(freq, F64), (n,))
    return TAU * np.cumsum(freq) / SR


def sine(freq, n, phase0=0.0):
    return np.sin(phase_from_freq(freq, n) + phase0)


def saw(freq, n, harmonics=48, phase0=0.0):
    """Band-limited sawtooth by additive synthesis.

    A naive ramp aliases ferociously: at 200 Hz it folds everything above
    24 kHz back down into the audible band as inharmonic rubbish.  Summing
    only the partials that fit below Nyquist costs more but is clean.
    """
    ph = phase_from_freq(freq, n) + phase0
    f0 = float(np.mean(np.broadcast_to(np.asarray(freq, F64), (n,))))
    kmax = max(1, min(harmonics, int(0.45 * SR / max(f0, 1e-6))))
    out = np.zeros(n, F64)
    for k in range(1, kmax + 1):
        out += np.sin(ph * k) / k
    return out * (2.0 / math.pi)


def square(freq, n, harmonics=32, phase0=0.0):
    ph = phase_from_freq(freq, n) + phase0
    f0 = float(np.mean(np.broadcast_to(np.asarray(freq, F64), (n,))))
    kmax = max(1, min(harmonics, int(0.45 * SR / max(f0, 1e-6))))
    out = np.zeros(n, F64)
    for k in range(1, kmax + 1, 2):
        out += np.sin(ph * k) / k
    return out * (4.0 / math.pi)


def triangle(freq, n, harmonics=24, phase0=0.0):
    ph = phase_from_freq(freq, n) + phase0
    f0 = float(np.mean(np.broadcast_to(np.asarray(freq, F64), (n,))))
    kmax = max(1, min(harmonics, int(0.45 * SR / max(f0, 1e-6))))
    out = np.zeros(n, F64)
    sign = 1.0
    for k in range(1, kmax + 1, 2):
        out += sign * np.sin(ph * k) / (k * k)
        sign = -sign
    return out * (8.0 / (math.pi * math.pi))


# ==========================================================================
# delay and reverberation
# ==========================================================================
def delay(x, time, feedback=0.35, taps=8, mix=0.35, damp=5200.0):
    """Feedback delay, unrolled into a finite tap sum.

    A feedback delay line is an IIR filter with a very long period; after
    ``taps`` repeats the level is ``feedback^taps``, which for the values
    music uses is already far below the noise floor.  Unrolling it into a
    sum of delayed, damped copies avoids any per-sample recursion.
    """
    n = x.shape[0]
    d = n_samples(time)
    if d <= 0:
        return x
    wet = np.zeros(n, F64)
    cur = x
    g = feedback
    for k in range(1, taps + 1):
        cur = lowpass(cur, damp, order=2)
        shifted = np.zeros(n, F64)
        off = d * k
        if off >= n:
            break
        shifted[off:] = cur[:n - off]
        wet += shifted * g
        g *= feedback
    return x * (1.0 - mix) + wet * mix


def ping_pong(x, time, feedback=0.42, taps=10, mix=0.4, damp=4800.0):
    """Stereo delay whose repeats alternate between the channels."""
    n = x.shape[0]
    d = n_samples(time)
    left = np.zeros(n, F64)
    right = np.zeros(n, F64)
    cur = x
    g = feedback
    for k in range(1, taps + 1):
        cur = lowpass(cur, damp, order=2)
        off = d * k
        if off >= n:
            break
        tgt = left if k % 2 else right
        tgt[off:] += cur[:n - off] * g
        g *= feedback
    dry = x * (1.0 - mix)
    return np.stack([dry + left * mix, dry + right * mix], axis=1)


def reverb_ir(dur, rng, decay=3.0, lo=90.0, hi=6500.0, predelay=0.02,
              diffusion=0.85):
    """Synthesise a stereo impulse response.

    An exponentially decaying noise burst *is* the impulse response of an
    idealised diffuse room; everything a real algorithmic reverb does
    beyond that — all-pass chains, modulated delays — exists to
    approximate it cheaply in real time.  Since the whole score is
    rendered offline there is no reason not to build the ideal response
    directly and convolve.

    Two decorrelated channels give a wide, stable image.  The decay is
    frequency dependent, with the top end dying faster than the bottom,
    because air absorption and soft surfaces both roll off the treble.
    """
    n = n_samples(dur)
    out = np.zeros((n, 2), F64)
    t = time_axis(n)
    env = np.exp(-t * (6.9 / max(decay, 1e-3)))
    # early build-up: a real room takes a few milliseconds to become
    # diffuse, which is what stops the tail sounding like a gated sample
    env *= 1.0 - np.exp(-t / 0.012)
    for ch in range(2):
        v = velvet(n, rng, density=2400.0) * diffusion \
            + white(n, rng) * (1.0 - diffusion)
        v = bandpass(v, lo, hi, order=2)
        # treble decays faster: multiply a second, shorter envelope into
        # the high band only
        hi_part = highpass(v, 1800.0, order=2) * np.exp(
            -t * (6.9 / max(decay * 0.45, 1e-3)))
        lo_part = lowpass(v, 1800.0, order=2) * env
        out[:, ch] = lo_part + hi_part * 0.7
    pd = n_samples(predelay)
    if pd > 0:
        out = np.concatenate([np.zeros((pd, 2), F64), out])[:n]
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out /= peak
    return out


def convolve_stereo(x, ir, mix=0.3, wet_gain=1.0):
    """FFT convolution of a mono source with a stereo impulse response."""
    n = x.shape[0]
    m = ir.shape[0]
    size = 1
    while size < n + m:
        size *= 2
    X = np.fft.rfft(x, size)
    out = np.empty((n, 2), F64)
    for ch in range(2):
        H = np.fft.rfft(ir[:, ch], size)
        y = np.fft.irfft(X * H, size)[:n]
        out[:, ch] = y
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out *= float(np.max(np.abs(x))) / peak
    dry = np.stack([x, x], axis=1)
    return dry * (1.0 - mix) + out * (mix * wet_gain)


# ==========================================================================
# stereo
# ==========================================================================
def pan(x, position):
    """Constant-power pan; ``position`` runs -1 (left) to +1 (right)."""
    p = (float(np.clip(position, -1.0, 1.0)) + 1.0) * 0.25 * math.pi
    return np.stack([x * math.cos(p), x * math.sin(p)], axis=1)


def haas(x, ms=14.0, side=1.0):
    """Widen by delaying one channel a few milliseconds.

    Below about thirty milliseconds the ear fuses the two arrivals into
    one event and reads the delay as direction rather than as an echo —
    the precedence effect.  It is the cheapest convincing width there is,
    and unlike a phase-inverted side channel it survives a mono fold-down
    without cancelling.
    """
    d = n_samples(ms / 1000.0)
    n = x.shape[0]
    a = x.copy()
    b = np.zeros(n, F64)
    if 0 < d < n:
        b[d:] = x[:n - d]
    else:
        b = x.copy()
    if side < 0:
        a, b = b, a
    return np.stack([a, b], axis=1)


def widen(stereo, amount=1.0):
    """Mid-side width control."""
    mid = (stereo[:, 0] + stereo[:, 1]) * 0.5
    sideb = (stereo[:, 0] - stereo[:, 1]) * 0.5 * amount
    return np.stack([mid + sideb, mid - sideb], axis=1)


# ==========================================================================
# dynamics
# ==========================================================================
def soft_clip(x, drive=1.0):
    """Cubic soft clipper — adds low-order harmonics, never squares off."""
    y = x * drive
    return np.where(np.abs(y) < 1.0,
                    y - (y ** 3) / 3.0,
                    np.sign(y) * (2.0 / 3.0)) * 1.5


def compress(x, threshold=0.35, ratio=3.0, attack=0.008, release=0.18):
    """Feed-forward compressor with a smoothed gain envelope.

    The envelope follower is a convolution with a one-sided exponential,
    which is the closed form of the usual one-pole smoother, so the whole
    detector is two FFTs instead of a sample loop.
    """
    n = x.shape[0] if x.ndim == 1 else x.shape[0]
    level = np.abs(x) if x.ndim == 1 else np.max(np.abs(x), axis=1)
    k_at = max(n_samples(attack), 1)
    k_rl = max(n_samples(release), 1)
    k = max(k_at, k_rl)
    kern = np.exp(-np.arange(k * 4, dtype=F64) / k)
    kern /= kern.sum()
    size = 1
    while size < n + kern.shape[0]:
        size *= 2
    env = np.fft.irfft(np.fft.rfft(level, size)
                       * np.fft.rfft(kern, size), size)[:n]
    env = np.maximum(env, 1e-9)
    over = env / threshold
    gain = np.where(over > 1.0, over ** (1.0 / ratio - 1.0), 1.0)
    return x * (gain if x.ndim == 1 else gain[:, None])


def limit(x, ceiling=0.97, lookahead=0.004, smooth=0.012):
    """Look-ahead brick-wall limiter.

    The gain reduction curve is computed from the *future* peak within
    the look-ahead window — a running maximum — and then smoothed, so the
    gain is already down by the time the transient arrives and nothing
    clips.
    """
    n = x.shape[0]
    peak = np.abs(x) if x.ndim == 1 else np.max(np.abs(x), axis=1)
    w = max(n_samples(lookahead), 1)
    pad = np.concatenate([peak, np.zeros(w, F64)])
    # running maximum over the look-ahead window, by repeated pairwise
    # maxima: log2(w) passes instead of w
    run = pad.copy()
    step = 1
    while step < w:
        run = np.maximum(run, np.concatenate(
            [run[step:], np.zeros(step, F64)]))
        step *= 2
    run = run[:n]
    need = np.where(run > ceiling, ceiling / np.maximum(run, 1e-9), 1.0)
    k = max(n_samples(smooth), 1)
    kern = np.exp(-np.arange(k * 4, dtype=F64) / k)
    kern /= kern.sum()
    size = 1
    while size < n + kern.shape[0]:
        size *= 2
    g = np.fft.irfft(np.fft.rfft(need, size)
                     * np.fft.rfft(kern, size), size)[:n]
    g = np.minimum(g, need)
    return x * (g if x.ndim == 1 else g[:, None])


# ==========================================================================
# mixing
# ==========================================================================
def add_at(bus, sig, t, gain=1.0):
    """Mix ``sig`` into ``bus`` starting at time ``t`` seconds."""
    i = n_samples(t)
    if i < 0:
        sig = sig[-i:]
        i = 0
    if sig.shape[0] == 0 or i >= bus.shape[0]:
        return bus
    k = min(sig.shape[0], bus.shape[0] - i)
    if bus.ndim == 2 and sig.ndim == 1:
        bus[i:i + k] += sig[:k, None] * gain
    elif bus.ndim == 2:
        bus[i:i + k] += sig[:k] * gain
    else:
        bus[i:i + k] += sig[:k] * gain
    return bus


__all__ = [
    "SR", "TAU", "n_samples", "seconds", "db", "silence", "time_axis",
    "lerp", "smoothstep", "smootherstep", "normalize", "mono_to_stereo",
    "white", "pink", "brown", "velvet", "spectral_filter", "lowpass",
    "highpass", "bandpass", "tilt", "resonant_peak", "sweep_lowpass",
    "sweep_bandpass", "adsr", "exp_decay", "percussive", "swell",
    "fade_in", "fade_out", "phase_from_freq", "sine", "saw", "square",
    "triangle", "delay", "ping_pong", "reverb_ir", "convolve_stereo",
    "pan", "haas", "widen", "soft_clip", "compress", "limit", "add_at",
]
