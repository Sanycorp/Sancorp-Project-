"""
cosmos.audio.voices
===================

The instruments.

Each function returns a mono float64 array at ``dsp.SR`` and takes care
of its own envelope, so the score reads as a list of events rather than
as a graph of modules.  Nothing here is a sample: every sound is
synthesised from an oscillator, a noise source or a physical model.
"""
from __future__ import annotations

import math

import numpy as np

from . import dsp as D

F64 = np.float64
TAU = 2.0 * math.pi


# ==========================================================================
# sustained voices
# ==========================================================================
def supersaw(freq, dur, voices=7, detune=0.14, spread=1.0, rng=None,
             cutoff=None, res=0.0):
    """A stack of slightly detuned sawtooths.

    The characteristic shimmer comes from the beating between the
    detuned copies; with seven voices and a detune of a few cents the
    beat frequencies are spread over a couple of hertz, which the ear
    hears as width and motion rather than as out-of-tune.
    """
    rng = rng or np.random.default_rng(0)
    n = D.n_samples(dur)
    out = np.zeros(n, F64)
    base = np.broadcast_to(np.asarray(freq, F64), (n,))
    for i in range(voices):
        c = (i - (voices - 1) * 0.5) / max(voices - 1, 1) * 2.0
        f = base * (1.0 + c * detune * 0.01 * spread)
        # a slow independent drift per voice keeps the chorus alive
        drift = 1.0 + 0.0009 * np.sin(
            D.time_axis(n) * TAU * (0.07 + 0.05 * i)
            + float(rng.random()) * TAU)
        out += D.saw(f * drift, n, harmonics=36,
                     phase0=float(rng.random()) * TAU)
    out /= voices
    if cutoff is not None:
        out = D.sweep_lowpass(out, cutoff, order=4)
    if res > 0.0 and cutoff is not None:
        cut = np.broadcast_to(np.asarray(cutoff, F64), (n,))
        out += D.sweep_bandpass(out, cut, q=7.0) * res
    return out


def additive_drone(f0, dur, partials=None, rng=None, inharmonicity=0.0,
                   shimmer=0.0035):
    """A drone built partial by partial.

    ``partials`` is a sequence of ``(harmonic number, amplitude)``.
    Giving each partial its own slow amplitude modulation at an
    incommensurate rate is what stops an additive tone sounding like an
    organ: real sustained sources have partials that breathe
    independently.
    """
    rng = rng or np.random.default_rng(1)
    n = D.n_samples(dur)
    t = D.time_axis(n)
    if partials is None:
        partials = [(1, 1.0), (2, 0.42), (3, 0.26), (4, 0.16),
                    (5, 0.11), (6, 0.07), (8, 0.05), (9, 0.035),
                    (12, 0.02)]
    out = np.zeros(n, F64)
    for k, amp in partials:
        # stiff strings and gas columns alike stretch their upper
        # partials slightly sharp; a touch of it adds body
        fk = f0 * k * math.sqrt(1.0 + inharmonicity * k * k)
        lfo = 1.0 + shimmer * np.sin(
            t * TAU * (0.11 + 0.173 * k) + float(rng.random()) * TAU)
        wob = 1.0 + 0.25 * np.sin(
            t * TAU * (0.037 + 0.041 * k) + float(rng.random()) * TAU)
        out += np.sin(TAU * fk * t * lfo
                      + float(rng.random()) * TAU) * amp * (0.75 + 0.25 * wob)
    return out / max(sum(a for _, a in partials), 1e-9)


def choir(freqs, dur, rng=None, breath=0.22, vowel=(730.0, 1090.0, 2440.0)):
    """Formant-filtered saw stack — a synthetic vocal pad.

    Three resonant peaks at the formant frequencies of a vowel turn a
    buzzy sawtooth into something the ear classifies as a voice.  The
    default triple is the vowel in "father", which is the most open and
    therefore the most choir-like.
    """
    rng = rng or np.random.default_rng(2)
    n = D.n_samples(dur)
    out = np.zeros(n, F64)
    for f in freqs:
        v = D.saw(f * (1.0 + 0.0014 * (rng.random() - 0.5)), n,
                  harmonics=60, phase0=float(rng.random()) * TAU)
        # vibrato, delayed so the onset is straight
        t = D.time_axis(n)
        vib = 1.0 + 0.004 * np.sin(t * TAU * 5.1) * D.smoothstep(0.25, 1.1, t)
        v = D.saw(f * vib, n, harmonics=60,
                  phase0=float(rng.random()) * TAU) * 0.6 + v * 0.4
        out += v
    out /= max(len(freqs), 1)
    for i, fmt in enumerate(vowel):
        out = D.resonant_peak(out, fmt, q=7.0 - i, gain_db=11.0 - 2.0 * i)
    out = D.lowpass(out, 4200.0, order=3)
    if breath > 0.0:
        air = D.bandpass(D.white(n, rng), 1800.0, 7000.0, order=2)
        out += air * breath * 0.08
    return out


def glass_pad(freqs, dur, rng=None, brightness=1.0):
    """Bell-like sustained tone: FM with a low index and a long decay."""
    rng = rng or np.random.default_rng(3)
    n = D.n_samples(dur)
    t = D.time_axis(n)
    out = np.zeros(n, F64)
    for f in freqs:
        ratio = 2.01
        index = 1.6 * brightness * np.exp(-t / (dur * 0.42))
        mod = np.sin(TAU * f * ratio * t + float(rng.random()) * TAU)
        out += np.sin(TAU * f * t + index * mod)
        out += 0.33 * np.sin(TAU * f * 2.0 * t
                             + 0.8 * index * mod) * brightness
    return out / max(len(freqs) * 1.33, 1e-9)


# ==========================================================================
# transient voices
# ==========================================================================
def fm_bell(f, dur, ratio=3.51, index=6.0, rng=None, tau=None):
    """A struck bell.

    The inharmonic ratio is what makes it a bell rather than a note: a
    tubular bell's partials sit near 2, 3.01, 4.17 and 5.43 times the
    fundamental, and a modulator at an irrational-looking ratio
    reproduces that cluster in one oscillator pair.  The modulation index
    decays faster than the carrier, so the strike is bright and the tail
    is pure — exactly how a real bell behaves.
    """
    n = D.n_samples(dur)
    t = D.time_axis(n)
    tau = tau if tau is not None else dur * 0.30
    ei = index * np.exp(-t / max(tau * 0.30, 1e-4))
    ec = np.exp(-t / max(tau, 1e-4))
    mod = np.sin(TAU * f * ratio * t)
    y = np.sin(TAU * f * t + ei * mod) * ec
    y += 0.3 * np.sin(TAU * f * 2.0 * t + 0.5 * ei * mod) * ec ** 1.6
    atk = min(n, D.n_samples(0.004))
    if atk:
        y[:atk] *= np.linspace(0.0, 1.0, atk)
    return y


def sub_hit(f, dur=2.2, drop=0.42, rng=None):
    """A sub-bass impact with a pitch drop.

    Every cinematic impact is the same trick: a sine that falls about an
    octave in the first eighty milliseconds.  The drop is what the ear
    reads as mass.
    """
    n = D.n_samples(dur)
    t = D.time_axis(n)
    freq = f * (1.0 + drop * np.exp(-t / 0.055))
    y = np.sin(D.phase_from_freq(freq, n))
    y *= D.percussive(n, attack=0.003, tau=dur * 0.33)
    y += D.soft_clip(y, 1.6) * 0.22
    return y


def impact(dur=3.2, rng=None, bright=1.0, tune=55.0):
    """Broadband impact: noise burst, pitch drop and a resonant tail."""
    rng = rng or np.random.default_rng(4)
    n = D.n_samples(dur)
    t = D.time_axis(n)
    nz = D.white(n, rng)
    body = D.sweep_lowpass(nz, 220.0 + 5200.0 * bright * np.exp(-t / 0.09))
    body *= D.percussive(n, attack=0.001, tau=0.42)
    low = sub_hit(tune, dur, drop=0.55)
    ring = np.zeros(n, F64)
    for k, (mult, amp, tau) in enumerate((
            (1.0, 1.0, dur * 0.42), (1.48, 0.5, dur * 0.3),
            (2.37, 0.28, dur * 0.21), (3.71, 0.16, dur * 0.14))):
        ring += np.sin(TAU * tune * mult * t
                       + float(rng.random()) * TAU) * amp * np.exp(-t / tau)
    return D.normalize(body * 0.55 + low * 0.9 + ring * 0.35)


def riser(dur, f0=90.0, f1=4800.0, rng=None, noise_mix=0.55):
    """A rising tension sweep: noise through a climbing band-pass."""
    rng = rng or np.random.default_rng(5)
    n = D.n_samples(dur)
    u = np.linspace(0.0, 1.0, n)
    curve = u ** 2.1
    centre = f0 * (f1 / f0) ** curve
    nz = D.sweep_bandpass(D.white(n, rng), centre, q=2.6)
    tone = np.sin(D.phase_from_freq(centre * 0.5, n))
    tone += 0.5 * np.sin(D.phase_from_freq(centre * 0.75, n))
    y = nz * noise_mix + tone * (1.0 - noise_mix) * 0.5
    y *= (0.04 + 0.96 * u ** 1.7)
    return D.normalize(y)


def downlifter(dur, f0=3200.0, f1=60.0, rng=None):
    """A falling sweep, used to land a cut."""
    rng = rng or np.random.default_rng(6)
    n = D.n_samples(dur)
    u = np.linspace(0.0, 1.0, n)
    centre = f0 * (f1 / f0) ** (u ** 0.8)
    y = D.sweep_bandpass(D.white(n, rng), centre, q=3.2)
    y += np.sin(D.phase_from_freq(centre, n)) * 0.45
    y *= np.exp(-u * 2.2)
    return D.normalize(y)


def reverse_swell(dur, f=220.0, rng=None):
    """A reversed cymbal-like swell that resolves on the downbeat."""
    rng = rng or np.random.default_rng(7)
    n = D.n_samples(dur)
    nz = D.bandpass(D.white(n, rng), f, 11000.0, order=2)
    nz *= np.linspace(0.0, 1.0, n) ** 2.6
    nz[-D.n_samples(0.02):] *= np.linspace(1.0, 0.0, D.n_samples(0.02))
    return D.normalize(nz)


def tick(dur=0.06, f=2400.0, rng=None):
    """A short instrument click — the sound of the readout updating."""
    rng = rng or np.random.default_rng(8)
    n = D.n_samples(dur)
    y = D.bandpass(D.white(n, rng), f * 0.6, f * 2.4, order=2)
    y += np.sin(TAU * f * D.time_axis(n)) * 0.4
    return y * D.percussive(n, attack=0.0005, tau=dur * 0.25)


def pluck(f, dur=1.4, bright=1.0, rng=None, damping=0.48):
    """Karplus-Strong plucked string.

    A delay line of length ``SR/f`` filled with noise and filtered a
    little each time round converges on a harmonic series whose upper
    partials die first — which is the physics of a real string, where
    high modes lose energy to the bridge fastest.  The loop is over
    *periods*, not samples, so a 1.4-second note at 220 Hz costs three
    hundred vectorised iterations rather than sixty-seven thousand
    scalar ones.
    """
    rng = rng or np.random.default_rng(9)
    n = D.n_samples(dur)
    period = max(int(round(D.SR / max(f, 20.0))), 2)
    buf = rng.standard_normal(period)
    buf = D.lowpass(buf, 6000.0 * bright, order=1) if period > 16 else buf
    buf -= buf.mean()
    out = np.empty(((n // period) + 1) * period, F64)
    pos = 0
    cur = buf.copy()
    while pos < out.shape[0]:
        out[pos:pos + period] = cur
        # one pass of the loop filter: average each sample with its
        # neighbour, then scale
        cur = (cur + np.roll(cur, 1)) * 0.5 * (1.0 - damping * 0.02)
        pos += period
    y = out[:n]
    y *= np.exp(-D.time_axis(n) / (dur * 0.5))
    return D.normalize(y) * 0.9


# ==========================================================================
# textures
# ==========================================================================
def wind(dur, rng=None, lo=120.0, hi=2600.0, motion=0.25):
    """Filtered noise with a wandering band — interstellar hiss."""
    rng = rng or np.random.default_rng(10)
    n = D.n_samples(dur)
    t = D.time_axis(n)
    centre = np.exp(np.log(lo) + (np.log(hi) - np.log(lo))
                    * (0.5 + 0.5 * np.sin(t * TAU * motion * 0.37)
                       * np.sin(t * TAU * motion * 0.11 + 1.3)))
    y = D.sweep_bandpass(D.pink(n, rng), centre, q=1.1)
    return D.normalize(y)


def shimmer(dur, f0, rng=None, count=24, spread=3.0):
    """A cloud of high sine grains — starlight.

    Each grain is a short windowed sinusoid at a harmonic of ``f0``,
    scattered in time.  Granular synthesis in its simplest useful form.
    """
    rng = rng or np.random.default_rng(11)
    n = D.n_samples(dur)
    out = np.zeros(n, F64)
    for _ in range(count):
        gl = D.n_samples(rng.uniform(0.18, 0.75))
        if gl >= n:
            gl = n // 2
        start = int(rng.integers(0, max(n - gl, 1)))
        k = float(rng.choice([2, 3, 4, 5, 6, 8, 9, 12, 16]))
        f = f0 * k * (1.0 + rng.normal(0.0, 0.0015))
        if f > 0.45 * D.SR:
            continue
        g = np.sin(TAU * f * np.arange(gl) / D.SR + rng.random() * TAU)
        g *= np.hanning(gl) ** 1.5
        out[start:start + gl] += g * (spread / max(k, 1.0)) * rng.uniform(
            0.35, 1.0)
    return D.normalize(out)


def chirp_gw(dur, f_start, f_end, rng=None, merge_frac=0.86,
             ringdown_q=11.0):
    """A gravitational-wave inspiral, chirp and ringdown.

    For two bodies spiralling together under radiation reaction the
    gravitational-wave frequency obeys

        f(t) ∝ (t_coalesce - t)^(-3/8)

    — the canonical result of the quadrupole formula, and the reason the
    detected signals are called chirps.  Amplitude rises as ``f^(2/3)``.
    At merger the signal cuts off and is replaced by the ringdown: the
    remnant's least-damped quasi-normal mode, an exponentially decaying
    sinusoid whose quality factor depends on the final spin.

    Transposed into the audio band this is literally what LIGO heard,
    which is why it belongs in the final phase of the film rather than a
    synthesised approximation of it.
    """
    rng = rng or np.random.default_rng(12)
    n = D.n_samples(dur)
    nm = int(n * merge_frac)
    nr = n - nm
    t = np.arange(nm, dtype=F64) / D.SR
    tc = nm / D.SR * (1.0 + 1e-3)
    # f = A (tc - t)^(-3/8); fix A from the requested start frequency
    a = f_start * (tc ** 0.375)
    f = a * np.maximum(tc - t, 1e-6) ** (-0.375)
    f = np.minimum(f, f_end)
    insp = np.sin(D.phase_from_freq(f, nm))
    insp *= (f / max(f_start, 1e-6)) ** (2.0 / 3.0)
    insp /= max(float(np.abs(insp).max()), 1e-9)
    out = np.zeros(n, F64)
    out[:nm] = insp
    if nr > 2:
        tr = np.arange(nr, dtype=F64) / D.SR
        f_ring = float(f[-1]) * 1.05
        tau = ringdown_q / (math.pi * max(f_ring, 1.0))
        out[nm:] = np.sin(TAU * f_ring * tr) * np.exp(-tr / tau)
    # the second polarisation, ninety degrees out of phase, gives the
    # stereo image its rotation
    return out


def doppler_bed(dur, f0, beta_curve, rng=None, partials=6):
    """A drone transposed by the relativistic Doppler factor.

    For motion straight toward a source the observed frequency is
    multiplied by ``sqrt((1 + beta) / (1 - beta))``.  Feeding the same
    beta the camera uses in Phase IV means the music accelerates with the
    picture, and by exactly the right amount.
    """
    rng = rng or np.random.default_rng(13)
    n = D.n_samples(dur)
    beta = np.clip(np.asarray(beta_curve, F64), -0.999, 0.999)
    beta = np.interp(np.linspace(0.0, 1.0, n),
                     np.linspace(0.0, 1.0, beta.shape[0]), beta)
    shift = np.sqrt((1.0 + beta) / (1.0 - beta))
    out = np.zeros(n, F64)
    for k in range(1, partials + 1):
        amp = 1.0 / (k ** 1.35)
        out += np.sin(D.phase_from_freq(f0 * k * shift, n)
                      + float(rng.random()) * TAU) * amp
    return D.normalize(out)


__all__ = ["supersaw", "additive_drone", "choir", "glass_pad", "fm_bell",
           "sub_hit", "impact", "riser", "downlifter", "reverse_swell",
           "tick", "pluck", "wind", "shimmer", "chirp_gw", "doppler_bed"]
