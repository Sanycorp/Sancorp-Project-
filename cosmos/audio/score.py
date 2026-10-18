"""
cosmos.audio.score
==================

The score.

Five movements, cut to the five phases, with no cross-fades — the music
cuts where the picture cuts.

Everything that can be derived from the simulation is derived from it
rather than invented.  That is not a conceit: a film about a physical
model should sound like the model, and the constants involved happen to
be unusually generous.

**The tuning comes from the Sun.**  The solar surface rings with acoustic
modes trapped below the photosphere, with the power concentrated in a
broad band around 3.3 millihertz — the famous five-minute oscillation.
Raised seventeen octaves, ``0.0033 Hz * 2^17``, that band lands on
**432.5 Hz**, a few cents above concert A.  Every pitch in the score is
referred to it.

**The second movement is an orrery.**  Each planet is given a tone at
its orbital frequency, folded up into the audible range by octaves, and
repeats at a rate proportional to that same frequency.  Mercury is a
fast ostinato, Neptune a single sustained pedal, and the intervals
between them are the real period ratios — including the 5:2 near
resonance between Jupiter and Saturn that perturbs both their orbits and
that Laplace spent twenty years explaining.

**The fourth movement is Doppler-shifted by the camera's own velocity.**
Phase IV accelerates the viewpoint to 0.88 c; the drone underneath it is
multiplied by ``sqrt((1 + beta) / (1 - beta))``, the relativistic
longitudinal Doppler factor, using the same beta curve the renderer
uses.  The music rises two octaves because the picture says it must.

**The fifth movement is a gravitational-wave chirp.**  Frequency sweeps
as ``(t_c - t)^(-3/8)``, amplitude as ``f^(2/3)``, and the whole thing
terminates in the ringdown of the remnant's least-damped quasi-normal
mode.  This is the waveform LIGO recorded, transposed, not an impression
of it.
"""
from __future__ import annotations

import math

import numpy as np

from . import dsp as D
from . import voices as V

F64 = np.float64
TAU = 2.0 * math.pi

# ==========================================================================
# tuning
# ==========================================================================
#: Peak of the solar p-mode power spectrum, in hertz.
SOLAR_NU_MAX = 3.30e-3

#: Seventeen octaves up: the reference pitch of the whole score.
A_REF = SOLAR_NU_MAX * (2.0 ** 17)        # 432.5 Hz

#: Semitone offsets from A within an octave, for name lookup.
_NAMES = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4,
          "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}


def hz(name):
    """Pitch name to frequency, referred to the solar reference pitch.

    ``"D3"`` is the D in the octave below middle C, following scientific
    pitch notation in which the reference is A4.
    """
    i = 1
    while i < len(name) and not (name[i].isdigit() or name[i] == "-"):
        i += 1
    letter = name[:i]
    octave = int(name[i:])
    semis = _NAMES[letter] + (octave - 4) * 12
    return A_REF * (2.0 ** (semis / 12.0))


# ==========================================================================
# orbital data for the second movement
# ==========================================================================
#: Sidereal orbital periods in Julian years.
PERIODS = {
    "Mercury": 0.2408467,
    "Venus": 0.6151973,
    "Earth": 1.0000174,
    "Mars": 1.8808476,
    "Jupiter": 11.862615,
    "Saturn": 29.447498,
    "Uranus": 84.016846,
    "Neptune": 164.79132,
}


def fold_to_octave(ratio, lo=1.0):
    """Multiply or divide by two until the ratio lies in ``[lo, 2 lo)``."""
    r = float(ratio)
    while r < lo:
        r *= 2.0
    while r >= 2.0 * lo:
        r *= 0.5
    return r


def planet_pitches(base_hz):
    """Audible pitch for each planet, folded from its orbital frequency.

    The orbital frequency ``1/T`` is multiplied by whatever power of two
    brings it into a single octave, then scaled to ``base_hz``.  Octave
    equivalence is the one transformation that leaves a frequency ratio
    musically recognisable, so the intervals that come out are the real
    ratios between the planets' years.
    """
    out = {}
    for name, t in PERIODS.items():
        out[name] = base_hz * fold_to_octave(1.0 / t)
    return out


# ==========================================================================
# movement timing
# ==========================================================================
MOVEMENTS = (
    ("I", 0.00, 11.00),
    ("II", 11.00, 22.00),
    ("III", 22.00, 33.00),
    ("IV", 33.00, 42.00),
    ("V", 42.00, 54.00),
)
TOTAL = 54.00


class Buses:
    """Four parallel mixes, each with its own reverb character.

    Separating them is what keeps a dense score legible: the bass must
    stay dry or the low end turns to mud, while the pads want a long
    tail, and a single global reverb cannot do both.
    """

    def __init__(self, total, rng):
        n = D.n_samples(total)
        self.n = n
        self.pad = np.zeros(n, F64)
        self.bass = np.zeros(n, F64)
        self.perc = np.zeros(n, F64)
        self.texture = np.zeros(n, F64)
        self.wide = np.zeros((n, 2), F64)
        self.rng = rng

    def mixdown(self):
        rng = self.rng
        long_ir = D.reverb_ir(4.5, rng, decay=4.6, lo=70.0, hi=7200.0,
                              predelay=0.035)
        short_ir = D.reverb_ir(1.6, rng, decay=1.5, lo=140.0, hi=9000.0,
                               predelay=0.012)

        pad = D.convolve_stereo(self.pad, long_ir, mix=0.42)
        pad = D.widen(pad, 1.25)

        tex = D.convolve_stereo(self.texture, long_ir, mix=0.55)
        tex = D.widen(tex, 1.45)

        perc = D.convolve_stereo(self.perc, short_ir, mix=0.26)

        # The bass stays mono and dry.  Below about 120 Hz the ear cannot
        # localise anyway, and any stereo content down there only steals
        # headroom from the loudest part of the spectrum.
        bass = D.lowpass(self.bass, 240.0, order=2)
        bass = np.stack([bass, bass], axis=1)

        mix = pad * 0.92 + tex * 0.72 + perc * 0.95 + bass * 1.0 \
            + self.wide
        return mix


# ==========================================================================
# movement I — galaxy
# ==========================================================================
def movement_galaxy(bus, t0, t1, rng):
    """Vast and slow: a minor pad, a sub pedal and a field of grains."""
    dur = t1 - t0
    n = D.n_samples(dur)

    # --- the sub pedal -------------------------------------------------
    root = hz("D1")
    sub = V.additive_drone(root, dur + 1.0, rng=rng,
                           partials=[(1, 1.0), (2, 0.30), (3, 0.10),
                                     (4, 0.05)])
    sub *= D.swell(sub.shape[0], peak=0.45, curve=1.1)
    D.add_at(bus.bass, sub, t0, 0.46)

    # --- the chord -----------------------------------------------------
    # D minor with an added ninth, voiced wide: the ninth is what makes a
    # minor triad sound like distance rather than sadness.
    voicing_a = [hz("D2"), hz("A2"), hz("D3"), hz("F3"), hz("A3")]
    voicing_b = voicing_a + [hz("E4"), hz("A4")]

    pad = V.choir(voicing_a, dur, rng=rng, breath=0.30)
    pad *= D.swell(pad.shape[0], peak=0.55, curve=1.4)
    D.add_at(bus.pad, pad, t0, 0.30)

    late = V.glass_pad(voicing_b, dur * 0.62, rng=rng, brightness=0.9)
    late *= D.swell(late.shape[0], peak=0.42, curve=1.8)
    D.add_at(bus.pad, late, t0 + dur * 0.36, 0.17)

    sweep = np.exp(np.linspace(math.log(320.0), math.log(3400.0), n))
    saws = V.supersaw([hz("D2")] * 1, dur, voices=7, detune=0.20, rng=rng,
                      cutoff=sweep)
    saws *= D.swell(saws.shape[0], peak=0.7, curve=2.0)
    D.add_at(bus.pad, saws, t0, 0.11)

    # --- opening strike -------------------------------------------------
    D.add_at(bus.perc, V.impact(4.0, rng=rng, bright=0.8,
                                tune=hz("D1")), t0 + 0.08, 0.52)
    D.add_at(bus.perc, V.reverse_swell(1.6, rng=rng), t0 + 0.0, 0.10)

    # --- stars ----------------------------------------------------------
    grains = V.shimmer(dur, hz("D4"), rng=rng, count=46, spread=2.4)
    D.add_at(bus.texture, grains, t0, 0.17)

    # a bell on the second and third phrase marks, which are placed at
    # the golden section of the movement and of its remainder
    g = 0.6180339887
    for k, (at, f, amp) in enumerate((
            (dur * (1.0 - g), hz("A4"), 0.17),
            (dur * (1.0 - g * g), hz("D5"), 0.12),
            (dur * 0.90, hz("F4"), 0.09))):
        D.add_at(bus.perc, V.fm_bell(f, 3.4, ratio=3.51, index=5.0),
                 t0 + at, amp)

    # --- interstellar hiss ----------------------------------------------
    air = V.wind(dur, rng=rng, lo=240.0, hi=3400.0, motion=0.2)
    air *= D.swell(air.shape[0], peak=0.5, curve=1.0)
    D.add_at(bus.texture, air, t0, 0.055)


# ==========================================================================
# movement II — the solar system
# ==========================================================================
def movement_orrery(bus, t0, t1, rng):
    """An audible orrery: one voice per planet, each at its own period."""
    dur = t1 - t0

    base = hz("D3")
    pitches = planet_pitches(base)

    # One Earth year lasts this many seconds of film.  Everything else
    # follows from the real period ratios, so Mercury runs four times as
    # fast as Earth because it does.
    year = 1.10

    plan = (
        # name,        gain,  bright, pan
        ("Mercury", 0.150, 1.30, 0.62),
        ("Venus", 0.195, 1.05, -0.45),
        ("Earth", 0.240, 0.95, 0.18),
        ("Mars", 0.190, 0.85, -0.68),
        ("Jupiter", 0.270, 0.60, 0.30),
        ("Saturn", 0.240, 0.50, -0.25),
    )
    for name, gain, bright, p in plan:
        period = PERIODS[name] / PERIODS["Earth"] * year
        f = pitches[name]
        # keep the fast inner planets from sitting on top of the pad
        while f > 1400.0:
            f *= 0.5
        while f < 70.0:
            f *= 2.0
        t = 0.0
        k = 0
        while t < dur:
            note = min(period * 0.92, 2.6)
            if period > dur * 0.6:
                note = dur - t
            y = V.pluck(f, max(note, 0.18), bright=bright, rng=rng)
            # every fourth note of the inner planets is accented, which
            # gives the ostinato a metre without imposing one
            acc = 1.0 + 0.35 * (k % 4 == 0)
            st = D.pan(y, p)
            D.add_at(bus.wide, st, t0 + t, gain * acc)
            t += period
            k += 1

    # Uranus and Neptune are too slow to articulate: they become the
    # pedal the rest of the movement sits on.
    for name, gain in (("Uranus", 0.20), ("Neptune", 0.26)):
        f = pitches[name]
        while f > 170.0:
            f *= 0.5
        y = V.additive_drone(f, dur, rng=rng,
                             partials=[(1, 1.0), (2, 0.34), (3, 0.14),
                                       (5, 0.06)])
        y *= D.swell(y.shape[0], peak=0.5, curve=1.0)
        D.add_at(bus.bass, y, t0, gain)

    # --- harmonic bed ----------------------------------------------------
    pad = V.choir([hz("D2"), hz("A2"), hz("D3"), hz("E3"), hz("A3")],
                  dur, rng=rng, breath=0.18,
                  vowel=(520.0, 1190.0, 2560.0))
    pad *= D.swell(pad.shape[0], peak=0.6, curve=1.2)
    D.add_at(bus.pad, pad, t0, 0.30)

    # --- instrument ticks -------------------------------------------------
    # the readout updating: one click per simulated Earth month
    month = year / 12.0
    t = 0.0
    while t < dur:
        D.add_at(bus.perc, V.tick(0.05, 3100.0, rng=rng), t0 + t, 0.048)
        t += month

    D.add_at(bus.perc, V.impact(2.6, rng=rng, bright=1.0, tune=hz("D1")),
             t0, 0.42)


# ==========================================================================
# movement III — the Sun
# ==========================================================================
def movement_sun(bus, t0, t1, rng):
    """Mass and heat: the solar acoustic spectrum, plus flares."""
    dur = t1 - t0
    n = D.n_samples(dur)

    # --- the p-mode cluster ----------------------------------------------
    # The Sun's oscillation power is spread over a band roughly 2.5 to
    # 4.5 mHz wide, split into a comb of modes separated by the large
    # frequency spacing, about 135 microhertz.  Transposed up seventeen
    # octaves that comb becomes a cluster of partials around 430 Hz
    # separated by about 18 Hz, which is exactly the spacing that
    # produces audible beating rather than a chord.
    shift = 2.0 ** 17
    nu0 = 2.55e-3
    dnu = 134.9e-6
    cluster = np.zeros(n, F64)
    t = D.time_axis(n)
    k = 0
    while nu0 + k * dnu < 4.6e-3:
        f = (nu0 + k * dnu) * shift
        # the envelope of solar power is a broad Gaussian about nu_max
        w = math.exp(-(((nu0 + k * dnu) - SOLAR_NU_MAX) / 0.85e-3) ** 2)
        amp = w / (1.0 + 0.4 * k)
        lfo = 1.0 + 0.09 * np.sin(t * TAU * (0.21 + 0.07 * k)
                                  + rng.random() * TAU)
        cluster += np.sin(TAU * f * t + rng.random() * TAU) * amp * lfo
        k += 1
    cluster = D.normalize(cluster)
    cluster *= D.swell(n, peak=0.42, curve=1.1)
    D.add_at(bus.pad, cluster, t0, 0.17)

    # --- the body of the star --------------------------------------------
    sweep = np.exp(np.linspace(math.log(220.0), math.log(2600.0), n))
    body = V.supersaw([hz("D1")], dur, voices=9, detune=0.26, rng=rng,
                      cutoff=sweep, res=0.22)
    body *= D.swell(n, peak=0.55, curve=1.3)
    D.add_at(bus.bass, D.lowpass(body, 420.0, order=2), t0, 0.38)
    D.add_at(bus.pad, D.highpass(body, 420.0, order=2), t0, 0.13)

    # a rising fifth in the brass register as the camera dives
    horn = V.supersaw([hz("A2")], dur * 0.55, voices=5, detune=0.10,
                      rng=rng,
                      cutoff=np.exp(np.linspace(math.log(400.0),
                                                math.log(3200.0),
                                                D.n_samples(dur * 0.55))))
    horn *= D.swell(horn.shape[0], peak=0.65, curve=1.6)
    D.add_at(bus.pad, horn, t0 + dur * 0.30, 0.16)

    # --- flares ------------------------------------------------------------
    # Six of them, matching the six fast-rise exponential-decay events in
    # the picture, and shaped the same way: a sharp onset and a long tail.
    for at, amp, f in ((0.14, 0.26, 1.0), (0.26, 0.17, 1.5),
                       (0.41, 0.30, 0.75), (0.58, 0.21, 1.25),
                       (0.72, 0.36, 0.6), (0.88, 0.30, 0.9)):
        y = V.impact(2.8, rng=rng, bright=1.3 * f, tune=hz("D1") * f)
        D.add_at(bus.perc, y, t0 + dur * at, amp)
        D.add_at(bus.texture,
                 V.riser(min(1.2, dur * at + 0.3), rng=rng, f0=300.0,
                         f1=5200.0 * f),
                 t0 + dur * at - min(1.2, dur * at + 0.3), 0.09)

    # --- the mass ejection -------------------------------------------------
    cme = V.downlifter(3.0, f0=5200.0, f1=48.0, rng=rng)
    D.add_at(bus.texture, cme, t0 + dur * 0.80, 0.26)
    D.add_at(bus.bass, V.sub_hit(hz("D0") * 2.0, 4.0, drop=0.62),
             t0 + dur * 0.82, 0.42)

    # --- plasma hiss --------------------------------------------------------
    hiss = V.wind(dur, rng=rng, lo=600.0, hi=8200.0, motion=0.55)
    hiss *= 0.35 + 0.65 * D.swell(n, peak=0.7, curve=1.0)
    D.add_at(bus.texture, hiss, t0, 0.075)


# ==========================================================================
# movement IV — the transition
# ==========================================================================
def movement_transit(bus, t0, t1, rng):
    """Acceleration to 0.88 c, with the music Doppler-shifted to match."""
    dur = t1 - t0
    n = D.n_samples(dur)

    # the same curve the renderer uses to drive the aberration
    u = np.linspace(0.0, 1.0, 512)
    beta = 0.88 * (u * u * (3.0 - 2.0 * u)) ** 1.15

    bed = V.doppler_bed(dur, hz("D1"), beta, rng=rng, partials=7)
    bed *= D.swell(n, peak=0.72, curve=1.0)
    D.add_at(bus.bass, D.lowpass(bed, 700.0, order=2), t0, 0.34)
    D.add_at(bus.pad, D.highpass(bed, 700.0, order=2), t0, 0.14)

    # a second bed an octave up, panned opposite, so the shift is audible
    # as motion across the stereo field as well as upward in pitch
    bed2 = V.doppler_bed(dur, hz("A1"), beta, rng=rng, partials=5)
    bed2 *= D.swell(n, peak=0.78, curve=1.4)
    D.add_at(bus.wide, D.haas(bed2, ms=17.0), t0, 0.10)

    # --- the accelerating pulse ----------------------------------------
    # Interval contracts as 1/gamma, which is the time dilation the
    # camera would actually experience: the clock ticks faster in the
    # frame the music is written in.
    t = 0.0
    k = 0
    while t < dur - 0.05:
        b = float(np.interp(t / dur, u, beta))
        gamma = 1.0 / math.sqrt(max(1.0 - b * b, 1e-6))
        interval = 0.62 / gamma
        f = 1800.0 * math.sqrt((1.0 + b) / (1.0 - b))
        D.add_at(bus.perc, V.tick(0.05, min(f, 9000.0), rng=rng),
                 t0 + t, 0.085 * (0.5 + 0.5 * b))
        t += max(interval, 0.035)
        k += 1

    # --- the long riser ---------------------------------------------------
    rise = V.riser(dur, f0=110.0, f1=7400.0, rng=rng, noise_mix=0.62)
    D.add_at(bus.texture, rise, t0, 0.19)

    # --- shock crossings ---------------------------------------------------
    # three boundaries: termination shock, heliopause, bow wave
    for at, amp in ((0.22, 0.26), (0.52, 0.34), (0.80, 0.42)):
        D.add_at(bus.perc,
                 V.impact(2.4, rng=rng, bright=1.5, tune=hz("A0") * 2.0),
                 t0 + dur * at, amp)
        D.add_at(bus.texture, V.reverse_swell(0.9, rng=rng),
                 t0 + dur * at - 0.9, 0.12)

    D.add_at(bus.texture, V.wind(dur, rng=rng, lo=900.0, hi=11000.0,
                                 motion=0.9), t0, 0.065)


# ==========================================================================
# movement V — the black hole
# ==========================================================================
def movement_black_hole(bus, t0, t1, rng):
    """Inspiral, merger, ringdown, and then almost nothing."""
    dur = t1 - t0
    n = D.n_samples(dur)

    # --- the chirp --------------------------------------------------------
    chirp_dur = dur * 0.60
    ch = V.chirp_gw(chirp_dur, f_start=hz("D1"), f_end=hz("A4"), rng=rng,
                    merge_frac=0.88, ringdown_q=11.0)
    # the two polarisations are ninety degrees apart, which is what makes
    # a real detection rotate through the detector network; here it gives
    # the sound a slow circular motion between the speakers
    ch2 = np.concatenate([np.zeros(D.n_samples(0.004), F64), ch])[:ch.shape[0]]
    D.add_at(bus.wide, np.stack([ch, ch2], axis=1), t0 + dur * 0.16, 0.26)
    D.add_at(bus.bass, D.lowpass(ch, 180.0, order=2), t0 + dur * 0.16, 0.30)

    # --- the hole itself ---------------------------------------------------
    # Two octaves below the rest of the score and mostly felt rather than
    # heard, which is the point.
    sub = V.additive_drone(hz("D0"), dur, rng=rng,
                           partials=[(1, 1.0), (2, 0.42), (3, 0.17),
                                     (4, 0.08), (6, 0.04)])
    sub *= D.swell(n, peak=0.52, curve=0.9)
    D.add_at(bus.bass, sub, t0, 0.50)

    # --- the accretion disc ------------------------------------------------
    # A bright, unstable cluster that beats against itself, detuned by
    # the same ratio the Doppler beaming imposes on the disc: the
    # approaching limb against the receding one.
    g_hi, g_lo = 1.86, 0.38
    disc = np.zeros(n, F64)
    t = D.time_axis(n)
    for k, amp in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.18), (6, 0.1)):
        f = hz("D3") * k
        disc += np.sin(TAU * f * g_lo * 2.0 * t + rng.random() * TAU) * amp
        disc += np.sin(TAU * f * g_hi * 0.5 * t + rng.random() * TAU) * amp
    disc = D.normalize(disc)
    disc *= D.swell(n, peak=0.62, curve=1.5)
    D.add_at(bus.pad, D.lowpass(disc, 3200.0, order=3), t0, 0.15)

    # --- the choir ---------------------------------------------------------
    voicing = [hz("D2"), hz("A2"), hz("D3"), hz("F3"), hz("C4"), hz("D4")]
    pad = V.choir(voicing, dur * 0.82, rng=rng, breath=0.26)
    pad *= D.swell(pad.shape[0], peak=0.5, curve=1.3)
    D.add_at(bus.pad, pad, t0, 0.26)

    # --- the merger --------------------------------------------------------
    merge_at = dur * 0.16 + chirp_dur * 0.88
    D.add_at(bus.perc, V.impact(5.0, rng=rng, bright=1.6, tune=hz("D0")),
             t0 + merge_at, 0.68)
    D.add_at(bus.bass, V.sub_hit(hz("D0"), 6.0, drop=0.70),
             t0 + merge_at, 0.60)
    D.add_at(bus.texture, V.riser(2.4, rng=rng, f0=200.0, f1=9000.0),
             t0 + merge_at - 2.4, 0.22)

    # --- the horizon -------------------------------------------------------
    # After the merger the score empties out: one bell, one drone, and
    # the reverb tail.  Nothing escapes.
    tail_at = merge_at + 0.5
    tail = max(dur - tail_at, 0.5)
    D.add_at(bus.perc, V.fm_bell(hz("D4"), min(tail, 6.0), ratio=3.51,
                                 index=4.0), t0 + tail_at, 0.16)
    D.add_at(bus.perc, V.fm_bell(hz("A3"), min(tail, 6.0), ratio=2.76,
                                 index=3.2), t0 + tail_at + 0.4, 0.11)
    last = V.additive_drone(hz("D1"), tail, rng=rng,
                            partials=[(1, 1.0), (3, 0.2), (5, 0.07)])
    last *= np.linspace(1.0, 0.0, last.shape[0]) ** 1.5
    D.add_at(bus.pad, last, t0 + tail_at, 0.22)

    D.add_at(bus.texture, V.shimmer(dur, hz("D5"), rng=rng, count=30,
                                    spread=1.8), t0, 0.09)


# ==========================================================================
# assembly
# ==========================================================================
def build(seed=20261018, total=TOTAL):
    """Render the complete score and return interleaved stereo float64."""
    rng = np.random.default_rng(seed)
    bus = Buses(total, rng)

    movement_galaxy(bus, 0.00, 11.00, rng)
    movement_orrery(bus, 11.00, 22.00, rng)
    movement_sun(bus, 22.00, 33.00, rng)
    movement_transit(bus, 33.00, 42.00, rng)
    movement_black_hole(bus, 42.00, 54.00, rng)

    mix = bus.mixdown()

    # --- master chain -------------------------------------------------
    # A gentle high-pass clears the inaudible rumble that the sub drones
    # put below twenty hertz, where it does nothing but eat headroom.
    for ch in range(2):
        mix[:, ch] = D.highpass(mix[:, ch], 22.0, order=2)
    mix = D.compress(mix, threshold=0.30, ratio=2.6, attack=0.012,
                     release=0.22)
    # a broad tilt, a little darker than flat: the picture is mostly
    # black and a bright mix fights it
    for ch in range(2):
        mix[:, ch] = D.tilt(mix[:, ch], -0.45, pivot=900.0)
        mix[:, ch] = D.resonant_peak(mix[:, ch], 62.0, q=1.1, gain_db=2.4)
    mix = D.soft_clip(mix, 0.90) * 0.97
    mix = D.limit(mix, ceiling=0.955, lookahead=0.005, smooth=0.015)

    D.fade_in(mix, 0.40)
    D.fade_out(mix, 0.55)

    n = D.n_samples(total)
    if mix.shape[0] < n:
        mix = np.concatenate([mix, np.zeros((n - mix.shape[0], 2), F64)])
    return mix[:n]


def write_wav(path, data, bits=16):
    """Write a 16-bit PCM WAV without touching the standard library."""
    import struct
    x = np.clip(np.asarray(data, F64), -1.0, 1.0)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    pcm = (x * 32767.0).astype("<i2")
    raw = pcm.tobytes()
    ch = x.shape[1]
    byte_rate = D.SR * ch * 2
    hdr = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, D.SR, byte_rate,
                                 ch * 2, 16)
    hdr += b"data" + struct.pack("<I", len(raw))
    with open(path, "wb") as fh:
        fh.write(hdr)
        fh.write(raw)
    return path


def _report(mix):
    peak = float(np.max(np.abs(mix)))
    rms = float(np.sqrt(np.mean(mix ** 2)))
    print("  durasi      %.3f s" % (mix.shape[0] / D.SR))
    print("  puncak      %.4f" % peak)
    print("  RMS         %.4f  (%.1f dBFS)"
          % (rms, 20.0 * math.log10(max(rms, 1e-9))))
    print("  nada acuan  %.3f Hz  (p-mode surya x 2^17)" % A_REF)
    for name in ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"):
        print("  %-8s %8.3f Hz" % (name, planet_pitches(hz("D3"))[name]))
    return peak, rms


if __name__ == "__main__":
    import os
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "out", "kosmik_score.wav")
    print("membangun skor ...", flush=True)
    mix = build()
    _report(mix)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_wav(out, mix)
    print("ditulis: %s" % out)


__all__ = ["A_REF", "SOLAR_NU_MAX", "PERIODS", "hz", "fold_to_octave",
           "planet_pitches", "MOVEMENTS", "TOTAL", "Buses", "build",
           "write_wav", "movement_galaxy", "movement_orrery",
           "movement_sun", "movement_transit", "movement_black_hole"]
