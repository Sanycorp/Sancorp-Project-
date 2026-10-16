"""
cosmos.timeline
===============

Sequencing: which phase owns a given film time, how the junctions between
them are handled, and the instrument overlay drawn on top.

The five phases are rendered independently and joined by short fades
through black rather than by cross-dissolves.  That is a deliberate
choice and not a shortcut: a dissolve would require both phases to be
rendered for every overlapping frame, roughly doubling the cost of the
most expensive seconds in the film, and it would superimpose two entirely
different spatial scales — a galaxy and a planet — which reads as a
mistake rather than as a transition.  A fall to black between scales is
how every observational documentary does it, for the same reason.

The overlay is a simulation readout: the current phase, the physical scale
in force, and the quantity that phase is actually solving for.  It is part
of the film, not documentation — the brief asks for a sequence that shows
a simulation engine running, and a running engine has instruments.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

F32 = np.float32

_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src", "fonts")
_FONT_FALLBACK = "/usr/share/fonts/truetype/dejavu"


def _font(name, size):
    for base in (_FONT_DIR, _FONT_FALLBACK):
        path = os.path.join(base, name)
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


# ==========================================================================
# slate text
# ==========================================================================
@dataclass
class Slate:
    """The readout shown while a phase is on screen."""

    numeral: str
    title: str
    scale: str
    solving: str
    readout: tuple


SLATES = {
    "I · GALAKSI": Slate(
        "I", "GALAKSI",
        "1 unit = 1 kiloparsec · 3.086e19 m",
        "GELOMBANG KERAPATAN · ORBIT EPISIKLIK",
        ("v_flat 220 km/s", "Omega_p 0.0259 rad/Myr",
         "korotasi 8.7 kpc", "165 000 bintang")),
    "II · TATA SURYA": Slate(
        "II", "TATA SURYA",
        "1 unit = 1 satuan astronomi · 1.496e11 m",
        "GRAVITASI N-BENDA · VERLET SIMPLEKTIK",
        ("15 benda", "G = 4 pi^2 AU^3/M_sun/thn^2",
         "dt_maks 3e-5 thn", "dE/E < 5e-6")),
    "III · MATAHARI": Slate(
        "III", "MATAHARI",
        "1 unit = 1 jari-jari surya · 6.957e8 m",
        "TERMODINAMIKA PLASMA · REKONEKSI MAGNETIK",
        ("T_eff 5772 K", "korona 1.6e6 K",
         "rotasi diferensial 24.5-35 hari", "CME 0.62 R_sun/s")),
    "IV · TRANSISI": Slate(
        "IV", "TRANSISI",
        "heliopause 121 AU · menuju gradien ekstrem",
        "ABERASI RELATIVISTIK · PERGESERAN DOPPLER",
        ("kejut terminasi 94 AU", "beta -> 0.88",
         "gamma 2.11", "pancaran D^3")),
    "V · LUBANG HITAM": Slate(
        "V", "LUBANG HITAM",
        "1 unit = 1 jari-jari Schwarzschild · 2GM/c^2",
        "GEODESIK NOL SCHWARZSCHILD · RK4",
        ("bayangan b = 2.598 r_s", "cakram foton r = 1.5 r_s",
         "ISCO r = 3 r_s", "v_ISCO = 0.500 c")),
}


# ==========================================================================
# the overlay
# ==========================================================================
class Overlay:
    """Draws the instrument readout into an 8-bit RGB frame.

    PIL is used rather than hand-rolled glyph rasterisation because the
    text is composited once per frame over an already finished image, so
    its cost is negligible, and because kerned type is one of the few
    things worth not reinventing.
    """

    def __init__(self, width, height, accent=(255, 92, 48)):
        self.w = int(width)
        self.h = int(height)
        self.accent = tuple(accent)
        s = self.h / 1080.0
        self.s = s
        self.f_num = _font("BebasNeue.ttf", int(150 * s))
        self.f_title = _font("Anton.ttf", int(54 * s))
        self.f_label = _font("JetBrainsMono-Bold.ttf", int(17 * s))
        self.f_small = _font("JetBrainsMono-Regular.ttf", int(15 * s))
        self.f_tick = _font("JetBrainsMono-Regular.ttf", int(13 * s))

    # ----------------------------------------------------------------
    @staticmethod
    def _ease(x):
        x = min(max(x, 0.0), 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _alpha(self, local, duration, hold=3.6, fade=0.55):
        """Slate visibility: in at the top of a phase, out after a hold."""
        a = self._ease(local / fade)
        b = 1.0 - self._ease((local - hold) / fade)
        return max(0.0, min(a, b))

    # ----------------------------------------------------------------
    def draw(self, frame_u8, slate: Slate, local, duration, film_t,
             total, frame_index):
        a_slate = self._alpha(local, duration)
        img = Image.fromarray(frame_u8)
        d = ImageDraw.Draw(img, "RGBA")
        s = self.s
        w, h = self.w, self.h
        acc = self.accent

        # --- persistent frame furniture -------------------------------
        m = int(54 * s)
        ln = int(26 * s)
        col = (210, 216, 228, 54)
        for (x, y, dx, dy) in ((m, m, 1, 0), (m, m, 0, 1),
                               (w - m, m, -1, 0), (w - m, m, 0, 1),
                               (m, h - m, 1, 0), (m, h - m, 0, -1),
                               (w - m, h - m, -1, 0), (w - m, h - m, 0, -1)):
            d.line([(x, y), (x + dx * ln, y + dy * ln)], fill=col,
                   width=max(1, int(1.6 * s)))

        # --- running clock, bottom right ------------------------------
        tc = "%02d:%02d:%02d" % (int(film_t) // 60, int(film_t) % 60,
                                 int((film_t % 1.0) * 60))
        d.text((w - m - int(6 * s), h - m - int(26 * s)), tc,
               font=self.f_small, fill=(198, 206, 220, 150), anchor="rs")
        d.text((w - m - int(6 * s), h - m - int(6 * s)),
               "SANCORP · MESIN SIMULASI KOSMIK", font=self.f_tick,
               fill=(150, 158, 174, 120), anchor="rs")

        # --- progress rule --------------------------------------------
        y = h - m + int(10 * s)
        d.line([(m, y), (w - m, y)], fill=(150, 158, 174, 50),
               width=max(1, int(1.4 * s)))
        px = m + (w - 2 * m) * min(max(film_t / max(total, 1e-6), 0.0), 1.0)
        d.line([(m, y), (px, y)], fill=acc + (190,),
               width=max(1, int(1.4 * s)))

        # --- the slate -------------------------------------------------
        if a_slate > 0.004:
            al = int(255 * a_slate)
            x = m + int(10 * s)
            ybase = int(h * 0.60)
            # drifting entrance
            x -= int((1.0 - a_slate) * 26 * s)

            d.text((x, ybase), slate.numeral, font=self.f_num,
                   fill=acc + (int(al * 0.92),), anchor="ls")
            nw = d.textlength(slate.numeral, font=self.f_num)
            tx = x + nw + int(26 * s)
            d.line([(tx - int(13 * s), ybase - int(104 * s)),
                    (tx - int(13 * s), ybase + int(6 * s))],
                   fill=(210, 216, 228, int(al * 0.42)),
                   width=max(1, int(1.6 * s)))
            d.text((tx, ybase - int(54 * s)), slate.title,
                   font=self.f_title, fill=(238, 242, 250, al), anchor="ls")
            d.text((tx, ybase - int(26 * s)), slate.scale,
                   font=self.f_label, fill=(176, 186, 204, int(al * 0.90)),
                   anchor="ls")
            d.text((tx, ybase - int(4 * s)), slate.solving,
                   font=self.f_small, fill=acc + (int(al * 0.88),),
                   anchor="ls")

            # the numeric column, revealed line by line
            yy = ybase + int(34 * s)
            for i, line in enumerate(slate.readout):
                la = self._ease((local - 0.22 - i * 0.11) / 0.42) * a_slate
                if la <= 0.004:
                    continue
                d.line([(x, yy - int(5 * s)), (x + int(5 * s),
                                               yy - int(5 * s))],
                       fill=acc + (int(255 * la * 0.8),),
                       width=max(1, int(1.4 * s)))
                d.text((x + int(14 * s), yy), line, font=self.f_tick,
                       fill=(188, 196, 212, int(255 * la * 0.82)),
                       anchor="ls")
                yy += int(19 * s)

        return np.asarray(img, dtype=np.uint8)


# ==========================================================================
# the sequence
# ==========================================================================
@dataclass
class Segment:
    phase: object
    start: float
    duration: float
    fade_in: float = 0.42
    fade_out: float = 0.42

    @property
    def end(self):
        return self.start + self.duration


class Timeline:
    """Maps film time onto a phase, a local time and a fade multiplier."""

    def __init__(self, phases, gap=0.0):
        self.segments = []
        t = 0.0
        for i, ph in enumerate(phases):
            seg = Segment(ph, t, ph.duration,
                          fade_in=0.0 if i == 0 else 0.40,
                          fade_out=0.55)
            self.segments.append(seg)
            t += ph.duration + gap
        self.total = t - (gap if phases else 0.0)

    # ----------------------------------------------------------------
    def at(self, film_t):
        """Return ``(segment, local_time, fade)`` for a film time."""
        for seg in self.segments:
            if film_t < seg.end or seg is self.segments[-1]:
                local = min(max(film_t - seg.start, 0.0), seg.duration)
                fade = 1.0
                if seg.fade_in > 0.0 and local < seg.fade_in:
                    fade *= Overlay._ease(local / seg.fade_in)
                tail = seg.duration - local
                if seg.fade_out > 0.0 and tail < seg.fade_out:
                    fade *= Overlay._ease(tail / seg.fade_out)
                return seg, local, fade
        raise RuntimeError("timeline exhausted")

    def frames(self, fps):
        return int(round(self.total * fps))


def build_phases():
    """Instantiate the five phases in the order the brief demands."""
    from .phases.p1_galaxy import PhaseGalaxy
    from .phases.p2_solar import PhaseSolarSystem
    from .phases.p3_sun import PhaseSun
    from .phases.p4_transit import PhaseTransit
    from .phases.p5_blackhole import PhaseBlackHole
    return [PhaseGalaxy(), PhaseSolarSystem(), PhaseSun(),
            PhaseTransit(), PhaseBlackHole()]


#: volumes each phase needs, so a worker loads only what it will use
PHASE_VOLUMES = {
    "I · GALAKSI": ("nebula", "nebula_fine", "dust", "spiral", "filament"),
    "II · TATA SURYA": ("nebula_fine", "plasma", "granule", "disc_fine"),
    "III · MATAHARI": ("plasma", "plasma_fine", "corona", "granule",
                       "supergran"),
    "IV · TRANSISI": ("nebula", "nebula_fine", "filament", "dust"),
    "V · LUBANG HITAM": ("disc", "disc_fine", "plasma"),
}


def volumes_for(phases):
    names = []
    for ph in phases:
        for v in PHASE_VOLUMES.get(ph.name, ()):
            if v not in names:
                names.append(v)
    return names


__all__ = ["Timeline", "Segment", "Overlay", "Slate", "SLATES",
           "build_phases", "volumes_for", "PHASE_VOLUMES"]
