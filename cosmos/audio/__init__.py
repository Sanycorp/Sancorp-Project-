"""
cosmos.audio
============

The score for the cosmic sequence: a synthesiser, a set of instruments
and a composition, all of it generated from the same physical constants
the picture is generated from.

Three modules:

``dsp``
    Signal processing — oscillators, filters, envelopes, delay, reverb,
    dynamics.  Everything is vectorised; there is not a single
    per-sample loop in the package.

``voices``
    Instruments.  Each returns a mono array and owns its envelope.

``score``
    The composition itself, cut into five movements matching the five
    phases, plus the master chain and the WAV writer.

Typical use::

    from cosmos.audio import score
    mix = score.build()
    score.write_wav("out/kosmik_score.wav", mix)
"""
from __future__ import annotations

from . import dsp
from . import score
from . import voices

SR = dsp.SR


def build_score(seed=20261018, total=score.TOTAL):
    """Render the score and return interleaved stereo float64."""
    return score.build(seed=seed, total=total)


def render_to(path, seed=20261018, total=score.TOTAL):
    """Render the score and write it to ``path`` as 16-bit PCM WAV."""
    return score.write_wav(path, score.build(seed=seed, total=total))


__all__ = ["dsp", "voices", "score", "SR", "build_score", "render_to"]
