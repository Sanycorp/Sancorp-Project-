"""
cosmos.core.camera
==================

The virtual camera: ray generation, animated paths, and the handheld /
breathing motion that keeps a synthetic shot from looking locked off.

Ray generation
--------------
Rays are built from a pixel grid in normalised device coordinates, mapped
through the camera basis.  The grid is cached per (width, height, fov)
combination because it is identical on every frame of a shot and rebuilding
it costs more than marching several hundred rays.

Sub-pixel jitter is applied per frame from a rotated-grid offset sequence.
Across the film's 60 fps this acts as free temporal antialiasing: an edge
that would crawl at a fixed sample position instead dithers below the
threshold of perception.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import vmath as V

F32 = np.float32


# ==========================================================================
# ray grid cache
# ==========================================================================
_GRID_CACHE = {}


def _ndc_grid(width, height):
    """Cached pixel-centre grid in NDC, aspect-corrected."""
    key = (int(width), int(height))
    g = _GRID_CACHE.get(key)
    if g is None:
        aspect = width / float(height)
        xs = (np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0
        ys = 1.0 - (np.arange(height, dtype=np.float32) + 0.5) / height * 2.0
        gx, gy = np.meshgrid(xs * aspect, ys)
        g = (np.ascontiguousarray(gx.ravel()), np.ascontiguousarray(gy.ravel()))
        _GRID_CACHE[key] = g
    return g


# Rotated-grid 8-phase sub-pixel offsets.  A plain Halton sequence clusters
# visibly at low sample counts; this pattern is the one used for 8x MSAA and
# distributes far more evenly over a pixel footprint.
_JITTER = np.array([
    (0.0625, -0.1875), (-0.0625, 0.1875), (0.3125, 0.0625), (-0.1875, -0.3125),
    (-0.3125, 0.3125), (-0.4375, -0.0625), (0.1875, 0.4375), (0.4375, -0.4375),
], np.float32)


# ==========================================================================
# camera
# ==========================================================================
@dataclass
class Camera:
    """A perspective camera positioned by eye / target / up."""

    eye: np.ndarray = field(default_factory=lambda: np.array([0, 0, 5], F32))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, F32))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], F32))
    fov_deg: float = 50.0
    roll_deg: float = 0.0
    # physical lens properties, consumed by the post chain
    focal_distance: float = 0.0
    aperture: float = 0.0
    exposure: float = 1.0

    def basis(self):
        """Orthonormal (right, up, forward), with roll applied."""
        fwd = np.asarray(self.target, F32) - np.asarray(self.eye, F32)
        r, u, f = V.basis_from_forward(fwd, self.up)
        if abs(self.roll_deg) > 1e-6:
            m = V.rot_axis(f, math.radians(self.roll_deg))
            r = (m @ r).astype(F32)
            u = (m @ u).astype(F32)
        return r, u, f

    def generate_rays(self, width, height, jitter_phase=None, subset=None):
        """Build primary rays.

        Returns ``(origins, directions, index)``.  ``origins`` is a single
        broadcast row rather than an (N,3) copy where possible, because for a
        pinhole camera every ray shares an origin and materialising N copies
        of it wastes 25 MB a frame at 1080p.
        """
        gx, gy = _ndc_grid(width, height)
        if subset is not None:
            gx = gx[subset]
            gy = gy[subset]
            index = np.asarray(subset, np.int32)
        else:
            index = np.arange(gx.size, dtype=np.int32)

        if jitter_phase is not None:
            jx, jy = _JITTER[int(jitter_phase) % len(_JITTER)]
            aspect = width / float(height)
            gx = gx + jx * (2.0 * aspect / width)
            gy = gy + jy * (2.0 / height)

        scale = math.tan(math.radians(self.fov_deg) * 0.5)
        r, u, f = self.basis()
        d = (f[None, :]
             + r[None, :] * (gx[:, None] * scale)
             + u[None, :] * (gy[:, None] * scale))
        V.normalize_(d)
        o = np.broadcast_to(np.asarray(self.eye, F32), d.shape)
        return np.ascontiguousarray(o), d, index

    def project(self, points):
        """World -> pixel-space NDC, for placing sharp billboards and flares.

        Returns ``(ndc_x, ndc_y, depth, visible)``.  ``visible`` is false for
        anything behind the camera, which must be filtered or a star behind
        the eye will be mirrored into frame.
        """
        r, u, f = self.basis()
        rel = np.asarray(points, F32) - np.asarray(self.eye, F32)
        z = rel @ f
        x = rel @ r
        y = rel @ u
        scale = math.tan(math.radians(self.fov_deg) * 0.5)
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        return (x / (safe * scale), y / (safe * scale), z, z > 1e-6)

    def copy(self):
        return Camera(np.array(self.eye, F32), np.array(self.target, F32),
                      np.array(self.up, F32), self.fov_deg, self.roll_deg,
                      self.focal_distance, self.aperture, self.exposure)


# ==========================================================================
# animated camera rigs
# ==========================================================================
@dataclass
class CameraKey:
    t: float
    eye: tuple
    target: tuple
    fov: float = 50.0
    roll: float = 0.0
    up: tuple = (0.0, 1.0, 0.0)


class CameraTrack:
    """Keyframed camera with smooth interpolation between poses.

    Eye and target are interpolated on Catmull-Rom splines so the motion has
    continuous velocity through the keys; FOV and roll use a smootherstep
    ease, which has zero second derivative at the endpoints and therefore
    never produces the visible 'snap out' that a cubic ease leaves behind.
    """

    def __init__(self, keys):
        self.keys = sorted(keys, key=lambda k: k.t)
        if not self.keys:
            raise ValueError("CameraTrack needs at least one key")
        self._times = np.array([k.t for k in self.keys], np.float64)
        self._eyes = np.array([k.eye for k in self.keys], np.float64)
        self._targets = np.array([k.target for k in self.keys], np.float64)
        self._fovs = np.array([k.fov for k in self.keys], np.float64)
        self._rolls = np.array([k.roll for k in self.keys], np.float64)
        self._ups = np.array([k.up for k in self.keys], np.float64)

    def _segment(self, t):
        ts = self._times
        if t <= ts[0]:
            return 0, 0.0
        if t >= ts[-1]:
            return len(ts) - 2 if len(ts) > 1 else 0, 1.0
        i = int(np.searchsorted(ts, t) - 1)
        i = max(0, min(i, len(ts) - 2))
        span = max(ts[i + 1] - ts[i], 1e-9)
        return i, float((t - ts[i]) / span)

    @staticmethod
    def _spline(points, i, f):
        m = len(points)
        if m == 1:
            return points[0]

        def at(k):
            return points[max(0, min(k, m - 1))]

        p0, p1, p2, p3 = at(i - 1), at(i), at(i + 1), at(i + 2)
        f2 = f * f
        f3 = f2 * f
        return 0.5 * ((2.0 * p1)
                      + (-p0 + p2) * f
                      + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f2
                      + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f3)

    def at(self, t):
        i, f = self._segment(float(t))
        s = float(V.smootherstep(0.0, 1.0, f))
        eye = self._spline(self._eyes, i, s)
        tgt = self._spline(self._targets, i, s)
        j = min(i + 1, len(self._fovs) - 1)
        fov = self._fovs[i] + (self._fovs[j] - self._fovs[i]) * s
        roll = self._rolls[i] + (self._rolls[j] - self._rolls[i]) * s
        up = self._ups[i] + (self._ups[j] - self._ups[i]) * s
        return Camera(eye.astype(F32), tgt.astype(F32), up.astype(F32),
                      float(fov), float(roll))


# ==========================================================================
# procedural motion
# ==========================================================================
def handheld(t, amp_pos=1.0, amp_rot=1.0, seed=0.0):
    """Three-octave positional and rotational drift.

    Sums of incommensurate sines rather than filtered noise: the result is
    perfectly reproducible across worker processes rendering different frame
    ranges, which filtered noise with per-process state would not be.
    """
    s = float(seed)
    px = (math.sin(t * 1.70 + 1.3 + s) + math.sin(t * 3.13 + 0.4) * 0.45
          + math.sin(t * 5.90 + 2.2) * 0.22) * amp_pos
    py = (math.cos(t * 1.41 + 0.7 + s) + math.cos(t * 2.71 + 1.9) * 0.40
          + math.cos(t * 6.29 + 0.2) * 0.18) * amp_pos
    pz = (math.sin(t * 1.17 + 2.6 + s) + math.sin(t * 2.33 + 1.1) * 0.38) * amp_pos
    roll = (math.sin(t * 0.63 + 0.9 + s) * 0.7
            + math.sin(t * 1.51 + 2.4) * 0.3) * amp_rot
    return np.array([px, py, pz], F32), float(roll)


def breathe(t, amount=0.0035, rate=0.78, phase=0.0):
    """Slow FOV oscillation — the lens equivalent of a held breath."""
    return 1.0 + amount * math.sin(t * rate + phase)


def apply_handheld(cam, t, pos_amp=0.0, rot_amp=0.0, fov_breath=0.0, seed=0.0):
    """Return a copy of ``cam`` with procedural life added."""
    out = cam.copy()
    if pos_amp > 0.0 or rot_amp > 0.0:
        dp, droll = handheld(t, pos_amp, rot_amp, seed)
        out.eye = (np.asarray(out.eye, F32) + dp).astype(F32)
        out.roll_deg = float(out.roll_deg + droll)
    if fov_breath > 0.0:
        out.fov_deg = float(out.fov_deg * breathe(t, fov_breath))
    return out


def orbit(centre, radius, azimuth, elevation, up=(0.0, 1.0, 0.0)):
    """Build an eye position on a sphere around ``centre``."""
    c = np.asarray(centre, F32)
    ce = math.cos(elevation)
    eye = c + np.array([radius * ce * math.cos(azimuth),
                        radius * math.sin(elevation),
                        radius * ce * math.sin(azimuth)], F32)
    return Camera(eye, c.copy(), np.asarray(up, F32))


def dolly_toward(cam, distance):
    """Move the eye along its own forward axis without changing the target."""
    out = cam.copy()
    _, _, f = cam.basis()
    out.eye = (np.asarray(cam.eye, F32) + f * F32(distance)).astype(F32)
    return out


def log_approach(t, r_start, r_end):
    """Geometric interpolation of distance.

    Flying from a galaxy to a star spans eight orders of magnitude.  Linear
    interpolation of radius would spend 99.99 percent of the shot in the
    first decade and then cross the remaining seven in the final frames.
    Interpolating the logarithm makes the apparent rate of approach constant,
    which is what reads as a smooth continuous zoom.
    """
    t = float(np.clip(t, 0.0, 1.0))
    return float(math.exp(math.log(max(r_start, 1e-12)) * (1.0 - t)
                          + math.log(max(r_end, 1e-12)) * t))


__all__ = ["Camera", "CameraKey", "CameraTrack", "handheld", "breathe",
           "apply_handheld", "orbit", "dolly_toward", "log_approach"]
