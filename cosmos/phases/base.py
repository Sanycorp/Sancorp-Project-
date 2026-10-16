"""
cosmos.phases.base
==================

Shared scaffolding for the five phases: the render context, the phase
protocol, and the point-splatting rasteriser that draws stars.

Why stars are not volumetric
----------------------------
A star subtends far less than a pixel.  Marching a volume to find it would
need a step size smaller than the star, which is hopeless — the sampling
would miss almost every star and the ones it hit would alias violently.
Stars are therefore *projected and splatted*: each is transformed to screen
space analytically and accumulated into a full-resolution buffer with a
sub-pixel point-spread function.  That gives exact positions, correct
parallax, no aliasing, and costs a few milliseconds for a quarter million
stars.

The splat uses ``np.bincount`` rather than ``np.add.at``.  Both scatter with
accumulation, but ``add.at`` falls back to an un-vectorised loop and is
roughly forty times slower at these counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import spectrum as S
from ..core import vmath as V

F32 = np.float32


# ==========================================================================
# render context
# ==========================================================================
@dataclass
class RenderContext:
    """Everything a phase needs that is not phase-specific."""

    width: int = 1920
    height: int = 1080
    march_width: int = 854
    march_height: int = 480
    fps: int = 60
    steps: int = 56
    seed: int = 20261016
    lib: object = None                      # QuantLibrary
    rng: object = None                      # numpy Generator, per frame
    frame: int = 0
    time: float = 0.0
    quality: float = 1.0                    # global multiplier on step counts

    def march_steps(self, base=None):
        return max(8, int(round((base or self.steps) * self.quality)))

    @property
    def march_pixels(self):
        return self.march_width * self.march_height

    @property
    def aspect(self):
        return self.width / float(self.height)


class Phase:
    """Base class for a timeline segment.

    ``render`` returns a linear-HDR ``(height, width, 3)`` float32 image.
    Post-processing, grading and encoding happen once, globally, downstream —
    phases never tonemap.
    """

    name = "phase"
    duration = 10.0

    def setup(self, ctx):
        """Build and cache anything expensive.  Called once per process."""

    def camera_at(self, ctx, lt):
        raise NotImplementedError

    def render(self, ctx, lt):
        raise NotImplementedError

    def exposure_at(self, ctx, lt):
        """Scene-referred exposure multiplier for this moment.

        A camera flying from intergalactic space into the midplane of a
        galaxy crosses roughly six stops of surface brightness.  No fixed
        exposure survives that, and no tonemapper can rescue it afterwards
        — the highlights are already clipped by then.  Phases therefore
        publish an exposure curve, applied before the tonemap, which is
        exactly what a real camera operator would ride on the iris.
        """
        return 1.0

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def progress(lt, duration):
        return float(np.clip(lt / max(duration, 1e-6), 0.0, 1.0))


# ==========================================================================
# point splatting
# ==========================================================================
def splat_points(width, height, ndc_x, ndc_y, rgb, size=None,
                 soft=True, out=None):
    """Accumulate coloured points into a full-resolution HDR buffer.

    ``ndc_x`` / ``ndc_y`` are in [-1, 1] (aspect already applied to x).
    ``rgb`` is ``(N, 3)`` linear radiance.  ``size`` optionally gives a
    per-point radius in pixels; points larger than one pixel are drawn with
    a 3x3 tent so bright stars gain a visible core without a per-star loop.
    """
    n = width * height
    if out is None:
        out = np.zeros((n, 3), F32)

    px = (ndc_x * 0.5 + 0.5) * width
    py = (0.5 - ndc_y * 0.5) * height
    ix = np.floor(px).astype(np.int32)
    iy = np.floor(py).astype(np.int32)
    # the -1 margin lets the bilinear footprint address ix+1 / iy+1 without
    # any clamping inside the hot path
    inside = (ix >= 0) & (ix < width - 1) & (iy >= 0) & (iy < height - 1)
    if not inside.any():
        return out

    ix = ix[inside]
    iy = iy[inside]
    c = np.ascontiguousarray(rgb[inside], np.float64)
    flat = iy * width + ix

    if not soft:
        for k in range(3):
            out[:, k] += np.bincount(flat, weights=c[:, k],
                                     minlength=n).astype(F32)
        return out

    # Energy-conserving footprint.  A point is split bilinearly over the
    # four pixels it straddles (so a slow pan slides stars smoothly instead
    # of snapping them pixel to pixel), and bright points additionally bleed
    # a fraction `spread` into the eight surrounding pixels so they read as
    # discs rather than dots.  The core is scaled by (1 - spread), which
    # keeps total deposited energy exactly equal to the input radiance.
    fx = px[inside] - ix
    fy = py[inside] - iy
    core = np.ones(ix.shape[0])
    spread = None
    if size is not None:
        spread = np.clip((size[inside].astype(np.float64) - 1.0) * 0.40,
                         0.0, 0.62)
        core = 1.0 - spread

    idx_parts = [flat, flat + 1, flat + width, flat + width + 1]
    w_parts = [core * (1.0 - fx) * (1.0 - fy),
               core * fx * (1.0 - fy),
               core * (1.0 - fx) * fy,
               core * fx * fy]

    if spread is not None and np.any(spread > 1e-3):
        # 3x3 tent, normalised over the eight ring taps only
        ring = ((-1, -1, 0.0742), (0, -1, 0.1758), (1, -1, 0.0742),
                (-1, 0, 0.1758), (1, 0, 0.1758),
                (-1, 1, 0.0742), (0, 1, 0.1758), (1, 1, 0.0742))
        bx = np.clip(ix, 1, width - 2)
        by = np.clip(iy, 1, height - 2)
        bflat = by * width + bx
        for dx, dy, w in ring:
            idx_parts.append(bflat + dy * width + dx)
            w_parts.append(spread * w)

    all_idx = np.concatenate(idx_parts)
    all_w = np.concatenate(w_parts)
    # tile the colours to match the replicated indices, then a single
    # bincount per channel: three 2M-element reductions instead of dozens
    reps = len(idx_parts)
    ctile = np.tile(c, (reps, 1))
    ctile *= all_w[:, None]
    np.clip(all_idx, 0, n - 1, out=all_idx)
    for k in range(3):
        out[:, k] += np.bincount(all_idx, weights=ctile[:, k],
                                 minlength=n).astype(F32)
    return out


def project_and_splat(cam, points, rgb_emit, width, height, *,
                      falloff=True, ref_distance=1.0, max_gain=600.0,
                      near_clip=1e-4, size_scale=0.0):
    """Project world-space points and splat them.

    ``rgb_emit`` is intrinsic linear radiance.  With ``falloff`` the result
    is divided by distance squared and referenced to ``ref_distance`` — the
    inverse-square law, which is what makes flying into a star field feel
    like genuine motion rather than a zoom on a texture.
    """
    nx, ny, z, vis = cam.project(points)
    vis = vis & (z > near_clip)
    # a generous margin keeps stars that are just offscreen contributing to
    # bloom, which stops them popping as they cross the frame edge
    aspect = width / float(height)
    vis &= (np.abs(nx) < aspect * 1.25) & (np.abs(ny) < 1.25)
    if not vis.any():
        return np.zeros((height * width, 3), F32), vis

    nxv = nx[vis]
    nyv = ny[vis]
    c = rgb_emit[vis].astype(F32, copy=True)
    if falloff:
        d2 = np.maximum(z[vis] ** 2, 1e-12)
        gain = np.minimum((ref_distance * ref_distance) / d2, max_gain)
        c *= gain[:, None]

    size = None
    if size_scale > 0.0:
        lum = S.luminance(c)
        size = np.sqrt(np.maximum(lum, 0.0)) * F32(size_scale)

    buf = splat_points(width, height, nxv / aspect, nyv, c, size=size)
    return buf, vis


def diffraction_spikes(img_hw3, cam, points, rgb_emit, width, height,
                       count=140, length=46, strength=0.5, arms=4,
                       angle_deg=0.0):
    """Draw spikes on the brightest few stars.

    Purely an optical lie — real stars have no spikes, the telescope's
    secondary-mirror spider does — but the eye reads them as "this is very
    bright", and they cost one short loop over a hundred points rather than
    a convolution over two million pixels.
    """
    import math
    nx, ny, z, vis = cam.project(points)
    lum = S.luminance(rgb_emit) / np.maximum(z ** 2, 1e-9)
    lum = np.where(vis, lum, 0.0)
    if count <= 0 or not np.any(lum > 0):
        return img_hw3
    k = min(int(count), int((lum > 0).sum()))
    if k <= 0:
        return img_hw3
    top = np.argpartition(lum, -k)[-k:]
    aspect = width / float(height)
    img = img_hw3.reshape(height, width, 3)

    for i in top:
        if not vis[i]:
            continue
        px = int((nx[i] / aspect * 0.5 + 0.5) * width)
        py = int((0.5 - ny[i] * 0.5) * height)
        if not (0 <= px < width and 0 <= py < height):
            continue
        amp = float(lum[i]) * strength
        if amp < 1e-4:
            continue
        col = rgb_emit[i] / max(float(S.luminance(rgb_emit[i:i + 1])[0]), 1e-6)
        ln = int(length * min(1.0, (amp ** 0.25)))
        for a in range(arms):
            th = math.radians(angle_deg) + a * math.pi / (arms * 0.5) * 0.5
            dx, dy = math.cos(th), math.sin(th)
            for s in range(1, ln):
                f = 1.0 - s / ln
                w = amp * f * f * 0.09
                xx = int(px + dx * s)
                yy = int(py + dy * s)
                if 0 <= xx < width and 0 <= yy < height:
                    img[yy, xx] += col * w
                xx = int(px - dx * s)
                yy = int(py - dy * s)
                if 0 <= xx < width and 0 <= yy < height:
                    img[yy, xx] += col * w
    return img.reshape(-1, 3) if img_hw3.ndim == 2 else img


# ==========================================================================
# low-resolution march -> full-resolution upsample
# ==========================================================================
def upsample_linear(img_small, width, height):
    """Bilinear upsample of a low-res HDR volumetric layer.

    Volumetrics are low-frequency by nature, so marching at roughly a
    quarter of the output resolution and upsampling is visually free while
    being four times cheaper.  PIL cannot resize three-channel float32
    directly, so each channel goes through mode 'F'.
    """
    from PIL import Image
    h, w, _ = img_small.shape
    if (w, h) == (width, height):
        return img_small
    out = np.empty((height, width, 3), F32)
    for k in range(3):
        im = Image.fromarray(np.ascontiguousarray(img_small[:, :, k]), "F")
        out[:, :, k] = np.asarray(im.resize((width, height), Image.BILINEAR),
                                  F32)
    return out


def march_layer(ctx, cam, bound_fn, shade_fn, steps=None, jitter_phase=0):
    """Generate rays at march resolution, bound them, march, return an image.

    Returns ``(rgb_small, transmittance_small)`` at the march resolution.
    """
    from ..core import march as M
    mw, mh = ctx.march_width, ctx.march_height
    o, d, idx = cam.generate_rays(mw, mh, jitter_phase=jitter_phase)
    rays = M.rays_from_camera(o, d, idx)
    rays = bound_fn(rays)
    rgb = np.zeros((mw * mh, 3), F32)
    trn = np.ones(mw * mh, F32)
    if len(rays) == 0:
        return rgb.reshape(mh, mw, 3), trn.reshape(mh, mw)
    c, t = M.march(rays, shade_fn, steps=ctx.march_steps(steps), rng=ctx.rng)
    rgb[rays.index] = c
    trn[rays.index] = t
    return rgb.reshape(mh, mw, 3), trn.reshape(mh, mw)


# ==========================================================================
# emission ramps
# ==========================================================================
def build_emission_ramp(temps, gains, n=1024):
    """Bake a 1-D blackbody colour ramp.

    Calling ``blackbody_lut`` per sample costs 53 ms per million.  Baking the
    physics into a ramp indexed by whatever scalar the shader already has —
    density, radius, turbulence — reduces that to a single 14 ms gather,
    while keeping the colours derived from Planck's law rather than guessed.

    ``temps`` and ``gains`` are sampled across [0, 1] and interpolated.
    """
    u = np.linspace(0.0, 1.0, n)
    kt = np.linspace(0.0, 1.0, len(temps))
    t = np.interp(u, kt, np.asarray(temps, np.float64))
    g = np.interp(u, np.linspace(0.0, 1.0, len(gains)),
                  np.asarray(gains, np.float64))
    rgb = S.blackbody_lut(t.astype(F32)) * g[:, None].astype(F32)
    return np.ascontiguousarray(rgb.astype(F32))


def ramp_lookup(ramp, u):
    """Index a baked ramp with a float in [0, 1]."""
    n = ramp.shape[0]
    i = (np.clip(u, 0.0, 1.0) * F32(n - 1)).astype(np.int32)
    return ramp[i]


__all__ = ["RenderContext", "Phase", "splat_points", "project_and_splat",
           "diffraction_spikes", "upsample_linear", "march_layer",
           "build_emission_ramp", "ramp_lookup"]
