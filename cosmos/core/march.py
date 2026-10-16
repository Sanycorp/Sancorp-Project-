"""
cosmos.core.march
=================

The volumetric ray-marching framework.

Integration model
-----------------
Each ray accumulates the emission-absorption integral front to back:

    C  =  sum_i  T_i * c_i * alpha_i
    T_{i+1} = T_i * (1 - alpha_i)
    alpha_i = 1 - exp(-sigma_i * dt)

Front-to-back is chosen over back-to-front specifically so the loop can
*stop early*: once transmittance drops below a threshold nothing behind it
can contribute, and that ray can be removed from the batch entirely.

Two optimisations carry this renderer
-------------------------------------
**Analytic bounds.**  Every phase hands the marcher a near/far interval from
a closed-form intersection (sphere, annulus, cylinder, slab).  Rays that
miss the volume never enter the loop at all, and rays that hit it march only
the occupied span rather than the whole depth range.  In a typical cosmic
frame two thirds of the pixels are empty sky, so this alone is a 3x saving.

**Active-ray compaction.**  Inside a dense nebula or the inner accretion
disc, most rays saturate within the first third of their steps.  Every
``compact_every`` iterations the batch is rebuilt from the survivors.  The
gather costs something, so it is not done every step — measured across this
film, every 8 steps is the point where the saving stops paying for itself.

Shaders
-------
A shader is any callable with the signature

    shade(pos, t, step_fraction) -> (emission, extinction)

where ``pos`` is ``(M, 3)`` world positions, ``t`` is ``(M,)`` distance along
the ray, ``step_fraction`` is the normalised step index, and the returns are
``(M, 3)`` linear-RGB emission and ``(M,)`` extinction coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32


# ==========================================================================
# ray batches
# ==========================================================================
@dataclass
class RayBatch:
    """A flat collection of rays with per-ray integration bounds."""

    origin: np.ndarray            # (N, 3) float32
    direction: np.ndarray         # (N, 3) float32, unit length
    t_near: np.ndarray            # (N,)   float32
    t_far: np.ndarray             # (N,)   float32
    index: np.ndarray = field(default=None)   # (N,) int32 into the framebuffer

    def __post_init__(self):
        n = self.origin.shape[0]
        if self.index is None:
            self.index = np.arange(n, dtype=np.int32)
        self.t_near = np.ascontiguousarray(self.t_near, F32).reshape(-1)
        self.t_far = np.ascontiguousarray(self.t_far, F32).reshape(-1)

    def __len__(self):
        return self.origin.shape[0]

    def select(self, mask_or_idx):
        """Return a new batch containing only the selected rays."""
        k = mask_or_idx
        return RayBatch(self.origin[k], self.direction[k],
                        self.t_near[k], self.t_far[k], self.index[k])

    def valid(self, min_span=1e-6):
        """Mask of rays whose interval is non-empty and in front of the eye."""
        return (self.t_far > np.maximum(self.t_near, 0.0) + min_span)

    def clipped(self, min_span=1e-6):
        """Drop empty rays and clamp t_near to the eye plane."""
        b = self.select(self.valid(min_span))
        np.maximum(b.t_near, 0.0, out=b.t_near)
        return b


def rays_from_camera(origins, directions, index=None):
    return RayBatch(np.ascontiguousarray(origins, F32),
                    np.ascontiguousarray(directions, F32),
                    np.zeros(len(origins), F32),
                    np.full(len(origins), np.float32(np.inf)),
                    index)


# ==========================================================================
# phase functions
# ==========================================================================
def henyey_greenstein(cos_theta, g=0.3):
    """HG scattering phase function.

    ``g`` in (-1, 1): positive is forward scattering.  Dust in nebulae and
    the solar corona are both strongly forward-scattering, which is why a
    star seen *through* dust gets a halo rather than a uniform wash.
    """
    g = F32(g)
    g2 = g * g
    denom = F32(1.0) + g2 - F32(2.0) * g * np.asarray(cos_theta, F32)
    return (F32(1.0) - g2) / (F32(4.0 * np.pi) * np.maximum(denom, 1e-6) ** F32(1.5))


def rayleigh_phase(cos_theta):
    c = np.asarray(cos_theta, F32)
    return F32(3.0 / (16.0 * np.pi)) * (F32(1.0) + c * c)


def beer_lambert(sigma, dt):
    """Transmittance through a homogeneous slab."""
    return np.exp(-np.asarray(sigma, F32) * F32(dt))


def alpha_from_sigma(sigma, dt):
    """Opacity of one march step.

    ``-expm1(-x)`` rather than ``1 - exp(-x)``: at the small extinction
    values that dominate a thin nebula the naive form loses most of its
    significant digits, and the error shows up as faint banding.
    """
    return -np.expm1(-np.asarray(sigma, F32) * dt)


# ==========================================================================
# the marcher
# ==========================================================================
def march(rays, shade, steps=64, rng=None, *,
          transmittance_cutoff=0.004,
          compact_every=8,
          jitter=1.0,
          max_sigma_dt=8.0,
          collect_depth=False,
          background=None):
    """March a batch of rays through an emissive, absorbing medium.

    Parameters
    ----------
    rays : RayBatch
        Rays with finite ``t_near`` / ``t_far`` bounds.
    shade : callable
        ``shade(pos, t, u) -> (emission (M,3), extinction (M,))``.
    steps : int
        Samples per ray across its own interval.  Because each ray gets its
        own ``dt = (t_far - t_near) / steps``, a ray that only clips the
        edge of a volume spends the same number of samples resolving a much
        shorter span — the sampling adapts to the geometry for free.
    rng : numpy Generator
        Source for the stratification jitter.  Required when ``jitter > 0``.
    transmittance_cutoff : float
        Rays dimmer than this are retired.
    compact_every : int
        Rebuild the active set this often.  Set to 0 to disable.

    Returns
    -------
    Results are returned in *batch order* (one row per input ray).  The
    caller scatters them into the framebuffer with ``rays.index``.

    color : (N, 3) float32   accumulated linear-RGB radiance
    trans : (N,)  float32    remaining transmittance (1 = nothing hit)
    depth : (N,)  float32    first-contact distance, or inf  (optional)
    """
    n_total = len(rays)
    color = np.zeros((n_total, 3), F32)
    trans = np.ones(n_total, F32)
    depth = np.full(n_total, np.float32(np.inf)) if collect_depth else None
    if n_total == 0:
        return (color, trans, depth) if collect_depth else (color, trans)

    # `slot` maps each live ray to its row in the output arrays.  This is
    # deliberately distinct from RayBatch.index, which is the caller's
    # framebuffer index and must survive untouched through compaction.
    valid = rays.valid()
    slot = np.flatnonzero(valid).astype(np.int64)
    live = rays.select(valid)
    np.maximum(live.t_near, 0.0, out=live.t_near)
    if len(live) == 0:
        return (color, trans, depth) if collect_depth else (color, trans)

    # Per-ray step size.  Rays with short spans get fine steps automatically.
    span = live.t_far - live.t_near
    dt = span / F32(steps)

    # Local accumulators for the active set only.
    acc_c = np.zeros((len(live), 3), F32)
    acc_t = np.ones(len(live), F32)
    acc_d = np.full(len(live), np.float32(np.inf)) if collect_depth else None

    if rng is None:
        rng = np.random.default_rng(0)

    for i in range(int(steps)):
        m = len(live)
        if m == 0:
            break
        u = F32((i + 0.5) / steps)
        if jitter > 0.0:
            off = (rng.random(m, dtype=np.float32) - F32(0.5)) * F32(jitter)
            tt = live.t_near + (F32(i) + F32(0.5) + off) * dt
        else:
            tt = live.t_near + (F32(i) + F32(0.5)) * dt
        np.clip(tt, live.t_near, live.t_far, out=tt)

        pos = live.origin + live.direction * tt[:, None]
        emission, sigma = shade(pos, tt, u)

        sd = sigma * dt
        np.clip(sd, 0.0, F32(max_sigma_dt), out=sd)
        alpha = -np.expm1(-sd)
        w = acc_t * alpha

        acc_c += emission * w[:, None]
        if collect_depth:
            first = np.isinf(acc_d) & (alpha > 0.01)
            acc_d[first] = tt[first]
        acc_t *= (F32(1.0) - alpha)

        if compact_every and (i % compact_every == compact_every - 1) \
                and i < steps - 1:
            keep = acc_t > F32(transmittance_cutoff)
            if not keep.all():
                # flush the finished rays into the framebuffer
                done = ~keep
                if done.any():
                    out = slot[done]
                    color[out] = acc_c[done]
                    trans[out] = acc_t[done]
                    if collect_depth:
                        depth[out] = acc_d[done]
                slot = slot[keep]
                live = live.select(keep)
                dt = dt[keep]
                acc_c = acc_c[keep]
                acc_t = acc_t[keep]
                if collect_depth:
                    acc_d = acc_d[keep]

    if len(live):
        color[slot] = acc_c
        trans[slot] = acc_t
        if collect_depth:
            depth[slot] = acc_d

    if background is not None:
        color += np.asarray(background, F32) * trans[:, None]

    return (color, trans, depth) if collect_depth else (color, trans)


def march_layered(rays, layers, rng=None, background=None):
    """March several independent media and composite them by depth order.

    ``layers`` is a sequence of ``(shade, steps, t_near, t_far)`` tuples
    already sorted front to back.  Used where two volumes overlap but have
    very different feature scales — for instance the sun's dense photosphere
    against its enormous, sparse corona, which would waste most of its steps
    if marched on one shared interval.
    """
    n = len(rays)
    color = np.zeros((n, 3), F32)
    trans = np.ones(n, F32)
    for shade, steps, tn, tf in layers:
        sub = RayBatch(rays.origin, rays.direction,
                       np.asarray(tn, F32), np.asarray(tf, F32), rays.index)
        c, t = march(sub, shade, steps=steps, rng=rng)
        color += c * trans[:, None]
        trans *= t
    if background is not None:
        color += np.asarray(background, F32) * trans[:, None]
    return color, trans


# ==========================================================================
# bounding helpers used by the phases
# ==========================================================================
def bound_sphere(rays, centre, radius):
    """Clip a batch to a sphere.  Returns a new batch of the hits only."""
    from .vmath import ray_sphere
    hit, tn, tf = ray_sphere(rays.origin, rays.direction, centre, radius)
    hit = hit.reshape(-1)
    tn = tn.reshape(-1)
    tf = tf.reshape(-1)
    ok = hit & (tf > 0.0)
    b = rays.select(ok)
    b.t_near = np.maximum(np.maximum(tn[ok], 0.0), b.t_near)
    b.t_far = np.minimum(tf[ok], b.t_far)
    return b.clipped()


def bound_shell(rays, centre, r_inner, r_outer):
    """Clip to a spherical shell.

    A ray through a shell has two disjoint spans.  Rather than track both,
    we return the full outer span and let the shader return zero density
    inside the hole — simpler, and the inner region is usually opaque anyway
    (a star's photosphere, a black hole's horizon), so the saving would be
    recovered by early termination regardless.
    """
    b = bound_sphere(rays, centre, r_outer)
    return b


def bound_disc(rays, centre, normal, r_inner, r_outer, half_thickness):
    """Clip to a thick annulus — accretion discs and planetary rings.

    Intersects the slab defined by the disc plane and its thickness, then
    further clips against the outer cylinder.  Without the cylinder clip,
    rays travelling nearly parallel to the disc would march hundreds of
    units of empty space before reaching any material.
    """
    from .vmath import ray_cylinder_y
    o = rays.origin - np.asarray(centre, F32)
    d = rays.direction
    # slab in the normal direction
    nrm = np.asarray(normal, F32)
    on = o @ nrm
    dn = d @ nrm
    safe = np.where(np.abs(dn) < 1e-12, 1e-12, dn)
    t_a = (-half_thickness - on) / safe
    t_b = (half_thickness - on) / safe
    ts_lo = np.minimum(t_a, t_b)
    ts_hi = np.maximum(t_a, t_b)
    parallel = np.abs(dn) < 1e-12
    inside = np.abs(on) <= half_thickness
    ts_lo = np.where(parallel, np.where(inside, 0.0, np.inf), ts_lo)
    ts_hi = np.where(parallel, np.where(inside, np.inf, -np.inf), ts_hi)

    hit_c, tc_lo, tc_hi = ray_cylinder_y(o, d, r_outer, -half_thickness * 4.0,
                                         half_thickness * 4.0)
    tc_lo = tc_lo.reshape(-1)
    tc_hi = tc_hi.reshape(-1)
    hit_c = hit_c.reshape(-1)

    tn = np.maximum(ts_lo, np.where(hit_c, tc_lo, np.inf))
    tf = np.minimum(ts_hi, np.where(hit_c, tc_hi, -np.inf))
    ok = hit_c & (tf > np.maximum(tn, 0.0))

    b = rays.select(ok)
    b.t_near = np.maximum(np.maximum(tn[ok], 0.0), b.t_near)
    b.t_far = np.minimum(tf[ok], b.t_far)
    return b.clipped()


# ==========================================================================
# compositing
# ==========================================================================
def over(dst_rgb, dst_alpha, src_rgb, src_alpha):
    """Standard source-over for premultiplied radiance + coverage."""
    a = np.asarray(src_alpha, F32)[..., None] if np.ndim(src_alpha) == 1 \
        else np.asarray(src_alpha, F32)
    out_rgb = np.asarray(src_rgb, F32) + np.asarray(dst_rgb, F32) * (1.0 - a)
    out_a = np.asarray(src_alpha, F32) + np.asarray(dst_alpha, F32) * (
        1.0 - np.asarray(src_alpha, F32))
    return out_rgb, out_a


def scatter_to_image(values, index, height, width, channels=3, fill=0.0):
    """Scatter a flat per-ray result back into an image buffer."""
    img = np.full((height * width, channels), F32(fill), F32) if channels > 1 \
        else np.full(height * width, F32(fill), F32)
    img[index] = values
    return img.reshape((height, width, channels) if channels > 1
                       else (height, width))


__all__ = [
    "RayBatch", "rays_from_camera", "march", "march_layered",
    "henyey_greenstein", "rayleigh_phase", "beer_lambert", "alpha_from_sigma",
    "bound_sphere", "bound_shell", "bound_disc",
    "over", "scatter_to_image",
]
