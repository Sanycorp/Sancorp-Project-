"""
cosmos.core.vmath
=================

Vectorised 3-D mathematics for the cosmic renderer.

Every routine in this module is written to operate on *batches* of vectors
held as ``(N, 3)`` float32 arrays, because the renderer marches millions of
rays at once and a per-ray Python loop would be several thousand times too
slow.  Scalar convenience overloads are provided where they cost nothing.

Conventions
-----------
* Vectors are row-major ``(..., 3)`` arrays of float32.
* The world is right-handed: +X right, +Y up, -Z forward (OpenGL style).
* Angles are radians unless a function name says ``_deg``.
* Functions never mutate their inputs unless the name ends in ``_``.
"""
from __future__ import annotations

import math

import numpy as np

F32 = np.float32
EPS = F32(1e-9)

# Canonical basis vectors, kept as module constants so hot loops do not
# rebuild them on every call.
X_AXIS = np.array([1.0, 0.0, 0.0], F32)
Y_AXIS = np.array([0.0, 1.0, 0.0], F32)
Z_AXIS = np.array([0.0, 0.0, 1.0], F32)
ORIGIN = np.array([0.0, 0.0, 0.0], F32)


# ==========================================================================
# construction
# ==========================================================================
def vec3(x, y=None, z=None):
    """Build a float32 3-vector (or broadcast a scalar into one)."""
    if y is None and z is None:
        if np.isscalar(x):
            return np.array([x, x, x], F32)
        return np.asarray(x, F32)
    return np.array([x, y, z], F32)


def zeros(n):
    return np.zeros((int(n), 3), F32)


def full(n, v):
    out = np.empty((int(n), 3), F32)
    out[:] = np.asarray(v, F32)
    return out


# ==========================================================================
# elementary algebra
# ==========================================================================
def dot(a, b, keepdims=True):
    """Row-wise dot product of two ``(N, 3)`` stacks."""
    return np.einsum("...i,...i->...", a, b)[..., None] if keepdims else \
        np.einsum("...i,...i->...", a, b)


def length(a, keepdims=True):
    return np.sqrt(np.maximum(dot(a, a, keepdims), 0.0))


def length_sq(a, keepdims=True):
    return dot(a, a, keepdims)


def normalize(a, out=None):
    """Unit-length version of each row.  Zero rows stay zero."""
    n = length(a)
    np.maximum(n, EPS, out=n)
    if out is None:
        return (a / n).astype(F32, copy=False)
    np.divide(a, n, out=out)
    return out


def normalize_(a):
    """In-place normalisation."""
    n = length(a)
    np.maximum(n, EPS, out=n)
    np.divide(a, n, out=a)
    return a


def cross(a, b):
    return np.cross(a, b).astype(F32, copy=False)


def distance(a, b):
    return length(a - b)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)


def saturate(x):
    return np.clip(x, 0.0, 1.0)


def reflect(incident, normal):
    """Mirror ``incident`` about ``normal`` (both assumed unit length)."""
    return incident - 2.0 * dot(incident, normal) * normal


def refract(incident, normal, eta):
    """Snell refraction.  Returns zeros where total internal reflection occurs."""
    ci = -dot(incident, normal)
    k = 1.0 - eta * eta * (1.0 - ci * ci)
    tir = k < 0.0
    k = np.maximum(k, 0.0)
    out = eta * incident + (eta * ci - np.sqrt(k)) * normal
    out[np.broadcast_to(tir, out.shape)] = 0.0
    return out


def face_forward(n, i):
    """Flip ``n`` so it opposes ``i``."""
    s = np.where(dot(i, n) < 0.0, 1.0, -1.0)
    return n * s


# ==========================================================================
# interpolation & easing
# ==========================================================================
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / np.maximum(e1 - e0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(e0, e1, x):
    """Ken Perlin's C2-continuous variant — no second-derivative pop."""
    t = np.clip((x - e0) / np.maximum(e1 - e0, 1e-12), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def remap(x, a0, a1, b0, b1, clamped=True):
    t = (x - a0) / np.maximum(a1 - a0, 1e-12)
    if clamped:
        t = np.clip(t, 0.0, 1.0)
    return b0 + (b1 - b0) * t


def ease_in_cubic(t):
    return t * t * t


def ease_out_cubic(t):
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t):
    return np.where(t < 0.5, 4.0 * t ** 3, 1.0 - (-2.0 * t + 2.0) ** 3 * 0.5)


def ease_out_expo(t):
    return np.where(t >= 1.0, 1.0, 1.0 - np.exp2(-11.0 * t))


def ease_in_expo(t):
    return np.where(t <= 0.0, 0.0, np.exp2(11.0 * (t - 1.0)))


def ease_in_out_expo(t):
    t = np.clip(t, 0.0, 1.0)
    return np.where(
        t < 0.5,
        np.exp2(20.0 * t - 10.0) * 0.5,
        (2.0 - np.exp2(-20.0 * t + 10.0)) * 0.5,
    )


# ==========================================================================
# rotation
# ==========================================================================
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], F32)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], F32)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], F32)


def rot_axis(axis, angle):
    """Rodrigues rotation matrix about an arbitrary unit axis."""
    ax = np.asarray(axis, F32)
    ax = ax / max(float(np.linalg.norm(ax)), 1e-12)
    x, y, z = float(ax[0]), float(ax[1]), float(ax[2])
    c, s = math.cos(angle), math.sin(angle)
    ic = 1.0 - c
    return np.array([
        [c + x * x * ic,     x * y * ic - z * s, x * z * ic + y * s],
        [y * x * ic + z * s, c + y * y * ic,     y * z * ic - x * s],
        [z * x * ic - y * s, z * y * ic + x * s, c + z * z * ic],
    ], F32)


def apply_mat(m, v):
    """Apply a 3x3 matrix to a ``(N, 3)`` stack of row vectors."""
    return (np.asarray(v, F32) @ np.asarray(m, F32).T).astype(F32, copy=False)


def basis_from_forward(forward, up=Y_AXIS):
    """Orthonormal (right, up, forward) basis, Gram-Schmidt stabilised."""
    f = np.asarray(forward, F32)
    f = f / max(float(np.linalg.norm(f)), 1e-12)
    u = np.asarray(up, F32)
    if abs(float(np.dot(f, u / max(float(np.linalg.norm(u)), 1e-12)))) > 0.9995:
        u = X_AXIS if abs(f[0]) < 0.9 else Y_AXIS
    r = np.cross(f, u)
    r /= max(float(np.linalg.norm(r)), 1e-12)
    u2 = np.cross(r, f)
    u2 /= max(float(np.linalg.norm(u2)), 1e-12)
    return r.astype(F32), u2.astype(F32), f.astype(F32)


# ==========================================================================
# quaternions  (w, x, y, z)
# ==========================================================================
def quat_from_axis(axis, angle):
    ax = np.asarray(axis, np.float64)
    ax = ax / max(float(np.linalg.norm(ax)), 1e-12)
    h = angle * 0.5
    s = math.sin(h)
    return np.array([math.cos(h), ax[0] * s, ax[1] * s, ax[2] * s], np.float64)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], np.float64)


def quat_normalize(q):
    return q / max(float(np.linalg.norm(q)), 1e-12)


def quat_slerp(a, b, t):
    """Shortest-arc spherical interpolation; falls back to lerp when close."""
    a = quat_normalize(np.asarray(a, np.float64))
    b = quat_normalize(np.asarray(b, np.float64))
    d = float(np.dot(a, b))
    if d < 0.0:
        b = -b
        d = -d
    if d > 0.9995:
        return quat_normalize(a + (b - a) * t)
    th0 = math.acos(max(-1.0, min(1.0, d)))
    th = th0 * t
    s0 = math.sin(th0)
    return (a * (math.sin(th0 - th) / s0)) + (b * (math.sin(th) / s0))


def quat_to_mat(q):
    w, x, y, z = quat_normalize(np.asarray(q, np.float64))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], F32)


# ==========================================================================
# coordinates
# ==========================================================================
def spherical_to_cart(radius, theta, phi):
    """theta = polar from +Y, phi = azimuth in the XZ plane."""
    st = np.sin(theta)
    return np.stack([radius * st * np.cos(phi),
                     radius * np.cos(theta),
                     radius * st * np.sin(phi)], axis=-1).astype(F32, copy=False)


def cart_to_spherical(v):
    r = length(v, keepdims=False)
    safe = np.maximum(r, EPS)
    theta = np.arccos(np.clip(v[..., 1] / safe, -1.0, 1.0))
    phi = np.arctan2(v[..., 2], v[..., 0])
    return r, theta, phi


def cyl_to_cart(radius, phi, y):
    return np.stack([radius * np.cos(phi), y, radius * np.sin(phi)],
                    axis=-1).astype(F32, copy=False)


def cart_to_cyl(v):
    """Returns (radial distance in XZ, azimuth, height)."""
    rad = np.sqrt(v[..., 0] ** 2 + v[..., 2] ** 2)
    phi = np.arctan2(v[..., 2], v[..., 0])
    return rad, phi, v[..., 1]


def fibonacci_sphere(n, radius=1.0, dtype=F32):
    """`n` near-uniformly distributed points on a sphere (golden-angle spiral)."""
    i = np.arange(n, dtype=np.float64)
    y = 1.0 - 2.0 * i / max(n - 1.0, 1.0)
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    ga = math.pi * (3.0 - math.sqrt(5.0))
    th = ga * i
    pts = np.stack([np.cos(th) * r, y, np.sin(th) * r], axis=-1) * radius
    return pts.astype(dtype, copy=False)


def fibonacci_disk(n, radius=1.0, dtype=F32):
    """Sunflower sampling of a disk — the lowest-discrepancy cheap option."""
    i = np.arange(n, dtype=np.float64) + 0.5
    r = radius * np.sqrt(i / n)
    th = math.pi * (3.0 - math.sqrt(5.0)) * i
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=-1).astype(dtype, copy=False)


# ==========================================================================
# ray / primitive intersection
# ==========================================================================
def ray_sphere(origin, direction, centre, radius):
    """Analytic ray-sphere test.

    Returns ``(hit, t_near, t_far)``.  Directions are assumed normalised, so
    the quadratic's leading coefficient is 1 and drops out of the algebra.
    Uses the numerically stable form that avoids catastrophic cancellation
    when the ray origin is far from the sphere — which is the normal case in
    an astronomical scene where distances span many orders of magnitude.
    """
    oc = origin - np.asarray(centre, F32)
    b = dot(oc, direction)
    c = dot(oc, oc) - F32(radius) * F32(radius)
    disc = b * b - c
    hit = disc > 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))
    # stable roots: q = -(b + sign(b)*sq); t0 = q, t1 = c/q
    q = -(b + np.sign(b) * sq)
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(np.abs(q) > 1e-20, q, np.inf)
        t1 = np.where(np.abs(q) > 1e-20, c / np.where(q == 0, 1.0, q), np.inf)
    t_near = np.minimum(t0, t1)
    t_far = np.maximum(t0, t1)
    return hit, t_near, t_far


def ray_slab(origin, direction, bmin, bmax):
    """Axis-aligned box slab test.  Returns ``(hit, t_enter, t_exit)``."""
    inv = 1.0 / np.where(np.abs(direction) < 1e-20, 1e-20, direction)
    t1 = (np.asarray(bmin, F32) - origin) * inv
    t2 = (np.asarray(bmax, F32) - origin) * inv
    tmin = np.minimum(t1, t2).max(axis=-1, keepdims=True)
    tmax = np.maximum(t1, t2).min(axis=-1, keepdims=True)
    return (tmax > np.maximum(tmin, 0.0)), tmin, tmax


def ray_plane(origin, direction, point, normal):
    """Returns ``(hit, t)`` for an infinite plane."""
    nd = dot(direction, normal)
    par = np.abs(nd) < 1e-12
    t = dot(np.asarray(point, F32) - origin, normal) / np.where(par, 1.0, nd)
    return (~par) & (t > 0.0), t


def ray_disk(origin, direction, centre, normal, r_inner, r_outer):
    """Annulus test — the workhorse for accretion discs and planetary rings."""
    hit, t = ray_plane(origin, direction, centre, normal)
    p = origin + direction * t
    d = length(p - np.asarray(centre, F32))
    inside = (d >= r_inner) & (d <= r_outer)
    return hit & inside, t, d


def ray_cylinder_y(origin, direction, radius, y_lo, y_hi):
    """Y-aligned finite cylinder; used to bound galactic and jet volumes."""
    ox, oz = origin[..., 0:1], origin[..., 2:3]
    dx, dz = direction[..., 0:1], direction[..., 2:3]
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius * radius
    disc = b * b - 4.0 * a * c
    hit = (disc > 0.0) & (a > 1e-20)
    sq = np.sqrt(np.maximum(disc, 0.0))
    inv2a = 1.0 / np.where(a > 1e-20, 2.0 * a, 1.0)
    t0 = (-b - sq) * inv2a
    t1 = (-b + sq) * inv2a
    # clip against the end caps
    dy = np.where(np.abs(direction[..., 1:2]) < 1e-20, 1e-20, direction[..., 1:2])
    ty0 = (y_lo - origin[..., 1:2]) / dy
    ty1 = (y_hi - origin[..., 1:2]) / dy
    tlo = np.minimum(ty0, ty1)
    thi = np.maximum(ty0, ty1)
    t_enter = np.maximum(t0, tlo)
    t_exit = np.minimum(t1, thi)
    return hit & (t_exit > np.maximum(t_enter, 0.0)), t_enter, t_exit


# ==========================================================================
# splines — camera paths and parameter automation
# ==========================================================================
def catmull_rom(points, t, closed=False, tension=0.5):
    """Evaluate a Catmull-Rom spline through `points` at normalised ``t``.

    ``points`` is an ``(M, D)`` array.  ``t`` may be a scalar or an array;
    the result keeps the trailing dimension D.
    """
    p = np.asarray(points, np.float64)
    m = len(p)
    if m == 0:
        raise ValueError("catmull_rom needs at least one control point")
    if m == 1:
        return np.broadcast_to(p[0], np.shape(t) + (p.shape[1],)).copy()

    t = np.asarray(t, np.float64)
    segs = m if closed else (m - 1)
    u = np.clip(t, 0.0, 1.0) * segs
    i = np.floor(u).astype(np.int64)
    i = np.minimum(i, segs - 1)
    f = (u - i)[..., None]

    def idx(k):
        if closed:
            return p[np.mod(k, m)]
        return p[np.clip(k, 0, m - 1)]

    p0, p1, p2, p3 = idx(i - 1), idx(i), idx(i + 1), idx(i + 2)
    tau = tension * 2.0
    f2 = f * f
    f3 = f2 * f
    return (
        p1
        + 0.5 * tau * (p2 - p0) * f
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * 0.5 * tau * f2 * (1.0 / tension)
        * 0.5
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * 0.5 * tau * f3 * (1.0 / tension) * 0.5
    )


def bezier_cubic(p0, p1, p2, p3, t):
    t = np.asarray(t, np.float64)[..., None]
    u = 1.0 - t
    return (u ** 3 * np.asarray(p0, np.float64)
            + 3.0 * u * u * t * np.asarray(p1, np.float64)
            + 3.0 * u * t * t * np.asarray(p2, np.float64)
            + t ** 3 * np.asarray(p3, np.float64))


def arc_length_table(fn, n=512):
    """Build a cumulative arc-length table for a parametric curve ``fn(t)``.

    Returns ``(ts, cumulative_length)`` so a camera can be driven at constant
    speed instead of constant parameter — without this, Catmull-Rom cameras
    visibly accelerate through tight control points.
    """
    ts = np.linspace(0.0, 1.0, n)
    pts = np.asarray([fn(float(x)) for x in ts], np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return ts, cum


def reparam_by_length(ts, cum, s):
    """Map normalised arc length ``s`` back to the curve parameter."""
    total = cum[-1] if cum[-1] > 0 else 1.0
    return np.interp(np.clip(s, 0.0, 1.0) * total, cum, ts)


# ==========================================================================
# stochastic helpers
# ==========================================================================
def hash_u32(x):
    """Integer avalanche hash (Wang / Jenkins style), vectorised."""
    x = np.asarray(x, np.uint32)
    x = (x ^ np.uint32(61)) ^ (x >> np.uint32(16))
    x = x + (x << np.uint32(3))
    x = x ^ (x >> np.uint32(4))
    x = x * np.uint32(0x27D4EB2D)
    x = x ^ (x >> np.uint32(15))
    return x


def hash_unit(x):
    """Deterministic float in [0, 1) from an integer stream."""
    return (hash_u32(x) >> np.uint32(8)).astype(np.float32) * np.float32(1.0 / 16777216.0)


def stratified_jitter(n, rng, strength=1.0):
    """Stratified [0,1) samples — far lower variance than pure uniform.

    Ray-marching with unjittered steps produces visible concentric banding;
    stratification removes it for the price of one addition per sample.
    """
    base = (np.arange(n, dtype=np.float32) + 0.5) / n
    return base + (rng.random(n, dtype=np.float32) - 0.5) * (strength / n)


def blue_noise_tile(size, rng, iterations=12):
    """Void-and-cluster-ish blue-noise tile used to dither march offsets.

    A true void-and-cluster solver is overkill here; iteratively swapping the
    brightest and darkest low-pass neighbourhoods converges close enough for
    dithering and costs milliseconds.
    """
    v = rng.random((size, size), dtype=np.float32)
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
    k /= k.sum()
    for _ in range(iterations):
        lp = np.zeros_like(v)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                lp += np.roll(np.roll(v, dy, 0), dx, 1) * k[dy + 1, dx + 1]
        hi = v - lp
        order = np.argsort(hi, axis=None)
        flat = v.reshape(-1).copy()
        ranks = np.empty(order.size, np.float32)
        ranks[order] = (np.arange(order.size, dtype=np.float32) + 0.5) / order.size
        v = ranks.reshape(size, size)
    return v


__all__ = [n for n in dir() if not n.startswith("_")]
