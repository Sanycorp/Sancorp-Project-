"""
cosmos.core.noise
=================

Procedural noise bank: value, gradient (Perlin), simplex, cellular (Worley),
and the fractal / curl / warp operators built on top of them.

Design notes
------------
Volumetric ray-marching evaluates noise more often than anything else in the
renderer — a single 640x360 frame at 48 steps is 11 million field lookups,
and a fractal field multiplies that by its octave count.  Three decisions
follow from that:

1.  **No permutation table.**  A classic Perlin implementation indexes a
    256-entry permutation array, which in NumPy becomes a gather over a
    multi-million-element index array — memory-latency bound and slow.  We
    use an integer avalanche hash computed arithmetically instead, so the
    gradient lookup stays in registers and vectorises cleanly.

2.  **float32 end to end.**  Halving the bandwidth very nearly halves the
    runtime of these memory-bound kernels.

3.  **Pre-allocated accumulators.**  The fractal operators add octaves into
    one buffer with ``out=`` rather than building a new temporary per octave.

All sampling functions take ``p`` as an ``(N, 3)`` float32 array and return
an ``(N,)`` float32 array in a documented range.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32
U32 = np.uint32

# Three large odd constants; multiplying each axis by its own prime before
# mixing decorrelates the lattice axes and kills the axis-aligned streaking
# that a naive (x + y*57 + z*131) hash produces.
_PX = U32(0x9E3779B1)
_PY = U32(0x85EBCA77)
_PZ = U32(0xC2B2AE3D)
_PS = U32(0x27D4EB2F)


# ==========================================================================
# integer hashing
# ==========================================================================
def _hash3(ix, iy, iz, seed):
    """Avalanche hash of an integer lattice coordinate -> uint32."""
    h = (ix.astype(U32) * _PX) ^ (iy.astype(U32) * _PY) ^ (iz.astype(U32) * _PZ)
    h = h ^ U32((int(seed) * 0x27D4EB2F) & 0xFFFFFFFF)
    h ^= h >> U32(15)
    h = h * U32(0x2C1B3C6D)
    h ^= h >> U32(12)
    h = h * U32(0x297A2D39)
    h ^= h >> U32(15)
    return h


def _hash_unit(ix, iy, iz, seed):
    """Hash to float32 in [0, 1)."""
    return (_hash3(ix, iy, iz, seed) >> U32(8)).astype(F32) * F32(1.0 / 16777216.0)


def _hash_signed(ix, iy, iz, seed):
    """Hash to float32 in [-1, 1)."""
    return _hash_unit(ix, iy, iz, seed) * F32(2.0) - F32(1.0)


# ==========================================================================
# interpolation kernels
# ==========================================================================
def _fade(t):
    """Perlin's quintic 6t^5 - 15t^4 + 10t^3 — C2 continuous."""
    return t * t * t * (t * (t * F32(6.0) - F32(15.0)) + F32(10.0))


def _fade_(t, out):
    """In-place quintic fade to avoid three temporaries per axis."""
    np.multiply(t, F32(6.0), out=out)
    np.subtract(out, F32(15.0), out=out)
    np.multiply(out, t, out=out)
    np.add(out, F32(10.0), out=out)
    np.multiply(out, t, out=out)
    np.multiply(out, t, out=out)
    np.multiply(out, t, out=out)
    return out


# ==========================================================================
# value noise
# ==========================================================================
def value3(p, seed=0):
    """Trilinearly interpolated value noise.  Range approximately [-1, 1]."""
    p = np.asarray(p, F32)
    pf = np.floor(p)
    i = pf.astype(np.int32)
    f = p - pf
    u = _fade(f)

    ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

    c000 = _hash_signed(ix, iy, iz, seed)
    c100 = _hash_signed(ix + 1, iy, iz, seed)
    c010 = _hash_signed(ix, iy + 1, iz, seed)
    c110 = _hash_signed(ix + 1, iy + 1, iz, seed)
    c001 = _hash_signed(ix, iy, iz + 1, seed)
    c101 = _hash_signed(ix + 1, iy, iz + 1, seed)
    c011 = _hash_signed(ix, iy + 1, iz + 1, seed)
    c111 = _hash_signed(ix + 1, iy + 1, iz + 1, seed)

    x00 = c000 + (c100 - c000) * ux
    x10 = c010 + (c110 - c010) * ux
    x01 = c001 + (c101 - c001) * ux
    x11 = c011 + (c111 - c011) * ux
    y0 = x00 + (x10 - x00) * uy
    y1 = x01 + (x11 - x01) * uy
    return y0 + (y1 - y0) * uz


# ==========================================================================
# gradient (Perlin) noise
# ==========================================================================
# Perlin's 12 edge-midpoint gradients, padded to 16 so the index mask is a
# cheap bitwise AND rather than a modulo.
_GRAD = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1],
], F32)
_GX = np.ascontiguousarray(_GRAD[:, 0])
_GY = np.ascontiguousarray(_GRAD[:, 1])
_GZ = np.ascontiguousarray(_GRAD[:, 2])


def _grad_dot(ix, iy, iz, fx, fy, fz, seed):
    """Dot the hashed lattice gradient with the offset vector."""
    h = _hash3(ix, iy, iz, seed) & U32(15)
    return _GX[h] * fx + _GY[h] * fy + _GZ[h] * fz


def perlin3(p, seed=0):
    """Improved Perlin gradient noise.  Range approximately [-1, 1]."""
    p = np.asarray(p, F32)
    pf = np.floor(p)
    i = pf.astype(np.int32)
    f = p - pf

    ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    ux, uy, uz = _fade(fx), _fade(fy), _fade(fz)
    fx1, fy1, fz1 = fx - F32(1.0), fy - F32(1.0), fz - F32(1.0)
    ix1, iy1, iz1 = ix + 1, iy + 1, iz + 1

    n000 = _grad_dot(ix, iy, iz, fx, fy, fz, seed)
    n100 = _grad_dot(ix1, iy, iz, fx1, fy, fz, seed)
    n010 = _grad_dot(ix, iy1, iz, fx, fy1, fz, seed)
    n110 = _grad_dot(ix1, iy1, iz, fx1, fy1, fz, seed)
    n001 = _grad_dot(ix, iy, iz1, fx, fy, fz1, seed)
    n101 = _grad_dot(ix1, iy, iz1, fx1, fy, fz1, seed)
    n011 = _grad_dot(ix, iy1, iz1, fx, fy1, fz1, seed)
    n111 = _grad_dot(ix1, iy1, iz1, fx1, fy1, fz1, seed)

    x00 = n000 + (n100 - n000) * ux
    x10 = n010 + (n110 - n010) * ux
    x01 = n001 + (n101 - n001) * ux
    x11 = n011 + (n111 - n011) * ux
    y0 = x00 + (x10 - x00) * uy
    y1 = x01 + (x11 - x01) * uy
    return (y0 + (y1 - y0) * uz) * F32(1.1547)      # normalise toward unit range


# ==========================================================================
# simplex noise
# ==========================================================================
_F3 = F32(1.0 / 3.0)
_G3 = F32(1.0 / 6.0)


def simplex3(p, seed=0):
    """3-D simplex noise.

    Costs four gradient evaluations instead of Perlin's eight and has no
    axis-aligned directional bias, which matters for plasma and nebula
    fields where lattice artefacts would read as a visible grid.
    Range approximately [-1, 1].
    """
    p = np.asarray(p, F32)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]

    s = (x + y + z) * _F3
    i = np.floor(x + s).astype(np.int32)
    j = np.floor(y + s).astype(np.int32)
    k = np.floor(z + s).astype(np.int32)

    t = (i + j + k).astype(F32) * _G3
    x0 = x - (i.astype(F32) - t)
    y0 = y - (j.astype(F32) - t)
    z0 = z - (k.astype(F32) - t)

    # determine which of the six simplex tetrahedra we are inside
    ge_xy = x0 >= y0
    ge_yz = y0 >= z0
    ge_xz = x0 >= z0

    i1 = (ge_xy & ge_xz).astype(np.int32)
    j1 = (~ge_xy & ge_yz).astype(np.int32)
    k1 = (~ge_xz & ~ge_yz).astype(np.int32)

    i2 = (ge_xy & (ge_xz | ge_yz)).astype(np.int32)
    j2 = ((ge_xy & ge_yz) | (~ge_xy & ge_yz) | (~ge_xy & ge_xz)).astype(np.int32)
    k2 = ((~ge_yz) | (~ge_xz)).astype(np.int32)
    # the three corner offsets must sum to one and two respectively
    i2 = np.minimum(i2 + (1 - j2) * (1 - k2) * i1, 1)
    j2 = np.minimum(j2, 1)
    k2 = np.minimum(k2, 1)

    x1 = x0 - i1.astype(F32) + _G3
    y1 = y0 - j1.astype(F32) + _G3
    z1 = z0 - k1.astype(F32) + _G3
    x2 = x0 - i2.astype(F32) + F32(2.0) * _G3
    y2 = y0 - j2.astype(F32) + F32(2.0) * _G3
    z2 = z0 - k2.astype(F32) + F32(2.0) * _G3
    x3 = x0 - F32(1.0) + F32(3.0) * _G3
    y3 = y0 - F32(1.0) + F32(3.0) * _G3
    z3 = z0 - F32(1.0) + F32(3.0) * _G3

    def corner(ii, jj, kk, xx, yy, zz):
        t0 = F32(0.6) - xx * xx - yy * yy - zz * zz
        m = t0 > 0.0
        t0 = np.where(m, t0, F32(0.0))
        t4 = t0 * t0 * t0 * t0
        return t4 * _grad_dot(ii, jj, kk, xx, yy, zz, seed)

    n = (corner(i, j, k, x0, y0, z0)
         + corner(i + i1, j + j1, k + k1, x1, y1, z1)
         + corner(i + i2, j + j2, k + k2, x2, y2, z2)
         + corner(i + 1, j + 1, k + 1, x3, y3, z3))
    return n * F32(32.0)


# ==========================================================================
# cellular / Worley noise
# ==========================================================================
def worley3(p, seed=0, feature="f1", jitter=1.0):
    """Cellular noise over a 3x3x3 neighbourhood.

    ``feature`` selects the returned statistic:

    ``f1``     distance to the nearest feature point (range ~[0, 1.2])
    ``f2``     distance to the second nearest
    ``f2f1``   F2 - F1, the classic "cracked cell wall" pattern
    ``cell``   a flat random value per cell, for id-style masking

    Used for solar granulation, nebula clumping and disc turbulence.
    """
    p = np.asarray(p, F32)
    pf = np.floor(p)
    i = pf.astype(np.int32)
    f = p - pf

    n = p.shape[0]
    best1 = np.full(n, F32(1e9), F32)
    best2 = np.full(n, F32(1e9), F32)
    cell_id = np.zeros(n, F32)

    jx = F32(jitter)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cx = i[:, 0] + dx
                cy = i[:, 1] + dy
                cz = i[:, 2] + dz
                # three decorrelated hashes give the feature point offset
                ox = _hash_unit(cx, cy, cz, seed) * jx
                oy = _hash_unit(cx, cy, cz, seed + 7919) * jx
                oz = _hash_unit(cx, cy, cz, seed + 15485863) * jx
                rx = F32(dx) + ox - f[:, 0]
                ry = F32(dy) + oy - f[:, 1]
                rz = F32(dz) + oz - f[:, 2]
                d = rx * rx + ry * ry + rz * rz
                closer = d < best1
                best2 = np.where(closer, best1, np.minimum(best2, d))
                if feature == "cell":
                    cell_id = np.where(
                        closer, _hash_unit(cx, cy, cz, seed + 104729), cell_id)
                best1 = np.where(closer, d, best1)

    if feature == "cell":
        return cell_id
    if feature == "f2":
        return np.sqrt(best2)
    if feature == "f2f1":
        return np.sqrt(best2) - np.sqrt(best1)
    return np.sqrt(best1)


# ==========================================================================
# fractal operators
# ==========================================================================
_BASIS = {
    "value": value3,
    "perlin": perlin3,
    "simplex": simplex3,
}


def fbm3(p, octaves=5, lacunarity=2.02, gain=0.5, seed=0, basis="perlin",
         amplitude=1.0, frequency=1.0, normalise=True):
    """Fractional Brownian motion.

    ``lacunarity`` is deliberately defaulted to 2.02 rather than exactly 2.0:
    an exact doubling makes successive octaves share lattice alignment, which
    produces faint grid-locked ridges in large smooth fields such as nebulae.
    """
    fn = _BASIS[basis]
    p = np.asarray(p, F32)
    total = np.zeros(p.shape[0], F32)
    amp = F32(amplitude)
    freq = F32(frequency)
    norm = F32(0.0)
    for o in range(int(octaves)):
        total += fn(p * freq, seed + o * 1013) * amp
        norm += amp
        amp *= F32(gain)
        freq *= F32(lacunarity)
    if normalise and norm > 0.0:
        total /= norm
    return total


def turbulence3(p, octaves=5, lacunarity=2.02, gain=0.5, seed=0,
                basis="perlin", frequency=1.0):
    """Sum of absolute octaves — billowy, cloud-like, always positive."""
    fn = _BASIS[basis]
    p = np.asarray(p, F32)
    total = np.zeros(p.shape[0], F32)
    amp = F32(1.0)
    freq = F32(frequency)
    norm = F32(0.0)
    for o in range(int(octaves)):
        total += np.abs(fn(p * freq, seed + o * 1013)) * amp
        norm += amp
        amp *= F32(gain)
        freq *= F32(lacunarity)
    if norm > 0.0:
        total /= norm
    return total


def ridged3(p, octaves=5, lacunarity=2.02, gain=0.5, seed=0, basis="perlin",
            frequency=1.0, sharpness=1.0, weight_boost=1.6):
    """Ridged multifractal.

    Each octave is weighted by the previous one, so detail concentrates on
    the ridges instead of smearing uniformly.  This is what gives solar
    filaments and accretion-disc streamers their stringy structure.
    """
    fn = _BASIS[basis]
    p = np.asarray(p, F32)
    total = np.zeros(p.shape[0], F32)
    amp = F32(1.0)
    freq = F32(frequency)
    norm = F32(0.0)
    prev = np.ones(p.shape[0], F32)
    for o in range(int(octaves)):
        s = F32(1.0) - np.abs(fn(p * freq, seed + o * 1013))
        s *= s
        if sharpness != 1.0:
            s = s ** F32(sharpness)
        s *= np.clip(prev * F32(weight_boost), F32(0.0), F32(1.0))
        total += s * amp
        norm += amp
        prev = s
        amp *= F32(gain)
        freq *= F32(lacunarity)
    if norm > 0.0:
        total /= norm
    return total


def billow3(p, octaves=5, lacunarity=2.02, gain=0.5, seed=0, basis="perlin",
            frequency=1.0):
    """Billow — |fbm| remapped to [-1, 1]; puffier than plain fbm."""
    return turbulence3(p, octaves, lacunarity, gain, seed, basis, frequency) \
        * F32(2.0) - F32(1.0)


# ==========================================================================
# derivative-based operators
# ==========================================================================
def gradient3(p, fn, eps=1e-2, **kw):
    """Central-difference gradient of a scalar field.  Costs six samples."""
    p = np.asarray(p, F32)
    e = F32(eps)
    ex = np.zeros_like(p); ex[:, 0] = e
    ey = np.zeros_like(p); ey[:, 1] = e
    ez = np.zeros_like(p); ez[:, 2] = e
    inv = F32(0.5 / eps)
    gx = (fn(p + ex, **kw) - fn(p - ex, **kw)) * inv
    gy = (fn(p + ey, **kw) - fn(p - ey, **kw)) * inv
    gz = (fn(p + ez, **kw) - fn(p - ez, **kw)) * inv
    return np.stack([gx, gy, gz], axis=-1).astype(F32, copy=False)


def curl3(p, eps=1e-2, seed=0, octaves=3, frequency=1.0, scale=1.0):
    """Curl of a 3-component fBm potential field.

    The curl of any vector field is divergence-free by construction, so
    advecting particles or density along it produces incompressible,
    swirling motion with no sources or sinks.  This is the single most
    useful primitive for plasma, coronal loops and disc turbulence.

    Costs six fBm evaluations; returns an ``(N, 3)`` velocity field.
    """
    p = np.asarray(p, F32) * F32(frequency)
    e = F32(eps)
    ex = np.zeros_like(p); ex[:, 0] = e
    ey = np.zeros_like(p); ey[:, 1] = e
    ez = np.zeros_like(p); ez[:, 2] = e
    inv = F32(0.5 / eps)

    def pot(q, s):
        return fbm3(q, octaves=octaves, seed=s, basis="perlin")

    # dPz/dy - dPy/dz, dPx/dz - dPz/dx, dPy/dx - dPx/dy
    pz_y = (pot(p + ey, 3) - pot(p - ey, 3)) * inv
    py_z = (pot(p + ez, 2) - pot(p - ez, 2)) * inv
    px_z = (pot(p + ez, 1) - pot(p - ez, 1)) * inv
    pz_x = (pot(p + ex, 3) - pot(p - ex, 3)) * inv
    py_x = (pot(p + ex, 2) - pot(p - ex, 2)) * inv
    px_y = (pot(p + ey, 1) - pot(p - ey, 1)) * inv

    return np.stack([pz_y - py_z, px_z - pz_x, py_x - px_y],
                    axis=-1).astype(F32, copy=False) * F32(scale)


def domain_warp(p, strength=1.0, seed=0, octaves=3, frequency=1.0):
    """Offset sample positions by a vector noise field.

    Two levels of warp (applying this twice) is what turns bland fBm into
    the filamentary, stretched structure real nebulae show.
    """
    p = np.asarray(p, F32)
    q = p * F32(frequency)
    wx = fbm3(q, octaves=octaves, seed=seed + 11, basis="perlin")
    wy = fbm3(q, octaves=octaves, seed=seed + 23, basis="perlin")
    wz = fbm3(q, octaves=octaves, seed=seed + 37, basis="perlin")
    w = np.stack([wx, wy, wz], axis=-1).astype(F32, copy=False)
    return p + w * F32(strength)


# ==========================================================================
# tileable / spherical helpers
# ==========================================================================
def flow_noise3(p, t, seed=0, octaves=4, frequency=1.0, rate=0.35):
    """Noise that rotates its lattice over time.

    Scrolling a 3-D field along one axis is the usual cheap animation, but
    the eye reads the constant drift direction immediately.  Advancing two
    decorrelated fields in opposite directions and cross-fading hides it.
    """
    p = np.asarray(p, F32) * F32(frequency)
    phase = F32(t) * F32(rate)
    a = p.copy()
    a[:, 2] += phase
    b = p.copy()
    b[:, 0] -= phase * F32(0.77)
    b[:, 1] += phase * F32(0.41)
    k = F32(0.5) + F32(0.5) * np.float32(np.sin(float(phase) * 0.6))
    na = fbm3(a, octaves=octaves, seed=seed)
    nb = fbm3(b, octaves=octaves, seed=seed + 5501)
    return na * (F32(1.0) - k) + nb * k


def spherical_fbm(dirs, octaves=5, seed=0, frequency=1.0, gain=0.5):
    """fBm evaluated on unit directions — seamless over a sphere.

    Sampling a 3-D field on the unit sphere avoids every UV seam and polar
    pinch that a lat-long texture would introduce on stars and planets.
    """
    return fbm3(np.asarray(dirs, F32) * F32(frequency), octaves=octaves,
                gain=gain, seed=seed, basis="perlin")


__all__ = [
    "value3", "perlin3", "simplex3", "worley3",
    "fbm3", "turbulence3", "ridged3", "billow3",
    "gradient3", "curl3", "domain_warp",
    "flow_noise3", "spherical_fbm",
]
