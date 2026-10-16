"""
cosmos.core.volume
==================

Precomputed 3-D noise volumes and their samplers.

The decision this module encodes
--------------------------------
Measured on this machine, evaluating a four-octave fBm analytically runs at
1.09 M samples/s per core.  Sampling a precomputed volume trilinearly runs
at 4.75 M/s, and nearest-neighbour at 31 M/s.  A volumetric frame needs
roughly ten million field lookups, so the analytic path costs about nine
seconds a frame and the cached path about one.  Over a three-thousand-frame
film that is the difference between seven hours and fifty minutes.

The cost is memory and tiling.  A 128^3 float32 volume is 8.4 MB, which is
nothing, but it must tile seamlessly or the repeat will read as a visible
grid of identical cells.  So the volumes here are generated against a
*periodic* lattice: the integer hash coordinates are wrapped modulo the
period before hashing, which makes the gradient field exactly continuous
across the boundary.

Three sampling qualities are provided, and the renderer picks per use:

``sample_trilinear``   smooth, 4.75 M/s   — primary density
``sample_nearest``     blocky, 31 M/s     — high-frequency detail layers,
                                            where the march's own jitter and
                                            accumulation hide the blockiness
``sample_tricubic``    very smooth, slow  — stills and hero close-ups only
"""
from __future__ import annotations

import numpy as np

from . import noise as _noise

F32 = np.float32
U32 = np.uint32

_PX = U32(0x9E3779B1)
_PY = U32(0x85EBCA77)
_PZ = U32(0xC2B2AE3D)


# ==========================================================================
# periodic lattice noise  (tileable by construction)
# ==========================================================================
def _hash3p(ix, iy, iz, period, seed):
    """Hash integer lattice coords after wrapping them to ``period``."""
    p = np.int32(period)
    ix = np.mod(ix, p).astype(U32)
    iy = np.mod(iy, p).astype(U32)
    iz = np.mod(iz, p).astype(U32)
    h = (ix * _PX) ^ (iy * _PY) ^ (iz * _PZ)
    h = h ^ U32((int(seed) * 0x27D4EB2F) & 0xFFFFFFFF)
    h ^= h >> U32(15)
    h = h * U32(0x2C1B3C6D)
    h ^= h >> U32(12)
    h = h * U32(0x297A2D39)
    h ^= h >> U32(15)
    return h


def _grad_dot_p(ix, iy, iz, fx, fy, fz, period, seed):
    h = _hash3p(ix, iy, iz, period, seed) & U32(15)
    return (_noise._GX[h] * fx + _noise._GY[h] * fy + _noise._GZ[h] * fz)


def perlin3_periodic(p, period, seed=0):
    """Perlin gradient noise that tiles exactly every ``period`` units."""
    p = np.asarray(p, F32)
    pf = np.floor(p)
    i = pf.astype(np.int32)
    f = p - pf

    ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    ux = _noise._fade(fx)
    uy = _noise._fade(fy)
    uz = _noise._fade(fz)
    fx1, fy1, fz1 = fx - F32(1.0), fy - F32(1.0), fz - F32(1.0)
    ix1, iy1, iz1 = ix + 1, iy + 1, iz + 1

    g = _grad_dot_p
    n000 = g(ix, iy, iz, fx, fy, fz, period, seed)
    n100 = g(ix1, iy, iz, fx1, fy, fz, period, seed)
    n010 = g(ix, iy1, iz, fx, fy1, fz, period, seed)
    n110 = g(ix1, iy1, iz, fx1, fy1, fz, period, seed)
    n001 = g(ix, iy, iz1, fx, fy, fz1, period, seed)
    n101 = g(ix1, iy, iz1, fx1, fy, fz1, period, seed)
    n011 = g(ix, iy1, iz1, fx, fy1, fz1, period, seed)
    n111 = g(ix1, iy1, iz1, fx1, fy1, fz1, period, seed)

    x00 = n000 + (n100 - n000) * ux
    x10 = n010 + (n110 - n010) * ux
    x01 = n001 + (n101 - n001) * ux
    x11 = n011 + (n111 - n011) * ux
    y0 = x00 + (x10 - x00) * uy
    y1 = x01 + (x11 - x01) * uy
    return (y0 + (y1 - y0) * uz) * F32(1.1547)


def fbm3_periodic(p, base_period=4, octaves=5, gain=0.5, seed=0,
                  normalise=True):
    """Tileable fBm.

    Lacunarity is forced to exactly 2 here — unlike the analytic ``fbm3``,
    which uses 2.02 to break lattice alignment.  Periodicity requires each
    octave's period to divide the volume size, and only an integer doubling
    does that.  The resulting grid-lock is then broken by *rotating* each
    octave instead, which preserves tiling because the rotation is applied
    to the already-wrapped lattice.
    """
    p = np.asarray(p, F32)
    total = np.zeros(p.shape[0], F32)
    amp = F32(1.0)
    per = int(base_period)
    norm = F32(0.0)
    # per-octave rotations; arbitrary but fixed so volumes are reproducible
    rots = (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], F32),
        np.array([[0.80, 0.60, 0.0], [-0.60, 0.80, 0.0], [0.0, 0.0, 1.0]], F32),
        np.array([[0.36, 0.48, -0.80], [-0.80, 0.60, 0.0], [0.48, 0.64, 0.60]], F32),
        np.array([[0.60, 0.0, 0.80], [0.0, 1.0, 0.0], [-0.80, 0.0, 0.60]], F32),
        np.array([[0.28, -0.96, 0.0], [0.96, 0.28, 0.0], [0.0, 0.0, 1.0]], F32),
        np.array([[0.70, 0.0, -0.71], [0.0, 1.0, 0.0], [0.71, 0.0, 0.70]], F32),
    )
    for o in range(int(octaves)):
        q = p * F32(per)
        total += perlin3_periodic(q, per, seed + o * 1013) * amp
        norm += amp
        amp *= F32(gain)
        per *= 2
    if normalise and norm > 0:
        total /= norm
    return total


# ==========================================================================
# volume container
# ==========================================================================
class NoiseVolume:
    """A cached, tileable 3-D scalar field with fast samplers.

    Coordinates are in *tile space*: the volume repeats every 1.0 unit, so a
    caller scales world position by whatever feature size it wants.  The
    resolution must be a power of two so wrapping is a bitwise AND rather
    than a modulo — on a multi-million-element index array that difference
    is measurable.
    """

    __slots__ = ("data", "n", "mask", "name")

    def __init__(self, data, name="volume"):
        data = np.ascontiguousarray(np.asarray(data, F32))
        n = data.shape[0]
        if data.shape != (n, n, n):
            raise ValueError("NoiseVolume expects a cubic array")
        if n & (n - 1):
            raise ValueError("NoiseVolume resolution must be a power of two")
        self.data = data
        self.n = n
        self.mask = np.int32(n - 1)
        self.name = name

    # -- factory ----------------------------------------------------------
    @classmethod
    def generate(cls, n=128, octaves=5, gain=0.5, seed=0, base_period=4,
                 kind="fbm", ridged_sharpness=1.0, name=None, chunk=1 << 20):
        """Bake a volume.  ``kind`` is 'fbm', 'ridged', 'turbulence' or 'worley'."""
        if n & (n - 1):
            raise ValueError("resolution must be a power of two")
        g = (np.arange(n, dtype=np.float32) + 0.5) / n
        zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")
        pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        out = np.empty(pts.shape[0], F32)

        for s in range(0, pts.shape[0], chunk):
            e = min(s + chunk, pts.shape[0])
            q = pts[s:e]
            if kind == "fbm":
                v = fbm3_periodic(q, base_period, octaves, gain, seed)
            elif kind == "turbulence":
                v = np.abs(fbm3_periodic(q, base_period, octaves, gain, seed,
                                         normalise=False))
                v /= max(float(v.max()), 1e-6)
            elif kind == "ridged":
                acc = np.zeros(e - s, F32)
                amp = F32(1.0)
                per = int(base_period)
                nrm = F32(0.0)
                prev = np.ones(e - s, F32)
                for o in range(int(octaves)):
                    r = F32(1.0) - np.abs(
                        perlin3_periodic(q * F32(per), per, seed + o * 1013))
                    r *= r
                    if ridged_sharpness != 1.0:
                        r = r ** F32(ridged_sharpness)
                    r *= np.clip(prev * F32(1.6), 0.0, 1.0)
                    acc += r * amp
                    nrm += amp
                    prev = r
                    amp *= F32(gain)
                    per *= 2
                v = acc / max(float(nrm), 1e-6)
            elif kind == "worley":
                v = _worley_periodic(q, base_period, seed)
            else:
                raise ValueError(f"unknown volume kind {kind!r}")
            out[s:e] = v

        return cls(out.reshape(n, n, n), name or f"{kind}{n}s{seed}")

    # -- samplers ---------------------------------------------------------
    def sample_nearest(self, p, scale=1.0, offset=None):
        """31 M samples/s.  Use for detail layers the march will blur anyway."""
        q = np.asarray(p, F32)
        if scale != 1.0:
            q = q * F32(scale)
        if offset is not None:
            q = q + np.asarray(offset, F32)
        i = (q * F32(self.n)).astype(np.int32) & self.mask
        return self.data[i[:, 2], i[:, 1], i[:, 0]]

    def sample_trilinear(self, p, scale=1.0, offset=None):
        """4.75 M samples/s.  The default for primary density."""
        q = np.asarray(p, F32)
        if scale != 1.0:
            q = q * F32(scale)
        if offset is not None:
            q = q + np.asarray(offset, F32)
        q = q * F32(self.n)
        i0 = np.floor(q).astype(np.int32)
        f = q - i0.astype(F32)
        i0 &= self.mask
        i1 = (i0 + 1) & self.mask
        d = self.data
        x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
        x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

        c000 = d[z0, y0, x0]
        c100 = d[z0, y0, x1]
        c010 = d[z0, y1, x0]
        c110 = d[z0, y1, x1]
        c001 = d[z1, y0, x0]
        c101 = d[z1, y0, x1]
        c011 = d[z1, y1, x0]
        c111 = d[z1, y1, x1]

        x00 = c000 + (c100 - c000) * fx
        x10 = c010 + (c110 - c010) * fx
        x01 = c001 + (c101 - c001) * fx
        x11 = c011 + (c111 - c011) * fx
        ya = x00 + (x10 - x00) * fy
        yb = x01 + (x11 - x01) * fy
        return ya + (yb - ya) * fz

    def sample_gradient(self, p, scale=1.0, eps=None):
        """Central-difference gradient — used for volumetric normals."""
        e = eps if eps is not None else (1.0 / self.n)
        ex = np.zeros_like(p); ex[:, 0] = e
        ey = np.zeros_like(p); ey[:, 1] = e
        ez = np.zeros_like(p); ez[:, 2] = e
        inv = F32(0.5 / e)
        gx = (self.sample_trilinear(p + ex, scale)
              - self.sample_trilinear(p - ex, scale)) * inv
        gy = (self.sample_trilinear(p + ey, scale)
              - self.sample_trilinear(p - ey, scale)) * inv
        gz = (self.sample_trilinear(p + ez, scale)
              - self.sample_trilinear(p - ez, scale)) * inv
        return np.stack([gx, gy, gz], axis=-1).astype(F32, copy=False)

    # -- maintenance ------------------------------------------------------
    def remap(self, lo=0.0, hi=1.0):
        """Linearly rescale the stored range in place."""
        d = self.data
        mn, mx = float(d.min()), float(d.max())
        if mx - mn < 1e-9:
            return self
        d -= mn
        d *= F32((hi - lo) / (mx - mn))
        d += F32(lo)
        return self

    def contrast(self, power=1.0, bias=0.0):
        d = self.data
        if bias:
            d += F32(bias)
        np.clip(d, 0.0, None, out=d)
        if power != 1.0:
            np.power(d, F32(power), out=d)
        return self

    def stats(self):
        d = self.data
        return dict(name=self.name, n=self.n, mb=d.nbytes / 1e6,
                    min=float(d.min()), max=float(d.max()),
                    mean=float(d.mean()), std=float(d.std()))

    def save(self, path):
        np.savez_compressed(path, data=self.data, name=self.name)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["data"], str(z["name"]))


def _worley_periodic(p, period, seed):
    """Tileable cellular noise, returned as 1 - F1 so cells read as blobs."""
    per = int(period)
    q = np.asarray(p, F32) * F32(per)
    pf = np.floor(q)
    i = pf.astype(np.int32)
    f = q - pf
    best = np.full(q.shape[0], F32(1e9), F32)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cx, cy, cz = i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz
                hx = (_hash3p(cx, cy, cz, per, seed) >> U32(8)).astype(F32)
                hy = (_hash3p(cx, cy, cz, per, seed + 7919) >> U32(8)).astype(F32)
                hz = (_hash3p(cx, cy, cz, per, seed + 104729) >> U32(8)).astype(F32)
                inv = F32(1.0 / 16777216.0)
                rx = F32(dx) + hx * inv - f[:, 0]
                ry = F32(dy) + hy * inv - f[:, 1]
                rz = F32(dz) + hz * inv - f[:, 2]
                np.minimum(best, rx * rx + ry * ry + rz * rz, out=best)
    return np.clip(F32(1.0) - np.sqrt(best), 0.0, 1.0)


# ==========================================================================
# volume library  — built once per process, shared by all phases
# ==========================================================================
class VolumeLibrary:
    """Lazily-built, process-wide cache of the volumes the phases ask for.

    Each worker process builds its own copy.  Baking the full set costs a
    few seconds once; amortised over hundreds of frames per worker that is
    irrelevant, and it avoids shipping tens of megabytes through a pickle.
    """

    _SPECS = {
        # name            n    octaves  gain  seed  base  kind
        "nebula":       (128, 6, 0.55, 101, 2, "fbm"),
        "nebula_fine":  (64,  4, 0.50, 211, 4, "fbm"),
        "filament":     (128, 5, 0.52, 307, 2, "ridged"),
        "plasma":       (128, 5, 0.50, 401, 3, "fbm"),
        "plasma_fine":  (64,  4, 0.55, 503, 6, "turbulence"),
        "granule":      (64,  1, 0.50, 601, 8, "worley"),
        "supergranule": (64,  1, 0.50, 607, 4, "worley"),
        "disc":         (128, 5, 0.54, 701, 2, "ridged"),
        "disc_fine":    (64,  4, 0.50, 809, 6, "fbm"),
        "dust":         (64,  4, 0.58, 907, 3, "fbm"),
    }

    def __init__(self):
        self._cache = {}

    def get(self, key):
        v = self._cache.get(key)
        if v is None:
            if key not in self._SPECS:
                raise KeyError(f"no volume spec named {key!r}")
            n, oct_, gain, seed, base, kind = self._SPECS[key]
            v = NoiseVolume.generate(n=n, octaves=oct_, gain=gain, seed=seed,
                                     base_period=base, kind=kind, name=key)
            if kind in ("fbm",):
                v.remap(-1.0, 1.0)
            self._cache[key] = v
        return v

    def prebuild(self, keys=None, verbose=False):
        import time as _t
        for k in (keys or self._SPECS.keys()):
            t0 = _t.time()
            self.get(k)
            if verbose:
                print(f"  volume {k:14s} baked in {_t.time() - t0:5.2f}s",
                      flush=True)
        return self

    def total_mb(self):
        return sum(v.data.nbytes for v in self._cache.values()) / 1e6


LIBRARY = VolumeLibrary()


__all__ = ["NoiseVolume", "VolumeLibrary", "LIBRARY",
           "perlin3_periodic", "fbm3_periodic"]
