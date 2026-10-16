"""
cosmos.phases.p3_sun
====================

FASE III — MATAHARI.

Plasma thermodynamics, magnetic structure, flares and a coronal mass
ejection.  The Sun is the one object in this film where the magnetic field
matters more than gravity, and the phase is built around that fact: almost
every visible feature of the solar atmosphere is the shape of a field line
made luminous.

The physical model
------------------
**Photosphere.**  A hard surface at ``r = 1`` with limb darkening from the
Eddington grey atmosphere, convective granulation from a Worley cell field,
and sunspot umbrae where the field is strong enough to choke convection.  A
spot is about 1500 K cooler than the surrounding photosphere, and because
the emitted colour is a blackbody at the local temperature, the umbra comes
out dark orange *for free* rather than being painted on.

**Differential rotation.**  The Sun is not solid.  Its equator turns in
24.5 days and its poles in nearly 35, which is what shears the global
magnetic field and ultimately drives the whole activity cycle.  The surface
shader applies the measured law

    Omega(theta) = A + B sin^2(theta) + C sin^4(theta)

with the Snodgrass coefficients, so features at different latitudes drift
apart over the shot.

**Corona.**  A marched volume from the limb out to several solar radii.
Density is organised by the magnetic field rather than by radius alone: for
a dipole the field line through a point is labelled by its L-shell,
``L = r / sin^2(theta)``, and the line is closed if ``L`` is small enough
to return to the surface.  Closed field traps plasma — that is a helmet
streamer.  Open field lets it escape — that is a coronal hole, and it is
why the poles of the real corona are dark.  Nothing about that structure is
drawn by hand; it falls out of one formula.

**Active regions and loops.**  Bipolar magnetic pairs on the surface, with
plasma-filled arcs joining their footpoints.  The arcs are baked once into
a volume at setup rather than evaluated per sample, because the distance
from a point to a circle costs far too much to run ten times inside the
inner loop.

**Flares and the coronal mass ejection.**  Magnetic reconnection releases
energy impulsively: a fast rise and a slow exponential decay.  The CME is a
flux rope launched from an active region that expands self-similarly at
constant speed, which is what a real one does once it clears the corona.

Colour
------
The photosphere is a blackbody.  The corona is not: at one to two million
kelvin its visible output is dominated by forbidden emission lines of
highly ionised iron — Fe XIV at 530.3 nm (green) and Fe X at 637.4 nm
(red) — plus Thomson-scattered photospheric light, which is white.  Using
Planck's law for the corona would make it a deep blue that no coronagraph
has ever recorded, so the coronal ramp is built from the actual line
wavelengths through the CIE observer.

Units
-----
One scene unit is one solar radius.  Time is in hours.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import camera as CAM
from ..core import march as M
from ..core import spectrum as S
from ..core import vmath as V
from ..core import vcache
from .base import (Phase, build_emission_ramp, march_layer, project_and_splat,
                   ramp_lookup, upsample_linear)

F32 = np.float32
TAU = 2.0 * math.pi

#: kilometres per solar radius
KM_PER_RSUN = 695700.0
#: effective temperature of the photosphere, K
T_EFF = 5772.0


# ==========================================================================
# parameters
# ==========================================================================
@dataclass
class SunParams:
    """Physical and structural parameters of the star."""

    # --- rotation (Snodgrass 1990, magnetic tracers), degrees per day ---
    omega_a: float = 14.713
    omega_b: float = -2.396
    omega_c: float = -1.787
    tilt_deg: float = 7.25          # inclination of the solar equator

    # --- atmosphere -----------------------------------------------------
    r_photosphere: float = 1.0
    h_chromo: float = 0.0035        # chromospheric scale height, R_sun
    r_corona: float = 7.0           # outer bound of the marched corona
    h_corona: float = 0.52          # coronal density scale height
    t_corona: float = 1.6e6         # coronal temperature, K

    # --- magnetic structure ---------------------------------------------
    dipole_tilt: float = 11.0       # magnetic axis offset from rotation
    l_shell_max: float = 3.4        # largest closed field line
    streamer_sharpness: float = 2.2
    n_active: int = 7               # active-region bipoles
    n_loops: int = 26               # coronal loops
    spot_strength: float = 1.0

    # --- granulation ----------------------------------------------------
    granule_scale: float = 44.0
    supergranule_scale: float = 7.0

    # --- timing (hours of simulated time per second of film) -----------
    hours_per_second: float = 2.1


def differential_rotation(sin_lat, p: SunParams):
    """Angular velocity in radians per hour at a given latitude.

    ``Omega = A + B sin^2 + C sin^4`` degrees per day, converted.  The
    equator laps the poles roughly every four months, which is the shear
    that winds up the global field and makes the activity cycle run.
    """
    s2 = sin_lat * sin_lat
    deg_per_day = p.omega_a + p.omega_b * s2 + p.omega_c * s2 * s2
    return np.radians(deg_per_day) / 24.0


# ==========================================================================
# magnetic field model
# ==========================================================================
class MagneticField:
    """Global dipole plus localised bipolar active regions.

    The global field sets where plasma is confined and where it escapes;
    the bipoles set where the interesting things happen.  Both are analytic,
    so the field can be evaluated anywhere without a grid.
    """

    def __init__(self, p: SunParams, seed=5):
        self.p = p
        rng = np.random.default_rng(seed)
        tilt = math.radians(p.dipole_tilt)
        self.axis = np.array([math.sin(tilt), math.cos(tilt), 0.0], F32)

        # Active regions cluster in two activity belts either side of the
        # equator — the butterfly diagram — never at the poles.
        n = p.n_active
        lat = (rng.normal(0.0, 0.17, n)
               + np.where(rng.random(n) < 0.5, -0.24, 0.24))
        lon = rng.random(n) * TAU
        self.ar_lat = lat.astype(F32)
        self.ar_lon = lon.astype(F32)
        self.ar_strength = (0.45 + rng.random(n) * 0.85).astype(F32)
        self.ar_sep = (0.07 + rng.random(n) * 0.09).astype(F32)
        self.ar_tilt = (rng.normal(0.0, 0.35, n)).astype(F32)
        self.ar_size = (0.055 + rng.random(n) * 0.075).astype(F32)

    # ----------------------------------------------------------------
    def footpoints(self):
        """Surface positions of the two poles of each active region.

        Joy's law: the leading spot of a bipolar group sits closer to the
        equator than the trailing one, and the tilt grows with latitude.
        """
        lat, lon = self.ar_lat, self.ar_lon
        tilt = self.ar_tilt + np.sign(lat) * np.abs(lat) * F32(0.55)
        d = self.ar_sep * F32(0.5)
        out = []
        for sign in (-1.0, 1.0):
            la = lat + F32(sign) * d * np.sin(tilt)
            lo = lon + F32(sign) * d * np.cos(tilt) / np.maximum(
                np.cos(lat), F32(0.25))
            cl = np.cos(la)
            out.append(np.stack([cl * np.cos(lo), np.sin(la),
                                 cl * np.sin(lo)], axis=1).astype(F32))
        return out[0], out[1]

    def l_shell(self, pos, r):
        """Dipole L-shell of each point: ``L = r / sin^2(colatitude)``.

        Points on the same field line share an L value, so this single
        scalar separates the closed, plasma-filled part of the corona from
        the open, evacuated part without tracing a single line.
        """
        cos_t = np.einsum('ij,j->i', pos, self.axis) / np.maximum(r, 1e-6)
        sin2 = np.maximum(F32(1.0) - cos_t * cos_t, F32(1e-4))
        return r / sin2, cos_t

    def surface_strength(self, dirs):
        """Unsigned field strength on the photosphere, for sunspots."""
        p1, p2 = self.footpoints()
        acc = np.zeros(dirs.shape[0], F32)
        for pts, sign in ((p1, 1.0), (p2, -1.0)):
            for k in range(pts.shape[0]):
                d = np.einsum('ij,j->i', dirs, pts[k])
                w = np.exp(-(F32(1.0) - d) / F32(self.ar_size[k] ** 2 * 0.5
                                                 + 1e-5))
                acc += w * self.ar_strength[k]
        return acc


# ==========================================================================
# coronal loops, baked
# ==========================================================================
def bake_loop_volume(p: SunParams, mag: MagneticField, n=128, seed=5,
                     extent=3.0):
    """Rasterise plasma-filled magnetic arcs into a 3-D volume.

    The distance from a point to a circular arc is cheap for one arc and
    ruinous for twenty-six evaluated per sample inside the march.  Since
    the loops are static in the frame that co-rotates with the surface,
    they are baked once here and then sampled like any other volume, with
    the lookup coordinates rotated to follow the rotation.

    ``extent`` is the half-width of the baked cube in solar radii.
    """
    rng = np.random.default_rng(seed + 17)
    f1, f2 = mag.footpoints()
    n_ar = f1.shape[0]

    # choose loops: several per active region, with a spread of heights
    loops = []
    for k in range(p.n_loops):
        i = k % n_ar
        jitter = rng.normal(0.0, 0.018, 3).astype(np.float64)
        a = f1[i].astype(np.float64) + jitter
        b = f2[i].astype(np.float64) - jitter
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        # scale the separation so loops of different heights nest
        grow = 0.55 + 1.75 * (k // n_ar) / max(1, p.n_loops // n_ar)
        mid = (a + b) * 0.5
        nm = np.linalg.norm(mid)
        if nm < 1e-6:
            continue
        mid /= nm
        half = (b - a) * 0.5 * grow
        a2 = mid - half
        b2 = mid + half
        a2 /= np.linalg.norm(a2)
        b2 /= np.linalg.norm(b2)
        centre = (a2 + b2) * 0.5
        radius = float(np.linalg.norm(b2 - a2) * 0.5)
        if radius < 1e-4:
            continue
        axis = np.cross(b2 - a2, centre)
        na = np.linalg.norm(axis)
        if na < 1e-9:
            continue
        axis /= na
        bright = float(mag.ar_strength[i]) * (0.45 + rng.random() * 0.9)
        tube = 0.016 + rng.random() * 0.030
        loops.append((centre, radius, axis, tube, bright))

    # rasterise
    lin = (np.arange(n, dtype=np.float32) + 0.5) / n * (2.0 * extent) - extent
    gz, gy, gx = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    acc = np.zeros(pts.shape[0], np.float32)

    for centre, radius, axis, tube, bright in loops:
        rel = pts - centre.astype(np.float32)[None, :]
        ax = np.einsum('ij,j->i', rel, axis.astype(np.float32))
        # distance to the circle of radius `radius` in the plane
        perp = rel - ax[:, None] * axis.astype(np.float32)[None, :]
        rp = np.sqrt(np.einsum('ij,ij->i', perp, perp))
        d = np.sqrt((rp - np.float32(radius)) ** 2 + ax * ax)
        acc += np.float32(bright) * np.exp(-(d / np.float32(tube)) ** 2)

    # loops cannot exist below the surface
    rr = np.sqrt(np.einsum('ij,ij->i', pts, pts))
    acc *= np.clip((rr - 0.995) * 60.0, 0.0, 1.0)
    acc = acc.reshape(n, n, n)
    peak = float(acc.max())
    if peak > 0:
        acc /= peak
    return vcache.QuantVolume.from_float(acc.astype(np.float32),
                                         name="loops"), extent


# ==========================================================================
# transient events
# ==========================================================================
@dataclass
class Flare:
    """An impulsive brightening at an active region.

    The light curve is the canonical fast-rise / exponential-decay shape of
    magnetic reconnection: energy is released far faster than the corona
    can radiate it away, then cools on its own thermal timescale.
    """

    t0: float                    # film-time onset, seconds
    rise: float                  # rise time, seconds
    decay: float                 # e-folding decay, seconds
    direction: np.ndarray        # unit vector to the active region
    peak: float = 1.0
    height: float = 0.10
    width: float = 0.12

    def amplitude(self, t):
        if t < self.t0:
            return 0.0
        u = (t - self.t0) / max(self.rise, 1e-4)
        if u < 1.0:
            return self.peak * (u * u * (3.0 - 2.0 * u))
        return self.peak * math.exp(-(t - self.t0 - self.rise)
                                    / max(self.decay, 1e-4))


@dataclass
class CME:
    """A coronal mass ejection: a flux rope launched off the limb.

    Once clear of the low corona a CME expands self-similarly — its radius
    grows in proportion to the distance its leading edge has travelled —
    and coasts at roughly constant speed.  Both behaviours are reproduced
    here directly rather than keyframed.
    """

    t0: float
    direction: np.ndarray
    speed: float = 0.62          # solar radii per second of film
    width: float = 0.62          # angular half-width, radians
    rope_turns: float = 1.6
    peak: float = 1.0

    def front(self, t):
        return 1.0 + max(0.0, t - self.t0) * self.speed

    def amplitude(self, t):
        if t < self.t0:
            return 0.0
        u = (t - self.t0)
        return self.peak * (1.0 - math.exp(-u * 2.2)) * math.exp(-u * 0.11)


# ==========================================================================
# the corona shader
# ==========================================================================
class CoronaShader:
    """Emission and extinction of the solar atmosphere above the surface."""

    def __init__(self, p: SunParams, mag: MagneticField, lib, loop_vol,
                 loop_extent, seed=5):
        self.p = p
        self.mag = mag
        self.loops = loop_vol
        self.loop_extent = float(loop_extent)
        self.v_plasma = lib.get("plasma")
        self.v_fine = lib.get("plasma_fine")
        self.v_corona = lib.get("corona")
        self.v_gran = lib.get("granule")

        # Coronal emission.  Forbidden lines of highly ionised iron plus
        # Thomson scattering, not a blackbody — see the module docstring.
        self.ramp_corona = self._corona_ramp()
        # Chromosphere and prominences: optically thick hydrogen alpha.
        self.ramp_chromo = self._chromo_ramp()
        # Flare plasma reaches tens of millions of kelvin; its visible
        # continuum really is a hot blackbody, so this ramp is Planckian.
        self.ramp_flare = build_emission_ramp(
            temps=(6000, 9000, 14000, 22000, 34000),
            gains=(0.30, 0.95, 1.90, 3.20, 4.80), n=256)

        self.flares = []
        self.cmes = []
        self.jitter = None
        self._t = 0.0
        self._rot = np.eye(3, dtype=F32)
        self._spin = 0.0

    # ----------------------------------------------------------------
    @staticmethod
    def _cie_rgb(lam):
        M = np.array([[3.2406, -1.5372, -0.4986],
                      [-0.9689, 1.8758, 0.0415],
                      [0.0557, -0.2040, 1.0570]], np.float64)
        xb, yb, zb = S.cie_xyz_bar(np.array([lam], np.float32))
        xyz = np.array([float(xb[0]), float(yb[0]), float(zb[0])], np.float64)
        c = np.clip(M @ xyz, 0.0, None)
        return c / max(c.max(), 1e-9)

    def _corona_ramp(self, n=512):
        green = self._cie_rgb(530.3)      # Fe XIV, the green line
        red = self._cie_rgb(637.4)        # Fe X, the red line
        white = np.array([1.0, 0.985, 0.95])   # Thomson-scattered sunlight
        u = np.linspace(0.0, 1.0, n)[:, None]
        # cool, quiet corona is dominated by scattering; hot active regions
        # light up the iron lines
        mix = white * (1.0 - 0.45 * u) + green * u * 0.26 + red * (
            u * (1.0 - u)) * 0.34
        mix /= np.maximum(mix.max(axis=1, keepdims=True), 1e-9)
        gain = (0.16 + 2.4 * u.ravel() ** 1.25)[:, None]
        return np.ascontiguousarray((mix * gain).astype(F32))

    def _chromo_ramp(self, n=256):
        ha = self._cie_rgb(656.3)
        hb = self._cie_rgb(486.1)
        u = np.linspace(0.0, 1.0, n)[:, None]
        mix = ha * (1.0 - 0.30 * u) + hb * (0.30 * u)
        gain = (0.45 + 3.1 * u.ravel() ** 1.1)[:, None]
        return np.ascontiguousarray((mix * gain).astype(F32))

    # ----------------------------------------------------------------
    def set_time(self, t_film, t_hours):
        self._t = float(t_film)
        self._hours = float(t_hours)
        # equatorial rotation of the co-rotating frame used for the loops
        self._spin = float(differential_rotation(np.float32(0.0), self.p)
                           * t_hours)
        self._rot = V.rot_y(F32(-self._spin)).astype(F32)
        self._drift = np.array([0.0, 0.0, t_film * 0.012], F32)

    def new_step(self, rng):
        self.jitter = vcache.new_jitter(rng)

    # ----------------------------------------------------------------
    def __call__(self, pos, t, u):
        p = self.p
        r2 = np.einsum('ij,ij->i', pos, pos)
        r = np.sqrt(r2)
        inv_r = np.reciprocal(np.maximum(r, F32(1e-5)))
        dirs = pos * inv_r[:, None]

        above = np.maximum(r - F32(1.0), F32(0.0))

        # ---- radial density profile ------------------------------------
        # Baumbach-Allen: the electron corona's brightness goes as a sum of
        # steep inverse powers of radius, roughly r^-7 close in with a much
        # shallower r^-2.5 F-corona term taking over further out.  An
        # exponential scale height is the wrong model entirely — it decays
        # far too slowly and fogs the whole frame with uniform grey, which
        # is exactly what the first version of this shader did.
        ir2 = inv_r * inv_r
        ir4 = ir2 * ir2
        k_corona = ir4 * ir2 * inv_r          # r^-7, the K corona
        f_corona = ir2 * inv_r * F32(0.030)   # r^-3, the dust F corona
        base = k_corona + f_corona

        # Coronagraphs cannot show that range either: the inner corona is a
        # thousand times brighter than the outer, so every instrument from
        # Lyot onward puts a radially graded filter in the beam.  This is
        # the same filter, applied for the same reason.
        vignette = r * r * np.sqrt(r) * F32(1.35)
        base = base * vignette

        # ---- magnetic organisation -------------------------------------
        lsh, cos_t = self.mag.l_shell(pos, r)
        # closed field traps plasma, open field evacuates: this single
        # expression produces helmet streamers at the equator and dark
        # coronal holes over the poles
        closed = np.clip((F32(p.l_shell_max) - lsh)
                         * F32(1.0 / (p.l_shell_max * 0.42)), 0.0, 1.0)
        closed = closed * closed * closed
        # Polar coronal holes are genuinely dark — a factor of five or more
        # below the equatorial streamer belt in white light.
        streamer = F32(0.055) + F32(0.945) * closed

        # ---- texture ----------------------------------------------------
        q = pos @ self._rot
        # Low tiling frequencies.  The volumes are periodic, and the corona
        # is marched out to seven solar radii, so a scale of 1.35 wrapped
        # the texture nineteen times across the frame and printed a visible
        # square grid over the whole sky.
        ntex = self.v_corona.sample(q * F32(0.235), jitter=self.jitter)
        nfine = self.v_fine.sample(q * F32(0.78) + self._drift,
                                   jitter=self.jitter)
        # Combed along the field: sampled on the unit sphere of directions
        # with only a slow drift in radius, so features stretch radially
        # into streamer rays instead of looking like isotropic cloud.
        radial = self.v_plasma.sample(dirs * F32(1.15) + above[:, None]
                                      * F32(0.085), jitter=self.jitter)

        # Coronal plasma is combed into radial filaments by the field, so
        # the texture is stretched along r and compressed across it.
        fine_u = np.clip(nfine * F32(2.5) + F32(0.5), 0.0, 1.0)
        rad_u = np.clip(radial * F32(2.6) + F32(0.5), 0.0, 1.0)
        tex = ntex * F32(0.58) + fine_u * F32(0.42)
        tex = np.clip(tex * F32(1.45) - F32(0.22), 0.0, 1.6)
        # Polar plumes and equatorial rays: the corona is not a cloud, it
        # is a bundle of filaments combed straight out along open field.
        # Raising the radial term to a power is what turns a soft haze into
        # the fan of rays an eclipse photograph actually shows.
        ray = rad_u * rad_u
        ray = ray * ray * F32(2.6) + F32(0.22)
        tex *= ray

        dens = base * streamer * tex * F32(0.62)

        # ---- coronal loops ----------------------------------------------
        inside = (np.abs(q[:, 0]) < F32(self.loop_extent)) & \
                 (np.abs(q[:, 1]) < F32(self.loop_extent)) & \
                 (np.abs(q[:, 2]) < F32(self.loop_extent))
        loop = np.zeros_like(r)
        if inside.any():
            lp = (q[inside] * F32(0.5 / self.loop_extent)) + F32(0.5)
            loop[inside] = self.loops.sample(lp, jitter=self.jitter)
        loop = np.maximum(loop, F32(0.0))
        loop *= F32(0.55) + F32(0.45) * fine_u

        # ---- chromosphere / spicules -------------------------------------
        # The chromosphere is only about two thousand kilometres deep —
        # three thousandths of a solar radius — which is why it is visible
        # as a thin crimson ring at the instant of second contact and at no
        # other time.  Spicules reach several times higher.
        chromo = np.exp(-above * F32(1.0 / 0.0055))
        chromo += np.exp(-above * F32(1.0 / 0.024)) * F32(0.30)
        chromo *= F32(0.30) + F32(1.05) * np.clip(
            ntex * F32(2.2) + F32(0.40), 0.0, 1.0)
        chromo *= F32(0.55) + F32(0.75) * rad_u

        # ---- flares -------------------------------------------------------
        flare = np.zeros_like(r)
        for fl in self.flares:
            amp = fl.amplitude(self._t)
            if amp <= 1e-4:
                continue
            d = np.einsum('ij,j->i', dirs, fl.direction)
            ang = np.clip((d - F32(1.0 - fl.width ** 2 * 0.5))
                          / F32(fl.width ** 2 * 0.5 + 1e-6), 0.0, 1.0)
            h = np.exp(-above / F32(fl.height))
            flare += F32(amp) * ang * ang * h

        # ---- coronal mass ejection ----------------------------------------
        cme = np.zeros_like(r)
        for ej in self.cmes:
            amp = ej.amplitude(self._t)
            if amp <= 1e-4:
                continue
            front = F32(ej.front(self._t))
            d = np.einsum('ij,j->i', dirs, ej.direction)
            cone = np.clip((d - F32(math.cos(ej.width)))
                           / F32(1.0 - math.cos(ej.width) + 1e-6), 0.0, 1.0)
            cone = np.sqrt(cone)          # broaden the flanks
            # The classic three-part structure seen in every coronagraph
            # movie: a bright compressed front, a dark evacuated cavity
            # behind it, and the cool prominence core that used to be the
            # filament trailing at the bottom.
            thick = F32(0.13) * front
            shell = np.exp(-((r - front) / thick) ** 2)
            cavity = F32(1.0) - F32(0.78) * np.exp(
                -((r - front * F32(0.70)) / (thick * F32(1.9))) ** 2)
            core = np.exp(-((r - front * F32(0.46))
                            / (thick * F32(1.25))) ** 2) * F32(0.85)
            # the flux rope is twisted, so the front is not a smooth arc
            rope = F32(0.55) + F32(0.95) * np.clip(
                nfine * F32(3.4) + F32(0.5), 0.0, 1.0)
            cme += F32(amp) * cone * cone * (shell * cavity + core) * rope

        # ---- composition ---------------------------------------------------
        total = dens + loop * F32(1.9) + chromo * F32(1.1) \
            + flare * F32(2.4) + cme * F32(1.3)

        # ramp coordinate: how energetic is this parcel
        heat = np.clip(loop * F32(0.55) + flare * F32(0.9)
                       + cme * F32(0.45) + closed * F32(0.22), 0.0, 1.0)
        emis = ramp_lookup(self.ramp_corona, heat)
        emis = emis * (dens + cme * F32(1.2))[:, None]

        # Coronal loops change colour with height for a real reason: the
        # plasma in a low, cool loop is chromospheric hydrogen emitting
        # H-alpha, which is why quiescent prominences photograph crimson,
        # while the same field higher up holds million-kelvin plasma that
        # shines in the forbidden iron lines.  The crossover is near a
        # tenth of a solar radius.
        low = np.exp(-above * F32(1.0 / 0.085))
        prom = ramp_lookup(self.ramp_chromo,
                           np.clip(loop * F32(1.3), 0.0, 1.0))
        hot = ramp_lookup(self.ramp_corona,
                          np.clip(loop * F32(0.8) + F32(0.25), 0.0, 1.0))
        emis += (prom * low[:, None]
                 + hot * (F32(1.0) - low)[:, None]) * (loop * F32(2.3)
                                                       )[:, None]
        emis += ramp_lookup(self.ramp_chromo,
                            np.clip(chromo * F32(1.4), 0.0, 1.0)) \
            * (chromo * F32(0.45))[:, None]
        if float(flare.max()) > 1e-5:
            emis += ramp_lookup(self.ramp_flare,
                                np.clip(flare * F32(0.6), 0.0, 1.0)) \
                * (flare * F32(1.8))[:, None]

        # The corona is optically thin at every visible wavelength — its
        # electron density is about 10^14 per cubic metre, fourteen orders
        # below the photosphere.  Extinction is therefore tiny, and the
        # source-function division keeps the emission independent of it.
        # Electron density in the corona is about 10^14 per cubic metre,
        # fourteen orders below the photosphere: it is optically thin at
        # every visible wavelength, and a sigma large enough to fog the
        # frame would be physically absurd.
        sigma = total * F32(0.016) + F32(1e-5)
        source = emis / sigma[:, None]
        return source.astype(F32, copy=False), sigma.astype(F32, copy=False)


# ==========================================================================
# the photosphere
# ==========================================================================
class Photosphere:
    """The visible surface: granulation, spots, limb darkening."""

    def __init__(self, p: SunParams, mag: MagneticField, lib, seed=5):
        self.p = p
        self.mag = mag
        self.v_gran = lib.get("granule")
        self.v_super = lib.get("supergran")
        self.v_fine = lib.get("plasma_fine")
        self.rng = np.random.default_rng(seed)

    def _n(self, vol, q, scale):
        return vol.sample(np.ascontiguousarray(q * F32(scale), F32),
                          jitter=vcache.new_jitter(self.rng))

    def shade(self, dirs, view_dir, t_hours, flares=()):
        """Radiance of the photosphere for a set of surface directions."""
        p = self.p

        # --- differential rotation of the surface pattern -------------
        sin_lat = np.clip(dirs[:, 1], -1.0, 1.0)
        omega = differential_rotation(sin_lat, p)
        phi = -omega * F32(t_hours)
        c, s = np.cos(phi), np.sin(phi)
        qx = dirs[:, 0] * c - dirs[:, 2] * s
        qz = dirs[:, 0] * s + dirs[:, 2] * c
        q = np.stack([qx, dirs[:, 1], qz], axis=1).astype(F32)

        # --- granulation ------------------------------------------------
        # Convection cells about a thousand kilometres across, with cooler
        # darker lanes between them where the gas sinks.  A Worley cell
        # field is the natural generator: its value is the distance to the
        # nearest seed, which is exactly the geometry of a convection cell.
        gran = self._n(self.v_gran, q, p.granule_scale)
        gran2 = self._n(self.v_gran, q, p.granule_scale * 2.35)
        sup = self._n(self.v_super, q, p.supergranule_scale)
        mottle = (gran - F32(0.48)) * F32(2.30) \
            + (gran2 - F32(0.48)) * F32(0.85) \
            + (sup - F32(0.48)) * F32(0.95)
        fine = self._n(self.v_fine, q, p.granule_scale * 5.0)
        mottle += fine * F32(0.70)

        # --- sunspots ----------------------------------------------------
        bfield = self.mag.surface_strength(q)
        # Convection is suppressed once the magnetic pressure rivals the
        # gas pressure; the umbra is about 1500 K cooler than its
        # surroundings, which is why it looks dark but is still glowing.
        umbra = np.clip((bfield - F32(0.72)) * F32(2.6), 0.0, 1.0)
        penumbra = np.clip((bfield - F32(0.40)) * F32(1.5), 0.0, 1.0)
        spot = umbra * F32(0.80) + penumbra * F32(0.22)
        spot *= F32(p.spot_strength)
        # filamentary penumbral structure, radial about the umbra
        pen_tex = self._n(self.v_fine, q, 95.0)
        spot *= F32(1.0) + pen_tex * F32(0.55) * (penumbra - umbra)

        # --- limb darkening -----------------------------------------------
        # I(mu)/I(0) = 1 - u(1 - mu), the Eddington grey result; u = 0.6 is
        # the solar value in the visible.  The physical cause is that a
        # slanted line of sight reaches optical depth unity higher up,
        # where the gas is cooler.
        mu = np.abs(np.einsum('ij,ij->i', dirs, view_dir))
        limb = F32(1.0) - F32(0.62) * (F32(1.0) - mu)

        # --- temperature and colour ----------------------------------------
        temp = F32(T_EFF) * (F32(1.0) + mottle * F32(0.085))
        temp *= F32(1.0) - spot * F32(0.30)
        temp *= np.power(np.maximum(limb, F32(0.02)), F32(0.25))

        for fl in flares:
            amp = fl.amplitude_surface if hasattr(fl, "amplitude_surface") \
                else 0.0
            if amp:
                pass    # surface flare ribbons are added by the caller

        rgb = S.blackbody_lut(temp)
        # Stefan-Boltzmann: radiated power goes as T^4, so an 8 per cent
        # temperature modulation is a 36 per cent brightness modulation.
        # Applying it here is what makes granulation read as convection
        # rather than as a decal.
        rel = temp / F32(T_EFF)
        power = rel * rel
        power = power * power
        return rgb * (power * limb * F32(0.46))[:, None]


# ==========================================================================
# the phase
# ==========================================================================
class PhaseSun(Phase):
    """FASE III — the Sun: surface, corona, flares, mass ejection."""

    name = "III · MATAHARI"
    duration = 11.0

    def __init__(self, params=None, seed=5):
        self.p = params or SunParams()
        self.seed = int(seed)
        self.mag = None
        self.shader = None
        self.photo = None
        self.track = None
        self._loop_extent = 3.0

    # ----------------------------------------------------------------
    def setup(self, ctx):
        p = self.p
        self.mag = MagneticField(p, seed=self.seed)
        loops, extent = bake_loop_volume(p, self.mag, n=128,
                                         seed=self.seed)
        self._loop_extent = extent
        self.shader = CoronaShader(p, self.mag, ctx.lib, loops, extent,
                                   seed=self.seed)
        self.photo = Photosphere(p, self.mag, ctx.lib, seed=self.seed)
        self._build_events()
        self._build_track()

    def _build_events(self):
        """Schedule the flares and the mass ejection."""
        rng = np.random.default_rng(self.seed + 71)
        f1, f2 = self.mag.footpoints()
        mid = (f1 + f2) * 0.5
        mid /= np.maximum(np.linalg.norm(mid, axis=1, keepdims=True), 1e-6)
        n = mid.shape[0]

        times = [1.6, 3.1, 4.4, 5.3, 6.9, 8.2]
        self.shader.flares = []
        for k, t0 in enumerate(times):
            i = k % n
            self.shader.flares.append(Flare(
                t0=t0,
                rise=0.10 + 0.14 * float(rng.random()),
                decay=0.55 + 1.3 * float(rng.random()),
                direction=mid[i].astype(F32),
                peak=0.35 + 1.5 * float(rng.random()),
                height=0.045 + 0.10 * float(rng.random()),
                width=0.09 + 0.10 * float(rng.random())))

        # The headline event: a fast CME off the limb, timed to peak as
        # the camera pulls back.  The launch direction is deliberately set
        # roughly perpendicular to the line of sight — a CME aimed at the
        # observer is a "halo event", and a halo event photographs as a
        # featureless disc because the whole structure is projected on top
        # of itself.  Seen side-on the three-part structure is visible.
        d = np.array([-0.84, 0.50, -0.12], F32)
        d /= np.linalg.norm(d)
        self.shader.cmes = [CME(t0=6.35, direction=d, speed=0.62,
                                width=0.72, peak=0.95)]
        # a smaller precursor eruption
        d2 = np.array([0.76, -0.42, -0.10], F32)
        d2 /= np.linalg.norm(d2)
        self.shader.cmes.append(CME(t0=2.85, direction=d2, speed=0.30,
                                    width=0.40, peak=0.22))

    def _build_track(self):
        """Camera: a limb skim, a dive to the surface, then a pull-back.

        The shot has to do three different jobs — show the globe, show the
        granulation at close range, and show the ejection at scale — so it
        is choreographed as one continuous move rather than cuts, with the
        field of view carrying most of the scale change.
        """
        d = self.duration
        self.track = CAM.CameraTrack([
            CAM.CameraKey(0.00 * d, eye=(-1.05, 0.62, -4.35),
                          target=(0.0, 0.0, 0.0), fov=40.0, roll=-2.0),
            CAM.CameraKey(0.20 * d, eye=(-0.95, 0.34, -3.05),
                          target=(-0.05, 0.02, 0.0), fov=38.0, roll=-1.0),
            CAM.CameraKey(0.40 * d, eye=(-0.92, 0.17, -2.22),
                          target=(-0.22, 0.05, -0.10), fov=36.0, roll=0.6),
            CAM.CameraKey(0.55 * d, eye=(-0.86, 0.08, -1.72),
                          target=(-0.34, 0.03, -0.26), fov=35.0, roll=1.4),
            CAM.CameraKey(0.70 * d, eye=(-0.42, 0.26, -2.60),
                          target=(0.10, 0.06, -0.06), fov=42.0, roll=0.8),
            CAM.CameraKey(0.86 * d, eye=(0.45, 0.58, -4.30),
                          target=(0.25, 0.12, 0.0), fov=48.0, roll=-0.8),
            CAM.CameraKey(1.00 * d, eye=(1.15, 0.92, -6.60),
                          target=(0.30, 0.10, 0.0), fov=52.0, roll=-1.6),
        ])

    def camera_at(self, ctx, lt):
        cam = self.track.at(lt)
        return CAM.apply_handheld(cam, lt, pos_amp=0.0035, rot_amp=0.10,
                                  fov_breath=0.10, seed=53.0)

    def exposure_at(self, ctx, lt):
        """The photosphere is 10^5 times brighter than the corona.

        No single exposure shows both, which is precisely why a total
        eclipse or an occulting disc is needed to see a corona at all.  The
        curve below opens up as the camera leaves the disc behind.
        """
        u = self.progress(lt, self.duration)
        close = V.smootherstep(0.18, 0.56, u)        # diving in
        out = V.smootherstep(0.62, 1.0, u)           # pulling back
        ev = 0.62 * (1.0 - close) + 0.30 * close
        ev = ev * (1.0 - out) + 2.1 * out
        return float(ev)

    # ----------------------------------------------------------------
    def _bounds(self, rays):
        """Corona shell, with the photosphere as an opaque inner stop."""
        p = self.p
        o, dd = rays.origin, rays.direction
        c = np.zeros(3, F32)
        hit_o, tn_o, tf_o = V.ray_sphere(o, dd, c, F32(p.r_corona))
        hit_i, tn_i, tf_i = V.ray_sphere(o, dd, c, F32(1.0))
        hit_o = hit_o.ravel(); tn_o = tn_o.ravel(); tf_o = tf_o.ravel()
        hit_i = hit_i.ravel(); tn_i = tn_i.ravel()
        tn = np.maximum(tn_o, F32(1e-4))
        # a ray that strikes the surface stops there; nothing behind the
        # photosphere is visible through it
        tf = np.where(hit_i & (tn_i > 0.0), np.minimum(tf_o, tn_i), tf_o)
        rays.t_near = tn.astype(F32)
        rays.t_far = np.where(hit_o, tf, tn - 1.0).astype(F32)
        return rays.clipped()

    def _surface_layer(self, ctx, cam, t_hours, exposure):
        """Ray-trace the photosphere at full resolution."""
        w, h = ctx.width, ctx.height
        out = np.zeros((h * w, 3), F32)
        # bounding rectangle of the disc
        eye = np.asarray(cam.eye, np.float64)
        z = float(np.linalg.norm(eye))
        if z <= 1.0:
            return out.reshape(h, w, 3)
        f = 1.0 / math.tan(math.radians(cam.fov_deg) * 0.5)
        rad_ndc = math.tan(math.asin(min(1.0, 1.0 / z))) * f
        nx, ny, zz, vis = cam.project(np.zeros((1, 3), F32))
        aspect = w / float(h)
        cx = (float(nx[0]) / aspect * 0.5 + 0.5) * w
        cy = (0.5 - float(ny[0]) * 0.5) * h
        rpx = rad_ndc * 0.5 * h
        x0 = max(int(cx - rpx) - 4, 0); x1 = min(int(cx + rpx) + 5, w)
        y0 = max(int(cy - rpx) - 4, 0); y1 = min(int(cy + rpx) + 5, h)
        if x1 <= x0 or y1 <= y0:
            return out.reshape(h, w, 3)
        ys = np.arange(y0, y1); xs = np.arange(x0, x1)
        idx = (ys[:, None] * w + xs[None, :]).ravel()
        o, d, _ = cam.generate_rays(w, h, jitter_phase=None, subset=idx)
        hit, tn, tf = V.ray_sphere(o, d, np.zeros(3, F32), F32(1.0))
        hit = hit.ravel(); tn = tn.ravel()
        live = hit & (tn > 0.0)
        if not live.any():
            return out.reshape(h, w, 3)
        sel = np.flatnonzero(live)
        hp = o[sel] + d[sel] * tn[sel][:, None]
        nrm = hp / np.maximum(np.sqrt(np.einsum('ij,ij->i', hp, hp)),
                              1e-9)[:, None]
        rgb = self.photo.shade(nrm, d[sel], t_hours,
                               flares=self.shader.flares)

        # flare ribbons printed onto the surface itself
        for fl in self.shader.flares:
            amp = fl.amplitude(self.shader._t)
            if amp <= 1e-3:
                continue
            dd = np.einsum('ij,j->i', nrm, fl.direction)
            w2 = F32(fl.width * fl.width * 0.7 + 1e-6)
            band = np.exp(-((F32(1.0) - dd) / w2))
            hot = S.blackbody_lut(np.full(band.shape[0], F32(11000.0)))
            rgb = rgb + hot * (band * F32(amp * 1.5))[:, None]

        out[idx[sel]] = rgb * F32(exposure)
        return out.reshape(h, w, 3)

    # ----------------------------------------------------------------
    def render(self, ctx, lt):
        p = self.p
        t_hours = lt * p.hours_per_second
        self.shader.set_time(lt, t_hours)
        cam = self.camera_at(ctx, lt)
        exposure = self.exposure_at(ctx, lt)
        rng = ctx.rng or np.random.default_rng(ctx.frame * 7919 + 31)
        self.photo.rng = rng

        def shade(pos, t, u):
            self.shader.new_step(rng)
            return self.shader(pos, t, u)

        small, trn = march_layer(ctx, cam, self._bounds, shade,
                                 steps=ctx.march_steps(),
                                 jitter_phase=ctx.frame & 7)
        corona = upsample_linear(small, ctx.width, ctx.height) * F32(exposure)

        surface = self._surface_layer(ctx, cam, t_hours, exposure)
        # the corona is in front of the disc as well as around it, and the
        # marched transmittance already accounts for how much of the
        # surface it hides
        tr = upsample_linear(trn[:, :, None].repeat(3, axis=2),
                             ctx.width, ctx.height)
        return surface * tr + corona


__all__ = ["SunParams", "MagneticField", "Flare", "CME", "CoronaShader",
           "Photosphere", "PhaseSun", "bake_loop_volume",
           "differential_rotation", "T_EFF", "KM_PER_RSUN"]
