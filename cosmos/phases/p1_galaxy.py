"""
cosmos.phases.p1_galaxy
=======================

FASE I — GALAKSI.

A barred spiral rendered from its physics rather than painted: a stellar
population sampled from a real density law, spiral arms as a rotating
density wave, interstellar gas and dust as a marched volume, and every
colour derived from a blackbody temperature.

Structure of the model
----------------------
**Stars** are points.  A star subtends nanoarcseconds; no ray-marcher will
ever find one.  They are sampled once into a catalogue, then projected and
splatted every frame.  Four populations are sampled separately because they
have genuinely different distributions and ages:

  thin disc    exponential in radius, sech^2 in height, intermediate ages
  arm/OB       tight to the spiral ridge, very thin, hot and young
  bulge        Hernquist profile, old and cool, high velocity dispersion
  halo         sparse power-law sphere, oldest and most metal-poor

**Gas and dust** are volumetric.  The emitting gas follows the arms with a
phase offset — in a real density wave the gas shocks on the *trailing* edge
of the arm, which is why dust lanes sit on the concave side of the bright
blue arm and not on top of it.  That offset is modelled explicitly.

**Rotation** is differential for the stars and rigid for the pattern.  This
is the key insight of density-wave theory: the spiral is not made of fixed
material, it is a standing wave that stars orbit *through*.  Inside the
corotation radius stars overtake the pattern, outside it they fall behind.
The simulation reproduces that shear, so over the shot the arms stay crisp
while the star field slides past them.

Coordinates
-----------
One scene unit is one kiloparsec.  The disc lies in the XZ plane, galactic
north is +Y.  Times are in megayears; ``myr_per_second`` converts film time
to physical time.
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

# km/s/kpc -> rad/Myr.  (1 km/s = 1.02271 pc/Myr, so v/r in km/s/kpc is
# 1.02271e-3 rad/Myr.)
KMS_PER_KPC_TO_RAD_PER_MYR = 1.02271e-3


# ==========================================================================
# parameters
# ==========================================================================
@dataclass
class GalaxyParams:
    """Structural parameters of the galaxy.

    Defaults are Milky-Way-like: a 15 kpc optical disc, a 3.2 kpc scale
    length, a 300 pc thin-disc scale height, a 12 degree pitch angle and a
    220 km/s flat rotation curve.
    """

    r_disc: float = 15.0            # outer optical radius, kpc
    r_inner: float = 0.55           # inner truncation of the gas disc
    h_r: float = 3.2                # radial scale length, kpc
    h_z: float = 0.30               # thin-disc scale height, kpc
    h_z_thick: float = 0.95         # thick-disc scale height, kpc
    r_bulge: float = 1.25           # Hernquist scale radius, kpc
    bulge_flatten: float = 0.62     # axis ratio c/a of the bulge
    r_halo: float = 24.0            # stellar halo extent, kpc

    arms: int = 2                   # grand-design arm count
    pitch_deg: float = 12.5         # logarithmic spiral pitch angle
    arm_sharpness: float = 3.2      # exponent on the arm profile
    arm_width_grow: float = 0.045   # arms broaden with radius
    spur_strength: float = 0.34     # 4-arm harmonic (inter-arm spurs)
    dust_lag_deg: float = 9.0       # dust lane offset, trailing edge
    hii_lead_deg: float = -5.0      # OB associations lead the dust shock

    v_flat: float = 220.0           # asymptotic rotation speed, km/s
    r_turn: float = 2.1             # rotation-curve turnover radius, kpc
    corotation_frac: float = 0.58   # corotation as a fraction of r_disc

    bar_strength: float = 0.30      # central bar amplitude
    bar_length: float = 3.1         # bar semi-major axis, kpc
    bar_angle_deg: float = 28.0     # bar position angle at t = 0

    warp_amp: float = 0.55          # outer-disc warp amplitude, kpc
    warp_start: float = 9.0         # radius where the warp begins

    myr_per_second: float = 1.6     # film time -> physical time

    @property
    def inv_tan_pitch(self):
        return 1.0 / math.tan(math.radians(self.pitch_deg))

    @property
    def omega_pattern(self):
        """Pattern speed in rad/Myr, set by the corotation radius."""
        rc = self.corotation_frac * self.r_disc
        return angular_speed(np.array([rc], F32), self)[0]


def rotation_curve(r, p: GalaxyParams):
    """Circular speed in km/s.

    ``v(r) = v_flat * tanh(r / r_turn)`` rises linearly through the bulge
    (solid-body, as a constant-density core must) and flattens outside it,
    which is the observed shape and the signature of a dark-matter halo.
    """
    return F32(p.v_flat) * np.tanh(np.asarray(r, F32) / F32(p.r_turn))


def angular_speed(r, p: GalaxyParams):
    """Angular speed in rad/Myr."""
    r = np.maximum(np.asarray(r, F32), F32(1e-3))
    return (rotation_curve(r, p) / r) * F32(KMS_PER_KPC_TO_RAD_PER_MYR)


def epicyclic_frequency(r, p: GalaxyParams):
    """Radial oscillation frequency kappa, in rad/Myr.

    kappa^2 = (2 Omega / r) d(r^2 Omega)/dr.  Used to give disc stars
    eccentric, precessing orbits instead of perfect circles.
    """
    r = np.maximum(np.asarray(r, F32), F32(1e-3))
    dr = F32(0.02)
    om0 = angular_speed(r, p)
    om1 = angular_speed(r + dr, p)
    d = ((r + dr) ** 2 * om1 - r ** 2 * om0) / dr
    return np.sqrt(np.maximum(2.0 * om0 / r * d, 1e-9)).astype(F32)


# ==========================================================================
# spiral structure
# ==========================================================================
class SpiralField:
    """Analytic logarithmic-spiral density wave with a baked phase table.

    The arm phase is ``theta(r) = ln(r / r0) / tan(pitch)``, and the arm
    strength is ``cos^2`` of ``m(phi - theta)``.  Evaluating that naively
    costs an ``arctan2`` and two transcendentals per sample, which at the
    sample counts involved is the single most expensive line in the frame.

    Two tricks remove all of it:

    * For ``m = 2``, ``cos 2phi`` and ``sin 2phi`` are rational functions of
      the Cartesian coordinates — ``(x^2 - z^2)/r^2`` and ``2xz/r^2`` — so
      the azimuth never has to be formed.
    * ``cos 2theta`` and ``sin 2theta`` depend only on ``ln r``, so they are
      baked into a 4096-entry table indexed by log radius.  Advancing the
      pattern in time is a constant added to ``theta``, i.e. a rebuild of a
      4096-element table once per frame rather than per sample.

    What remains per sample is one logarithm (0.6 ms per million), one
    table gather and a handful of multiplies.
    """

    LUT_N = 4096

    def __init__(self, p: GalaxyParams, r_min=0.04, r_max=40.0):
        self.p = p
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.ln_min = math.log(self.r_min)
        self.ln_scale = (self.LUT_N - 1) / (math.log(self.r_max) - self.ln_min)
        self._u = np.linspace(self.ln_min, math.log(self.r_max), self.LUT_N)
        self.cos_lut = np.zeros(self.LUT_N, F32)
        self.sin_lut = np.zeros(self.LUT_N, F32)
        self.cos_dust = np.zeros(self.LUT_N, F32)
        self.sin_dust = np.zeros(self.LUT_N, F32)
        self.set_time(0.0)

    def set_time(self, t_myr):
        """Rebuild the phase table for the pattern's current orientation."""
        p = self.p
        m = float(p.arms)
        rot = p.omega_pattern * float(t_myr)
        theta = self._u * p.inv_tan_pitch + rot
        a = m * theta
        np.cos(a, out=None)
        self.cos_lut[:] = np.cos(a)
        self.sin_lut[:] = np.sin(a)
        # the dust shock sits a few degrees behind the stellar arm
        ad = a + math.radians(p.dust_lag_deg) * m
        self.cos_dust[:] = np.cos(ad)
        self.sin_dust[:] = np.sin(ad)
        self.t_myr = float(t_myr)

    def _index(self, r):
        li = (np.log(np.maximum(r, F32(self.r_min))) - F32(self.ln_min))
        li *= F32(self.ln_scale)
        return np.clip(li, 0, self.LUT_N - 1).astype(np.int32)

    def arm_cos(self, x, z, r2, inv_r2, dust=False):
        """``cos(m (phi - theta))`` evaluated without an arctangent."""
        c2phi = (x * x - z * z) * inv_r2
        s2phi = (2.0 * x * z) * inv_r2
        i = self._index(np.sqrt(r2))
        if dust:
            ct = self.cos_dust[i]
            st = self.sin_dust[i]
        else:
            ct = self.cos_lut[i]
            st = self.sin_lut[i]
        return c2phi * ct + s2phi * st

    def strength(self, x, z, r2, inv_r2, dust=False, sharpness=None):
        """Normalised arm strength in [0, 1]."""
        p = self.p
        c = self.arm_cos(x, z, r2, inv_r2, dust=dust)
        a = c * F32(0.5) + F32(0.5)
        if p.spur_strength > 0.0:
            # cos(4 psi) = 2 cos^2(2 psi) - 1 — the second harmonic that
            # produces inter-arm spurs, obtained for one multiply
            c4 = F32(2.0) * c * c - F32(1.0)
            a += F32(p.spur_strength) * (c4 * F32(0.5) + F32(0.5) - a)
        s = F32(p.arm_sharpness if sharpness is None else sharpness)
        # integer powers where possible: pow() costs 15 ms per million,
        # three multiplies cost 2
        if abs(float(s) - 2.0) < 1e-6:
            return a * a
        if abs(float(s) - 3.0) < 1e-6:
            return a * a * a
        return np.power(a, s, dtype=F32)

    def phase_for_points(self, r, phi, dust=False):
        """Scalar-path arm strength for catalogue sampling (not hot)."""
        p = self.p
        theta = np.log(np.maximum(r, self.r_min)) * p.inv_tan_pitch
        theta = theta + p.omega_pattern * self.t_myr
        if dust:
            theta = theta + math.radians(p.dust_lag_deg)
        c = np.cos(p.arms * (phi - theta))
        a = 0.5 + 0.5 * c
        if p.spur_strength > 0.0:
            c4 = 2.0 * c * c - 1.0
            a += p.spur_strength * (0.5 + 0.5 * c4 - a)
        return np.power(np.clip(a, 0.0, 1.0), p.arm_sharpness)

    def bar_factor(self, x, z, r2):
        """Central bar as an elongated quadrupole enhancement."""
        p = self.p
        if p.bar_strength <= 0.0:
            return None
        ang = math.radians(p.bar_angle_deg) + p.omega_pattern * self.t_myr
        ca, sa = math.cos(ang), math.sin(ang)
        bx = x * F32(ca) + z * F32(sa)
        bz = -x * F32(sa) + z * F32(ca)
        q = (bx / F32(p.bar_length)) ** 2 + (bz / F32(p.bar_length * 0.26)) ** 2
        return F32(p.bar_strength) * np.exp(-q)


def disc_warp(r, p: GalaxyParams, phase=0.0):
    """Integral-sign warp of the outer disc.

    Essentially every spiral galaxy's HI disc is warped — the Milky Way's by
    about a kiloparsec at 15 kpc.  Modelled as an m = 1 bend switched on
    beyond ``warp_start``.
    """
    s = np.clip((r - F32(p.warp_start)) / F32(max(p.r_disc - p.warp_start,
                                                  1e-3)), 0.0, 1.0)
    return F32(p.warp_amp) * s * s


# ==========================================================================
# stellar catalogue
# ==========================================================================
@dataclass
class StarCatalogue:
    """Static per-star data; positions are recomputed per frame."""

    radius: np.ndarray            # (N,) cylindrical radius, kpc
    phi0: np.ndarray              # (N,) azimuth at t = 0
    height: np.ndarray            # (N,) y offset, kpc
    omega: np.ndarray             # (N,) rad/Myr
    ecc: np.ndarray               # (N,) epicyclic amplitude, kpc
    kappa: np.ndarray             # (N,) epicyclic frequency
    ecc_phase: np.ndarray         # (N,) epicycle phase at t = 0
    temperature: np.ndarray       # (N,) K
    rgb: np.ndarray               # (N,3) unit-luminance blackbody colour
    brightness: np.ndarray        # (N,) render radiance
    warp: np.ndarray              # (N,) warp coefficient
    population: np.ndarray        # (N,) int8 tag, 0..3

    def __len__(self):
        return self.radius.shape[0]

    def positions(self, t_myr, p: GalaxyParams):
        """Evaluate world positions at a given physical time.

        Each star is on a slowly precessing epicyclic orbit: a guiding
        circle of radius ``radius`` rotating at ``omega``, plus a radial
        oscillation at ``kappa``.  That is the linearised solution to motion
        in an axisymmetric potential, and it is what gives the disc its
        velocity dispersion instead of a set of perfect rings.
        """
        t = F32(t_myr)
        ph = self.phi0 + self.omega * t
        ep = self.ecc_phase + self.kappa * t
        r = self.radius + self.ecc * np.cos(ep)
        # conservation of angular momentum tilts the epicycle: the azimuthal
        # excursion is twice the radial one, divided by two
        ph = ph - (2.0 * self.ecc / np.maximum(self.radius, 0.2)) * np.sin(ep)
        out = np.empty((r.shape[0], 3), F32)
        np.multiply(r, np.cos(ph), out=out[:, 0])
        np.multiply(r, np.sin(ph), out=out[:, 2])
        out[:, 1] = self.height + self.warp * np.sin(ph)
        return out


def _sample_radius_exponential(n, h_r, r_max, rng):
    """Inverse-transform sampling of a truncated exponential disc.

    The surface density is ``Sigma(r) = Sigma0 exp(-r/h)``, so the mass
    inside r goes as ``1 - (1 + r/h) exp(-r/h)``.  That has no closed-form
    inverse, so sample the 2-D density by rejection against its own peak —
    vectorised, two passes are enough for a 99.9% fill.
    """
    out = np.empty(0, np.float64)
    need = n
    while need > 0:
        m = int(need * 2.6) + 64
        r = rng.random(m) * r_max
        # pdf(r) proportional to r * exp(-r/h); peak at r = h
        pdf = (r / h_r) * np.exp(1.0 - r / h_r)
        keep = r[rng.random(m) < pdf]
        out = np.concatenate([out, keep])
        need = n - out.shape[0]
    return out[:n]


def _sample_hernquist(n, a, rng):
    """Radii from a Hernquist sphere: ``M(<r) = M r^2 / (r + a)^2``.

    Invertible in closed form, which is exactly why Hernquist proposed it
    as the analytic stand-in for the de Vaucouleurs profile.
    """
    u = rng.random(n)
    su = np.sqrt(u)
    return a * su / np.maximum(1.0 - su, 1e-6)


def build_catalogue(p: GalaxyParams, spiral: SpiralField, n_total=420000,
                    seed=7, hot_arm_bias=0.72):
    """Sample the four stellar populations."""
    rng = np.random.default_rng(seed)

    f_disc, f_arm, f_bulge, f_halo = 0.44, 0.30, 0.19, 0.07
    n_disc = int(n_total * f_disc)
    n_arm = int(n_total * f_arm)
    n_bulge = int(n_total * f_bulge)
    n_halo = n_total - n_disc - n_arm - n_bulge

    radius, phi0, height, pop, temp = [], [], [], [], []

    # -- thin + thick disc ------------------------------------------------
    r = _sample_radius_exponential(n_disc, p.h_r, p.r_disc * 1.30, rng)
    ph = rng.random(n_disc) * TAU
    # a mild arm bias even for the older disc: the density wave compresses
    # everything, not only the gas
    bias = spiral.phase_for_points(r, ph)
    ph = ph - 0.22 * np.sin(p.arms * ph) * bias
    thick = rng.random(n_disc) < 0.26
    hz = np.where(thick, p.h_z_thick, p.h_z)
    # sech^2 is the isothermal self-gravitating sheet; inverse CDF is atanh
    u = rng.random(n_disc) * 2.0 - 1.0
    y = hz * np.arctanh(np.clip(u, -0.999, 0.999))
    radius.append(r)
    phi0.append(ph)
    height.append(y)
    pop.append(np.where(thick, 1, 0).astype(np.int8))
    temp.append(S.sample_stellar_temperatures(n_disc, rng, hot_bias=0.10))

    # -- arm / OB associations -------------------------------------------
    # rejection sampling against the arm profile itself: propose uniformly
    # in the annulus, accept in proportion to arm strength
    acc_r = np.empty(0)
    acc_p = np.empty(0)
    while acc_r.shape[0] < n_arm:
        m = int((n_arm - acc_r.shape[0]) * 3.4) + 256
        rr = np.sqrt(rng.random(m)) * p.r_disc
        pp = rng.random(m) * TAU
        w = spiral.phase_for_points(rr, pp)
        w = w * np.exp(-rr / (p.h_r * 1.45))
        w = w * np.clip((rr - p.r_inner) / 1.2, 0.0, 1.0)
        k = rng.random(m) < w / max(w.max(), 1e-9)
        acc_r = np.concatenate([acc_r, rr[k]])
        acc_p = np.concatenate([acc_p, pp[k]])
    acc_r = acc_r[:n_arm]
    acc_p = acc_p[:n_arm]
    # young stars have not had time to scatter out of the plane
    y = rng.normal(0.0, p.h_z * 0.30, n_arm)
    radius.append(acc_r)
    phi0.append(acc_p + math.radians(p.hii_lead_deg))
    height.append(y)
    pop.append(np.full(n_arm, 2, np.int8))
    temp.append(S.sample_stellar_temperatures(n_arm, rng,
                                              hot_bias=hot_arm_bias))

    # -- bulge ------------------------------------------------------------
    rb = _sample_hernquist(n_bulge, p.r_bulge, rng)
    rb = np.minimum(rb, p.r_bulge * 7.0)
    dirs = V.fibonacci_sphere(n_bulge).astype(np.float64)
    dirs = dirs[rng.permutation(n_bulge)]
    bx = rb * dirs[:, 0]
    by = rb * dirs[:, 1] * p.bulge_flatten
    bz = rb * dirs[:, 2]
    # stretch along the bar
    ang = math.radians(p.bar_angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    u1 = bx * ca + bz * sa
    u2 = -bx * sa + bz * ca
    u1 *= 1.0 + p.bar_strength * 2.1
    bx = u1 * ca - u2 * sa
    bz = u1 * sa + u2 * ca
    radius.append(np.sqrt(bx * bx + bz * bz))
    phi0.append(np.arctan2(bz, bx))
    height.append(by)
    pop.append(np.full(n_bulge, 3, np.int8))
    # old, metal-rich, red
    temp.append(np.clip(rng.normal(4150.0, 950.0, n_bulge), 2600.0, 9000.0))

    # -- halo -------------------------------------------------------------
    u = rng.random(n_halo)
    rh = p.r_bulge * 2.0 + (p.r_halo - p.r_bulge * 2.0) * u ** 1.9
    dirs = V.fibonacci_sphere(n_halo).astype(np.float64)
    dirs = dirs[rng.permutation(n_halo)]
    radius.append(np.sqrt((rh * dirs[:, 0]) ** 2 + (rh * dirs[:, 2]) ** 2))
    phi0.append(np.arctan2(rh * dirs[:, 2], rh * dirs[:, 0]))
    height.append(rh * dirs[:, 1])
    pop.append(np.full(n_halo, 4, np.int8))
    temp.append(np.clip(rng.normal(4600.0, 1100.0, n_halo), 2800.0, 11000.0))

    radius = np.concatenate(radius).astype(F32)
    phi0 = np.concatenate(phi0).astype(F32)
    height = np.concatenate(height).astype(F32)
    population = np.concatenate(pop)
    temperature = np.concatenate(temp).astype(F32)

    # -- kinematics -------------------------------------------------------
    omega = angular_speed(radius, p)
    kappa = epicyclic_frequency(radius, p)
    disc_like = population <= 2
    # velocity dispersion rises with age: the halo and bulge barely follow
    # the disc's circular motion at all
    disp = np.where(population == 2, 0.018,
                    np.where(population == 0, 0.055,
                             np.where(population == 1, 0.105, 0.0))).astype(F32)
    ecc = (radius * disp * np.abs(rng.normal(1.0, 0.45,
                                             radius.shape[0]))).astype(F32)
    ecc = np.where(disc_like, ecc, 0.0).astype(F32)
    omega = np.where(disc_like, omega,
                     omega * F32(0.35)).astype(F32)      # pressure support
    ecc_phase = (rng.random(radius.shape[0]) * TAU).astype(F32)

    warp = np.where(disc_like, disc_warp(radius, p), 0.0).astype(F32)

    # -- photometry -------------------------------------------------------
    radius_pre = radius
    mass = S.mass_from_temperature(temperature)
    lum = S.luminosity_from_mass(mass)
    # Real stellar luminosity spans eleven orders of magnitude.  Reproduced
    # literally, a dozen O stars would be the only thing on screen and the
    # other four hundred thousand would be black.  The exponent compresses
    # that range the way a long photographic exposure does, preserving the
    # ordering and the colour-luminosity correlation while keeping the
    # faint population visible.
    bright = np.power(np.maximum(lum, 1e-6), 0.47).astype(F32)
    bright *= np.where(population == 2, 1.30,
                       np.where(population == 3, 0.80, 1.0)).astype(F32)
    bright *= rng.lognormal(0.0, 0.34, bright.shape[0]).astype(F32)
    # One O5 star outshines the Sun by a factor of 10^5.  Even after the
    # compression above a handful of outliers would each eat a quarter of
    # the frame, so the top of the distribution is clipped at a high
    # percentile.  Everything below it keeps its exact relative ordering.
    bright = np.minimum(bright, np.percentile(bright, 99.95)).astype(F32)
    # Taper the outermost *disc* so the star field does not end on a hard
    # circle.  Applying this to the halo as well would cut the halo sphere
    # at a cylindrical radius and print that cut into the frame as an arc.
    edge_fade = np.clip((p.r_disc * 1.26 - radius_pre) * 0.55, 0.0, 1.0)
    bright *= np.where(population <= 2, edge_fade, 1.0).astype(F32)
    rgb = S.blackbody_lut(temperature).astype(F32)
    # Blackbody colours are genuinely pale — the Planck locus is close to
    # white across the whole stellar range.  Summed over the thousands of
    # stars that land in one pixel of a distant galaxy they wash out
    # completely, so chroma is pushed before splatting.  The hue, which is
    # what carries the temperature information, is untouched.
    rgb = S.saturate_rgb(rgb, 1.65)
    # normalise so `brightness` alone controls output level
    lum_rgb = np.maximum(S.luminance(rgb), 1e-5)
    rgb = rgb / lum_rgb[:, None]

    return StarCatalogue(radius=radius, phi0=phi0, height=height,
                         omega=omega.astype(F32), ecc=ecc,
                         kappa=kappa.astype(F32), ecc_phase=ecc_phase,
                         temperature=temperature, rgb=rgb.astype(F32),
                         brightness=bright, warp=warp, population=population)


# ==========================================================================
# interstellar medium — the marched volume
# ==========================================================================
class ISMShader:
    """Emission and extinction of the galactic disc.

    Four physically distinct components share one march:

    * **Unresolved stellar continuum** — the integrated light of the
      billions of stars no instrument resolves.  This carries most of the
      image: a real galaxy photograph is mostly surface brightness, not
      point sources.  Blue in the arms, where star formation is ongoing and
      short-lived O and B stars dominate the light; yellow between them,
      where only long-lived K and M dwarfs are left.
    * **Ionised gas** — HII regions around OB associations.  Clumpy, tight
      to the arm ridge, and *red*, because the light is hydrogen
      recombination at 656 nm rather than a thermal continuum.
    * **Dust** — pure extinction, no emission, on the trailing edge of the
      arm where the gas shocks.  This is what carves the dark lanes;
      without it a spiral galaxy renders as a smear.
    * **Bulge** — a Hernquist-profile glow of old, metal-rich stars.

    Everything follows the fast-shader rules established by profiling:
    radii without ``arctan2``, colour via baked 1-D ramps instead of per
    sample blackbody evaluation, integer powers, in-place arithmetic, and
    stochastic nearest sampling of uint8 volumes instead of trilinear.
    """

    #: radius at which the exponential envelopes are normalised to 1
    R_REF = 4.0
    #: radius at which the bulge emission is tapered fully to zero, kpc
    BULGE_CUT = 7.0
    #: half-height of the bounding slab, kpc; must match PhaseGalaxy._bounds
    Y_SLAB = 2.1

    @staticmethod
    def _unit(n):
        """Map a signed fBm sample (mean 0, sigma ~0.18) onto [0, 1]."""
        return np.clip(n * F32(2.55) + F32(0.5), F32(0.0), F32(1.0))

    def __init__(self, p: GalaxyParams, spiral: SpiralField, lib,
                 star_gain=1.0, gas_gain=1.0, dust_gain=1.0,
                 bulge_gain=1.0):
        self.p = p
        self.spiral = spiral
        self.star_gain = float(star_gain)
        self.gas_gain = float(gas_gain)
        self.dust_gain = float(dust_gain)
        self.bulge_gain = float(bulge_gain)

        self.v_neb = lib.get("nebula")
        self.v_fine = lib.get("nebula_fine")
        self.v_fil = lib.get("filament")
        self.v_dust = lib.get("dust")

        # Stellar continuum.  The ramp runs from the old interarm
        # population to the young arm population, and the colours are
        # genuine blackbody loci for those effective temperatures.
        self.ramp_star = build_emission_ramp(
            temps=(3400, 3900, 4600, 5600, 7200, 10500, 17000, 26000),
            gains=(0.55, 0.70, 0.85, 1.00, 1.22, 1.55, 1.95, 2.35), n=1024)
        # Bulge: old, metal-rich, reddened by its own dust.
        self.ramp_bulge = build_emission_ramp(
            temps=(3000, 3450, 3900, 4400, 5000),
            gains=(0.22, 0.55, 1.00, 1.45, 1.85), n=512)
        # Ionised gas.  Not a blackbody — an HII region's visible output is
        # dominated by the hydrogen Balmer lines plus [OIII] 500.7 nm, so
        # the colour is taken from the CIE response to those wavelengths
        # directly.  Using Planck here would make nebulae teal, which is
        # the classic giveaway of a renderer that fakes its spectra.
        self.ramp_gas = self._line_ramp()

        self.jitter = None
        self._t_myr = 0.0
        self._drift = np.zeros(3, F32)

        self._exp_star = F32(math.exp(self.R_REF / p.h_r))
        self._exp_gas = F32(math.exp(self.R_REF / (p.h_r * 1.30)))
        self._exp_dust = F32(math.exp(self.R_REF / (p.h_r * 1.15)))

    # ----------------------------------------------------------------
    @staticmethod
    def _line_ramp(n=512):
        """Emission-line colour ramp: H-beta/[OIII] -> H-alpha.

        Low ionisation parameter gives the deep red of H-alpha alone; high
        excitation adds [OIII] and H-beta and shifts the mix toward
        magenta-pink, which is what the eye reads in a real HII region.
        """
        # sRGB primaries, D65 white point
        M = np.array([[3.2406, -1.5372, -0.4986],
                      [-0.9689, 1.8758, 0.0415],
                      [0.0557, -0.2040, 1.0570]], np.float64)

        def rgb_of(lam):
            xb, yb, zb = S.cie_xyz_bar(np.array([lam], np.float32))
            xyz = np.array([float(xb[0]), float(yb[0]), float(zb[0])],
                           np.float64)
            c = np.clip(M @ xyz, 0.0, None)
            return c / max(c.max(), 1e-9)

        ha = rgb_of(656.3)        # hydrogen alpha
        hb = rgb_of(486.1)        # hydrogen beta
        o3 = rgb_of(500.7)        # doubly ionised oxygen
        s2 = rgb_of(671.6)        # singly ionised sulphur, shock tracer
        u = np.linspace(0.0, 1.0, n)[:, None]
        base = 0.74 * ha + 0.26 * s2
        high = 0.46 * ha + 0.32 * o3 + 0.22 * hb
        mix = base * (1.0 - u) + high * u
        gain = (0.30 + 2.10 * u.ravel() ** 1.4)[:, None]
        return np.ascontiguousarray((mix * gain).astype(F32))

    def set_time(self, t_myr):
        self._t_myr = float(t_myr)
        # slow turbulent drift of the gas texture, decoupled from the
        # pattern so the medium does not look rigidly glued to the arms
        self._drift = np.array([0.013 * t_myr, 0.004 * t_myr,
                                -0.009 * t_myr], F32)

    def new_step(self, rng):
        self.jitter = vcache.new_jitter(rng)

    # ----------------------------------------------------------------
    def __call__(self, pos, t, u):
        p = self.p
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        r2 = x * x + z * z
        np.maximum(r2, F32(1e-6), out=r2)
        inv_r2 = np.reciprocal(r2)
        r = np.sqrt(r2)
        ay = np.abs(y)

        # ---- envelopes -------------------------------------------------
        # The stellar disc is thicker than the gas disc, and the gas disc is
        # radially more extended than the stars — both are observed facts
        # and both matter visually: the gas layer stays a thin bright line
        # when the disc is seen edge-on.
        env_star = np.exp(-(r * F32(1.0 / p.h_r)
                            + ay * F32(1.0 / (p.h_z * 1.45)))) * self._exp_star
        env_gas = np.exp(-(r * F32(1.0 / (p.h_r * 1.30))
                           + ay * F32(1.0 / (p.h_z * 0.70)))) * self._exp_gas

        # Fade everything to zero before the bounding slab.  Any
        # component still emitting at |y| = Y_SLAB draws the cylinder's
        # silhouette straight into the frame as a hard horizontal edge.
        vt = np.clip((F32(self.Y_SLAB) - ay) * F32(1.0 / (self.Y_SLAB * 0.55)),
                     0.0, 1.0)
        vt *= vt

        inner = np.clip((r - F32(p.r_inner)) * F32(1.5), 0.0, 1.0)
        outer = np.clip((F32(p.r_disc) - r) * F32(0.40), 0.0, 1.0)
        edge = inner * outer
        env_star *= edge
        env_gas *= edge

        # ---- arms ------------------------------------------------------
        arm = self.spiral.strength(x, z, r2, inv_r2)
        bar = self.spiral.bar_factor(x, z, r2)
        if bar is not None:
            arm = arm + bar * (F32(1.0) - arm)
        # arms broaden and blur outward as shear wins over self-gravity
        soften = np.clip(r * F32(p.arm_width_grow), 0.0, 0.38)
        arm = arm * (F32(1.0) - soften) + soften * F32(0.34)
        arm_sharp = arm * arm * arm

        # ---- texture ---------------------------------------------------
        # The baked fBm volumes are *signed*: zero mean, standard deviation
        # about 0.18, range roughly +/-0.6.  Treating them as if they lived
        # in [0, 1] is the easiest way to lose every feature in the frame,
        # so each one is explicitly mapped to unit range here.  The ridged
        # volumes ('filament') are already unipolar and are left alone.
        n1 = self._unit(self.v_neb.sample(pos * F32(0.085) + self._drift,
                                          jitter=self.jitter))
        n2 = self._unit(self.v_fine.sample(pos * F32(0.46) - self._drift,
                                           jitter=self.jitter))
        fil = self.v_fil.sample(pos * F32(0.23) + self._drift * F32(0.5),
                                jitter=self.jitter)

        # ---- unresolved stellar continuum ------------------------------
        # 34% smooth, 66% modulated by the arms: even between the arms a
        # disc is not empty, it is merely older and fainter.
        starlight = env_star * (F32(0.20) + F32(0.80) * arm)
        # Two-scale mottling.  Real spiral arms are not smooth ribbons —
        # they are chains of star-forming complexes separated by gaps, and
        # the ridged volume supplies the feathering and spurs that branch
        # off the main arms.
        starlight *= (F32(0.56) + F32(0.80) * n1)
        starlight *= (F32(0.76) + F32(0.52) * fil)
        starlight *= vt * F32(1.55 * self.star_gain)

        # ---- ionised gas -----------------------------------------------
        # Sparse, high-contrast knots: HII regions occupy a tiny fraction
        # of the disc's volume but dominate its H-alpha output.
        knot = n2 * F32(0.52) + fil * F32(0.48)
        knot = np.maximum(knot - F32(0.52), F32(0.0)) * F32(2.08)
        knot *= knot
        gas = env_gas * arm_sharp * knot * vt * F32(62.0 * self.gas_gain)

        # ---- bulge ------------------------------------------------------
        rb2 = r2 + (y * F32(1.0 / p.bulge_flatten)) ** 2
        rb = np.sqrt(rb2)
        hq = F32(p.r_bulge) / (rb + F32(p.r_bulge))
        bulge = hq * hq * hq
        # Taper to zero well inside the bounding sphere.  Without this the
        # sphere's silhouette prints into the frame as a hard ellipse where
        # the marched interval stops.
        bt = np.clip((F32(self.BULGE_CUT) - rb)
                     * F32(1.0 / (self.BULGE_CUT * 0.50)), 0.0, 1.0)
        bulge = bulge * bt * bt * F32(3.6 * self.bulge_gain)

        # ---- dust --------------------------------------------------------
        dust_arm = self.spiral.strength(x, z, r2, inv_r2, dust=True,
                                        sharpness=3.0)
        dn = self._unit(self.v_dust.sample(pos * F32(0.32)
                                           + self._drift * F32(0.3),
                                           jitter=self.jitter))
        dust_env = np.exp(-(r * F32(1.0 / (p.h_r * 1.15))
                            + ay * F32(1.0 / (p.h_z * 0.55)))) * self._exp_dust
        # dust is filamentary and patchy; the (1 - fil) term puts the
        # densest obscuration in the gaps between the bright filaments,
        # which is what gives the lanes their torn, ragged edges
        dtex = dn * F32(0.58) + (F32(1.0) - fil) * F32(0.42)
        dtex = np.clip(dtex * F32(1.55) - F32(0.42), 0.0, 1.0)
        dtex *= dtex
        dust = dust_env * edge * (F32(0.16) + F32(0.84) * dust_arm)
        dust *= dtex * vt * F32(3.4 * self.dust_gain)

        # ---- colour ------------------------------------------------------
        # Ramp coordinate for the continuum: arm proximity, i.e. how much of
        # this parcel's light comes from stars young enough to still be hot.
        # Scaled so the arm ridges actually reach the blue end of the ramp —
        # the blue-arm / yellow-interarm contrast is the single most
        # recognisable colour signature of a spiral galaxy.
        youth = np.clip(arm * F32(1.55) - F32(0.22) + n1 * F32(0.18),
                        0.0, 1.0)
        emis = ramp_lookup(self.ramp_star, youth)
        emis *= starlight[:, None]

        exc = np.clip(arm_sharp * F32(0.55) + knot * F32(0.70), 0.0, 1.0)
        emis += ramp_lookup(self.ramp_gas, exc) * gas[:, None]

        bcol = ramp_lookup(self.ramp_bulge,
                           np.clip(bulge * F32(0.55), 0.0, 1.0))
        emis += bcol * bulge[:, None]

        # ---- radiative transfer ------------------------------------------
        # The marcher integrates  C = sum T_i * c_i * alpha_i  with
        # alpha_i = 1 - exp(-sigma_i dt).  In that formulation c is the
        # *source function* S = j / sigma, not the emission per unit length:
        # the exact solution of dI/ds = j - sigma I over one step is
        #
        #     I_out = I_in e^{-sigma dt} + (j / sigma)(1 - e^{-sigma dt})
        #
        # Returning j directly is the classic mistake, and it makes every
        # optically thin medium invisible — as sigma falls the emission
        # falls with it, when physically a thin gas should emit j*dt
        # regardless of how transparent it is.  Dividing by sigma makes the
        # product c*alpha tend to j*dt in exactly that limit, so emission
        # and opacity become independent knobs.
        #
        # Stars and line emission barely absorb; dust does almost all of it.
        sigma = (starlight * F32(0.030) + gas * F32(0.010)
                 + bulge * F32(0.018) + dust * F32(9.0))
        np.maximum(sigma, F32(1e-5), out=sigma)
        source = emis / sigma[:, None]
        return source.astype(F32, copy=False), sigma.astype(F32, copy=False)


# ==========================================================================
# the phase
# ==========================================================================
class PhaseGalaxy(Phase):
    """FASE I — the galaxy: establish, descend, enter an arm."""

    name = "I · GALAKSI"
    duration = 11.0

    def __init__(self, params=None, n_stars=165000, seed=7):
        self.p = params or GalaxyParams()
        self.n_stars = int(n_stars)
        self.seed = int(seed)
        self.spiral = None
        self.cat = None
        self.shader = None
        self.track = None
        self._bg = None

    # ----------------------------------------------------------------
    def setup(self, ctx):
        self.spiral = SpiralField(self.p)
        self.cat = build_catalogue(self.p, self.spiral,
                                   n_total=self.n_stars, seed=self.seed)
        self.shader = ISMShader(self.p, self.spiral, ctx.lib,
                                star_gain=1.0, gas_gain=1.0,
                                dust_gain=1.0, bulge_gain=1.0)
        self._build_track()
        self._build_background(ctx)

    def _build_background(self, ctx):
        """A fixed shell of very distant field stars and galaxies.

        Without it the sky behind the disc is pure black, which reads as a
        rendering error rather than as intergalactic space.
        """
        rng = np.random.default_rng(self.seed + 991)
        n = 5200
        dirs = V.fibonacci_sphere(n).astype(F32)
        dirs = dirs[rng.permutation(n)]
        rad = F32(620.0) * (F32(1.0) + rng.random(n).astype(F32) * F32(0.3))
        pos = dirs * rad[:, None]
        t = np.clip(rng.normal(5200.0, 2600.0, n), 2400.0, 26000.0).astype(F32)
        rgb = S.saturate_rgb(S.blackbody_lut(t), 1.5)
        rgb /= np.maximum(S.luminance(rgb), 1e-5)[:, None]
        b = (rng.random(n).astype(F32) ** 7.5) * F32(0.30) + F32(0.002)
        self._bg = (pos, (rgb * b[:, None]).astype(F32))

    def _build_track(self):
        """Camera choreography.

        Opens three-quarters on, well outside the disc, so the spiral reads
        as a whole.  Falls toward the plane while pushing in — the dolly is
        logarithmic, which keeps the *angular* rate of approach constant and
        is what makes a long approach feel like steady flight instead of a
        deceleration.  Ends low and inside an arm with a wide lens, where
        parallax between near and far stars does the work.
        """
        p = self.p
        d = self.duration
        self.track = CAM.CameraTrack([
            CAM.CameraKey(0.00 * d, eye=(-6.0, 26.0, -36.0),
                          target=(0.0, 0.0, 0.0), fov=36.0, roll=-3.0),
            CAM.CameraKey(0.22 * d, eye=(-2.2, 19.5, -30.0),
                          target=(0.3, 0.0, 0.4), fov=38.0, roll=-1.6),
            CAM.CameraKey(0.46 * d, eye=(3.4, 10.2, -21.0),
                          target=(1.2, 0.0, -1.0), fov=43.0, roll=0.8),
            CAM.CameraKey(0.68 * d, eye=(6.6, 3.9, -13.2),
                          target=(2.6, 0.1, -3.4), fov=49.0, roll=2.2),
            CAM.CameraKey(0.86 * d, eye=(7.4, 1.25, -8.4),
                          target=(4.2, 0.15, -3.0), fov=55.0, roll=2.8),
            CAM.CameraKey(1.00 * d, eye=(7.2, 0.45, -5.9),
                          target=(5.4, 0.1, -1.4), fov=58.0, roll=2.4),
        ])

    def exposure_at(self, ctx, lt):
        """Stop down on the way in.

        At t = 0 the galaxy is a small faint object against black and needs
        all the gain it can get.  By t = 11 the camera is inside the disc
        with ten thousand times the surface brightness filling the frame.
        The curve is a smootherstep in log space, so the iris closes at a
        perceptually even rate rather than lurching at the end.
        """
        u = V.smootherstep(0.0, 1.0, self.progress(lt, self.duration))
        stops = -3.35 * float(u)
        return float(2.0 ** stops) * 2.35

    def camera_at(self, ctx, lt):
        cam = self.track.at(lt)
        # a very small amount of hand-held keeps a long slow move alive
        cam = CAM.apply_handheld(cam, lt, pos_amp=0.030, rot_amp=0.055,
                                 fov_breath=0.14, seed=11.0)
        return cam

    # ----------------------------------------------------------------
    def _bounds(self, rays):
        """Clip rays to the gas cylinder, then to the bulge sphere.

        The disc is thin; a cylinder bound removes the enormous empty volume
        above and below it.  Rays that miss the cylinder but hit the bulge
        still need marching, so the two intervals are unioned.
        """
        p = self.p
        o, dd = rays.origin, rays.direction
        # vmath's intersectors return (hit, t_near, t_far) with a trailing
        # keepdims axis; flatten to per-ray scalars
        hit1, tn1, tf1 = V.ray_cylinder_y(o, dd, F32(p.r_disc * 1.02),
                                          F32(-ISMShader.Y_SLAB), F32(ISMShader.Y_SLAB))
        centre = np.zeros(3, F32)
        hit2, tn2, tf2 = V.ray_sphere(o, dd, centre,
                                      F32(ISMShader.BULGE_CUT * 1.18))
        hit1 = hit1.ravel(); tn1 = tn1.ravel(); tf1 = tf1.ravel()
        hit2 = hit2.ravel(); tn2 = tn2.ravel(); tf2 = tf2.ravel()
        hit = hit1 | hit2
        tn = np.where(hit1 & hit2, np.minimum(tn1, tn2),
                      np.where(hit1, tn1, tn2))
        tf = np.where(hit1 & hit2, np.maximum(tf1, tf2),
                      np.where(hit1, tf1, tf2))
        tn = np.maximum(tn, F32(1e-3))
        rays.t_near = tn.astype(F32)
        rays.t_far = np.where(hit, tf, tn - 1.0).astype(F32)
        return rays.clipped()

    def _star_layer(self, ctx, cam, t_myr, trn_small):
        """Project, attenuate and splat the stellar populations."""
        p = self.p
        pos = self.cat.positions(t_myr, p)
        rgb = self.cat.rgb * self.cat.brightness[:, None]

        # --- interstellar extinction of starlight ----------------------
        # The marched layer already knows how much of each line of sight is
        # obscured.  A star is dimmed by whatever fraction of that column
        # lies in front of it, and reddened because dust scatters blue light
        # far more efficiently than red (A_lambda roughly proportional to
        # 1/lambda).  This is the single detail that makes the dust lanes
        # read as dust rather than as black paint.
        eye = cam.eye.astype(F32)
        delta = pos - eye
        dist = np.sqrt(np.einsum('ij,ij->i', delta, delta))
        ddir = delta / np.maximum(dist, 1e-6)[:, None]
        hit, tn, tf = V.ray_cylinder_y(
            np.broadcast_to(eye, pos.shape).astype(F32), ddir,
            F32(p.r_disc * 1.02), F32(-ISMShader.Y_SLAB), F32(ISMShader.Y_SLAB))
        hit = hit.ravel(); tn = tn.ravel(); tf = tf.ravel()
        span = np.maximum(tf - tn, 1e-4)
        frac = np.clip((dist - tn) / span, 0.0, 1.0)
        frac = np.where(hit, frac, 0.0).astype(F32)

        nx, ny, zz, vis = cam.project(pos)
        aspect = ctx.width / float(ctx.height)
        mw, mh = ctx.march_width, ctx.march_height
        sx = np.clip(((nx / aspect) * 0.5 + 0.5) * mw, 0, mw - 1).astype(np.int32)
        sy = np.clip((0.5 - ny * 0.5) * mh, 0, mh - 1).astype(np.int32)
        T = trn_small[sy, sx]
        tau = -np.log(np.maximum(T, F32(1e-4))) * frac
        ext = np.empty_like(rgb)
        for k, kl in enumerate((0.74, 1.0, 1.36)):
            np.exp(-tau * F32(kl), out=ext[:, k])
        rgb = rgb * ext

        buf, _ = project_and_splat(cam, pos, rgb, ctx.width, ctx.height,
                                   falloff=True, ref_distance=9.0,
                                   max_gain=90.0, size_scale=2.4)

        bgpos, bgrgb = self._bg
        bgw = bgpos + eye            # keep the shell centred on the camera
        bbuf, _ = project_and_splat(cam, bgw, bgrgb, ctx.width, ctx.height,
                                    falloff=False, size_scale=0.0)
        buf += bbuf
        return buf.reshape(ctx.height, ctx.width, 3)

    # ----------------------------------------------------------------
    def render(self, ctx, lt):
        t_myr = lt * self.p.myr_per_second
        self.spiral.set_time(t_myr)
        self.shader.set_time(t_myr)
        cam = self.camera_at(ctx, lt)

        rng = ctx.rng or np.random.default_rng(ctx.frame * 7919 + 17)

        def shade(pos, t, u):
            self.shader.new_step(rng)
            return self.shader(pos, t, u)

        small, trn = march_layer(ctx, cam, self._bounds, shade,
                                 steps=ctx.march_steps(),
                                 jitter_phase=ctx.frame & 7)
        vol = upsample_linear(small, ctx.width, ctx.height)
        stars = self._star_layer(ctx, cam, t_myr, trn)
        out = vol + stars
        out *= F32(self.exposure_at(ctx, lt))
        return out


__all__ = ["GalaxyParams", "SpiralField", "StarCatalogue", "ISMShader",
           "PhaseGalaxy", "rotation_curve", "angular_speed",
           "epicyclic_frequency", "build_catalogue", "disc_warp"]
