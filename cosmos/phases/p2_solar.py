"""
cosmos.phases.p2_solar
======================

FASE II — TATA SURYA.

Planetary motion integrated from Newtonian gravity, not keyframed.  Every
body in the table below is given its real mass and its real osculating
orbital elements at J2000, converted to a state vector, and then handed to
a symplectic integrator that evaluates all mutual interactions.  Nothing in
this phase follows a predetermined path: Jupiter perturbs Saturn, the
Galilean moons perturb each other, and the Sun itself wobbles about the
barycentre by rather more than its own radius.

Why a symplectic integrator
---------------------------
The obvious choice — Runge-Kutta 4 — is more accurate per step on a short
interval and completely wrong on a long one.  RK4 is not symplectic: it
does not preserve phase-space volume, so orbital energy drifts secularly
and after a few thousand orbits the planets have visibly spiralled.  The
velocity-Verlet / leapfrog scheme used here is only second order, but it is
symplectic and time-reversible, so the energy error stays bounded and
oscillatory forever.  For orbital mechanics that property matters far more
than the order of the local truncation error.

Rendering
---------
This phase does not ray-march.  A planet is a hard surface, and marching a
volume to find its boundary would be both slower and softer than simply
solving the quadratic.  Each body is projected to a screen-space bounding
rectangle, rays are generated for that rectangle only, and the sphere is
intersected analytically.  A depth buffer resolves occlusion between
bodies, so moons pass correctly in front of and behind their primaries.

Shadows are analytic too: a surface point is in shadow if the segment
joining it to the Sun intersects another body, which is one ray-sphere test
per caster.  That gives real solar eclipses, Jupiter's moons dropping their
shadows onto the cloud deck, and Saturn's rings casting a band across the
planet — none of which are drawn, all of which emerge from the geometry.

Units
-----
Astronomical units, years, solar masses.  In that system Newton's constant
is exactly ``4 pi^2``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import camera as CAM
from ..core import spectrum as S
from ..core import vmath as V
from ..core import vcache
from .base import (Phase, build_emission_ramp, project_and_splat,
                   ramp_lookup, splat_points, upsample_linear)

F32 = np.float32
TAU = 2.0 * math.pi

#: Newton's constant in AU^3 / (solar mass * year^2)
G_AU = 4.0 * math.pi * math.pi
#: kilometres per astronomical unit
KM_PER_AU = 1.495978707e8
#: hours per year
HOURS_PER_YEAR = 365.25 * 24.0


# ==========================================================================
# body table
# ==========================================================================
@dataclass
class Body:
    """One gravitating body with its physical and visual properties."""

    name: str
    mass: float                  # solar masses
    radius_km: float
    parent: str = ""             # "" for heliocentric orbits

    # osculating elements at J2000 (heliocentric, ecliptic)
    a: float = 0.0               # semi-major axis, AU
    e: float = 0.0               # eccentricity
    inc: float = 0.0             # inclination, degrees
    lan: float = 0.0             # longitude of ascending node, degrees
    argp: float = 0.0            # argument of periapsis, degrees
    mean_anom: float = 0.0       # mean anomaly at epoch, degrees

    rot_hours: float = 24.0      # sidereal rotation period (negative = retrograde)
    tilt: float = 0.0            # axial tilt, degrees
    tilt_node: float = 0.0       # node of the tilt, degrees

    draw_scale: float = 1.0      # display magnification of the radius
    kind: str = "rock"           # rock | gas | ice | ocean | star | moon

    # surface appearance
    base_temp: float = 5200.0    # colour temperature of the albedo ramp
    albedo: tuple = (0.42, 0.40, 0.38)
    albedo2: tuple = (0.22, 0.20, 0.19)
    roughness: float = 0.9
    atmosphere: float = 0.0      # rim-scattering strength
    atmo_rgb: tuple = (0.30, 0.52, 1.00)
    band_count: float = 0.0      # latitudinal banding for gas giants
    spot: float = 0.0            # great-red-spot style vortex
    ring_inner: float = 0.0      # ring radii in planetary radii
    ring_outer: float = 0.0
    night_glow: float = 0.0      # city lights / internal heat
    noise_scale: float = 2.2
    emissive: float = 0.0        # self-luminous (the Sun)

    @property
    def radius_au(self):
        return self.radius_km / KM_PER_AU

    @property
    def draw_radius(self):
        return self.radius_au * self.draw_scale

    @property
    def spin_rate(self):
        """Radians per year."""
        if abs(self.rot_hours) < 1e-9:
            return 0.0
        return TAU * HOURS_PER_YEAR / self.rot_hours


#: The real solar system.  Masses, radii and elements are J2000 values,
#: used unmodified; see ``SCALE_NOTE`` for how readability is handled.
SOLAR_SYSTEM = [
    Body("Sun", 1.0, 695700.0, draw_scale=1.0, kind="star",
         base_temp=5772.0, emissive=1.0, rot_hours=609.12,
         albedo=(1.0, 0.93, 0.84), noise_scale=5.0),

    Body("Mercury", 1.66012e-7, 2439.7, a=0.3870983, e=0.2056302,
         inc=7.00487, lan=48.3317, argp=29.1241, mean_anom=174.796,
         rot_hours=1407.6, tilt=0.034, draw_scale=1.0, kind="rock",
         albedo=(0.41, 0.38, 0.35), albedo2=(0.20, 0.18, 0.17),
         noise_scale=5.5),

    Body("Venus", 2.44784e-6, 6051.8, a=0.7233298, e=0.0067719,
         inc=3.39471, lan=76.6807, argp=54.8523, mean_anom=50.115,
         rot_hours=-5832.5, tilt=177.36, draw_scale=1.0, kind="rock",
         albedo=(0.93, 0.85, 0.66), albedo2=(0.80, 0.70, 0.50),
         atmosphere=0.9, atmo_rgb=(1.0, 0.88, 0.62), noise_scale=1.9),

    Body("Earth", 3.00348e-6, 6371.0, a=1.0000011, e=0.0167086,
         inc=0.00005, lan=-11.26064, argp=114.20783, mean_anom=358.617,
         rot_hours=23.9345, tilt=23.4393, draw_scale=1.0, kind="ocean",
         albedo=(0.12, 0.30, 0.17), albedo2=(0.03, 0.07, 0.16),
         atmosphere=1.0, atmo_rgb=(0.26, 0.48, 1.00), night_glow=0.55,
         noise_scale=2.6),

    Body("Mars", 3.22715e-7, 3389.5, a=1.5236883, e=0.0934006,
         inc=1.84969, lan=49.5785, argp=286.5016, mean_anom=19.373,
         rot_hours=24.6229, tilt=25.19, draw_scale=1.0, kind="rock",
         albedo=(0.72, 0.36, 0.20), albedo2=(0.45, 0.22, 0.13),
         atmosphere=0.22, atmo_rgb=(0.85, 0.60, 0.45), noise_scale=4.2),

    Body("Jupiter", 9.54588e-4, 69911.0, a=5.2026032, e=0.0484979,
         inc=1.30327, lan=100.4739, argp=273.8670, mean_anom=20.020,
         rot_hours=9.9250, tilt=3.13, draw_scale=1.0, kind="gas",
         albedo=(0.86, 0.74, 0.58), albedo2=(0.52, 0.34, 0.22),
         band_count=9.0, spot=1.0, atmosphere=0.45,
         atmo_rgb=(0.95, 0.80, 0.62), noise_scale=1.4),

    Body("Saturn", 2.85802e-4, 58232.0, a=9.5549091, e=0.0555481,
         inc=2.48446, lan=113.6624, argp=339.3939, mean_anom=317.021,
         rot_hours=10.656, tilt=26.73, draw_scale=1.0, kind="gas",
         albedo=(0.91, 0.82, 0.62), albedo2=(0.70, 0.60, 0.42),
         band_count=6.0, atmosphere=0.38, atmo_rgb=(0.96, 0.88, 0.68),
         ring_inner=1.24, ring_outer=2.27, noise_scale=1.2),

    Body("Uranus", 4.36624e-5, 25362.0, a=19.2184461, e=0.0457290,
         inc=0.76986, lan=74.0170, argp=96.5410, mean_anom=142.239,
         rot_hours=-17.24, tilt=97.77, draw_scale=1.0, kind="ice",
         albedo=(0.52, 0.80, 0.84), albedo2=(0.38, 0.66, 0.74),
         band_count=3.0, atmosphere=0.5, atmo_rgb=(0.55, 0.85, 0.92),
         noise_scale=1.0),

    Body("Neptune", 5.15139e-5, 24622.0, a=30.1104, e=0.0112150,
         inc=1.76917, lan=131.7843, argp=273.2497, mean_anom=259.908,
         rot_hours=16.11, tilt=28.32, draw_scale=1.0, kind="ice",
         albedo=(0.22, 0.38, 0.80), albedo2=(0.13, 0.24, 0.62),
         band_count=4.0, atmosphere=0.55, atmo_rgb=(0.32, 0.50, 0.95),
         noise_scale=1.1),

    # --- satellites, elements relative to their primary -----------------
    Body("Luna", 3.69432e-8, 1737.4, parent="Earth", a=0.00256956,
         e=0.0549, inc=5.145, lan=125.08, argp=318.15, mean_anom=135.27,
         rot_hours=655.73, tilt=6.68, draw_scale=1.0, kind="moon",
         albedo=(0.38, 0.36, 0.33), albedo2=(0.17, 0.16, 0.15),
         noise_scale=6.0),

    Body("Io", 4.49e-8, 1821.6, parent="Jupiter", a=0.002819,
         e=0.0041, inc=0.05, lan=43.0, argp=84.1, mean_anom=342.0,
         rot_hours=42.46, draw_scale=1.0, kind="moon",
         albedo=(0.92, 0.80, 0.33), albedo2=(0.70, 0.42, 0.14),
         noise_scale=5.0),

    Body("Europa", 2.41e-8, 1560.8, parent="Jupiter", a=0.004486,
         e=0.0094, inc=0.47, lan=219.1, argp=88.9, mean_anom=171.0,
         rot_hours=85.23, draw_scale=1.0, kind="moon",
         albedo=(0.86, 0.82, 0.76), albedo2=(0.62, 0.58, 0.56),
         noise_scale=7.5),

    Body("Ganymede", 7.4497e-8, 2634.1, parent="Jupiter", a=0.007155,
         e=0.0013, inc=0.20, lan=63.6, argp=192.4, mean_anom=317.5,
         rot_hours=171.71, draw_scale=1.0, kind="moon",
         albedo=(0.55, 0.50, 0.45), albedo2=(0.32, 0.30, 0.29),
         noise_scale=5.2),

    Body("Callisto", 5.3938e-8, 2410.3, parent="Jupiter", a=0.012585,
         e=0.0074, inc=0.19, lan=298.0, argp=52.6, mean_anom=181.4,
         rot_hours=400.54, draw_scale=1.0, kind="moon",
         albedo=(0.30, 0.27, 0.24), albedo2=(0.14, 0.13, 0.12),
         noise_scale=6.5),

    Body("Titan", 6.7659e-8, 2574.7, parent="Saturn", a=0.008168,
         e=0.0288, inc=0.34, lan=28.1, argp=180.5, mean_anom=120.0,
         rot_hours=382.69, draw_scale=1.0, kind="moon",
         albedo=(0.78, 0.58, 0.26), albedo2=(0.58, 0.40, 0.18),
         atmosphere=0.6, atmo_rgb=(0.95, 0.72, 0.36), noise_scale=2.4),
]

SCALE_NOTE = """\
Everything in this phase is at true scale: orbits, radii and the ratios
between them.  The usual documentary trick of magnifying planetary radii is
refused here because it breaks the thing the phase exists to show — at a
magnification large enough to make Jupiter visible from one astronomical
unit, the Galilean moons end up orbiting *inside* the planet.

Readability is recovered the way an observer actually recovers it: a body
whose angular diameter falls below a pixel stops being traced as a sphere
and is splatted as a point whose brightness is its genuine reflected flux,
albedo * R^2 / (d_sun^2 d_cam^2), with a Lambert phase function.  That is
exactly what a planet looks like from far away, so nothing is faked - the
representation changes, the physics does not."""


# ==========================================================================
# orbital elements -> state vectors
# ==========================================================================
def solve_kepler(mean_anom, ecc, tol=1e-12, max_iter=60):
    """Solve M = E - e sin E for the eccentric anomaly.

    Newton-Raphson from a cubic starting guess.  Converges in three or four
    iterations for every eccentricity in the solar system; the iteration
    cap exists only for the hyperbolic-adjacent cases that this table does
    not contain.
    """
    m = np.mod(np.asarray(mean_anom, np.float64) + math.pi, TAU) - math.pi
    e = np.asarray(ecc, np.float64)
    # starting guess good to ~1e-3 even at e = 0.9
    ea = m + e * np.sin(m) / np.maximum(1.0 - np.sin(m + e) + np.sin(m), 0.3)
    for _ in range(max_iter):
        f = ea - e * np.sin(ea) - m
        fp = 1.0 - e * np.cos(ea)
        d = f / fp
        ea = ea - d
        if np.all(np.abs(d) < tol):
            break
    return ea


def elements_to_state(a, e, inc_deg, lan_deg, argp_deg, m0_deg, mu):
    """Convert Keplerian elements to a heliocentric state vector.

    Returns ``(position, velocity)`` in AU and AU/year.  ``mu`` is
    ``G (M_primary + M_body)``.
    """
    inc = math.radians(inc_deg)
    lan = math.radians(lan_deg)
    argp = math.radians(argp_deg)
    m0 = math.radians(m0_deg)

    ea = float(solve_kepler(np.array([m0]), np.array([e]))[0])
    # true anomaly from the eccentric anomaly
    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(ea * 0.5),
                          math.sqrt(1.0 - e) * math.cos(ea * 0.5))
    r = a * (1.0 - e * math.cos(ea))

    # position and velocity in the perifocal frame
    p_x = r * math.cos(nu)
    p_y = r * math.sin(nu)
    h = math.sqrt(mu * a * (1.0 - e * e))
    v_x = -mu / h * math.sin(nu)
    v_y = mu / h * (e + math.cos(nu))

    # perifocal -> ecliptic: Rz(lan) Rx(inc) Rz(argp)
    cO, sO = math.cos(lan), math.sin(lan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    m11 = cO * cw - sO * sw * ci
    m12 = -cO * sw - sO * cw * ci
    m21 = sO * cw + cO * sw * ci
    m22 = -sO * sw + cO * cw * ci
    m31 = sw * si
    m32 = cw * si

    pos = np.array([m11 * p_x + m12 * p_y,
                    m31 * p_x + m32 * p_y,
                    m21 * p_x + m22 * p_y], np.float64)
    vel = np.array([m11 * v_x + m12 * v_y,
                    m31 * v_x + m32 * v_y,
                    m21 * v_x + m22 * v_y], np.float64)
    # NOTE: the ecliptic z axis is mapped to the scene's y axis, so the
    # solar system lies in the renderer's XZ plane like everything else.
    return pos, vel


def state_to_elements(r_vec, v_vec, mu):
    """Recover osculating Keplerian elements from a state vector.

    The inverse of ``elements_to_state``.  Returns ``None`` for unbound or
    degenerate orbits, which this system never produces but a perturbed
    one could.
    """
    r_vec = np.asarray(r_vec, np.float64)
    v_vec = np.asarray(v_vec, np.float64)
    r = float(np.linalg.norm(r_vec))
    v2 = float(v_vec @ v_vec)
    if r < 1e-12:
        return None
    # specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = float(np.linalg.norm(h_vec))
    if h < 1e-14:
        return None
    # eccentricity vector: e = (v x h)/mu - r/|r|
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = float(np.linalg.norm(e_vec))
    energy = 0.5 * v2 - mu / r
    if energy >= 0.0:
        return None
    a = -mu / (2.0 * energy)
    # true anomaly: angle from periapsis to the body
    if e > 1e-10:
        cos_nu = float(e_vec @ r_vec) / (e * r)
        nu = math.acos(max(-1.0, min(1.0, cos_nu)))
        if float(r_vec @ v_vec) < 0.0:
            nu = TAU - nu
        p_hat = e_vec / e
    else:
        nu = 0.0
        p_hat = r_vec / r
    w_hat = h_vec / h
    q_hat = np.cross(w_hat, p_hat)
    return {"a": a, "e": e, "nu": nu, "p": p_hat, "q": q_hat, "w": w_hat,
            "h": h, "mu": mu}


def sample_orbit(el, centre, n=512):
    """Points along an osculating conic, in world coordinates.

    Sampled uniformly in true anomaly, which concentrates points near
    periapsis exactly where the curvature is highest.
    """
    nu = np.linspace(0.0, TAU, n, endpoint=False)
    a, e = el["a"], el["e"]
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(nu))
    pts = (np.asarray(centre, np.float64)[None, :]
           + r[:, None] * (np.cos(nu)[:, None] * el["p"][None, :]
                           + np.sin(nu)[:, None] * el["q"][None, :]))
    return pts, nu


# ==========================================================================
# the integrator
# ==========================================================================
class NBody:
    """Velocity-Verlet N-body integrator with all pairwise interactions.

    Softening is set to a thousandth of an AU — far below any real
    separation in the system, so it never alters the dynamics, but it keeps
    the acceleration finite if a close encounter is ever engineered.
    """

    def __init__(self, bodies, softening=1e-3, history=256):
        self.bodies = list(bodies)
        self.n = len(self.bodies)
        self.index = {b.name: i for i, b in enumerate(self.bodies)}
        self.mass = np.array([b.mass for b in self.bodies], np.float64)
        self.soft2 = float(softening) ** 2
        self.pos = np.zeros((self.n, 3), np.float64)
        self.vel = np.zeros((self.n, 3), np.float64)
        self.time = 0.0
        self.history_len = int(history)
        self.history = None
        self._hist_fill = 0
        self._init_state()
        self._acc = self.accelerations(self.pos)

    # ----------------------------------------------------------------
    def _init_state(self):
        """Seed positions and velocities from the osculating elements."""
        for i, b in enumerate(self.bodies):
            if b.a <= 0.0:
                continue
            if b.parent:
                j = self.index[b.parent]
                mu = G_AU * (self.bodies[j].mass + b.mass)
                p, v = elements_to_state(b.a, b.e, b.inc, b.lan, b.argp,
                                         b.mean_anom, mu)
                self.pos[i] = self.pos[j] + p
                self.vel[i] = self.vel[j] + v
            else:
                mu = G_AU * (self.mass[0] + b.mass)
                p, v = elements_to_state(b.a, b.e, b.inc, b.lan, b.argp,
                                         b.mean_anom, mu)
                self.pos[i] = p
                self.vel[i] = v
        # Satellites must be seeded after their primaries; the table is
        # ordered so that holds, but re-run the satellite pass to be safe
        # in case a caller reorders it.
        for i, b in enumerate(self.bodies):
            if b.parent and b.a > 0.0:
                j = self.index[b.parent]
                mu = G_AU * (self.bodies[j].mass + b.mass)
                p, v = elements_to_state(b.a, b.e, b.inc, b.lan, b.argp,
                                         b.mean_anom, mu)
                self.pos[i] = self.pos[j] + p
                self.vel[i] = self.vel[j] + v

        # Shift to the barycentric frame.  Without this the whole system
        # drifts across the screen at the Sun's recoil velocity, which is
        # physically correct but visually just a slow pan.
        total = self.mass.sum()
        self.pos -= (self.mass[:, None] * self.pos).sum(0) / total
        self.vel -= (self.mass[:, None] * self.vel).sum(0) / total

        self.history = np.repeat(self.pos[None, :, :], self.history_len,
                                 axis=0)

    # ----------------------------------------------------------------
    def accelerations(self, pos):
        """All-pairs Newtonian acceleration.

        ``(n, n, 3)`` difference tensor.  With twenty bodies that is 1200
        floats — vectorising it is both simpler and faster than any
        cleverer scheme, and tree codes only start winning in the
        thousands.
        """
        d = pos[None, :, :] - pos[:, None, :]
        r2 = np.einsum('ijk,ijk->ij', d, d) + self.soft2
        np.fill_diagonal(r2, np.inf)
        inv_r3 = r2 ** -1.5
        return G_AU * np.einsum('ij,ijk,j->ik', inv_r3, d, self.mass)

    def step(self, dt):
        """One velocity-Verlet step."""
        half = 0.5 * dt
        self.vel += self._acc * half
        self.pos += self.vel * dt
        self._acc = self.accelerations(self.pos)
        self.vel += self._acc * half
        self.time += dt

    def advance_to(self, t, max_dt=3.0e-5):
        """Integrate forward to absolute time ``t``.

        The step cap is set by the fastest orbit in the system — Io, at
        1.77 days.  Resolving it to better than a percent needs a step of
        about 2e-4 years, and the symplectic scheme then keeps every slower
        orbit essentially exact.
        """
        remaining = t - self.time
        if remaining <= 0.0:
            return
        n = max(1, int(math.ceil(remaining / max_dt)))
        dt = remaining / n
        for _ in range(n):
            self.step(dt)
            self.history[self._hist_fill % self.history_len] = self.pos
            self._hist_fill += 1

    # ----------------------------------------------------------------
    def energy(self):
        """Total mechanical energy — the integrator's conservation check."""
        ke = 0.5 * float((self.mass * np.einsum('ij,ij->i', self.vel,
                                                self.vel)).sum())
        d = self.pos[None, :, :] - self.pos[:, None, :]
        r = np.sqrt(np.einsum('ijk,ijk->ij', d, d))
        np.fill_diagonal(r, np.inf)
        pe = -G_AU * float((np.outer(self.mass, self.mass) / r).sum()) * 0.5
        return ke + pe

    def trail(self, i, count=160, stride=3):
        """Recent positions of body ``i``, oldest first."""
        if self._hist_fill == 0:
            return self.pos[i][None, :]
        n = min(self._hist_fill, self.history_len)
        order = (np.arange(n) + self._hist_fill) % self.history_len
        pts = self.history[order[-count * stride:], i]
        return pts[::stride]


# ==========================================================================
# surface rendering
# ==========================================================================
class SurfaceRenderer:
    """Analytic sphere rasteriser with physically-motivated shading.

    Only the screen rectangle each body actually covers is traced, so the
    cost scales with the area the planets occupy rather than with the frame.
    A full-resolution depth buffer resolves mutual occlusion.
    """

    #: Display slowdown applied to axial rotation.  Same justification as
    #: the radius magnification: at true rate Jupiter turns twice per second
    #: of film and reads as a strobing blur rather than a rotating planet.
    #: Orbital motion is *not* scaled — only the spin of each body on its
    #: own axis, which is kinematically independent of the dynamics.
    SPIN_SCALE = 0.055

    def __init__(self, lib, width, height, seed=3):
        self.lib = lib
        self.width = int(width)
        self.height = int(height)
        self.v_rock = lib.get("nebula_fine")
        self.v_band = lib.get("plasma")
        self.v_fine = lib.get("granule")
        self.v_cloud = lib.get("disc_fine")
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        n = self.width * self.height
        self.color = np.zeros((n, 3), F32)
        self.depth = np.full(n, np.inf, np.float32)

    # ----------------------------------------------------------------
    @staticmethod
    def screen_rect(cam, centre, radius, width, height, margin=6):
        """Bounding rectangle in pixels, or None if entirely offscreen."""
        nx, ny, z, vis = cam.project(centre[None, :].astype(F32))
        if z[0] <= radius * 1.001:
            return None                      # camera inside or behind
        aspect = width / float(height)
        # angular radius -> NDC radius, using the same projection the
        # camera's ray generator uses
        f = 1.0 / math.tan(math.radians(cam.fov_deg) * 0.5)
        ang = math.asin(min(1.0, radius / float(z[0])))
        rad_ndc = math.tan(ang) * f
        cx = (float(nx[0]) / aspect * 0.5 + 0.5) * width
        cy = (0.5 - float(ny[0]) * 0.5) * height
        rpx = rad_ndc * 0.5 * height / 1.0 * aspect / aspect
        rpx = rad_ndc * 0.5 * height
        x0 = int(math.floor(cx - rpx)) - margin
        x1 = int(math.ceil(cx + rpx)) + margin
        y0 = int(math.floor(cy - rpx)) - margin
        y1 = int(math.ceil(cy + rpx)) + margin
        x0 = max(x0, 0); y0 = max(y0, 0)
        x1 = min(x1, width); y1 = min(y1, height)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1, rpx

    def rect_rays(self, cam, rect):
        """Generate rays for a screen rectangle only."""
        x0, y0, x1, y1, _ = rect
        w, h = self.width, self.height
        ys = np.arange(y0, y1)
        xs = np.arange(x0, x1)
        idx = (ys[:, None] * w + xs[None, :]).ravel()
        o, d, _ = cam.generate_rays(w, h, jitter_phase=None, subset=idx)
        return o, d, idx

    # ----------------------------------------------------------------
    def body_frame(self, body: Body, t_years):
        """Rotation matrix taking body-fixed coordinates to world.

        Axial tilt applied about the node, then spin about the body's own
        pole.  Uranus, tilted 98 degrees, therefore rolls along its orbit
        with its pole pointing at the Sun for a quarter of its year — which
        is exactly what the real planet does.
        """
        tilt = math.radians(body.tilt)
        node = math.radians(body.tilt_node)
        spin = body.spin_rate * float(t_years) * self.SPIN_SCALE
        rz = V.rot_y(np.float32(node))
        rx = V.rot_x(np.float32(tilt))
        ry = V.rot_y(np.float32(spin))
        return (rz @ rx @ ry).astype(F32)

    # ----------------------------------------------------------------
    def shade(self, body: Body, hit_pos, normal, view_dir, sun_pos,
              t_years, shadow_casters, frame_mat):
        """Surface radiance for the hit points of one body."""
        # --- illumination geometry -----------------------------------
        to_sun = sun_pos[None, :].astype(F32) - hit_pos
        dist2 = np.einsum('ij,ij->i', to_sun, to_sun)
        dist = np.sqrt(dist2)
        ldir = to_sun / dist[:, None]
        ndl = np.einsum('ij,ij->i', normal, ldir)
        # Solar irradiance falls as 1/r^2.  Neptune receives one nine
        # hundredth of Earth's; the exposure curve, not the shader, is what
        # makes the outer planets visible.
        irr = F32(1.0) / np.maximum(dist2, F32(1e-9))

        lam = np.maximum(ndl, F32(0.0))
        # soften the terminator: a sphere lit by a disc of finite angular
        # size has a penumbra, and a hard terminator is the classic tell of
        # a point-light renderer
        ang_rad = F32(body.radius_au * 0.0 + 0.0046) / np.maximum(dist, 1e-6)
        soft = np.clip((ndl + ang_rad) / np.maximum(2.0 * ang_rad, 1e-4),
                       0.0, 1.0)
        lam = np.maximum(lam, soft * F32(0.5) * np.maximum(ndl + ang_rad, 0))

        # --- cast shadows ---------------------------------------------
        shadow = np.ones(hit_pos.shape[0], F32)
        for cpos, crad in shadow_casters:
            oc = hit_pos - cpos[None, :].astype(F32)
            b = np.einsum('ij,ij->i', oc, ldir)
            c = np.einsum('ij,ij->i', oc, oc) - F32(crad * crad)
            disc = b * b - c
            blocked = (disc > 0.0) & (b < 0.0)
            if blocked.any():
                # penumbra from the Sun's finite angular size
                pen = np.clip(np.sqrt(np.maximum(disc, 0.0))
                              / F32(crad * 0.22), 0.0, 1.0)
                shadow = np.where(blocked, np.minimum(shadow,
                                                      F32(1.0) - pen), shadow)

        # --- surface coordinates in the body frame --------------------
        local = normal @ frame_mat          # world normal -> body-fixed
        lat = np.clip(local[:, 1], -1.0, 1.0)

        alb = np.asarray(body.albedo, F32)
        alb2 = np.asarray(body.albedo2, F32)

        if body.kind in ("gas", "ice"):
            rgb = self._gas_surface(body, local, lat, alb, alb2)
            spec = F32(0.0)
        elif body.kind == "ocean":
            rgb, spec = self._ocean_surface(body, local, lat, alb, alb2,
                                            normal, ldir, view_dir)
        else:
            rgb = self._rock_surface(body, local, lat, alb, alb2)
            spec = F32(0.0)

        out = rgb * (lam * shadow * irr)[:, None]

        if np.ndim(spec) or float(np.max(spec)) > 0.0:
            out += (spec * lam * shadow * irr)[:, None] * F32(2.4)

        # --- atmospheric limb ------------------------------------------
        if body.atmosphere > 0.0:
            ndv = np.abs(np.einsum('ij,ij->i', normal, view_dir))
            rim = np.power(np.clip(1.0 - ndv, 0.0, 1.0), 2.6)
            # forward scattering: the limb glows most where it is also lit
            fwd = np.clip(ndl + F32(0.26), 0.0, 1.0)
            atmo = np.asarray(body.atmo_rgb, F32)[None, :]
            out += atmo * (rim * fwd * shadow * irr
                           * F32(1.9 * body.atmosphere))[:, None]

        # --- night side -------------------------------------------------
        if body.night_glow > 0.0:
            night = np.clip(-ndl * F32(3.2), 0.0, 1.0)
            # city lights cluster on land, not ocean
            land = np.clip((self._rock_mask(body, local) - F32(0.585))
                           * F32(8.0), 0.0, 1.0)
            glow = np.asarray((1.0, 0.72, 0.36), F32)[None, :]
            out += glow * (night * land * F32(0.0022 * body.night_glow)
                           )[:, None]

        if body.emissive > 0.0:
            out = self._star_surface(body, local, view_dir, normal)

        return out

    # ----------------------------------------------------------------
    def _noise(self, vol, p, scale):
        return vol.sample(np.ascontiguousarray(p * F32(scale), F32),
                          jitter=vcache.new_jitter(self.rng))

    def _rock_mask(self, body, local):
        """Continent / highland mask in [0, 1].

        ``v_rock`` is a signed fBm volume (zero mean, sigma about 0.18) and
        ``v_fine`` is a unipolar Worley volume, so the two need different
        remappings before they can be mixed — combining them as if both
        were in [0, 1] is what turns an ocean planet into a grey rock.
        """
        n = self._noise(self.v_rock, local, body.noise_scale)
        n2 = self._noise(self.v_fine, local, body.noise_scale * 3.1)
        m = n * F32(2.45) + F32(0.5) + (n2 - F32(0.48)) * F32(0.34)
        return np.clip(m, F32(0.0), F32(1.0))

    def _rock_surface(self, body, local, lat, alb, alb2):
        m = self._rock_mask(body, local)
        crater = self._noise(self.v_fine, local, body.noise_scale * 6.0)
        m = np.clip(m + crater * F32(0.30), 0.0, 1.0)
        rgb = alb2[None, :] + (alb - alb2)[None, :] * m[:, None]
        # polar frost
        if body.kind in ("rock", "moon"):
            ice = np.clip((np.abs(lat) - F32(0.80)) * F32(5.5), 0.0, 1.0)
            ice *= np.clip(F32(0.6) + m * F32(0.8), 0.0, 1.0)
            rgb = rgb + (F32(0.92) - rgb) * ice[:, None]
        return rgb

    def _gas_surface(self, body, local, lat, alb, alb2):
        """Zonal bands with turbulent shear.

        Jupiter's banding is a set of alternating prograde and retrograde
        jets.  The visual signature is not the bands themselves but the
        turbulence *between* them, so the latitude coordinate is warped by
        noise before the band function is evaluated — which is what makes
        the belts billow and curl instead of lying in flat stripes.
        """
        warp = self._noise(self.v_band, local, body.noise_scale * 1.7)
        warp2 = self._noise(self.v_cloud, local, body.noise_scale * 4.3)
        wlat = lat + warp * F32(0.085) + warp2 * F32(0.028)
        bands = np.sin(wlat * F32(body.band_count * math.pi))
        bands = bands * F32(0.5) + F32(0.5)
        bands = bands * bands * (F32(3.0) - F32(2.0) * bands)   # smoothstep
        rgb = alb2[None, :] + (alb - alb2)[None, :] * bands[:, None]

        # fine storm texture, strongest in the belts
        storm = np.clip(warp2 * F32(2.0) + F32(0.5), 0.0, 1.0)
        rgb *= (F32(0.70) + F32(0.62) * storm)[:, None]

        # polar hoods are darker and bluer
        hood = np.clip((np.abs(lat) - F32(0.70)) * F32(3.0), 0.0, 1.0)
        rgb = rgb * (F32(1.0) - hood * F32(0.45))[:, None]

        if body.spot > 0.0:
            # an anticyclone: a long-lived ellipse in the south tropical
            # zone, elongated in longitude because the zonal winds shear it
            sx = local[:, 0]
            sy = local[:, 1] + F32(0.33)
            sz = local[:, 2]
            lon = np.arctan2(sz, sx)
            d = ((lon - F32(1.1)) / F32(0.55)) ** 2 + (sy / F32(0.14)) ** 2
            blob = np.exp(-d * F32(1.6))
            spot_rgb = np.asarray((0.86, 0.34, 0.20), F32)[None, :]
            rgb = rgb + (spot_rgb - rgb) * (blob * F32(body.spot))[:, None]
        return rgb

    def _ocean_surface(self, body, local, lat, alb, alb2, normal, ldir,
                       view_dir):
        """Land, ocean, ice caps and a specular sun glint."""
        m = self._rock_mask(body, local)
        land = np.clip((m - F32(0.585)) * F32(11.0), 0.0, 1.0)
        rgb = alb2[None, :] + (alb - alb2)[None, :] * land[:, None]

        # vegetation / desert variation on land
        veg = self._noise(self.v_cloud, local, body.noise_scale * 2.3)
        dry = np.asarray((0.52, 0.42, 0.24), F32)[None, :]
        rgb = rgb + (dry - rgb) * (np.clip(veg * F32(2.2) + F32(0.35), 0, 1)
                                   * land * F32(0.55))[:, None]

        ice = np.clip((np.abs(lat) - F32(0.74)) * F32(6.0), 0.0, 1.0)
        rgb = rgb + (F32(0.95) - rgb) * ice[:, None]

        # cloud deck, a separate layer above the surface
        cl = self._noise(self.v_band, local, body.noise_scale * 1.35)
        cl2 = self._noise(self.v_fine, local, body.noise_scale * 4.0)
        cloud = np.clip((cl * F32(2.4) + F32(0.5)) * F32(0.72)
                        + cl2 * F32(0.5), 0.0, 1.0)
        cloud = np.clip((cloud - F32(0.46)) * F32(2.6), 0.0, 1.0)
        rgb = rgb + (F32(1.0) - rgb) * cloud[:, None]

        # Specular glint off water only — Blinn-Phong with a tight lobe.
        half = ldir - view_dir
        half /= np.maximum(np.sqrt(np.einsum('ij,ij->i', half, half)),
                           1e-9)[:, None]
        ndh = np.clip(np.einsum('ij,ij->i', normal, half), 0.0, 1.0)
        water = (F32(1.0) - land) * (F32(1.0) - cloud) * (F32(1.0) - ice)
        spec = np.power(ndh, F32(190.0)) * water * F32(1.0)
        return rgb, spec

    def _star_surface(self, body, local, view_dir, normal):
        """The photosphere, seen from outside: limb darkening plus granules.

        Limb darkening is the ``I(mu)/I(0) = 1 - u(1 - mu)`` law, with the
        solar coefficient u = 0.6.  It exists because a glancing line of
        sight reaches optical depth one higher in the atmosphere, where the
        gas is cooler.
        """
        mu = np.abs(np.einsum('ij,ij->i', normal, view_dir))
        limb = F32(1.0) - F32(0.62) * (F32(1.0) - mu)
        gran = self._noise(self.v_fine, local, 26.0)
        gran2 = self._noise(self.v_band, local, 7.0)
        mottle = F32(1.0) + gran * F32(0.42) + gran2 * F32(0.26)
        temp = F32(5772.0) * np.power(np.clip(limb * mottle, 0.05, 3.0),
                                      F32(0.25))
        rgb = S.blackbody_lut(temp)
        return rgb * (limb * mottle * F32(1.0))[:, None]

    # ----------------------------------------------------------------
    def draw_body(self, cam, body: Body, pos, sun_pos, t_years,
                  shadow_casters, exposure=1.0):
        """Trace and composite one body into the buffers."""
        radius = F32(body.draw_radius)
        rect = self.screen_rect(cam, pos.astype(F32), float(radius),
                                self.width, self.height)
        if rect is None:
            return 0
        o, d, idx = self.rect_rays(cam, rect)
        hit, tn, tf = V.ray_sphere(o, d, pos.astype(F32), radius)
        hit = hit.ravel(); tn = tn.ravel()
        live = hit & (tn > 0.0)
        if not live.any():
            return 0
        sel = np.flatnonzero(live)
        t = tn[sel]
        oi = o[sel]
        di = d[sel]
        hp = oi + di * t[:, None]
        nrm = (hp - pos[None, :].astype(F32)) / radius
        px = idx[sel]

        closer = t < self.depth[px]
        if not closer.any():
            return 0
        sel2 = np.flatnonzero(closer)
        hp = hp[sel2]; nrm = nrm[sel2]; px = px[sel2]; t = t[sel2]
        di = di[sel2]

        frame_mat = self.body_frame(body, t_years)
        rgb = self.shade(body, hp, nrm, di, sun_pos.astype(F32), t_years,
                         shadow_casters, frame_mat)
        self.color[px] = rgb * F32(exposure)
        self.depth[px] = t
        return px.shape[0]

    # ----------------------------------------------------------------
    def draw_rings(self, cam, body: Body, pos, sun_pos, t_years,
                   occluders, exposure=1.0, samples=3):
        """Saturn's rings: an annulus of orbiting ice, not a texture.

        Opacity varies with radius through the real gap structure (the
        Cassini Division at 1.95 planetary radii, the Encke gap at 2.21),
        the rings are shadowed by the planet, and they shadow it back.
        """
        if body.ring_outer <= 0.0:
            return 0
        r_in = body.radius_au * body.draw_scale * body.ring_inner
        r_out = body.radius_au * body.draw_scale * body.ring_outer
        frame_mat = self.body_frame(body, 0.0)
        normal = frame_mat @ np.array([0.0, 1.0, 0.0], F32)
        normal = normal / np.linalg.norm(normal)

        rect = self.screen_rect(cam, pos.astype(F32), float(r_out),
                                self.width, self.height, margin=8)
        if rect is None:
            return 0
        o, d, idx = self.rect_rays(cam, rect)
        hit, t, dist = V.ray_disk(o, d, pos.astype(F32), normal.astype(F32),
                                  F32(r_in), F32(r_out))
        hit = hit.ravel(); t = t.ravel(); dist = dist.ravel()
        live = hit & (t > 0.0)
        if not live.any():
            return 0
        sel = np.flatnonzero(live)
        px = idx[sel]
        tt = t[sel]
        rr = dist[sel]
        hp = o[sel] + d[sel] * tt[:, None]

        u = (rr - F32(r_in)) / F32(max(r_out - r_in, 1e-9))
        # Radial opacity profile: C ring, B ring, Cassini Division, A ring,
        # Encke gap.  Positions are the real ones expressed in the
        # normalised radial coordinate of this annulus.
        tau = (np.exp(-((u - F32(0.10)) / F32(0.10)) ** 2) * F32(0.30)
               + np.exp(-((u - F32(0.38)) / F32(0.17)) ** 2) * F32(1.35)
               + np.exp(-((u - F32(0.78)) / F32(0.13)) ** 2) * F32(0.78))
        tau *= F32(1.0) - np.exp(-((u - F32(0.615)) / F32(0.035)) ** 2
                                 ) * F32(0.88)      # Cassini Division
        tau *= F32(1.0) - np.exp(-((u - F32(0.885)) / F32(0.012)) ** 2
                                 ) * F32(0.80)      # Encke gap
        # Fine ringlet structure.  Saturn's rings are not three smooth
        # annuli but thousands of ringlets, carved by density waves at
        # resonances with the inner moons; the radial power spectrum is
        # what gives the rings their grooved, phonographic look.
        ringlets = (np.sin(u * F32(430.0)) * F32(0.19)
                    + np.sin(u * F32(167.0) + F32(1.7)) * F32(0.15)
                    + np.sin(u * F32(61.0) + F32(0.4)) * F32(0.11))
        grain = self._noise(self.v_fine, hp - pos[None, :].astype(F32),
                            900.0)
        tau *= F32(1.0) + grain * F32(0.42) + ringlets
        alpha = F32(1.0) - np.exp(-np.maximum(tau, 0.0) * F32(1.5))

        # illumination: ring particles scatter forward strongly
        to_sun = sun_pos[None, :].astype(F32) - hp
        dist2 = np.einsum('ij,ij->i', to_sun, to_sun)
        ldir = to_sun / np.sqrt(dist2)[:, None]
        irr = F32(1.0) / np.maximum(dist2, 1e-9)
        incl = np.abs(np.einsum('j,ij->i', normal.astype(F32), ldir))
        # planet shadow on the rings
        shadow = np.ones(px.shape[0], F32)
        for cpos, crad in occluders:
            oc = hp - cpos[None, :].astype(F32)
            b = np.einsum('ij,ij->i', oc, ldir)
            c = np.einsum('ij,ij->i', oc, oc) - F32(crad * crad)
            disc = b * b - c
            blocked = (disc > 0.0) & (b < 0.0)
            shadow = np.where(blocked, F32(0.06), shadow)

        # The C ring is darker and greyer than the B ring, which is the
        # cleanest water ice in the solar system; the A ring is in
        # between.  Tint follows the optical depth rather than the radius
        # alone, because the two are physically the same thing.
        ice = np.asarray((0.98, 0.93, 0.80), F32)[None, :]
        dirty = np.asarray((0.62, 0.58, 0.55), F32)[None, :]
        purity = np.clip(tau * F32(0.9), 0.0, 1.0)[:, None]
        col = dirty + (ice - dirty) * purity
        col = col * (F32(0.70) + F32(0.45) * np.clip(
            u * F32(1.2), 0.0, 1.0)[:, None])
        rgb = col * (incl * shadow * irr * F32(1.5))[:, None]

        # composite with depth: rings are translucent, so blend rather than
        # replace, and only where they are in front of whatever is there
        front = tt < self.depth[px]
        a = np.where(front, alpha, alpha * F32(0.0))
        self.color[px] = self.color[px] * (F32(1.0) - a)[:, None] + \
            rgb * (a * F32(exposure))[:, None]
        self.depth[px] = np.where(front & (a > 0.5), tt, self.depth[px])
        return px.shape[0]

    # ----------------------------------------------------------------
    def point_flux(self, body: Body, pos, sun_pos, eye):
        """Reflected flux of an unresolved body, and its colour.

        A planet shines by reflecting sunlight.  The power it intercepts is
        the solar flux at its distance times its cross-section pi R^2; the
        fraction it sends back is its geometric albedo; and what reaches
        the camera falls as the inverse square of the viewing distance
        again.  The phase angle between Sun and observer then decides how
        much of the lit hemisphere is actually turned toward us, which is
        why Venus is brightest as a crescent and not when full.
        """
        d_cam = float(np.linalg.norm(pos - eye))
        if d_cam < 1e-9:
            return None
        r = body.radius_au
        if body.emissive > 0.0:
            # The Sun does not reflect, it radiates, so it must be handled
            # before the Sun-distance guard below: its own distance to
            # itself is zero and the reflected-light branch would divide
            # by it.  Leaving the guard first made the Sun vanish from
            # every frame in which it was smaller than a pixel, which is
            # most of the opening wide shot.
            flux = (r * r) / (d_cam * d_cam) * 6.0e4
            col = np.asarray(S.blackbody_lut(
                np.array([body.base_temp], F32))[0], np.float64)
            return (col * flux).astype(F32)
        d_sun = float(np.linalg.norm(pos - sun_pos))
        if d_sun < 1e-9:
            return None
        to_sun = (sun_pos - pos) / d_sun
        to_cam = (eye - pos) / d_cam
        cos_phase = float(np.clip(to_sun @ to_cam, -1.0, 1.0))
        phase_ang = math.acos(cos_phase)
        # Lambert sphere phase law
        phi = ((math.pi - phase_ang) * math.cos(phase_ang)
               + math.sin(phase_ang)) / math.pi
        alb = np.asarray(body.albedo, np.float64)
        flux = alb * (r * r) / (d_sun * d_sun * d_cam * d_cam) * phi
        return flux.astype(F32)

    def angular_radius_px(self, cam, pos, radius):
        """Apparent radius of a body in pixels, or 0 if behind the camera."""
        d = np.asarray(pos, np.float64) - np.asarray(cam.eye, np.float64)
        z = float(np.linalg.norm(d))
        if z <= radius:
            return 1e9
        f = 1.0 / math.tan(math.radians(cam.fov_deg) * 0.5)
        return math.tan(math.asin(min(1.0, radius / z))) * f * 0.5 * self.height

    def image(self):
        return self.color.reshape(self.height, self.width, 3)


@dataclass
class Shot:
    """One leg of the flight, framed on a body rather than on coordinates.

    The bodies are moving under their own dynamics and will not be where an
    absolute keyframe expects them, so the camera is always specified as an
    offset from whatever it is looking at.
    """

    t0: float
    t1: float
    focus: str
    offset: tuple
    d0: float
    d1: float
    fov: float
    absolute_au: bool = False
    #: exposure multiplier for this leg.  Derived from the subject's
    #: heliocentric irradiance (0.42 r^1.55, the light-meter reading on the
    #: planet) except for the two legs where the Sun itself is the subject,
    #: whose surface brightness does not depend on distance at all.
    ev: float = 1.0


# ==========================================================================
# the phase
# ==========================================================================
class PhaseSolarSystem(Phase):
    """FASE II — the solar system, flown inward from Neptune to the Sun."""

    name = "II · TATA SURYA"
    duration = 11.0

    #: years of simulated time per second of film
    YEARS_PER_SECOND = 0.0026
    #: reference flux for the photographic compression of unresolved bodies
    POINT_REF = 1.0e-10

    def __init__(self, seed=3):
        self.seed = int(seed)
        self.sim = None
        self.surf = None
        self._bg = None
        self._plan = None

    # ----------------------------------------------------------------
    def setup(self, ctx):
        self.sim = NBody(SOLAR_SYSTEM, history=384)
        self.e0 = self.sim.energy()
        self.surf = SurfaceRenderer(ctx.lib, ctx.width, ctx.height,
                                    seed=self.seed)
        self._build_background()
        self._build_plan()

    def _build_background(self):
        rng = np.random.default_rng(self.seed + 4242)
        n = 120000
        dirs = V.fibonacci_sphere(n).astype(F32)
        dirs = dirs[rng.permutation(n)]
        pos = dirs * F32(4000.0)
        t = np.clip(rng.normal(6200.0, 3200.0, n), 2600.0, 30000.0
                    ).astype(F32)
        rgb = S.saturate_rgb(S.blackbody_lut(t), 1.55)
        rgb /= np.maximum(S.luminance(rgb), 1e-5)[:, None]
        b = (rng.random(n).astype(F32) ** 5.0) * F32(3.4) + F32(0.055)
        self._bg = (pos, (rgb * b[:, None]).astype(F32))

    def _build_plan(self):
        """Segments of the flight, each framed on a body.

        The camera is specified relative to the bodies rather than in
        absolute coordinates, because the bodies are moving under their own
        dynamics and will not be where a static keyframe expects them.
        """
        # (t0, t1, focus, offset direction, start distance, end distance,
        #  fov).  Distances are in multiples of the focused body's radius,
        #  except where `absolute_au` is set, which is how the opening wide
        #  shot frames the whole system rather than any one object.
        self._plan = [
            Shot(0.00, 0.22, "Sun", (-0.30, 0.44, -0.85),
                 24.0, 12.5, 52.0, absolute_au=True, ev=1.0),
            Shot(0.22, 0.42, "Saturn", (-0.60, 0.235, -0.765),
                 46.0, 6.4, 46.0, ev=13.0),
            Shot(0.42, 0.58, "Saturn", (0.56, 0.115, -0.82),
                 6.0, 3.3, 44.0, ev=13.0),
            Shot(0.58, 0.78, "Jupiter", (-0.76, 0.13, -0.64),
                 26.0, 9.2, 54.0, ev=5.4),
            Shot(0.78, 0.90, "Earth", (-0.52, 0.185, -0.835),
                 17.0, 3.6, 42.0, ev=0.46),
            Shot(0.90, 1.00, "Sun", (0.22, 0.155, -0.963),
                 38.0, 6.2, 50.0, ev=0.85),
        ]

    # ----------------------------------------------------------------
    def _focus_state(self, u):
        """Interpolate the flight plan at normalised time ``u``."""
        plan = self._plan
        for k, shot in enumerate(plan):
            t0, t1 = shot.t0, shot.t1
            if u <= t1 or k == len(plan) - 1:
                if k == 0 or u >= t0:
                    f = (u - t0) / max(t1 - t0, 1e-6)
                    f = float(np.clip(f, 0.0, 1.0))
                    # Hard cuts between shots.  An earlier version
                    # cross-faded the camera from one subject to the next,
                    # which quietly broke every approach: by the time the
                    # dolly was due to arrive it was already being pulled
                    # back out toward the following shot's start distance,
                    # so no planet was ever reached.  A cut is both more
                    # robust and better film grammar.
                    return self._eval_seg(shot, f), shot.focus
        return self._eval_seg(plan[-1], 1.0), plan[-1].focus

    def _eval_seg(self, shot, f):
        i = self.sim.index[shot.focus]
        body = self.sim.bodies[i]
        centre = self.sim.pos[i]
        unit = 1.0 if shot.absolute_au else body.draw_radius
        # Logarithmic approach.  Interpolating the distance linearly makes
        # a long dolly feel like a deceleration, because what the eye reads
        # is the *rate of change of angular size*, which is logarithmic in
        # range.  Interpolating log(d) instead gives a constant-feeling
        # closing speed over four orders of magnitude.
        g = V.smootherstep(0.0, 1.0, float(f))
        d = math.exp(math.log(shot.d0 * unit) * (1.0 - g)
                     + math.log(shot.d1 * unit) * g)
        o = np.asarray(shot.offset, np.float64)
        o = o / np.linalg.norm(o)
        eye = centre + o * d
        return (eye, np.asarray(centre, np.float64), shot.fov,
                shot.focus, shot.ev)

    def _lerp_seg(self, shot_a, shot_b, f, blend):
        ea, ca, fova, na, eva = self._eval_seg(shot_a, f)
        eb, cb, fovb, nb, evb = self._eval_seg(shot_b, 0.0)
        b = float(blend)
        eye = ea * (1.0 - b) + eb * b
        ctr = ca * (1.0 - b) + cb * b
        fov = fova * (1.0 - b) + fovb * b
        # exposure crossfades in log space — stops, not multipliers
        ev = math.exp(math.log(max(eva, 1e-6)) * (1.0 - b)
                      + math.log(max(evb, 1e-6)) * b)
        name = nb if b > 0.5 else na
        return (eye, ctr, fov, name, ev), name

    def camera_at(self, ctx, lt):
        u = self.progress(lt, self.duration)
        state, _ = self._focus_state(u)
        eye, ctr, fov, _, _ = state
        cam = CAM.Camera(eye=np.asarray(eye, F32),
                         target=np.asarray(ctr, F32),
                         up=np.array([0.0, 1.0, 0.0], F32),
                         fov_deg=float(fov))
        return CAM.apply_handheld(cam, lt, pos_amp=0.0, rot_amp=0.10,
                                  fov_breath=0.10, seed=29.0)

    def exposure_at(self, ctx, lt):
        """Ride the iris from Neptune's twilight to full sunlight.

        Irradiance at Neptune is 1/900 of Earth's.  The curve below is
        matched to the flight plan's heliocentric distance rather than to
        the clock, so it tracks the physics.
        """
        u = self.progress(lt, self.duration)
        state, _ = self._focus_state(u)
        return float(state[4])

    # ----------------------------------------------------------------
    def _draw_trails(self, ctx, cam, buf, exposure):
        """Draw each planet's osculating orbit.

        Not a decorative ellipse: the conic is recovered from the body's
        instantaneous position and velocity by ``state_to_elements``, so it
        is the path the body would follow if every perturbation switched
        off at this instant.  When Jupiter tugs on Saturn the drawn ellipse
        deforms, because the state vector it is derived from has changed.

        The arc is brightened toward the body's current location, which
        reads as direction of travel without any arrowheads.
        """
        sim = self.sim
        sun = sim.pos[sim.index["Sun"]]
        mu_sun = G_AU * sim.mass[sim.index["Sun"]]
        for name, tint, gain in (("Neptune", (0.42, 0.58, 1.0), 1.00),
                                 ("Uranus", (0.50, 0.85, 0.92), 1.00),
                                 ("Saturn", (1.0, 0.90, 0.66), 1.00),
                                 ("Jupiter", (1.0, 0.82, 0.60), 1.00),
                                 ("Mars", (1.0, 0.52, 0.32), 0.85),
                                 ("Earth", (0.45, 0.68, 1.0), 0.85),
                                 ("Venus", (1.0, 0.90, 0.70), 0.75),
                                 ("Mercury", (0.75, 0.72, 0.70), 0.65)):
            i = sim.index[name]
            el = state_to_elements(sim.pos[i] - sun, sim.vel[i] - sim.vel[0],
                                   mu_sun)
            if el is None:
                continue
            pts, nu = sample_orbit(el, sun, n=1500)
            # fade by angular distance behind the body
            dnu = np.mod(el["nu"] - nu, TAU) / TAU
            fade = (F32(0.08) + F32(0.92) * (1.0 - dnu) ** 3).astype(F32)
            col = np.asarray(tint, F32)[None, :] * fade[:, None]
            col *= F32(0.30 * gain * exposure)
            b, _ = project_and_splat(cam, pts.astype(F32), col,
                                     ctx.width, ctx.height,
                                     falloff=False, size_scale=0.0)
            buf += b.reshape(ctx.height, ctx.width, 3)

    def _draw_sun_glow(self, ctx, cam, buf, exposure):
        """Inverse-square halo around the Sun.

        The photosphere itself is a shaded sphere; this is the scattered
        light that makes it read as a source rather than a bright ball.
        """
        sim = self.sim
        sun = sim.pos[sim.index["Sun"]].astype(F32)
        nx, ny, z, vis = cam.project(sun[None, :])
        if not bool(vis[0]) or z[0] <= 0:
            return
        aspect = ctx.width / float(ctx.height)
        cx = (float(nx[0]) / aspect * 0.5 + 0.5) * ctx.width
        cy = (0.5 - float(ny[0]) * 0.5) * ctx.height
        yy, xx = np.mgrid[0:ctx.height, 0:ctx.width]
        dx = (xx - cx) / ctx.height
        dy = (yy - cy) / ctx.height
        r2 = dx * dx + dy * dy
        # Scattered light in the optics, scaled so the halo always subtends
        # a fixed multiple of the Sun's own angular radius.  Tying it to
        # absolute distance instead makes it swell into a white-out the
        # moment the camera closes in.
        sun_r = sim.bodies[sim.index["Sun"]].draw_radius
        ang = max(math.atan2(sun_r, float(z[0])), 1e-6)
        f = 1.0 / math.tan(math.radians(cam.fov_deg) * 0.5)
        rpx = math.tan(ang) * f * 0.5
        core = max(rpx * rpx, 1e-7)
        halo = (F32(0.85 * exposure) * core / (r2 + core)).astype(F32)
        halo += (F32(0.055 * exposure) * core
                 / (r2 + core * F32(34.0))).astype(F32)
        np.minimum(halo, F32(90.0), out=halo)
        warm = np.asarray((1.0, 0.86, 0.66), F32)
        buf += halo[:, :, None] * warm[None, None, :]

    # ----------------------------------------------------------------
    def render(self, ctx, lt):
        sim = self.sim
        sim.advance_to(lt * self.YEARS_PER_SECOND)
        cam = self.camera_at(ctx, lt)
        exposure = self.exposure_at(ctx, lt)

        self.surf.reset()
        self.surf.rng = np.random.default_rng(ctx.frame * 2693 + 11)
        sun_i = sim.index["Sun"]
        sun_pos = sim.pos[sun_i]
        t_years = sim.time

        # Shadow casters: the Sun is the light, so it never casts.  Each
        # body is tested against the others in its own gravitational
        # neighbourhood, which is all that can physically eclipse it.
        order = np.argsort(-np.array([
            np.linalg.norm(sim.pos[i] - cam.eye.astype(np.float64))
            for i in range(sim.n)]))

        points = []
        point_cols = []
        for i in order:
            body = sim.bodies[i]
            if body.name == "Sun":
                continue
            # A body smaller than about a pixel cannot be traced sensibly:
            # the rectangle would be one ray wide and the sphere would
            # flicker in and out between frames.  Below that threshold it
            # becomes a photometric point instead.
            rpx = self.surf.angular_radius_px(cam, sim.pos[i],
                                              body.draw_radius)
            if rpx < 1.1:
                fl = self.surf.point_flux(body, sim.pos[i], sun_pos,
                                          np.asarray(cam.eye, np.float64))
                if fl is not None:
                    points.append(sim.pos[i])
                    point_cols.append(fl)
                continue
            casters = []
            for j in range(sim.n):
                if j == i or sim.bodies[j].name == "Sun":
                    continue
                bj = sim.bodies[j]
                if bj.parent == body.name or body.parent == bj.name or \
                        (bj.parent and bj.parent == body.parent):
                    casters.append((sim.pos[j], bj.draw_radius))
            self.surf.draw_body(cam, body, sim.pos[i], sun_pos, t_years,
                                casters, exposure=exposure)
            if body.ring_outer > 0.0:
                self.surf.draw_rings(cam, body, sim.pos[i], sun_pos,
                                     t_years, [(sim.pos[i],
                                                body.draw_radius)],
                                     exposure=exposure)

        sun_rpx = self.surf.angular_radius_px(
            cam, sun_pos, sim.bodies[sun_i].draw_radius)
        if sun_rpx >= 1.1:
            self.surf.draw_body(cam, sim.bodies[sun_i], sun_pos, sun_pos,
                                t_years, [], exposure=exposure)
        else:
            fl = self.surf.point_flux(sim.bodies[sun_i], sun_pos, sun_pos,
                                      np.asarray(cam.eye, np.float64))
            if fl is not None:
                points.append(sun_pos)
                point_cols.append(fl)

        img = self.surf.image().copy()

        if points:
            pp = np.asarray(points, F32)
            pc = np.asarray(point_cols, F32)
            # The Sun outshines Neptune by eleven orders of magnitude at
            # this range.  Reproduced literally the frame holds one white
            # dot and nothing else, so point luminance is compressed by a
            # fractional power — the same response a long photographic
            # exposure has, and the same compression already applied to
            # the galaxy's stars in Phase I.  Hue and relative ordering
            # survive untouched; only the ratio of extremes is squeezed.
            lum = np.maximum(S.luminance(pc), F32(1e-30))
            disp = np.power(lum / F32(self.POINT_REF), F32(0.28))
            pc = pc * (disp / lum)[:, None] * F32(1.35)
            pb, _ = project_and_splat(cam, pp, pc, ctx.width, ctx.height,
                                      falloff=False, size_scale=3.0)
            img += pb.reshape(ctx.height, ctx.width, 3)

        bgpos, bgrgb = self._bg
        bb, _ = project_and_splat(cam, bgpos + cam.eye.astype(F32),
                                  bgrgb * F32(0.40 + 0.020 * exposure),
                                  ctx.width, ctx.height, falloff=False)
        img += bb.reshape(ctx.height, ctx.width, 3)

        self._draw_trails(ctx, cam, img, exposure)
        self._draw_sun_glow(ctx, cam, img, exposure)
        return img


__all__ = ["Body", "SOLAR_SYSTEM", "SCALE_NOTE", "solve_kepler",
           "elements_to_state", "state_to_elements", "sample_orbit",
           "NBody", "SurfaceRenderer", "PhaseSolarSystem", "G_AU",
           "KM_PER_AU"]
