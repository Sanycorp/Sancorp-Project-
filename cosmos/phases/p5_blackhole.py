"""
cosmos.phases.p5_blackhole
==========================

FASE V — LUBANG HITAM.

The relativistic finale.  Light does not travel in straight lines here, so
nothing in this phase is rasterised: every pixel is a null geodesic of the
Schwarzschild metric, integrated outward from the camera until it either
falls through the horizon, escapes to infinity, or passes through the
accretion disc on the way.

The geodesic equation
---------------------
In Schwarzschild spacetime every geodesic stays in a plane through the
centre, which reduces the problem from three dimensions to two.  Writing
``u = 1/r`` and using the azimuth ``phi`` as the independent variable, the
null geodesic obeys

    d2u/dphi2 + u = 3 M u^2

The left-hand side alone is Newtonian: it integrates to a conic section,
the orbit of a free particle under an inverse-square force.  The entire
difference between Newton and Einstein, for this problem, is the single
term ``3 M u^2`` on the right.  It is negligible when ``r >> M`` — which is
why the solar system looked Keplerian in Phase II — and it dominates near
the horizon, where it produces light bending strong enough to show the far
side of the accretion disc arching over the top of the near side, an
unstable circular photon orbit at ``r = 3M``, and a black shadow whose
angular radius is ``sqrt(27) M``, appreciably larger than the horizon
itself.

Solving it by brute force for every ray is wasteful, so rays are triaged
by impact parameter ``b = |r x v|``:

* ``b < b_crit = 3 sqrt(3) M``  — captured.  No integration: the ray cannot
  escape, and the pixel is black.  This is the shadow.
* ``b > b_weak``                — the deflection is below a quarter of a
  degree and the first-order Einstein result ``alpha = 4 M / b`` is
  indistinguishable from the exact answer.  One rotation, no integration.
* otherwise                     — integrated with RK4 in ``phi``, with the
  batch compacted as rays finish.

That triage typically removes three quarters of the frame from the
integrator while leaving the strongly lensed annulus — the part anyone will
actually look at — fully exact.

The accretion disc
------------------
A geometrically thin, optically thick disc between the innermost stable
circular orbit at ``r = 6M`` and ``r = 26M``.  Three relativistic effects
are applied to its emission, all of them composed from verified code in
``cosmos.core.spectrum``:

**Gravitational redshift.**  Light climbing out of the well loses energy by
``sqrt(1 - 2M/r)``.  At the ISCO that is a factor of 0.816.

**Doppler shift and beaming.**  The gas orbits at ``v = sqrt(M/(r - 2M))``
as measured locally, which is exactly ``c/2`` at the ISCO.  The side coming
toward the observer is blueshifted and brightened; the receding side is
reddened and dimmed.  Specific intensity scales as the fourth power of the
combined shift factor, so the asymmetry is dramatic — a factor of tens, not
percent.  This is the single most recognisable feature of a real image of
an accreting black hole, and it is a prediction, not a decoration.

**Temperature profile.**  Shakura-Sunyaev: ``T ~ r^-3/4 (1 - sqrt(r_isco/r))^1/4``,
which vanishes at the inner edge because there is no torque there, peaks a
little outside it, and falls away outward.  The colour at every radius is
then the blackbody of the shifted temperature.

Units
-----
Schwarzschild radii: ``r_s = 2GM/c^2 = 1``, hence ``M = 1/2``.  The horizon
is at ``r = 1``, the photon sphere at ``r = 1.5``, the shadow radius is
``b_crit = 3 sqrt(3) / 2 = 2.598``, and the ISCO is at ``r = 3``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core import camera as CAM
from ..core import spectrum as S
from ..core import vmath as V
from ..core import vcache
from .base import (Phase, build_emission_ramp, project_and_splat,
                   ramp_lookup, upsample_linear)

F32 = np.float32
TAU = 2.0 * math.pi

#: geometric mass in units where the Schwarzschild radius is 1
M_GEO = 0.5
#: critical impact parameter: the radius of the black hole's shadow
B_CRIT = 3.0 * math.sqrt(3.0) * M_GEO          # 2.598...
#: innermost stable circular orbit
R_ISCO = 6.0 * M_GEO                            # 3.0
#: unstable circular photon orbit
R_PHOTON = 3.0 * M_GEO                          # 1.5


# ==========================================================================
# parameters
# ==========================================================================
@dataclass
class BlackHoleParams:
    """Geometry and appearance of the system."""

    r_horizon: float = 1.0
    r_disc_in: float = R_ISCO
    r_disc_out: float = 19.0
    disc_h0: float = 0.055          # scale height at the inner edge
    disc_flare: float = 1.125       # h ~ r^(9/8), the thin-disc result

    #: Peak effective temperature of the disc, kelvin.  Fixes the mass and
    #: accretion rate implicitly: a stellar-mass hole at the Eddington rate
    #: peaks in soft X-rays and would be invisible, a supermassive one in
    #: the ultraviolet.  The value below corresponds to an intermediate
    #: hole and puts the peak in the visible, which is the only reason any
    #: of this is renderable.
    t_peak: float = 7400.0
    t_floor: float = 1550.0

    #: angular velocity scaling for the visible turbulence of the disc
    spin_rate: float = 1.0

    b_weak: float = 16.0            # above this, use the first-order result
    r_escape: float = 400.0         # treat as infinity
    max_steps: int = 340
    d_phi: float = 0.0315           # integration step in azimuth

    jet_strength: float = 0.55
    jet_opening: float = 0.30


# ==========================================================================
# analytic relations
# ==========================================================================
def deflection_weak(b):
    """First-order Einstein deflection ``alpha = 4 M / b``.

    Exact to better than one per cent for ``b > 20 M``; this is the result
    Eddington measured at the 1919 eclipse and it is twice the value
    Newtonian gravity predicts for a corpuscle.
    """
    return 4.0 * M_GEO / np.maximum(b, 1e-9)


def keplerian_speed(r):
    """Locally measured orbital speed of a circular geodesic.

    ``v = sqrt(M / (r - 2M))`` as seen by a static observer at that radius.
    It reaches exactly ``c/2`` at the innermost stable circular orbit, and
    ``c`` at the photon sphere — which is the statement that no massive
    particle can orbit there.
    """
    return np.sqrt(M_GEO / np.maximum(r - 2.0 * M_GEO, 1e-6))


def disc_temperature(r, p: BlackHoleParams):
    """Shakura-Sunyaev effective temperature profile.

    ``T^4 ~ r^-3 (1 - sqrt(r_in / r))`` — the inner boundary condition is
    zero torque, so the emission vanishes at the ISCO, rises to a peak a
    little outside it, and then falls as ``r^-3/4``.
    """
    r = np.maximum(np.asarray(r, F32), F32(p.r_disc_in))
    x = np.maximum(F32(1.0) - np.sqrt(F32(p.r_disc_in) / r), F32(0.0))
    prof = np.power(F32(p.r_disc_in) / r, F32(0.75)) * np.power(x, F32(0.25))
    # normalise so the maximum of the profile equals t_peak
    rr = np.linspace(p.r_disc_in * 1.0001, p.r_disc_out, 512, dtype=np.float32)
    xx = np.maximum(1.0 - np.sqrt(p.r_disc_in / rr), 0.0)
    pp = (p.r_disc_in / rr) ** 0.75 * xx ** 0.25
    peak = float(pp.max()) or 1.0
    return F32(p.t_floor) + (F32(p.t_peak) - F32(p.t_floor)) * (prof / peak)


# ==========================================================================
# the geodesic integrator
# ==========================================================================
class GeodesicTracer:
    """Integrates null geodesics and accumulates disc emission along them.

    The state is ``(u, u')`` as a function of ``phi``, carried per ray in
    the ray's own orbital plane.  Positions in three dimensions are
    recovered as ``r (cos phi e1 + sin phi e2)`` whenever they are needed —
    for the disc test, for the volume lookups, and for the final escape
    direction.
    """

    def __init__(self, p: BlackHoleParams, disc_shader):
        self.p = p
        self.disc = disc_shader

    # ----------------------------------------------------------------
    @staticmethod
    def _rhs(u, up):
        """Right-hand side of ``u'' = 3 M u^2 - u``."""
        return 3.0 * M_GEO * u * u - u

    def _rk4(self, u, up, h):
        """One classical fourth-order Runge-Kutta step in ``phi``.

        The equation is a second-order ODE with no first-derivative term,
        so a symplectic scheme would also work, but RK4 is used here
        because these are open trajectories traced over at most a couple of
        windings — there is no long-term energy drift to worry about, and
        the fourth-order accuracy keeps the photon ring sharp.
        """
        k1u = up
        k1v = self._rhs(u, up)
        k2u = up + 0.5 * h * k1v
        k2v = self._rhs(u + 0.5 * h * k1u, k2u)
        k3u = up + 0.5 * h * k2v
        k3v = self._rhs(u + 0.5 * h * k2u, k3u)
        k4u = up + h * k3v
        k4v = self._rhs(u + h * k3u, k4u)
        u_new = u + (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        up_new = up + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return u_new, up_new

    # ----------------------------------------------------------------
    def trace(self, origin, direction, rng, steps=None):
        """Trace a batch of rays.

        Returns ``(colour, escaped_direction, escaped_mask)``.  ``colour``
        holds everything the ray picked up on the way (disc, jet, infalling
        gas); the caller adds the lensed background for rays that escaped.
        """
        p = self.p
        n = origin.shape[0]
        colour = np.zeros((n, 3), F32)
        out_dir = np.array(direction, F32, copy=True)
        escaped = np.zeros(n, bool)

        r0 = np.sqrt(np.einsum('ij,ij->i', origin, origin))
        e1 = origin / np.maximum(r0, 1e-9)[:, None]
        d_rad = np.einsum('ij,ij->i', direction, e1)
        perp = direction - d_rad[:, None] * e1
        d_tan = np.sqrt(np.einsum('ij,ij->i', perp, perp))

        # impact parameter of the straight-line ray
        b = r0 * d_tan

        # --- triage ----------------------------------------------------
        radial = d_tan < 1e-5
        captured = (b < F32(B_CRIT)) & (d_rad < 0.0) & (~radial)
        captured |= radial & (d_rad < 0.0)
        weak = (b > F32(p.b_weak)) & (~captured)
        strong = (~captured) & (~weak) & (~radial)
        # outgoing radial rays simply leave
        escaped |= radial & (d_rad >= 0.0)

        e2 = np.zeros_like(e1)
        ok = d_tan > 1e-9
        e2[ok] = perp[ok] / d_tan[ok][:, None]

        # --- weak field: one analytic rotation --------------------------
        if weak.any():
            idx = np.flatnonzero(weak)
            alpha = deflection_weak(b[idx]).astype(F32)
            # the ray bends toward the hole, i.e. away from e2
            new_dir = (direction[idx] * np.cos(alpha)[:, None]
                       - e1[idx] * np.sin(alpha)[:, None])
            # (-e1 is the direction of the deflection for an outbound ray
            #  passing the mass on the e2 side)
            nn = np.maximum(np.sqrt(np.einsum('ij,ij->i', new_dir, new_dir)),
                            1e-9)
            out_dir[idx] = new_dir / nn[:, None]
            escaped[idx] = True
            # the disc can still be crossed far from the hole, so test the
            # straight segment analytically
            c = self._disc_analytic(origin[idx], out_dir[idx], rng)
            colour[idx] += c

        # --- strong field: integrate -------------------------------------
        if strong.any():
            idx = np.flatnonzero(strong)
            col, dirs, esc = self._integrate(origin[idx], e1[idx], e2[idx],
                                             r0[idx], d_rad[idx], d_tan[idx],
                                             rng,
                                             steps or p.max_steps)
            colour[idx] += col
            out_dir[idx] = dirs
            escaped[idx] = esc

        return colour, out_dir, escaped

    # ----------------------------------------------------------------
    def _integrate(self, origin, e1, e2, r0, d_rad, d_tan, rng, steps):
        p = self.p
        n = origin.shape[0]
        colour = np.zeros((n, 3), F32)
        out_dir = np.zeros((n, 3), F32)
        escaped = np.zeros(n, bool)

        u = (1.0 / np.maximum(r0, 1e-9)).astype(np.float64)
        # du/dphi = -(d.e1) / ((d.e2) r)
        up = (-d_rad / np.maximum(d_tan, 1e-12) * u).astype(np.float64)
        phi = np.zeros(n, np.float64)
        slot = np.arange(n)
        e1w = e1.astype(np.float64)
        e2w = e2.astype(np.float64)

        h = p.d_phi
        inv_horizon = 1.0 / p.r_horizon
        inv_escape = 1.0 / p.r_escape

        prev_pos = origin.astype(np.float64)
        jitter = vcache.new_jitter(rng)

        for step in range(steps):
            u, up = self._rk4(u, up, h)
            phi += h
            np.maximum(u, 1e-9, out=u)
            r = 1.0 / u
            cp = np.cos(phi)
            sp = np.sin(phi)
            pos = r[:, None] * (cp[:, None] * e1w + sp[:, None] * e2w)

            # ---- accumulate anything the segment passed through -------
            seg = pos - prev_pos
            ds = np.sqrt(np.einsum('ij,ij->i', seg, seg))
            mid = (pos + prev_pos) * 0.5
            contrib = self.disc.sample_segment(mid.astype(F32),
                                               ds.astype(F32),
                                               seg.astype(F32), jitter)
            colour[slot] += contrib
            prev_pos = pos

            # ---- termination -------------------------------------------
            fell = u > inv_horizon
            gone = (u < inv_escape) & (up < 0.0)
            done = fell | gone
            if done.any():
                di = np.flatnonzero(done)
                gi = slot[di]
                escaped[gi] = gone[di]
                # escape direction: d/dphi of the position, normalised
                dr = -up[di] / (u[di] * u[di])
                tangent = (dr[:, None] * (cp[di, None] * e1w[di]
                                          + sp[di, None] * e2w[di])
                           + r[di, None] * (-sp[di, None] * e1w[di]
                                            + cp[di, None] * e2w[di]))
                tn = np.maximum(np.sqrt(np.einsum('ij,ij->i', tangent,
                                                  tangent)), 1e-12)
                out_dir[gi] = (tangent / tn[:, None]).astype(F32)

                keep = ~done
                if not keep.any():
                    break
                u = u[keep]; up = up[keep]; phi = phi[keep]
                e1w = e1w[keep]; e2w = e2w[keep]
                prev_pos = prev_pos[keep]
                slot = slot[keep]
                jitter = vcache.new_jitter(rng)

        # anything still integrating after the step cap is treated as
        # captured: it has wound around the photon sphere many times and
        # is on its way in
        return colour, out_dir, escaped

    # ----------------------------------------------------------------
    def _disc_analytic(self, origin, direction, rng):
        """Disc crossing for weakly deflected rays, solved in closed form."""
        p = self.p
        oy = origin[:, 1]
        dy = direction[:, 1]
        safe = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
        t = -oy / safe
        hit = t > 0.0
        pos = origin + direction * t[:, None]
        rr = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
        hit &= (rr >= F32(p.r_disc_in)) & (rr <= F32(p.r_disc_out))
        out = np.zeros_like(origin)
        if not hit.any():
            return out
        i = np.flatnonzero(hit)
        # path length through a flared disc of local half-thickness h
        hh = self.disc.half_thickness(rr[i])
        ds = (2.0 * hh / np.maximum(np.abs(direction[i, 1]), 1e-3)
              ).astype(F32)
        jitter = vcache.new_jitter(rng)
        out[i] = self.disc.sample_segment(pos[i].astype(F32), ds,
                                          direction[i].astype(F32), jitter)
        return out


# ==========================================================================
# the accretion disc
# ==========================================================================
class DiscShader:
    """Emission of the accretion disc, jet and plunging region."""

    def __init__(self, p: BlackHoleParams, lib, seed=23):
        self.p = p
        self.v_disc = lib.get("disc")
        self.v_fine = lib.get("disc_fine")
        self.v_plasma = lib.get("plasma")
        self.rng = np.random.default_rng(seed)
        self._t = 0.0

        # Precompute the radial temperature table so the hot inner loop
        # does a gather instead of three powers and a square root.
        n = 1024
        self._r_tab = np.linspace(p.r_disc_in, p.r_disc_out, n).astype(F32)
        self._t_tab = disc_temperature(self._r_tab, p)
        self._rgb_tab = S.blackbody_lut(self._t_tab)
        self._rgb_tab = S.saturate_rgb(self._rgb_tab, 1.40)
        # Stefan-Boltzmann weighting: the inner disc is not merely bluer,
        # it is overwhelmingly brighter, as T^4.
        rel = (self._t_tab / F32(p.t_peak))
        bright = rel * rel
        bright = bright * bright
        self._rgb_tab = (self._rgb_tab * bright[:, None]).astype(F32)
        self._inv_dr = F32((n - 1) / (p.r_disc_out - p.r_disc_in))

        # Jet: synchrotron, not thermal.  A power-law electron population
        # radiating in a magnetic field gives a flat, slightly blue
        # continuum, so the jet is tinted rather than taken from Planck.
        self.jet_rgb = np.array([0.52, 0.70, 1.0], F32)

    def set_time(self, t):
        self._t = float(t)

    # ----------------------------------------------------------------
    def half_thickness(self, r):
        p = self.p
        return (F32(p.disc_h0)
                * np.power(np.maximum(r, F32(1e-3)) / F32(p.r_disc_in),
                           F32(p.disc_flare)))

    # ----------------------------------------------------------------
    def sample_segment(self, pos, ds, seg_dir, jitter):
        """Radiance picked up over one integration segment.

        ``pos`` is the segment midpoint, ``ds`` its proper length, and
        ``seg_dir`` the direction of travel, which is needed because every
        relativistic correction depends on the angle between the photon and
        the orbiting gas.
        """
        p = self.p
        out = np.zeros((pos.shape[0], 3), F32)

        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        rho2 = x * x + z * z
        rho = np.sqrt(np.maximum(rho2, F32(1e-9)))
        r = np.sqrt(np.maximum(rho2 + y * y, F32(1e-9)))

        # ---- the disc ---------------------------------------------------
        inside = (rho >= F32(p.r_disc_in)) & (rho <= F32(p.r_disc_out))
        if inside.any():
            i = np.flatnonzero(inside)
            rr = rho[i]
            hh = self.half_thickness(rr)
            vert = np.exp(-(y[i] / hh) ** 2 * F32(2.2))
            live = vert > 1e-3
            if live.any():
                i = i[live]
                rr = rr[live]
                vert = vert[live]
                hh = hh[live]

                # --- turbulent structure, orbiting with the gas ---------
                # Differential rotation: the inner disc laps the outer, so
                # the texture must be sampled in a frame that shears.  This
                # is what makes the disc look like it is actually orbiting
                # rather than spinning as a rigid plate.
                omega = np.sqrt(F32(M_GEO) / (rr * rr * rr))
                ang = np.arctan2(z[i], x[i]) - omega * F32(
                    self._t * 6.0 * p.spin_rate)
                ca, sa = np.cos(ang), np.sin(ang)
                q = np.stack([rr * ca * F32(0.19), y[i] * F32(1.9),
                              rr * sa * F32(0.19)], axis=1).astype(F32)
                n1 = self.v_disc.sample(q, jitter=jitter)
                n2 = self.v_fine.sample(q * F32(3.7), jitter=jitter)
                tex = n1 * F32(0.70) + np.clip(n2 * F32(2.4) + F32(0.5),
                                               0.0, 1.0) * F32(0.30)
                tex = np.clip(tex * F32(1.55) - F32(0.22), 0.0, 1.8)

                # --- colour and brightness from the radial table --------
                ti = np.clip((rr - F32(p.r_disc_in)) * self._inv_dr, 0,
                             self._rgb_tab.shape[0] - 1).astype(np.int32)
                base = self._rgb_tab[ti]

                # --- relativistic corrections ----------------------------
                # Orbital velocity direction: prograde, in the disc plane.
                vmag = keplerian_speed(rr).astype(F32)
                # unit tangent of the prograde circular orbit here
                tx = -z[i] / rr
                tz = x[i] / rr
                # cos of the angle between the gas velocity and the photon
                # The geodesic is integrated *outward* from the camera,
                # but the photon physically travelled the other way, from
                # the gas to the eye.  The Doppler angle must therefore use
                # the reversed direction — getting this sign wrong still
                # produces a convincing asymmetry, just with the wrong side
                # of the disc lit, which is the sort of error that survives
                # review precisely because it looks right.
                cos_th = -(tx * seg_dir[i, 0] + tz * seg_dir[i, 2])
                dop = S.doppler_factor(vmag, cos_th)
                grav = S.gravitational_redshift(rr, F32(1.0))
                g = np.clip(dop * grav, F32(0.04), F32(3.0))

                # I_nu / nu^3 is a Lorentz invariant, so the bolometric
                # intensity scales as g^4.  This is the term that makes one
                # side of the disc blindingly bright and the other almost
                # vanish.
                g2 = g * g
                boost = g2 * g2

                # the shift also moves the blackbody peak: colour follows
                shifted = S.blackbody_lut(np.clip(
                    self._t_tab[ti] * g, F32(600.0), F32(60000.0)))
                shifted = S.saturate_rgb(shifted, 1.40)
                lum = np.maximum(S.luminance(base), F32(1e-6))
                col = shifted * lum[:, None]

                amp = (vert * tex * boost * ds[i] * F32(1.85))
                out[i] = col * amp[:, None]

        # ---- plunging region, inside the ISCO ---------------------------
        plunge = (rho < F32(p.r_disc_in)) & (r > F32(p.r_horizon * 1.02))
        if plunge.any():
            i = np.flatnonzero(plunge)
            hh = self.half_thickness(np.full(i.shape[0], F32(p.r_disc_in)))
            vert = np.exp(-(y[i] / (hh * F32(1.8))) ** 2 * F32(1.6))
            fade = np.clip((rho[i] - F32(p.r_horizon)) /
                           F32(p.r_disc_in - p.r_horizon), 0.0, 1.0)
            grav = S.gravitational_redshift(np.maximum(r[i], F32(1.001)),
                                            F32(1.0))
            hot = S.blackbody_lut(np.clip(F32(p.t_peak) * F32(1.35)
                                          * grav, 600.0, 60000.0))
            amp = vert * fade * fade * grav ** 4 * ds[i] * F32(1.4)
            out[i] += hot * amp[:, None]

        # ---- relativistic jet --------------------------------------------
        if p.jet_strength > 0.0:
            ay = np.abs(y)
            cone = np.clip(F32(1.0) - rho / np.maximum(
                ay * F32(p.jet_opening) + F32(0.10), F32(1e-3)), 0.0, 1.0)
            live = (cone > 0.0) & (ay > F32(1.2)) & (r < F32(90.0))
            if live.any():
                i = np.flatnonzero(live)
                q = np.stack([x[i] * F32(0.5), y[i] * F32(0.085),
                              z[i] * F32(0.5)], axis=1).astype(F32)
                n = self.v_plasma.sample(q, jitter=jitter)
                tex = np.clip(n * F32(2.6) + F32(0.5), 0.0, 1.0)
                # the jet is collimated and fades with height
                fall = np.exp(-ay[i] * F32(0.052))
                amp = (cone[i] ** 2 * tex * fall * ds[i]
                       * F32(0.085 * p.jet_strength))
                out[i] += self.jet_rgb[None, :] * amp[:, None]

        return out


# ==========================================================================
# the phase
# ==========================================================================
class PhaseBlackHole(Phase):
    """FASE V — the black hole: lensing, accretion disc, event horizon."""

    name = "V · LUBANG HITAM"
    duration = 12.0

    def __init__(self, params=None, seed=23):
        self.p = params or BlackHoleParams()
        self.seed = int(seed)
        self.disc = None
        self.tracer = None
        self._sky = None

    # ----------------------------------------------------------------
    def setup(self, ctx):
        self.disc = DiscShader(self.p, ctx.lib, seed=self.seed)
        self.tracer = GeodesicTracer(self.p, self.disc)
        self._build_sky()

    def _build_sky(self):
        """Background star field, sampled by escaped geodesics.

        Because the rays are bent, this field is what gets lensed: the
        Einstein ring and the secondary images of the sky around the
        shadow are produced by looking it up in the *final* direction of
        each geodesic, not the initial one.
        """
        rng = np.random.default_rng(self.seed + 77)
        n = 78000
        dirs = V.fibonacci_sphere(n).astype(F32)
        dirs = dirs[rng.permutation(n)]
        t = S.sample_stellar_temperatures(n, rng, hot_bias=0.52).astype(F32)
        rgb = S.saturate_rgb(S.blackbody_lut(t), 1.5)
        rgb /= np.maximum(S.luminance(rgb), 1e-5)[:, None]
        b = (rng.random(n).astype(F32) ** 6.0) * F32(0.58) + F32(0.0025)
        self._sky = (dirs, (rgb * b[:, None]).astype(F32))
        # A direction-indexed equirectangular table, so an escaped geodesic
        # is coloured with one gather instead of a nearest-neighbour search.
        # It has to be fine enough that a star stays a star: at 2048 by
        # 1024 a texel is about a tenth of a degree, which is roughly the
        # angular size of a pixel at this field of view.
        self._sky_res = 2048
        self._sky_tab = self._bake_sky_table(self._sky_res, rng)

    def _bake_sky_table(self, res, rng=None):
        """Equirectangular table of the background sky.

        Splatting the catalogue into a table once is far cheaper than
        querying it per pixel.  A diffuse galactic band is added underneath
        the stars, and it earns its place: isolated point stars make the
        lensing hard to read, whereas a continuous band is visibly sheared
        into arcs as it passes behind the hole, which is the whole point of
        putting a background there at all.
        """
        dirs, rgb = self._sky
        h = res // 2
        theta = np.arccos(np.clip(dirs[:, 1], -1.0, 1.0))
        phi = np.arctan2(dirs[:, 2], dirs[:, 0])
        iy = np.clip((theta / math.pi * h).astype(np.int32), 0, h - 1)
        ix = np.clip(((phi / TAU + 0.5) * res).astype(np.int32), 0, res - 1)
        flat = iy * res + ix
        tab = np.zeros((h * res, 3), np.float64)
        for k in range(3):
            tab[:, k] = np.bincount(flat, weights=rgb[:, k].astype(np.float64),
                                    minlength=h * res)
        tab = tab.reshape(h, res, 3).astype(F32)

        # --- diffuse galactic band ------------------------------------
        yy = (np.arange(h, dtype=np.float32) + 0.5) / h * math.pi
        xx = (np.arange(res, dtype=np.float32) + 0.5) / res * TAU - math.pi
        lat = yy[:, None] - math.pi * 0.5
        # a great circle tilted out of the equator, so it crosses the disc
        # plane at an angle and the shear is unmistakable
        tilt = math.radians(32.0)
        band_lat = np.arcsin(np.clip(
            np.sin(lat) * math.cos(tilt)
            - np.cos(lat) * np.sin(xx[None, :]) * math.sin(tilt), -1, 1))
        prof = np.exp(-(band_lat / 0.19) ** 2) * 0.55 \
            + np.exp(-(band_lat / 0.52) ** 2) * 0.22
        # clumpy, not a clean stripe
        if rng is None:
            rng = np.random.default_rng(1)
        lumps = rng.random((h // 16, res // 16)).astype(np.float32)
        lumps = np.kron(lumps, np.ones((16, 16), np.float32))[:h, :res]
        for _ in range(3):
            lumps = (lumps * 0.4
                     + np.roll(lumps, 3, 0) * 0.15
                     + np.roll(lumps, -3, 0) * 0.15
                     + np.roll(lumps, 3, 1) * 0.15
                     + np.roll(lumps, -3, 1) * 0.15)
        prof = prof * (0.45 + 1.1 * lumps)
        band_rgb = np.array([0.62, 0.66, 0.80], np.float32)
        tab += (prof[:, :, None] * band_rgb[None, None, :]
                * np.float32(0.034)).astype(F32)

        # a light blur so the table does not alias into hard squares
        tab = (tab * 0.64
               + np.roll(tab, 1, axis=1) * 0.09
               + np.roll(tab, -1, axis=1) * 0.09
               + np.roll(tab, 1, axis=0) * 0.09
               + np.roll(tab, -1, axis=0) * 0.09).astype(F32)
        return tab

    def sample_sky(self, dirs):
        """Bilinear lookup into the background table.

        Bilinear and not nearest: gravitational lensing *magnifies*, and
        near the Einstein ring it magnifies enormously, so a single texel
        of the sky can be stretched across dozens of pixels.  With nearest
        sampling that turns the lensed star field into a mosaic of hard
        coloured squares — visible, and the single most conspicuous
        artefact in the most important shot of the film.
        """
        tab = self._sky_tab
        h, w, _ = tab.shape
        theta = np.arccos(np.clip(dirs[:, 1], -1.0, 1.0))
        phi = np.arctan2(dirs[:, 2], dirs[:, 0])
        fy = np.clip(theta / math.pi * h - 0.5, 0.0, h - 1.001)
        fx = (phi / TAU + 0.5) * w - 0.5
        fx = np.mod(fx, w)
        y0 = fy.astype(np.int32)
        x0 = fx.astype(np.int32)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = (x0 + 1) % w            # the sky wraps in longitude
        wy = (fy - y0).astype(F32)[:, None]
        wx = (fx - x0).astype(F32)[:, None]
        return (tab[y0, x0] * (1 - wy) * (1 - wx)
                + tab[y0, x1] * (1 - wy) * wx
                + tab[y1, x0] * wy * (1 - wx)
                + tab[y1, x1] * wy * wx)

    # ----------------------------------------------------------------
    def camera_at(self, ctx, lt):
        """Approach, descend into the disc plane, and orbit.

        The descent matters: seen from far above, the disc is an ordinary
        ring.  The famous image — the far side of the disc lifted up and
        wrapped over the top of the shadow — only appears when the line of
        sight is within a few degrees of the plane, because that is when
        the light from behind the hole has to bend through nearly a right
        angle to reach the camera.
        """
        u = self.progress(lt, self.duration)
        d = self.duration

        # distance: a long logarithmic approach from well outside
        g = V.smootherstep(0.0, 0.82, u)
        dist = math.exp(math.log(190.0) * (1.0 - g) + math.log(15.5) * g)

        # elevation above the disc plane, in degrees
        elev = 48.0 * (1.0 - V.smootherstep(0.06, 0.74, u)) + 3.2
        elev += 2.1 * math.sin(lt * 0.44)

        azim = 0.62 + 0.40 * u + 0.055 * math.sin(lt * 0.31)

        cam = CAM.orbit(np.zeros(3, F32), dist, azim, math.radians(elev))
        cam.fov_deg = float(34.0 + 22.0 * V.smootherstep(0.35, 1.0, u))
        cam.roll_deg = float(-2.4 + 5.0 * V.smootherstep(0.2, 1.0, u))
        return CAM.apply_handheld(cam, lt, pos_amp=0.012, rot_amp=0.14,
                                  fov_breath=0.08, seed=97.0)

    def exposure_at(self, ctx, lt):
        u = self.progress(lt, self.duration)
        # the disc fills more of the frame as the camera closes in
        return float(1.25 * (1.0 - 0.58 * V.smootherstep(0.25, 1.0, u)))

    # ----------------------------------------------------------------
    def render(self, ctx, lt):
        p = self.p
        self.disc.set_time(lt)
        cam = self.camera_at(ctx, lt)
        exposure = self.exposure_at(ctx, lt)
        rng = ctx.rng or np.random.default_rng(ctx.frame * 7919 + 59)

        mw, mh = ctx.march_width, ctx.march_height
        o, d, idx = cam.generate_rays(mw, mh, jitter_phase=ctx.frame & 7)

        colour, out_dir, escaped = self.tracer.trace(o, d, rng)

        if escaped.any():
            i = np.flatnonzero(escaped)
            colour[i] += self.sample_sky(out_dir[i])

        img = np.zeros((mh * mw, 3), F32)
        img[idx] = colour
        img = img.reshape(mh, mw, 3)
        return upsample_linear(img, ctx.width, ctx.height) * F32(exposure)


__all__ = ["BlackHoleParams", "GeodesicTracer", "DiscShader",
           "PhaseBlackHole", "deflection_weak", "keplerian_speed",
           "disc_temperature", "M_GEO", "B_CRIT", "R_ISCO", "R_PHOTON"]
