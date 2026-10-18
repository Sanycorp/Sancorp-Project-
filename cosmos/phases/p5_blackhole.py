"""
cosmos.phases.p5_blackhole
==========================

FASE V — LUBANG HITAM.

The relativistic finale, and the only phase in which nothing is
rasterised: every pixel is a null geodesic of the **Kerr** metric,
integrated backwards from the camera until it falls through the horizon,
escapes to infinity, or crosses the accretion disc on the way.

Why Kerr and not Schwarzschild
------------------------------
Because every black hole that exists rotates.  Accretion delivers angular
momentum along with mass, and the spins that have actually been measured
sit near the top of the allowed range — Cygnus X-1 above ``0.95``,
GRS 1915+105 above ``0.98``, M87* near ``0.9``.  A static hole is a
textbook idealisation.

Choosing ``a = 0.94`` changes the picture completely rather than
decoratively:

====================  ================  =================
quantity              Schwarzschild     this film (a=0.94)
====================  ================  =================
horizon               2.00 M            1.34 M
innermost orbit       6.00 M            2.02 M
orbital speed there   0.500 c           0.625 c
photon orbit          3.00 M            1.43 / 3.95 M
shadow                circle, 5.20 M    D-shape, -2.6..6.9
====================  ================  =================

The disc reaches three times deeper, so it is hotter and faster; the
shadow is visibly lopsided; and frame dragging winds the lensed sky into
a vortex instead of a symmetric ring.

What is drawn
-------------
**The disc.**  Geometrically thin, optically thick, between the prograde
ISCO and ``r_out``.  Its radiated flux is the Novikov-Thorne profile —
the exact relativistic generalisation of Shakura-Sunyaev, worked out by
Page and Thorne in 1974 — not an ``r^-3/4`` approximation.

**The shift.**  One exact expression, ``g = nu_obs / nu_em``, carries the
gravitational redshift, both Doppler effects and the frame-dragging
contribution together.  Bolometric intensity goes as ``g^4``, so the
approaching limb outshines the receding one by two orders of magnitude.
This is the single most recognisable feature of a real image of an
accreting hole, and it is a prediction, not a decoration.

**Higher-order images.**  A ray that passes over the hole and bends back
down strikes the far side of the disc from underneath; one that threads
closer still wraps around and strikes it again.  Nothing special is done
to produce them — the tracer simply reports every equatorial crossing
with a running transmittance, and the arcs appear.

**The plunging region.**  Inside the ISCO no circular orbit is stable, so
the gas spirals in, its angular velocity tending to the horizon's own.
It is dim, strongly redshifted, and partly transparent.

**The jet.**  A collimated, helically structured outflow along the spin
axis, emitting synchrotron rather than thermal radiation — so it is
tinted, not taken from Planck's law — and lensed along with everything
else, because it is accumulated inside the geodesic integration.

Units
-----
Geometric, with ``M = 1``.  Lengths are gravitational radii.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import camera as CAM
from ..core import kerr as KR
from ..core import spectrum as S
from ..core import vmath as V
from ..core import vcache
from .base import Phase, upsample_linear

F32 = np.float32
TAU = 2.0 * math.pi


# ==========================================================================
# parameters
# ==========================================================================
@dataclass
class BlackHoleParams:
    """Geometry, thermodynamics and appearance of the system."""

    #: Dimensionless spin ``a / M``.  Must satisfy ``|a| < 1`` or there is
    #: no horizon and the singularity is naked, which is believed not to
    #: happen and would in any case be impossible to light.
    spin: float = 0.94

    #: Outer edge of the luminous disc, in gravitational radii.  The inner
    #: edge is not a free parameter: it is the innermost stable circular
    #: orbit, which the spin fixes.
    r_disc_out: float = 17.0

    disc_h0: float = 0.055          # scale height at the inner edge
    disc_flare: float = 1.125       # h ~ r^(9/8), the thin-disc result

    #: Peak effective temperature, kelvin.  This fixes the mass and
    #: accretion rate implicitly: a stellar-mass hole radiating at the
    #: Eddington limit peaks in soft X-rays and would be invisible, a
    #: supermassive one in the ultraviolet.  The value below corresponds
    #: to an intermediate-mass hole and puts the peak in the visible,
    #: which is the only reason any of this can be photographed.
    t_peak: float = 9300.0
    t_floor: float = 1250.0

    #: Vertical optical depth at the surface-density maximum.  Finite
    #: rather than infinite so that the inner edge, where the zero-torque
    #: boundary condition drives the surface density to zero, is genuinely
    #: translucent and the secondary image shows through it.
    tau0: float = 3.1

    #: Scale factor between film seconds and the orbital clock.
    spin_rate: float = 1.0

    #: Spiral density waves: azimuthal mode number and pitch angle.
    wave_m: int = 2
    wave_pitch_deg: float = 22.0
    wave_amp: float = 0.15

    jet_strength: float = 0.85
    jet_opening: float = 0.165
    jet_core: float = 1.05
    jet_scale_height: float = 10.5
    jet_pitch: float = 0.40          # helical winding per unit height

    max_steps: int = 400
    tol: float = 0.030
    r_escape: float = 420.0

    # ---- derived ------------------------------------------------------
    def r_horizon(self):
        return KR.horizon_outer(self.spin)

    def r_isco(self):
        return KR.isco(self.spin, prograde=True)

    def r_photon(self):
        return KR.photon_circular(self.spin, prograde=True)


# ==========================================================================
# Novikov-Thorne radial structure
# ==========================================================================
class NovikovThorne:
    """Relativistic thin-disc flux profile (Page & Thorne 1974).

    The Newtonian answer, due to Shakura and Sunyaev, is

        F(r) ~ r^-3 [ 1 - sqrt(r_in / r) ]

    — the bracket is the zero-torque boundary condition, which forces the
    emission to vanish at the inner edge and makes the disc peak a little
    outside it instead of diverging.  In Kerr spacetime the same energy
    bookkeeping, done with the correct expressions for the specific energy
    and angular momentum of a circular geodesic, gives

        F(x) ∝ 1 / [ x^4 (x^3 - 3x + 2a) ] *
               { x - x0 - (3/2) a ln(x/x0)
                 - sum_i  3 (x_i - a)^2
                          / [ x_i (x_i - x_j)(x_i - x_k) ]
                          ln[ (x - x_i) / (x0 - x_i) ] }

    with ``x = sqrt(r)``, ``x0 = sqrt(r_isco)`` and ``x1, x2, x3`` the
    three roots of ``x^3 - 3x + 2a = 0``,

        x1 = 2 cos( arccos(a)/3 - pi/3 )
        x2 = 2 cos( arccos(a)/3 + pi/3 )
        x3 = -2 cos( arccos(a)/3 )

    At ``a = 0`` this reduces to the Shakura-Sunyaev form, which the
    self-test checks; at ``a = 0.94`` it is roughly five times more
    luminous per unit accreted mass, because the ISCO is three times
    deeper and the binding energy released is correspondingly larger.

    Evaluated once into a radial table at construction, so the shader
    costs one gather.
    """

    def __init__(self, a, r_in, r_out, n=2048):
        self.a = float(a)
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.n = int(n)

        ac = math.acos(max(-1.0, min(1.0, self.a)))
        x1 = 2.0 * math.cos(ac / 3.0 - math.pi / 3.0)
        x2 = 2.0 * math.cos(ac / 3.0 + math.pi / 3.0)
        x3 = -2.0 * math.cos(ac / 3.0)
        self.roots = (x1, x2, x3)

        r = np.linspace(self.r_in, self.r_out, self.n)
        self.r_tab = r.astype(F32)
        self.flux = self._flux(r)
        # effective temperature from Stefan-Boltzmann, normalised later
        t = np.power(np.maximum(self.flux, 0.0), 0.25)
        peak = float(t.max()) or 1.0
        self.shape = (t / peak).astype(np.float64)
        self.inv_dr = F32((self.n - 1) / max(self.r_out - self.r_in, 1e-6))

    # ------------------------------------------------------------------
    def _flux(self, r):
        a = self.a
        x1, x2, x3 = self.roots
        x = np.sqrt(np.maximum(r, 1e-9))
        x0 = math.sqrt(self.r_in)

        def term(xi, xj, xk):
            # the (xi - a)^2 numerator vanishes together with xi in the
            # limit a -> 0, where one root goes to zero; guard the
            # division so the Schwarzschild case stays finite
            denom = xi * (xi - xj) * (xi - xk)
            if abs(denom) < 1e-12:
                return np.zeros_like(x)
            num = 3.0 * (xi - a) ** 2 / denom
            arg = np.maximum((x - xi) / (x0 - xi), 1e-12)
            return num * np.log(arg)

        bracket = (x - x0 - 1.5 * a * np.log(np.maximum(x / x0, 1e-12))
                   - term(x1, x2, x3) - term(x2, x1, x3)
                   - term(x3, x1, x2))
        denom = x ** 4 * np.maximum(x ** 3 - 3.0 * x + 2.0 * a, 1e-9)
        return np.maximum(bracket, 0.0) / denom

    # ------------------------------------------------------------------
    def temperature(self, r, t_peak, t_floor):
        """Effective temperature at radius ``r``, gathered from the table."""
        i = np.clip((np.asarray(r, F32) - F32(self.r_in)) * self.inv_dr,
                    0, self.n - 1).astype(np.int32)
        s = self.shape[i].astype(F32)
        return F32(t_floor) + (F32(t_peak) - F32(t_floor)) * s

    def shape_at(self, r):
        i = np.clip((np.asarray(r, F32) - F32(self.r_in)) * self.inv_dr,
                    0, self.n - 1).astype(np.int32)
        return self.shape[i].astype(F32)


# ==========================================================================
# the accretion disc
# ==========================================================================
class DiscShader:
    """Emission, colour and opacity of the disc at an equatorial crossing.

    The tracer hands over, for each crossing: the Boyer-Lindquist radius
    and azimuth, the ray's conserved ``lambda``, the cosine of the angle
    at which it met the plane, and the sign of its radial motion.  That is
    everything needed, because in a stationary spacetime the photon's
    entire history is summarised by its constants of motion.
    """

    def __init__(self, p: BlackHoleParams, lib, seed=23):
        self.p = p
        self.a = float(p.spin)
        self.r_in = p.r_isco()
        self.r_out = float(p.r_disc_out)
        self.r_h = p.r_horizon()
        self.nt = NovikovThorne(self.a, self.r_in, self.r_out)

        self.v_disc = lib.get("disc")
        self.v_fine = lib.get("disc_fine")
        self.v_plasma = lib.get("plasma")

        self._t = 0.0
        self._cam_r = 1e4
        self._cam_mu = 0.0
        self._jitter = None

        # ---- baked radial tables ------------------------------------
        n = self.nt.n
        r = self.nt.r_tab
        self.t_tab = self.nt.temperature(r, p.t_peak, p.t_floor)
        # Stefan-Boltzmann: the inner disc is not merely bluer, it is
        # overwhelmingly brighter, as the fourth power of temperature.
        rel = self.t_tab / F32(p.t_peak)
        self.bright_tab = (rel * rel * rel * rel).astype(F32)
        # surface density window: zero torque at the inner edge drives
        # sigma to zero there, and the outer edge is where the disc runs
        # out of gas
        u = (r - F32(self.r_in)) / F32(max(self.r_out - self.r_in, 1e-6))
        inner = np.clip(u / 0.055, 0.0, 1.0) ** 1.35
        outer = np.clip((1.0 - u) / 0.14, 0.0, 1.0) ** 0.8
        self.sigma_tab = (inner * outer).astype(F32)
        self.omega_tab = KR.orbit_omega(r.astype(np.float64),
                                        self.a).astype(F32)
        self.inv_dr = self.nt.inv_dr

        # orbital angular velocity bounds used by the plunging region
        self.om_isco = float(KR.orbit_omega(self.r_in, self.a))
        self.om_horizon = self.a / (2.0 * self.r_h)
        self.ut_isco = float(KR.orbit_ut(self.r_in, self.a))
        self.lapse_isco = float(KR.lapse(np.array([self.r_in]),
                                         np.array([0.0]), self.a)[0])

        # Jet: a power-law electron population radiating synchrotron in a
        # tangled field gives a flat, slightly blue continuum, so the jet
        # is tinted rather than sampled from Planck's law.
        self.jet_rgb = np.array([0.46, 0.66, 1.0], F32)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def set_frame(self, t, cam_r, cam_mu, jitter):
        self._t = float(t)
        self._cam_r = float(cam_r)
        self._cam_mu = float(cam_mu)
        self._jitter = jitter
        # the camera-side half of the redshift factor is the same for
        # every ray in the frame
        w = float(KR.frame_dragging(np.array([cam_r]), np.array([cam_mu]),
                                    self.a)[0])
        al = float(KR.lapse(np.array([cam_r]), np.array([cam_mu]),
                            self.a)[0])
        self._cam_w = F32(w)
        self._cam_alpha = F32(al)

    # ------------------------------------------------------------------
    def _gather(self, r):
        i = np.clip((r - F32(self.r_in)) * self.inv_dr, 0,
                    self.nt.n - 1).astype(np.int32)
        return i

    # ------------------------------------------------------------------
    def half_thickness(self, r):
        return (F32(self.p.disc_h0)
                * np.power(np.maximum(r, F32(1e-3)) / F32(self.r_in),
                           F32(self.p.disc_flare)))

    # ------------------------------------------------------------------
    def redshift(self, lam, r_em):
        """``g = nu_obs / nu_em`` for prograde circular emitters.

            g = [ (1 - omega_c lambda) / alpha_c ] / [ u^t (1 - Omega lambda) ]

        Written out rather than delegated so the camera-side numerator,
        which is constant across the frame, is applied as a scalar.
        """
        om = F32(1.0) / (np.power(r_em, F32(1.5)) + F32(self.a))
        r3 = r_em * r_em * r_em
        rad = r3 - F32(3.0) * r_em * r_em + F32(2.0 * self.a) * np.power(
            r_em, F32(1.5))
        ut = (np.power(r_em, F32(1.5)) + F32(self.a)) / np.sqrt(
            np.maximum(rad, F32(1e-6)))
        denom = ut * (F32(1.0) - om * lam)
        num = (F32(1.0) - self._cam_w * lam) / self._cam_alpha
        return num / np.where(np.abs(denom) < F32(1e-5), F32(1e-5), denom)

    # ------------------------------------------------------------------
    def __call__(self, r, phi, lam, cos_inc, slot, pr):
        """Equatorial-crossing callback.  Returns ``(rgb, opacity)``."""
        n = r.shape[0]
        out = np.zeros((n, 3), F32)
        opac = np.zeros(n, F32)

        r = r.astype(F32)
        phi = phi.astype(F32)
        lam = lam.astype(F32)
        cos_inc = np.maximum(cos_inc.astype(F32), F32(0.022))

        disc = (r >= F32(self.r_in)) & (r <= F32(self.r_out))
        if disc.any():
            i = np.flatnonzero(disc)
            self._shade_disc(out, opac, i, r[i], phi[i], lam[i], cos_inc[i])

        plunge = (r > F32(self.r_h * 1.015)) & (r < F32(self.r_in))
        if plunge.any():
            i = np.flatnonzero(plunge)
            self._shade_plunge(out, opac, i, r[i], phi[i], lam[i],
                               cos_inc[i])

        return out, opac

    # ------------------------------------------------------------------
    def _shade_disc(self, out, opac, idx, r, phi, lam, cos_inc):
        p = self.p
        ti = self._gather(r)
        t_em = self.t_tab[ti]
        bright = self.bright_tab[ti]
        sig = self.sigma_tab[ti]
        omega = self.omega_tab[ti]

        # ---- turbulence, carried round with the gas --------------------
        # Differential rotation: the inner disc laps the outer many times
        # while the camera closes in, so the texture has to be sampled in
        # a frame that shears.  Sampling it in a rigid frame makes the
        # disc look like a painted plate being spun.
        #
        # Sampled in *log-polar* coordinates rather than Cartesian, which
        # matters for two reasons.  First, a shearing flow does not have
        # isotropic turbulence: every eddy is stretched into a long thin
        # filament wrapped around the hole, so the texture has to be
        # elongated azimuthally by roughly the ratio of the shear rate to
        # the eddy turnover rate.  A Cartesian lookup gives round blobs,
        # which read as a rocky surface rather than as orbiting gas.
        # Second, the azimuthal coordinate is made an exact integer
        # number of tile periods per turn, so the field closes on itself
        # at ``phi = 2 pi`` instead of showing a seam.
        ang = phi - omega * F32(self._t * 7.5 * p.spin_rate)
        lr = np.log(r)
        aw = ang * F32(1.0 / TAU)
        w0 = np.full_like(r, F32(self._t * 0.055))
        # Three octaves, each with five, seventeen and forty-seven
        # periods per turn and a progressively steeper log-radial shear,
        # which is what tilts the filaments into trailing spirals.
        q1 = np.stack([lr * F32(2.2),
                       aw * F32(5.0) + lr * F32(3.4),
                       w0], axis=1).astype(F32)
        q2 = np.stack([lr * F32(5.0),
                       aw * F32(23.0) + lr * F32(9.0),
                       w0 * F32(2.3) + F32(11.0)], axis=1).astype(F32)
        q3 = np.stack([lr * F32(13.0),
                       aw * F32(59.0) + lr * F32(24.0),
                       w0 * F32(4.1) + F32(5.0)], axis=1).astype(F32)
        n1 = self.v_disc.sample(q1, jitter=self._jitter)
        n2 = self.v_fine.sample(q2, jitter=self._jitter)
        n3 = self.v_plasma.sample(q3, jitter=self._jitter)
        tex = (n1 * F32(0.70)
               + np.clip(n2 * F32(1.9) + F32(0.5), 0.0, 1.0) * F32(0.32)
               + np.clip(n3 * F32(2.0) + F32(0.5), 0.0, 1.0) * F32(0.14))
        # a floor, because an optically thick disc never goes dark: the
        # turbulence modulates the emission, it does not switch it off
        tex = F32(0.08) + F32(1.62) * np.clip(tex, 0.0, 1.4)

        # ---- spiral density waves --------------------------------------
        # A real disc is not axisymmetric: self-gravity, the
        # magnetorotational instability and any companion all drive
        # trailing spiral waves.  A logarithmic spiral of pitch angle
        # ``i`` satisfies phi = ln(r) / tan(i), so the wave phase is
        # m (phi - ln r / tan i).
        pitch = math.tan(math.radians(p.wave_pitch_deg))
        wave = np.cos(F32(p.wave_m) * (ang - lr / F32(pitch)))
        tex = tex * (F32(1.0) + F32(p.wave_amp) * wave)

        # ---- the relativistic shift ------------------------------------
        g = self.redshift(lam, r)
        np.clip(g, F32(0.035), F32(4.5), out=g)
        g2 = g * g
        boost = g2 * g2

        # ---- colour at the shifted temperature -------------------------
        rgb = S.blackbody_lut(np.clip(t_em * g, F32(600.0), F32(60000.0)))
        rgb = S.saturate_rgb(rgb, 1.55)
        rgb = rgb / np.maximum(S.luminance(rgb), F32(1e-5))[:, None]

        amp = bright * boost * tex * F32(1.0)
        out[idx] = rgb * amp[:, None]

        # ---- opacity ----------------------------------------------------
        # Vertical optical depth divided by the cosine of the incidence
        # angle: a ray that grazes the plane travels much further through
        # the same column of gas.  The inner edge is translucent because
        # the zero-torque condition drives the surface density to zero
        # there, which is what lets the secondary image show through.
        tau = F32(p.tau0) * sig / cos_inc
        opac[idx] = F32(1.0) - np.exp(-np.minimum(tau, F32(12.0)))

    # ------------------------------------------------------------------
    def _shade_plunge(self, out, opac, idx, r, phi, lam, cos_inc):
        """Gas inside the ISCO, on its way through the horizon.

        No stable circular orbit exists here, so the flow spirals in with
        its angular velocity tending to the hole's own,
        ``Omega_H = a / 2 r+``.  It is optically thin, it is strongly
        redshifted by the collapsing lapse, and it dims to nothing at the
        horizon — not because the gas stops radiating, but because the
        light cannot climb out.
        """
        p = self.p
        span = max(self.r_in - self.r_h, 1e-3)
        u = np.clip((r - F32(self.r_h)) / F32(span), 0.0, 1.0)

        # angular velocity blends from the ISCO value to the horizon's
        om = F32(self.om_horizon) + (F32(self.om_isco)
                                     - F32(self.om_horizon)) * u ** F32(1.4)
        denom = F32(self.ut_isco) * (F32(1.0) - om * lam)
        num = (F32(1.0) - self._cam_w * lam) / self._cam_alpha
        g = num / np.where(np.abs(denom) < F32(1e-5), F32(1e-5), denom)
        # extra redshift from the collapsing lapse between here and the
        # ISCO, which is what actually extinguishes the plunging flow
        mu0 = np.zeros_like(r, np.float64)
        al = KR.lapse(r.astype(np.float64), mu0, self.a).astype(F32)
        g = g * np.power(al / F32(self.lapse_isco), F32(0.85))
        np.clip(g, F32(0.02), F32(4.5), out=g)
        g2 = g * g
        boost = g2 * g2

        ang = phi - om * F32(self._t * 7.5 * p.spin_rate)
        qv = np.stack([r * np.cos(ang) * F32(0.42), np.zeros_like(r),
                       r * np.sin(ang) * F32(0.42)], axis=1).astype(F32)
        tex = np.clip(self.v_plasma.sample(qv, jitter=self._jitter)
                      * F32(2.3) + F32(0.55), 0.0, 1.6)

        t_em = F32(p.t_peak) * F32(1.22)
        rgb = S.blackbody_lut(np.clip(t_em * g, F32(600.0), F32(60000.0)))
        rgb = S.saturate_rgb(rgb, 1.30)
        rgb = rgb / np.maximum(S.luminance(rgb), F32(1e-5))[:, None]

        amp = boost * tex * u * F32(0.55)
        out[idx] += rgb * amp[:, None]
        opac[idx] = np.maximum(
            opac[idx],
            (F32(1.0) - np.exp(-F32(0.9) * u / cos_inc)) * F32(0.75))


# ==========================================================================
# the jet
# ==========================================================================
class JetField:
    """Collimated outflow along the spin axis, sampled during the trace.

    Accumulated inside the geodesic integration rather than composited
    afterwards, so it is lensed: the base of the counter-jet, on the far
    side of the hole, is bent into view around the shadow.

    The structure is analytic — a cone whose radius grows with height,
    modulated by a helix — rather than a volume lookup, because this
    callback fires once per integration step for every live ray, several
    tens of millions of times a frame, and a texture fetch at that rate
    would cost more than the geodesics themselves.
    """

    def __init__(self, p: BlackHoleParams, rgb):
        self.p = p
        self.rgb = np.asarray(rgb, F32)
        self._t = 0.0
        self.enabled = p.jet_strength > 0.0

    def set_time(self, t):
        self._t = float(t)

    def __call__(self, r0, mu0, phi0, r1, mu1, phi1, it):
        p = self.p
        if not self.enabled:
            return None

        r = (r0 + r1) * 0.5
        mu = (mu0 + mu1) * 0.5
        s2 = np.maximum(1.0 - mu * mu, 1e-7)
        s = np.sqrt(s2)
        z = r * mu
        rho = r * s
        az = np.abs(z)

        radius = p.jet_core + az * p.jet_opening
        live = (rho < radius) & (az > 1.1) & (r < 95.0)
        if not live.any():
            return None

        i = np.flatnonzero(live)
        ri = r[i]
        zi = z[i]
        azi = az[i]
        rhoi = rho[i]
        rad = radius[i]

        # segment length in the pseudo-Cartesian embedding
        dr = r1[i] - r0[i]
        dmu = mu1[i] - mu0[i]
        dph = phi1[i] - phi0[i]
        si = s[i]
        ds = np.sqrt(dr * dr + (ri * dmu / si) ** 2 + (ri * si * dph) ** 2)

        # helical magnetic structure: the field is wound up by the
        # rotation of the hole, so the brightness pattern climbs the jet
        # as a corkscrew
        ph = phi0[i]
        helix = 0.62 + 0.38 * np.sin(p.jet_pitch * zi - 2.0 * ph
                                     + self._t * 1.35)
        # Transverse profile: limb-brightened, because the emission comes
        # from a sheath around an evacuated spine rather than filling the
        # cone.  Softened with a cosine rather than a hard rim, or the
        # jet renders as a drawn line instead of a column of plasma.
        x = np.clip(rhoi / np.maximum(rad, 1e-6), 0.0, 1.0)
        sheath = (0.05 + 0.95 * np.sin(x * 3.14159) ** 1.6) \
            * (1.0 - x * x) ** 0.8
        # the nozzle is the brightest part: the flow is still being
        # collimated there and the field is strongest
        nozzle = 1.0 + 2.6 * np.exp(-((azi - 3.0) / 5.5) ** 2)
        fall = np.exp(-azi / p.jet_scale_height) / np.maximum(rad, 1e-3)
        amp = (helix * sheath * nozzle * fall * ds
               * (0.0016 * p.jet_strength)).astype(F32)

        out = np.zeros((r.shape[0], 3), F32)
        out[i] = self.rgb[None, :] * amp[:, None]
        return out


# ==========================================================================
# background sky
# ==========================================================================
class LensedSky:
    """Equirectangular star field, looked up by the escaped geodesics.

    Splatting the catalogue into a table once is far cheaper than
    searching it per pixel, and the table is what makes the lensing
    legible: isolated points are hard to read, whereas a continuous
    galactic band is visibly sheared into arcs as it passes behind the
    hole.

    Sampled bilinearly, and that is not optional.  Lensing *magnifies*,
    enormously so near the Einstein ring, and a single texel of sky can
    end up stretched across dozens of pixels; with nearest-neighbour
    lookup the lensed field becomes a mosaic of hard coloured squares in
    the most important shot of the film.
    """

    def __init__(self, seed=77, n_stars=150000, res=2048):
        self.res = int(res)
        rng = np.random.default_rng(seed)
        dirs = V.fibonacci_sphere(n_stars).astype(F32)
        dirs = dirs[rng.permutation(n_stars)]
        t = S.sample_stellar_temperatures(n_stars, rng,
                                          hot_bias=0.78).astype(F32)
        rgb = S.saturate_rgb(S.blackbody_lut(t), 1.15)
        rgb /= np.maximum(S.luminance(rgb), 1e-5)[:, None]
        b = (rng.random(n_stars).astype(F32) ** 8.0) * F32(0.46) + F32(0.0035)
        self.catalogue = (dirs, (rgb * b[:, None]).astype(F32))
        self.table = self._bake(rng)

    # ------------------------------------------------------------------
    def _bake(self, rng):
        dirs, rgb = self.catalogue
        res = self.res
        h = res // 2
        theta = np.arccos(np.clip(dirs[:, 1], -1.0, 1.0))
        phi = np.arctan2(dirs[:, 2], dirs[:, 0])
        iy = np.clip((theta / math.pi * h).astype(np.int32), 0, h - 1)
        ix = np.clip(((phi / TAU + 0.5) * res).astype(np.int32), 0, res - 1)
        flat = iy * res + ix
        tab = np.zeros((h * res, 3), np.float64)
        for k in range(3):
            tab[:, k] = np.bincount(flat,
                                    weights=rgb[:, k].astype(np.float64),
                                    minlength=h * res)
        tab = tab.reshape(h, res, 3).astype(F32)

        # --- diffuse galactic band -----------------------------------
        yy = (np.arange(h, dtype=np.float32) + 0.5) / h * math.pi
        xx = (np.arange(res, dtype=np.float32) + 0.5) / res * TAU - math.pi
        lat = yy[:, None] - math.pi * 0.5
        tilt = math.radians(34.0)
        band_lat = np.arcsin(np.clip(
            np.sin(lat) * math.cos(tilt)
            - np.cos(lat) * np.sin(xx[None, :]) * math.sin(tilt), -1, 1))
        prof = (np.exp(-(band_lat / 0.17) ** 2) * 0.58
                + np.exp(-(band_lat / 0.50) ** 2) * 0.24)
        lumps = rng.random((h // 16, res // 16)).astype(np.float32)
        lumps = np.kron(lumps, np.ones((16, 16), np.float32))[:h, :res]
        for _ in range(3):
            lumps = (lumps * 0.4
                     + np.roll(lumps, 3, 0) * 0.15
                     + np.roll(lumps, -3, 0) * 0.15
                     + np.roll(lumps, 3, 1) * 0.15
                     + np.roll(lumps, -3, 1) * 0.15)
        prof = prof * (0.42 + 1.15 * lumps)
        # dust lanes bite into the band from the midplane
        dust = np.exp(-(band_lat / 0.055) ** 2) * (0.35 + 0.65 * lumps)
        prof = prof * (1.0 - 0.62 * dust)
        band_rgb = np.array([0.60, 0.65, 0.82], np.float32)
        tab += (prof[:, :, None] * band_rgb[None, None, :]
                * np.float32(0.036)).astype(F32)

        tab = (tab * 0.64
               + np.roll(tab, 1, axis=1) * 0.09
               + np.roll(tab, -1, axis=1) * 0.09
               + np.roll(tab, 1, axis=0) * 0.09
               + np.roll(tab, -1, axis=0) * 0.09).astype(F32)
        return tab

    # ------------------------------------------------------------------
    def sample(self, dirs):
        tab = self.table
        h, w, _ = tab.shape
        theta = np.arccos(np.clip(dirs[:, 1], -1.0, 1.0))
        phi = np.arctan2(dirs[:, 2], dirs[:, 0])
        fy = np.clip(theta / math.pi * h - 0.5, 0.0, h - 1.001)
        fx = (phi / TAU + 0.5) * w - 0.5
        fx = np.mod(fx, w)
        y0 = fy.astype(np.int32)
        # np.mod can return exactly w: a longitude a hair below zero maps
        # to w - epsilon, and in float32 that rounds straight up to w.
        x0 = np.minimum(fx.astype(np.int32), w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = (x0 + 1) % w
        wy = (fy - y0).astype(F32)[:, None]
        wx = (fx - x0).astype(F32)[:, None]
        return (tab[y0, x0] * (1 - wy) * (1 - wx)
                + tab[y0, x1] * (1 - wy) * wx
                + tab[y1, x0] * wy * (1 - wx)
                + tab[y1, x1] * wy * wx)


# ==========================================================================
# the phase
# ==========================================================================
class PhaseBlackHole(Phase):
    """FASE V — a rotating black hole: Kerr lensing, disc, horizon, jet."""

    name = "V · LUBANG HITAM"
    duration = 12.0

    def __init__(self, params=None, seed=23):
        self.p = params or BlackHoleParams()
        self.seed = int(seed)
        self.disc = None
        self.jet = None
        self.tracer = None
        self.sky = None

    # ------------------------------------------------------------------
    def setup(self, ctx):
        p = self.p
        self.disc = DiscShader(p, ctx.lib, seed=self.seed)
        self.jet = JetField(p, self.disc.jet_rgb)
        self.tracer = KR.KerrTracer(KR.TraceConfig(
            a=p.spin, max_steps=p.max_steps, tol=p.tol,
            r_escape=p.r_escape))
        self.sky = LensedSky(seed=self.seed + 77)

    # ------------------------------------------------------------------
    def camera_at(self, ctx, lt):
        """Approach, descend into the plane, and orbit.

        The descent is the whole shot.  Seen from well above, the disc is
        an ordinary ring with a dark hole in it.  The famous image — the
        far side of the disc lifted up and folded over the top of the
        shadow — only appears when the line of sight comes within a few
        degrees of the plane, because that is when light from behind the
        hole has to bend through nearly a right angle to reach the
        camera.  So the camera starts high, where the geometry is legible,
        and falls to within three degrees of the midplane by the end.
        """
        u = self.progress(lt, self.duration)

        g = V.smootherstep(0.0, 0.84, u)
        dist = math.exp(math.log(165.0) * (1.0 - g) + math.log(23.5) * g)

        elev = 42.0 * (1.0 - V.smootherstep(0.05, 0.76, u)) + 5.2
        elev += 1.9 * math.sin(lt * 0.44)

        azim = 0.55 + 0.46 * u + 0.05 * math.sin(lt * 0.31)

        cam = CAM.orbit(np.zeros(3, F32), dist, azim, math.radians(elev))
        cam.fov_deg = float(30.0 + 28.0 * V.smootherstep(0.30, 1.0, u))
        cam.roll_deg = float(-2.6 + 5.4 * V.smootherstep(0.2, 1.0, u))
        return CAM.apply_handheld(cam, lt, pos_amp=0.010, rot_amp=0.13,
                                  fov_breath=0.07, seed=97.0)

    def exposure_at(self, ctx, lt):
        u = self.progress(lt, self.duration)
        return float(1.30 * (1.0 - 0.62 * V.smootherstep(0.22, 1.0, u)))

    # ------------------------------------------------------------------
    def render(self, ctx, lt):
        cam = self.camera_at(ctx, lt)
        exposure = self.exposure_at(ctx, lt)
        rng = ctx.rng or np.random.default_rng(ctx.frame * 7919 + 59)

        eye = np.asarray(cam.eye, np.float64)
        cam_r = float(np.sqrt(eye @ eye))
        cam_mu = float(eye[1] / max(cam_r, 1e-9))

        self.disc.set_frame(lt, cam_r, cam_mu, vcache.new_jitter(rng))
        self.jet.set_time(lt)

        mw, mh = ctx.march_width, ctx.march_height
        o, d, idx = cam.generate_rays(mw, mh, jitter_phase=ctx.frame & 7)

        rgb, trans, out_dir, escaped = self.tracer.trace(
            o, d, on_equator=self.disc,
            on_step=self.jet if self.jet.enabled else None)

        if escaped.any():
            i = np.flatnonzero(escaped)
            rgb[i] += self.sky.sample(out_dir[i]) * trans[i][:, None]

        img = np.zeros((mh * mw, 3), F32)
        img[idx] = rgb
        img = img.reshape(mh, mw, 3)
        return upsample_linear(img, ctx.width, ctx.height) * F32(exposure)


# ==========================================================================
# self-test
# ==========================================================================
def _selftest():
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print("  %-46s %14.6f  %-6s" % (name, got, "OK" if good else "FAIL"),
              flush=True)

    print("phase V self-test", flush=True)
    p = BlackHoleParams()
    print("  spin %.3f  horizon %.4f  isco %.4f  photon %.4f"
          % (p.spin, p.r_horizon(), p.r_isco(), p.r_photon()), flush=True)

    # At a = 0 the series collapses to a closed form that can be written
    # out by hand, which is the sharpest available check on the roots and
    # the logarithms:
    #
    #   F ~ [ x - sqrt(6) + (sqrt3/2) ln( (x+sqrt3)(sqrt6-sqrt3)
    #                                   / ((x-sqrt3)(sqrt6+sqrt3)) ) ]
    #       / [ r^3.5 (1 - 3/r) ]
    nt0 = NovikovThorne(1e-7, 6.0, 400.0, 8192)
    r = nt0.r_tab[6:].astype(np.float64)
    x = np.sqrt(r)
    s3, s6 = math.sqrt(3.0), math.sqrt(6.0)
    closed = (x - s6 + 0.5 * s3 * np.log(
        ((x + s3) * (s6 - s3)) / ((x - s3) * (s6 + s3)))) \
        / (r ** 3.5 * (1.0 - 3.0 / r))
    a1 = nt0.flux[6:] / nt0.flux[6:].max()
    a2 = closed / closed.max()
    check("NT at a=0 vs closed form (max abs err)",
          float(np.abs(a1 - a2).max()), 0.0, 1e-4)

    # and far from the hole it must tend to the Newtonian answer
    far = r > 120.0
    ss = (1.0 - np.sqrt(6.0 / r[far])) / r[far] ** 3
    rel = nt0.flux[6:][far]
    check("NT -> Shakura-Sunyaev for r > 120 M",
          float(np.abs(rel / rel[0] - ss / ss[0]).max()), 0.0, 1.2e-2)

    nt = NovikovThorne(0.94, KR.isco(0.94), 17.0, 2048)
    ipk = int(np.argmax(nt.shape))
    print("  peak of T(r) at r = %.3f M  (isco %.3f)"
          % (nt.r_tab[ipk], KR.isco(0.94)), flush=True)
    check("T vanishes at the inner edge", float(nt.shape[0]), 0.0, 1e-9)

    # the redshift must span a large range across the disc
    a = 0.94
    rr = np.full(9, KR.isco(a) * 2.0)
    lam = np.linspace(-6.0, 6.0, 9)
    g = KR.redshift_factor(lam, rr, a, np.array([200.0]), np.array([0.05]))
    print("  g across lambda=-6..6 at r=%.2f : %s"
          % (rr[0], " ".join("%.2f" % v for v in g)), flush=True)
    check("beaming contrast (g^4 max / min)",
          float((g.max() / g.min()) ** 4) > 40.0, True, 0.5)

    print("phase V self-test %s" % ("PASSED" if ok else "FAILED"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())


__all__ = ["BlackHoleParams", "NovikovThorne", "DiscShader", "JetField",
           "LensedSky", "PhaseBlackHole"]
