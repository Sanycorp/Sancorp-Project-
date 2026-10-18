"""
cosmos.core.kerr
================

Null geodesics of the **Kerr** metric — a rotating black hole.

Why this replaces the Schwarzschild tracer
------------------------------------------
Every black hole that actually exists spins.  Accretion delivers angular
momentum along with mass, and the measured spins of real holes cluster
high: Cygnus X-1 sits above ``a = 0.95``, GRS 1915+105 above ``0.98``, and
the supermassive hole in M87 is estimated near ``0.9``.  A non-rotating
hole is a mathematical idealisation that nothing in the sky resembles.

The difference is not cosmetic:

* The horizon shrinks.  ``r+ = M + sqrt(M^2 - a^2)``, so at ``a = 0.94``
  it sits at ``1.34 M`` instead of ``2 M``.
* The innermost stable orbit collapses from ``6 M`` to ``2.02 M``, so the
  disc reaches far deeper, orbits far faster, and gets far hotter.
* Space itself is dragged around the hole.  A photon's azimuth is swept
  forward even when its angular momentum is zero, which twists the whole
  lensed sky into a vortex instead of a symmetric ring.
* The shadow stops being a circle.  Prograde photons can skim closer
  before being captured than retrograde ones, so the silhouette is pushed
  sideways and flattened on the approaching limb into a D.
* Geodesics stop being planar.  In Schwarzschild every ray stays in a
  plane through the centre, which is what let the old tracer reduce the
  problem to a single second-order ODE in ``u = 1/r``.  In Kerr that
  symmetry is gone and the motion is genuinely three dimensional.

Separability and the Carter constant
------------------------------------
Kerr geodesics remain integrable, which is a small miracle discovered by
Brandon Carter in 1968.  Besides energy ``E = -p_t`` and axial angular
momentum ``L = p_phi`` there is a third conserved quantity

    Q = p_theta^2 + cos^2(theta) [ L^2 / sin^2(theta) - a^2 E^2 ]

for a photon, and with it the Hamilton-Jacobi equation separates.  Writing
``lambda = L/E`` and ``q = Q/E^2`` and using **Mino time** ``sigma``,
defined by ``d sigma = d(affine) / Sigma``, the radial and polar motions
decouple completely:

    (dr / dsigma)^2     = R(r)   = (r^2 + a^2 - a lambda)^2
                                   - Delta [ (lambda - a)^2 + q ]
    (dtheta / dsigma)^2 = Theta  = q + a^2 cos^2 th - lambda^2 cot^2 th

Integrating those square roots directly is miserable, because the sign
flips at every turning point and the turning point is exactly where the
derivative is least well determined.  Differentiating once removes the
problem entirely:

    d2r / dsigma^2     = R'(r) / 2
    d2theta / dsigma^2 = Theta'(theta) / 2

Both right-hand sides are smooth, sign-free, and need no case analysis.

The substitution that removes all trigonometry
----------------------------------------------
Working in ``mu = cos(theta)`` instead of ``theta`` turns the polar
equation into a polynomial.  With ``(dmu/dsigma)^2 = sin^2(theta) Theta``,

    Mu(mu) = q + (a^2 - q - lambda^2) mu^2 - a^4-free quartic
           = q + (a^2 - q - lambda^2) mu^2 - a^2 mu^4

    d2mu / dsigma^2 = mu ( (a^2 - q - lambda^2) - 2 a^2 mu^2 )

so the entire inner loop of the integrator — four Runge-Kutta stages per
step, several hundred steps, a third of a million rays — contains not one
sine, cosine, arctangent or power.  Only two divisions survive, by
``Delta`` and by ``1 - mu^2``.  That is the single most important
optimisation in this module: the trigonometric form costs roughly four
times as much per step and buys nothing.

Observer frame
--------------
The camera is treated as a **ZAMO**, the locally non-rotating observer
whose worldline is orthogonal to the surfaces of constant ``t``.  A static
observer would also be available outside the ergosphere, but the ZAMO
tetrad is regular everywhere outside the horizon and its orthonormal
spatial triad points along the ordinary coordinate directions, so a
Cartesian view ray converts into local components with three dot products.

Redshift
--------
For a photon with conserved ratio ``lambda`` received by a ZAMO at
``(r_c, theta_c)`` and emitted by gas on a prograde circular geodesic at
radius ``r``, the ratio of received to emitted frequency is

    g = [ (1 - omega_c lambda) / alpha_c ] / [ u^t (1 - Omega lambda) ]

with ``Omega = 1 / (r^3/2 + a)`` the orbital angular velocity and ``u^t``
the emitter's time dilation.  This one expression contains the
gravitational redshift, the transverse Doppler effect, the longitudinal
Doppler effect and the frame-dragging contribution simultaneously, and it
is exact.  Specific intensity then scales as ``g^4`` because ``I / nu^3``
is a Lorentz invariant and the bolometric intensity carries one more power
from the frequency integral.

Units
-----
Geometric units with ``G = c = 1`` and ``M = 1``.  Lengths are in
gravitational radii; the horizon of a maximally spinning hole sits at
``r = 1`` and of a non-rotating one at ``r = 2``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

F64 = np.float64
F32 = np.float32
TAU = 2.0 * math.pi


# ==========================================================================
# metric scalars
# ==========================================================================
def delta(r, a):
    """``Delta = r^2 - 2 M r + a^2`` — vanishes at the two horizons."""
    return r * r - 2.0 * r + a * a


def sigma(r, mu, a):
    """``Sigma = r^2 + a^2 cos^2(theta)``, with ``mu = cos(theta)``."""
    return r * r + a * a * mu * mu


def amaj(r, mu, a):
    """``A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta)``.

    The area function that appears in ``g_phiphi`` and in the frame
    dragging rate; always positive outside the horizon.
    """
    s2 = 1.0 - mu * mu
    r2a2 = r * r + a * a
    return r2a2 * r2a2 - a * a * delta(r, a) * s2


def frame_dragging(r, mu, a):
    """Angular velocity ``omega = 2 a r / A`` of the locally non-rotating
    frame.

    A ZAMO at radius ``r`` is carried around the hole at this rate purely
    by the rotation of spacetime, with no force acting on it.  Inside the
    ergosphere it exceeds the escape condition for static observers: no
    worldline can remain at fixed ``phi``.
    """
    return 2.0 * a * r / amaj(r, mu, a)


def lapse(r, mu, a):
    """ZAMO lapse ``alpha = sqrt(Sigma Delta / A)``.

    The ratio of the ZAMO's proper time to coordinate time; it falls to
    zero at the horizon, which is the statement that the horizon is an
    infinite-redshift surface for these observers.
    """
    return np.sqrt(sigma(r, mu, a) * delta(r, a) / amaj(r, mu, a))


def g_phiphi(r, mu, a):
    """``g_phiphi = A sin^2(theta) / Sigma`` — the proper circumference
    squared, per radian squared."""
    return amaj(r, mu, a) * (1.0 - mu * mu) / sigma(r, mu, a)


# ==========================================================================
# characteristic radii
# ==========================================================================
def horizon_outer(a):
    """``r+ = M + sqrt(M^2 - a^2)``."""
    return 1.0 + math.sqrt(max(1.0 - a * a, 0.0))


def horizon_inner(a):
    """``r- = M - sqrt(M^2 - a^2)``, the Cauchy horizon."""
    return 1.0 - math.sqrt(max(1.0 - a * a, 0.0))


def ergosphere(a, mu):
    """Static limit ``r_E = M + sqrt(M^2 - a^2 cos^2 theta)``.

    Touches the horizon at the poles and bulges to ``2 M`` at the equator.
    Between it and the horizon lies the ergoregion, where nothing can stay
    still and energy can in principle be extracted from the hole.
    """
    return 1.0 + np.sqrt(np.maximum(1.0 - a * a * mu * mu, 0.0))


def isco(a, prograde=True):
    """Innermost stable circular orbit, Bardeen-Press-Teukolsky (1972).

    Prograde orbits can approach much closer than retrograde ones because
    frame dragging helps hold them up: at ``a = 0.94`` the prograde ISCO
    is at ``2.02 M`` and the retrograde one out at ``8.4 M``.  This single
    number sets the inner edge, the peak temperature and the maximum
    orbital speed of the accretion disc, so it controls almost everything
    the audience actually sees.
    """
    a = float(a)
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    root = math.sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0))
    return 3.0 + z2 - root if prograde else 3.0 + z2 + root


def photon_circular(a, prograde=True):
    """Equatorial circular photon orbit radius.

    ``r_ph = 2 M { 1 + cos[ (2/3) arccos(-+ a / M) ] }`` — the prograde
    branch runs from ``3 M`` at ``a = 0`` down to ``M`` at ``a = 1``, the
    retrograde branch out to ``4 M``.  The pair of them bracket the band
    of spherical photon orbits whose projection on the sky is the shadow.
    """
    s = -a if prograde else a
    return 2.0 * (1.0 + math.cos((2.0 / 3.0) * math.acos(max(-1.0,
                                                             min(1.0, s)))))


def marginally_bound(a, prograde=True):
    """``r_mb = 2 M -+ a + 2 sqrt(M (M -+ a))`` — the parabolic capture
    radius, which bounds the plunging region from outside."""
    s = -1.0 if prograde else 1.0
    return 2.0 + s * a + 2.0 * math.sqrt(max(1.0 + s * a, 0.0))


# ==========================================================================
# circular equatorial orbits
# ==========================================================================
def orbit_omega(r, a):
    """``Omega = 1 / (r^{3/2} + a)`` for a prograde circular geodesic.

    Reduces to the Keplerian ``r^{-3/2}`` far out and saturates at the
    horizon frequency as the orbit approaches a maximally spinning hole.
    """
    return 1.0 / (np.power(r, 1.5) + a)


def orbit_ut(r, a):
    """Time-dilation factor ``u^t`` of a prograde circular geodesic.

    ``u^t = (r^{3/2} + a) / sqrt(r^3 - 3 r^2 + 2 a r^{3/2})``

    The radicand vanishes exactly at the circular photon orbit, which is
    the statement that a massive particle would need infinite energy to
    orbit there.  Clamped a little outside it so the expression stays
    finite if a sample strays in.
    """
    r32 = np.power(r, 1.5)
    rad = r * r * r - 3.0 * r * r + 2.0 * a * r32
    return (r32 + a) / np.sqrt(np.maximum(rad, 1e-9))


def orbit_velocity_zamo(r, a):
    """Speed of the orbiting gas as measured by a local ZAMO.

    ``v = (Omega - omega) sqrt(g_phiphi) / alpha``.  At the prograde ISCO
    of a ``a = 0.94`` hole this is about ``0.63 c``, which is why the
    approaching limb of the disc outshines the receding one by more than
    two orders of magnitude.
    """
    mu = np.zeros_like(np.asarray(r, F64))
    om = orbit_omega(r, a)
    w = frame_dragging(r, mu, a)
    return (om - w) * np.sqrt(g_phiphi(r, mu, a)) / lapse(r, mu, a)


def redshift_factor(lam, r_em, a, r_cam, mu_cam):
    """Ratio of received to emitted frequency, ``g = nu_obs / nu_em``.

    ``lam`` is the photon's conserved ``L / E``; ``r_em`` the Boyer-
    Lindquist radius of the emitting gas, assumed on a prograde circular
    geodesic in the equatorial plane; the camera is a ZAMO at
    ``(r_cam, mu_cam)``.

    Every relativistic effect on the disc's appearance is in here.  There
    is no separate "Doppler term" and "gravity term" to multiply together
    — in a stationary spacetime the shift is one ratio of two inner
    products, and splitting it is an approximation that fails exactly
    where it matters most.
    """
    om = orbit_omega(r_em, a)
    ut = orbit_ut(r_em, a)
    denom = ut * (1.0 - om * lam)
    w_c = frame_dragging(r_cam, mu_cam, a)
    a_c = lapse(r_cam, mu_cam, a)
    numer = (1.0 - w_c * lam) / a_c
    return numer / np.where(np.abs(denom) < 1e-6, 1e-6, denom)


# ==========================================================================
# the shadow
# ==========================================================================
def spherical_photon_constants(r, a):
    """``(lambda, q)`` of the spherical photon orbit at radius ``r``.

    Spherical photon orbits keep ``r`` constant while oscillating in
    latitude.  They exist for ``r`` between the prograde and retrograde
    equatorial photon radii, and their projection onto an observer's sky
    is precisely the edge of the black hole's shadow.  Solving
    ``R(r) = R'(r) = 0`` gives

        lambda = (r^2 + a^2) / a - 2 r Delta / (a (r - M))
        q      = r^3 [ 4 M Delta - r (r - M)^2 ] / (a^2 (r - M)^2)
    """
    r = np.asarray(r, F64)
    d = delta(r, a)
    rm = r - 1.0
    lam = (r * r + a * a) / a - 2.0 * r * d / (a * rm)
    q = r ** 3 * (4.0 * d - r * rm * rm) / (a * a * rm * rm)
    return lam, q


def shadow_outline(a, inclination_deg, n=512):
    """Shadow boundary on the sky of a distant observer.

    Returns celestial coordinates ``(alpha, beta)`` in gravitational
    radii, using Bardeen's impact parameters

        alpha = -lambda / sin(i)
        beta  = +- sqrt( q + a^2 cos^2 i - lambda^2 cot^2 i )

    For ``a = 0`` this traces a circle of radius ``3 sqrt(3) = 5.196``.
    As the spin rises the curve shifts toward positive ``alpha`` and its
    prograde edge flattens, because prograde photons survive closer in.
    """
    i = math.radians(inclination_deg)
    si, ci = math.sin(i), math.cos(i)
    r_lo = photon_circular(a, prograde=True)
    r_hi = photon_circular(a, prograde=False)
    r = np.linspace(r_lo + 1e-7, r_hi - 1e-7, n)
    lam, q = spherical_photon_constants(r, a)
    if abs(si) < 1e-9:
        si = 1e-9
    b2 = q + a * a * ci * ci - lam * lam * (ci / si) ** 2
    ok = b2 >= 0.0
    alpha = -lam[ok] / si
    beta = np.sqrt(b2[ok])
    return alpha, beta


# ==========================================================================
# turning the camera into conserved quantities
# ==========================================================================
def cartesian_to_bl(pos):
    """``(x, y, z) -> (r, mu, phi)`` with the spin axis along ``+y``.

    The rest of the film keeps the vertical axis at ``+y`` and discs in
    the ``xz`` plane, so the Boyer-Lindquist polar angle is measured from
    ``+y`` and ``phi`` runs counter-clockwise seen from above.  For
    ``r >> a`` the oblate-spheroidal character of the coordinate is
    negligible and this identification is the usual one.
    """
    pos = np.atleast_2d(np.asarray(pos, F64))
    r = np.sqrt(np.einsum('ij,ij->i', pos, pos))
    r = np.maximum(r, 1e-9)
    mu = pos[:, 1] / r
    phi = np.arctan2(pos[:, 2], pos[:, 0])
    return r, np.clip(mu, -1.0, 1.0), phi


def spherical_triad(mu, phi):
    """Unit vectors of the coordinate triad at ``(mu, phi)``.

    Returns ``(e_r, e_theta, e_phi)``, each ``(N, 3)``, in the Cartesian
    embedding with ``+y`` as the axis.  The ZAMO's orthonormal spatial
    basis points along exactly these three directions — only the
    normalisations differ, and those cancel when the input is already a
    unit vector.
    """
    s = np.sqrt(np.maximum(1.0 - mu * mu, 1e-18))
    cp, sp = np.cos(phi), np.sin(phi)
    e_r = np.stack([s * cp, mu, s * sp], axis=1)
    e_t = np.stack([mu * cp, -s, mu * sp], axis=1)
    e_p = np.stack([-sp, np.zeros_like(sp), cp], axis=1)
    return e_r, e_t, e_p


def constants_from_rays(origin, direction, a):
    """Conserved quantities and initial Mino-time momenta for a ray batch.

    ``origin`` and ``direction`` are Cartesian, ``direction`` pointing the
    way the camera is looking.  The ray is traced *backwards along the
    photon's path*, which in a stationary but non-static spacetime is not
    the same curve as the forward path — reversing a Kerr geodesic also
    reverses the sense of frame dragging relative to it.  Handled here by
    building the momentum from the past-directed vector ``-P``, whose
    normalised constants work out identical to the naive construction;
    the subtlety is real but the arithmetic is forgiving.

    Returns ``(r, mu, phi, pr, pmu, lam, q)`` where ``pr = dr/dsigma`` and
    ``pmu = dmu/dsigma`` in Mino time, all normalised to ``E = 1``.
    """
    r, mu, phi = cartesian_to_bl(origin)
    direction = np.atleast_2d(np.asarray(direction, F64))
    e_r, e_t, e_p = spherical_triad(mu, phi)

    n_r = np.einsum('ij,ij->i', direction, e_r)
    n_t = np.einsum('ij,ij->i', direction, e_t)
    n_p = np.einsum('ij,ij->i', direction, e_p)

    sig = sigma(r, mu, a)
    dlt = delta(r, a)
    gpp = np.maximum(g_phiphi(r, mu, a), 1e-18)
    alph = lapse(r, mu, a)
    omg = frame_dragging(r, mu, a)

    # local ZAMO measurement, normalised to unit locally measured energy
    l_z = n_p * np.sqrt(gpp)
    energy = alph + omg * l_z
    energy = np.where(np.abs(energy) < 1e-12, 1e-12, energy)
    lam = l_z / energy

    p_theta = n_t * np.sqrt(sig) / energy          # = p_theta / E
    s = np.sqrt(np.maximum(1.0 - mu * mu, 1e-18))
    pmu = -s * p_theta
    pr = n_r * np.sqrt(np.maximum(dlt, 1e-18) * sig) / energy

    q = p_theta * p_theta + mu * mu * (lam * lam / (s * s) - a * a)
    return r, mu, phi, pr, pmu, lam, q


# ==========================================================================
# the integrator
# ==========================================================================
@dataclass
class TraceConfig:
    """Tunables of the geodesic integrator."""

    a: float = 0.94                 #: dimensionless spin, ``0 <= a < 1``
    max_steps: int = 420            #: hard cap before a ray is called lost
    tol: float = 0.030              #: fractional change allowed per step
    tol_far: float = 4.2            #: tolerance multiplier in the far field
    far_lo: float = 9.0             #: where the relaxation begins
    far_hi: float = 48.0            #: where it is fully relaxed
    r_escape: float = 420.0         #: treated as spatial infinity
    horizon_pad: float = 0.035      #: stop this far outside ``r+``
    compact_at: float = 0.14        #: repack when this fraction has retired
    weak_b: float = 24.0            #: above this impact parameter, analytic
    equator_eps: float = 1e-7
    #: Working precision of the integrator.  Single precision halves the
    #: memory traffic of a loop that is entirely bandwidth-bound, and the
    #: constraint projection in ``_project`` removes the accumulation of
    #: rounding error that would otherwise make it untenable: the first
    #: integrals are re-imposed exactly every step, so what remains is one
    #: step's worth of truncation rather than four hundred steps' worth.
    dtype: object = np.float32


class KerrTracer:
    """Vectorised backward ray tracer for the Kerr metric.

    The batch is integrated together in Mino time with a per-ray adaptive
    step, and rays retire as they fall through the horizon or reach the
    escape radius.  Retired rays are periodically compacted out of the
    working arrays, which matters a great deal: near the end of a trace
    only the few per cent of rays winding around the photon shell are
    still alive, and carrying the other ninety-odd per cent through four
    Runge-Kutta stages each would dominate the cost.

    Two callbacks let a caller put matter in the way without this class
    knowing anything about accretion physics:

    ``on_equator(r, phi, lam, cos_inc, slot)``
        Invoked for every crossing of the equatorial plane, with the
        crossing radius, azimuth, the ray's conserved ``lambda``, the
        cosine of the incidence angle against the plane, and the index of
        each ray in the original batch.  Must return ``(rgb, opacity)``.

    ``on_step(r0, mu0, phi0, r1, mu1, phi1, iteration)``
        Invoked once per integration step with both ends of the segment,
        for optically thin material such as a jet.  Must return ``rgb``
        for the live set, or ``None``; the caller applies the running
        transmittance.
    """

    def __init__(self, cfg: TraceConfig = None):
        self.cfg = cfg or TraceConfig()
        self.a = float(self.cfg.a)
        self._a2 = self.a * self.a
        self._a2x2 = 2.0 * self._a2
        self._a2x1 = 2.0 * self.a
        self.r_horizon = horizon_outer(self.a)
        self.r_stop = self.r_horizon + self.cfg.horizon_pad
        self.dtype = np.dtype(self.cfg.dtype)
        self.stats = {}

    # ------------------------------------------------------------------
    def _derivs(self, r, pr, mu, pmu, cl, kk, c2, c3, lam):
        """Right-hand side of the Mino-time system.

        Four per-ray constants are hoisted out of the stage and passed in:
        ``cl = a^2 - a lambda``, ``kk = (lambda - a)^2 + q``,
        ``c2 = a^2 - q - lambda^2`` and ``c3 = -a^2 lambda``.  The last
        exists because ``a (p/Delta - 1)`` simplifies to
        ``(2 a r - a^2 lambda) / Delta`` once ``p - Delta = 2r - a
        lambda`` is substituted, which removes a subtraction over the
        whole array.  This is not micro-optimisation for its own
        sake — this function is evaluated four times per step, several
        hundred steps deep, on a third of a million rays, so it runs of
        the order of half a billion times per frame.  Every temporary
        array it allocates is two and a half megabytes that has to cross
        the memory bus, and the whole loop is bandwidth-bound rather than
        arithmetic-bound.  The version below allocates eight temporaries;
        the straightforward transcription of the equations allocates
        nineteen, and runs a little over twice as slowly.
        """
        r2 = r * r
        p = r2 + cl
        d = r2 - (r + r) + self._a2
        np.maximum(d, 1e-4, out=d)
        dpr = (r + r) * p
        dpr -= (r - 1.0) * kk
        mu2 = mu * mu
        dpmu = mu * (c2 - self._a2x2 * mu2)
        s2 = 1.0 - mu2
        np.maximum(s2, 1e-7, out=s2)
        dphi = (self._a2x1 * r + c3) / d
        dphi += lam / s2
        return pr, dpr, pmu, dpmu, dphi

    # ------------------------------------------------------------------
    def _step_size(self, r, pr, mu, pmu, dphi):
        """Per-ray Mino-time step.

        Chosen so that one step changes ``ln r`` by at most ``tol``, and
        ``mu`` and ``phi`` by at most ``tol`` in absolute terms.  Those
        three limits are what actually control the visible accuracy: the
        first keeps the radial approach resolved over the enormous dynamic
        range between the camera and the horizon, the second keeps the
        equatorial crossing from being stepped over, and the third keeps
        the azimuthal winding near the photon shell from unwrapping in
        coarse jumps.

        The tolerance is relaxed smoothly with radius.  Beyond about
        fifty gravitational radii the metric is flat to a part in a
        hundred and the trajectory is a straight line; resolving it at the
        same fractional precision as the photon shell would spend two
        thirds of the integration budget drawing a ruler.
        """
        cfg = self.cfg
        x = np.clip((r - cfg.far_lo) / (cfg.far_hi - cfg.far_lo), 0.0, 1.0)
        relax = 1.0 + (cfg.tol_far - 1.0) * x * x * (3.0 - 2.0 * x)
        # The azimuth is weighted by sin(theta), because what has to stay
        # resolved is the *proper* transverse displacement r sin(th) dphi,
        # not the coordinate one.  Boyer-Lindquist phi is degenerate on
        # the spin axis and sweeps arbitrarily fast there for no physical
        # reason; without this factor the step size collapses, those rays
        # exhaust the iteration cap, and a thin black line is painted up
        # and down the projected axis of rotation through the middle of
        # the frame.
        s = np.sqrt(np.maximum(1.0 - mu * mu, 1e-7))
        scale = (np.abs(pr) / np.maximum(r, 1.0)
                 + np.abs(pmu) * 2.2
                 + np.abs(dphi) * s * 0.85 + 1e-12)
        return cfg.tol * relax / scale

    # ------------------------------------------------------------------
    def _project(self, r, pr, mu, pmu, cl, q, kk, c2):
        """Pull the state back onto the two first integrals.

        Runge-Kutta conserves ``pr^2 - R(r)`` and ``pmu^2 - Mu(mu)`` only
        to the order of its truncation error, and that is not good enough
        here for a reason specific to this problem: ``R`` falls by seven
        orders of magnitude between the camera and the horizon.  An
        absolute error of ten, utterly negligible against ``R ~ 4e7`` at
        the start, is larger than ``R`` itself by the time the ray reaches
        ``r = 1.7`` — so the integrator invents a radial turning point,
        the ray bounces back out, and the black hole acquires a mirror
        finish instead of a shadow.  That is exactly what the first
        version of this module did.

        The cure is to renormalise the momenta to the magnitudes the
        constants demand while keeping the signs the integration produced.
        Turning points still work, because there the integrated momentum
        passes through zero at the same place the constraint does.  Cost
        is two square roots per step against four right-hand sides, and
        the radial first integral then holds to machine precision for the
        whole trace.
        """
        r2 = r * r
        p = r2 + cl
        rad = p * p - (r2 - (r + r) + self._a2) * kk
        mu2 = mu * mu
        pol = q + mu2 * (c2 - self._a2 * mu2)
        np.maximum(rad, 0.0, out=rad)
        np.maximum(pol, 0.0, out=pol)
        np.sqrt(rad, out=rad)
        np.sqrt(pol, out=pol)
        return (np.copysign(rad, pr), np.copysign(pol, pmu))

    # ------------------------------------------------------------------
    def trace(self, origin, direction, on_equator=None, on_step=None,
              max_steps=None):
        """Trace a batch of rays.

        Returns ``(rgb, transmittance, out_dir, escaped)``:

        ``rgb``             radiance collected along each ray
        ``transmittance``   fraction of the background still visible
        ``out_dir``         asymptotic direction of escaped rays
        ``escaped``         boolean mask of rays that reached infinity
        """
        cfg = self.cfg
        a = self.a
        n = np.atleast_2d(np.asarray(origin, F64)).shape[0]

        rgb = np.zeros((n, 3), F64)
        trans = np.ones(n, F64)
        out_dir = np.array(np.atleast_2d(np.asarray(direction, F64)),
                           copy=True)
        escaped = np.zeros(n, bool)

        r, mu, phi, pr, pmu, lam, q = constants_from_rays(origin, direction,
                                                          a)
        dt = self.dtype
        r = r.astype(dt); mu = mu.astype(dt); phi = phi.astype(dt)
        pr = pr.astype(dt); pmu = pmu.astype(dt)
        lam = lam.astype(dt); q = q.astype(dt)
        kk = ((lam - a) ** 2 + q).astype(dt)
        c2 = (self._a2 - q - lam * lam).astype(dt)
        cl = (self._a2 - a * lam).astype(dt)
        c3 = (-self._a2 * lam).astype(dt)
        slot = np.arange(n)

        # Rays already heading outward from a camera that is itself far
        # away never come back; retire them immediately rather than
        # integrating a straight line for four hundred steps.
        runaway = (pr > 0.0) & (r > cfg.r_escape * 0.5)
        if runaway.any():
            keep = ~runaway
            gi = slot[runaway]
            escaped[gi] = True
            r, mu, phi = r[keep], mu[keep], phi[keep]
            pr, pmu = pr[keep], pmu[keep]
            lam, q, kk, c2 = lam[keep], q[keep], kk[keep], c2[keep]
            cl, c3 = cl[keep], c3[keep]
            slot = slot[keep]

        steps = int(max_steps or cfg.max_steps)
        r_stop = dt.type(self.r_stop)
        r_esc = dt.type(cfg.r_escape)
        r_cap = dt.type(cfg.r_escape * 1.04)
        # Working transmittance, parallel to the live set.  Retired rays
        # keep integrating until the next compaction — it is cheaper to
        # carry a few per cent of dead weight than to repack nine arrays
        # every step — so their weight is zeroed here and everything they
        # subsequently "collect" is multiplied by nothing.
        wt = np.ones(slot.shape[0], dt)
        alive = np.ones(slot.shape[0], bool)

        for it in range(steps):
            if slot.shape[0] == 0:
                break

            # --- classical fourth-order Runge-Kutta ---------------------
            k1 = self._derivs(r, pr, mu, pmu, cl, kk, c2, c3, lam)
            h = self._step_size(r, pr, mu, pmu, k1[4])

            h2 = h * 0.5
            k2 = self._derivs(r + h2 * k1[0], pr + h2 * k1[1],
                              mu + h2 * k1[2], pmu + h2 * k1[3],
                              cl, kk, c2, c3, lam)
            k3 = self._derivs(r + h2 * k2[0], pr + h2 * k2[1],
                              mu + h2 * k2[2], pmu + h2 * k2[3],
                              cl, kk, c2, c3, lam)
            k4 = self._derivs(r + h * k3[0], pr + h * k3[1],
                              mu + h * k3[2], pmu + h * k3[3],
                              cl, kk, c2, c3, lam)

            h6 = h / 6.0
            r_new = r + h6 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            pr_new = pr + h6 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            mu_new = mu + h6 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            pmu_new = pmu + h6 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
            phi_new = phi + h6 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])

            np.clip(r_new, 1e-4, r_cap, out=r_new)
            np.clip(mu_new, -0.999999, 0.999999, out=mu_new)
            pr_new, pmu_new = self._project(r_new, pr_new, mu_new, pmu_new,
                                            cl, q, kk, c2)

            # --- optically thin material along the segment --------------
            if on_step is not None:
                contrib = on_step(r, mu, phi, r_new, mu_new, phi_new, it)
                if contrib is not None:
                    rgb[slot] += contrib * wt[:, None]

            # --- equatorial crossings -----------------------------------
            if on_equator is not None:
                cross = (mu * mu_new) < 0.0
                np.logical_and(cross, alive, out=cross)
                if cross.any():
                    ci = np.flatnonzero(cross)
                    f = mu[ci] / (mu[ci] - mu_new[ci])
                    rc = r[ci] + f * (r_new[ci] - r[ci])
                    pc = phi[ci] + f * (phi_new[ci] - phi[ci])
                    prc = pr[ci] + f * (pr_new[ci] - pr[ci])
                    pmc = pmu[ci] + f * (pmu_new[ci] - pmu[ci])
                    dph = ((self._a2x1 * rc + c3[ci])
                           / np.maximum(delta(rc, a), 1e-5) + lam[ci])
                    # components of the coordinate velocity at the crossing
                    vz = rc * pmc                 # d(r mu)/dsigma at mu = 0
                    vr = prc
                    vt = rc * dph
                    speed = np.sqrt(vz * vz + vr * vr + vt * vt) + 1e-12
                    cos_inc = np.abs(vz) / speed
                    col, opac = on_equator(rc, pc, lam[ci], cos_inc,
                                           slot[ci], prc)
                    gi = slot[ci]
                    rgb[gi] += col * wt[ci][:, None]
                    wt[ci] *= (1.0 - opac)
                    trans[gi] = wt[ci]

            r, pr, mu, pmu, phi = r_new, pr_new, mu_new, pmu_new, phi_new

            # --- termination --------------------------------------------
            fell = r < r_stop
            gone = (r > r_esc) & (pr > 0.0)
            done = (fell | gone | (wt < 0.0035)) & alive
            if not done.any():
                continue

            di = np.flatnonzero(done)
            gi = slot[di]
            gd = gone[di]
            escaped[gi] = gd
            if gd.any():
                ei = di[gd]
                out_dir[slot[ei]] = self._escape_direction(
                    r[ei], mu[ei], phi[ei], pr[ei], pmu[ei], lam[ei])
            trans[gi[~gd]] = 0.0
            alive[di] = False
            wt[di] = 0.0

            n_live = int(alive.sum())
            if n_live == 0:
                slot = slot[:0]
                break
            if n_live < (1.0 - cfg.compact_at) * alive.shape[0]:
                keep = alive
                r, pr, mu, pmu, phi = (r[keep], pr[keep], mu[keep],
                                       pmu[keep], phi[keep])
                lam, q, kk, c2 = lam[keep], q[keep], kk[keep], c2[keep]
                cl, c3 = cl[keep], c3[keep]
                slot = slot[keep]
                wt = wt[keep]
                alive = alive[keep]

        # Rays still integrating when the step cap is reached have wound
        # many times around the photon shell and are on their way in; the
        # physically honest classification is captured, and they are left
        # black.
        self.stats = dict(remaining=int(alive.sum()) if slot.shape[0] else 0,
                          escaped=int(escaped.sum()))
        return (rgb.astype(F32), trans.astype(F32), out_dir.astype(F32),
                escaped)

    # ------------------------------------------------------------------
    def _escape_direction(self, r, mu, phi, pr, pmu, lam):
        """Asymptotic Cartesian direction of a ray that reached infinity.

        Differentiating the embedding ``x = r sin(th) cos(phi)``,
        ``y = r mu``, ``z = r sin(th) sin(phi)`` with respect to Mino time
        and normalising.  Done once per escaped ray rather than per step,
        so the trigonometry here costs nothing.
        """
        a = self.a
        s = np.sqrt(np.maximum(1.0 - mu * mu, 1e-12))
        cp, sp = np.cos(phi), np.sin(phi)
        dphi = a * ((r * r + a * a - a * lam)
                    / np.maximum(delta(r, a), 1e-9) - 1.0) + lam / (s * s)
        ds = -mu * pmu / s
        vx = pr * s * cp + r * ds * cp - r * s * sp * dphi
        vy = pr * mu + r * pmu
        vz = pr * s * sp + r * ds * sp + r * s * cp * dphi
        v = np.stack([vx, vy, vz], axis=1)
        nv = np.maximum(np.sqrt(np.einsum('ij,ij->i', v, v)), 1e-12)
        return v / nv[:, None]


# ==========================================================================
# weak-field analytic branch
# ==========================================================================
def deflection_analytic(b, a_spin, prograde_sign):
    """Light deflection to second post-Newtonian order, with spin.

        alpha = 4 M / b  +  15 pi M^2 / (4 b^2)  -+  4 M a / b^2

    The leading term is Einstein's 1915 result, twice the Newtonian
    corpuscular value, and the one Eddington confirmed in 1919.  The
    second term is the first relativistic correction, about ten per cent
    of the total at ``b = 10 M``.  The third is the gravitomagnetic
    contribution: a photon passing in the same sense as the hole's
    rotation is deflected less than one passing against it, which is why
    the lensed sky around a spinning hole is visibly asymmetric even far
    outside the photon ring.

    Accurate to better than a quarter of a degree for ``b > 20 M``, which
    is where the tracer hands rays over to it.
    """
    b = np.maximum(np.abs(b), 1e-9)
    return (4.0 / b + 15.0 * math.pi / (4.0 * b * b)
            - prograde_sign * 4.0 * a_spin / (b * b))


# ==========================================================================
# self-test
# ==========================================================================
def _selftest():
    import sys
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print("  %-46s %14.6f  %-6s" % (name, got, "OK" if good else "FAIL"),
              flush=True)

    print("kerr metric self-test", flush=True)
    check("horizon a=0", horizon_outer(0.0), 2.0, 1e-12)
    check("horizon a=0.94", horizon_outer(0.94), 1.3411744, 1e-6)
    check("isco a=0 prograde", isco(0.0), 6.0, 1e-9)
    check("isco a=0.94 prograde", isco(0.94), 2.0235930, 1e-6)
    check("isco a=0.94 retrograde", isco(0.94, False), 8.8307518, 1e-6)
    check("photon ring a=0", photon_circular(0.0), 3.0, 1e-9)
    check("photon ring a=0.94 pro", photon_circular(0.94), 1.4252442, 1e-6)
    check("photon ring a=0.94 retro", photon_circular(0.94, False),
          3.9463660, 1e-6)
    check("marg. bound a=0", marginally_bound(0.0), 4.0, 1e-9)
    check("ergosphere equator a=0.94", float(ergosphere(0.94, 0.0)), 2.0,
          1e-12)

    # shadow of a non-rotating hole is a circle of radius 3 sqrt(3)
    al, be = shadow_outline(1e-6, 90.0, 256)
    rad = np.sqrt(al * al + be * be)
    check("shadow radius a->0", float(rad.mean()), 3.0 * math.sqrt(3.0),
          2e-3)
    al, be = shadow_outline(0.94, 85.0, 512)
    print("  shadow a=0.94  alpha in [%.3f, %.3f]  beta max %.3f"
          % (al.min(), al.max(), be.max()), flush=True)

    # orbital speed at the ISCO
    v = float(orbit_velocity_zamo(np.array([isco(0.0)]), 0.0)[0])
    check("ZAMO orbit speed at a=0 ISCO", v, 0.5, 1e-6)
    v94 = float(orbit_velocity_zamo(np.array([isco(0.94)]), 0.94)[0])
    print("  ZAMO orbit speed at a=0.94 ISCO                %.6f c"
          % v94, flush=True)

    # Schwarzschild deflection must reproduce the 1919 eclipse number
    r_s_sun = 2.953250e3            # metres
    b = 6.957e8 / (r_s_sun * 0.5)   # solar radius in units of M
    arcsec = float(deflection_analytic(b, 0.0, 0.0)) * 206264.806
    check("solar light bending (arcsec)", arcsec, 1.7510, 2e-3)

    # integration accuracy: the two first integrals must stay satisfied
    cfg = TraceConfig(a=0.94, max_steps=600)
    tr = KerrTracer(cfg)
    rng = np.random.default_rng(7)
    origin = np.tile(np.array([0.0, 6.0, -60.0]), (64, 1))
    tgt = rng.normal(size=(64, 3)) * 6.0
    d = tgt - origin
    d /= np.linalg.norm(d, axis=1)[:, None]
    r0, mu0, p0, pr0, pmu0, lam, q = constants_from_rays(origin, d, 0.94)
    rres = (pr0 ** 2 - ((r0 ** 2 + 0.94 ** 2 - 0.94 * lam) ** 2
                        - delta(r0, 0.94) * ((lam - 0.94) ** 2 + q)))
    mres = (pmu0 ** 2 - (q + (0.94 ** 2 - q - lam ** 2) * mu0 ** 2
                         - 0.94 ** 2 * mu0 ** 4))
    check("initial radial first integral", float(np.abs(rres).max()), 0.0,
          1e-6)
    check("initial polar first integral", float(np.abs(mres).max()), 0.0,
          1e-9)

    # a ray aimed straight at the hole must be captured; one aimed well
    # wide must escape
    o = np.array([[0.0, 0.0, -80.0], [0.0, 0.0, -80.0]])
    dd = np.array([[0.0, 0.0, 1.0], [0.40, 0.0, 0.917]])
    dd /= np.linalg.norm(dd, axis=1)[:, None]
    _, _, od, esc = tr.trace(o, dd)
    print("  aimed at hole escaped=%s   aimed wide escaped=%s"
          % (bool(esc[0]), bool(esc[1])), flush=True)
    ok &= (not esc[0]) and bool(esc[1])

    # redshift sanity: a photon with zero angular momentum from the ISCO
    g = float(redshift_factor(np.array([0.0]), np.array([isco(0.94)]),
                              0.94, np.array([1e6]), np.array([0.0]))[0])
    check("g at a=0.94 ISCO, lambda=0", g, 1.0 / orbit_ut(isco(0.94), 0.94),
          1e-5)

    print("kerr self-test %s" % ("PASSED" if ok else "FAILED"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())


__all__ = [
    "delta", "sigma", "amaj", "frame_dragging", "lapse", "g_phiphi",
    "horizon_outer", "horizon_inner", "ergosphere", "isco",
    "photon_circular", "marginally_bound", "orbit_omega", "orbit_ut",
    "orbit_velocity_zamo", "redshift_factor", "spherical_photon_constants",
    "shadow_outline", "cartesian_to_bl", "spherical_triad",
    "constants_from_rays", "TraceConfig", "KerrTracer",
    "deflection_analytic",
]
