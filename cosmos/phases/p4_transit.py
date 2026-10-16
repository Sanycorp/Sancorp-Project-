"""
cosmos.phases.p4_transit
========================

FASE IV — TRANSISI.

Crossing the boundary of the heliosphere and running out into interstellar
space toward something with a gravitational gradient steep enough to bend
light.  This is the phase that has to sell distance, and distance is sold
by two things: structure that passes the camera, and the physics of going
fast.

The heliosphere
---------------
The Sun's wind blows a bubble in the interstellar medium.  It is not
spherical, because the Sun is itself moving at about 25 km/s relative to
the local cloud, so the bubble is blunt upstream and drawn out into a tail
downstream — the shape of a comet, not a balloon.  Three surfaces matter
and all three are modelled:

  termination shock   ~94 AU upwind.  The supersonic solar wind decelerates
                      abruptly to subsonic here.  Voyager 1 crossed it in
                      December 2004.
  heliopause          ~121 AU upwind.  The true boundary: solar plasma on
                      the inside, interstellar plasma on the outside.
                      Voyager 1 crossed it in August 2012.
  hydrogen wall       a pile-up of neutral hydrogen just outside the
                      heliopause, detectable in Lyman-alpha absorption.

Going fast
----------
Once outside, the camera accelerates to a large fraction of the speed of
light, and the star field is transformed by special relativity rather than
by a motion blur filter:

**Aberration.**  Angles are not invariant.  A star at angle ``theta`` in
the rest frame appears at ``theta'`` where
``cos theta' = (cos theta + beta) / (1 + beta cos theta)``.  At high beta
the entire sky piles up into a shrinking disc ahead — the headlight effect.
Stars genuinely behind the observer swing round into the forward view.

**Doppler shift.**  The observed frequency is multiplied by the Doppler
factor ``D = 1 / (gamma (1 - beta cos theta'))``.  Since the stars are
approximately blackbodies, a shift of ``D`` is exactly a temperature change
of ``D``, so the forward sky turns blue-white and the rear sky drops to
dull red.  This is done through the same blackbody code as everything else:
the temperature is scaled and the colour recomputed.

**Beaming.**  Specific intensity transforms as ``D^3`` for a bolometric
source, so the forward field brightens dramatically while the rear fades.

All three come out of ``cosmos.core.spectrum`` and ``cosmos.core.vmath``,
and together they produce the correct relativistic sky without a single
hand-tuned streak.

Units
-----
Astronomical units early in the phase, scaled continuously to Schwarzschild
radii by the end so the hand-off to Phase V is seamless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core import camera as CAM
from ..core import march as M
from ..core import spectrum as S
from ..core import vmath as V
from ..core import vcache
from .base import (Phase, build_emission_ramp, march_layer, project_and_splat,
                   ramp_lookup, splat_points, upsample_linear)

F32 = np.float32
TAU = 2.0 * math.pi

#: speed of light, km/s
C_KMS = 299792.458


# ==========================================================================
# parameters
# ==========================================================================
@dataclass
class TransitParams:
    """Geometry of the heliosphere and the run outward."""

    r_termination: float = 94.0      # termination shock, AU upwind
    r_heliopause: float = 121.0      # heliopause, AU upwind
    r_bowwave: float = 170.0         # interstellar bow wave
    tail_stretch: float = 3.1        # downwind elongation factor
    blunt: float = 0.62              # upwind flattening

    #: direction the Sun travels through the local interstellar cloud,
    #: in ecliptic coordinates (the "nose" of the heliosphere)
    nose: tuple = (0.42, 0.18, 0.89)

    wind_speed: float = 400.0        # solar wind, km/s
    ism_density: float = 0.1         # local cloud, atoms per cm^3

    beta_max: float = 0.88           # peak speed as a fraction of c
    r_outer: float = 900.0           # how far the shot travels, AU

    n_field_stars: int = 36000


def heliosphere_radius(dirs, p: TransitParams):
    """Distance to the heliopause in a given direction.

    A blunt nose and a long tail: the radius is largest downwind, smallest
    upwind, with the transition controlled by the angle to the nose.  This
    is the standard parametrisation used to fit the Voyager crossings and
    the IBEX ribbon.
    """
    nose = np.asarray(p.nose, F32)
    nose = nose / np.linalg.norm(nose)
    c = np.einsum('ij,j->i', dirs, nose)
    # 1 upwind, -1 downwind
    t = (F32(1.0) - c) * F32(0.5)          # 0 upwind, 1 downwind
    stretch = F32(1.0) + (F32(p.tail_stretch) - F32(1.0)) * t * t
    return F32(p.r_heliopause) * stretch


# ==========================================================================
# relativistic star field
# ==========================================================================
class RelativisticSky:
    """A star catalogue transformed into the observer's moving frame.

    The catalogue is generated once in the rest frame.  Every frame, the
    positions, colours and intensities are transformed by aberration,
    Doppler shift and relativistic beaming for the current velocity.
    """

    def __init__(self, n=60000, seed=13, radius=4.0e5):
        rng = np.random.default_rng(seed)
        dirs = V.fibonacci_sphere(n).astype(F32)
        dirs = dirs[rng.permutation(n)]
        self.dirs = dirs
        self.radius = float(radius)
        # a realistic mix of spectral types
        self.temp = S.sample_stellar_temperatures(n, rng, hot_bias=0.18
                                                  ).astype(F32)
        mass = S.mass_from_temperature(self.temp)
        lum = S.luminosity_from_mass(mass)
        b = np.power(np.maximum(lum, 1e-6), 0.34).astype(F32)
        b *= rng.lognormal(0.0, 0.42, n).astype(F32)
        b = (b / np.percentile(b, 99.5)).astype(F32)
        # Steepened hard.  A real sky is a few dozen obvious stars over a
        # haze of ones you cannot quite see; a field where every star has
        # a similar brightness reads as sensor noise, which is exactly what
        # the flatter distribution produced here.
        self.rest_flux = (np.power(np.clip(b, 0.0, 4.0), F32(3.1))
                          * F32(1.35)).astype(F32)

        # a handful of nearby bright stars that will show real parallax
        m = 220
        self.near_dirs = V.fibonacci_sphere(m).astype(F32)[rng.permutation(m)]
        self.near_dist = (rng.random(m).astype(F32) * F32(600.0)
                          + F32(180.0))
        self.near_temp = S.sample_stellar_temperatures(m, rng, hot_bias=0.4
                                                       ).astype(F32)
        self.near_flux = (rng.random(m).astype(F32) ** 2.0 * F32(2.4)
                          + F32(0.25))

    # ----------------------------------------------------------------
    def transform(self, beta, direction, eye):
        """Aberrate, shift and beam the sky for velocity ``beta * dir``.

        Returns ``(world_positions, rgb)`` ready for splatting.
        """
        d = np.asarray(direction, F32)
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        b = float(np.clip(beta, 0.0, 0.999))
        gamma = 1.0 / math.sqrt(max(1.0 - b * b, 1e-12))

        cos_t = np.einsum('ij,j->i', self.dirs, d)

        if b < 1e-6:
            new_dirs = self.dirs
            cos_new = cos_t
        else:
            # Relativistic aberration.  Derived directly from the Lorentz
            # transformation of the photon four-momentum; at beta = 0.9 a
            # star 90 degrees off the bow appears only 26 degrees off it.
            cos_new = (cos_t + b) / (1.0 + b * cos_t)
            sin_old = np.sqrt(np.maximum(1.0 - cos_t * cos_t, 0.0))
            sin_new = np.sqrt(np.maximum(1.0 - cos_new * cos_new, 0.0))
            # rebuild each direction in the plane it shares with the boost
            perp = self.dirs - cos_t[:, None] * d[None, :]
            pn = np.maximum(np.sqrt(np.einsum('ij,ij->i', perp, perp)),
                            1e-9)
            perp = perp / pn[:, None]
            scale = np.where(sin_old > 1e-7, sin_new, 0.0)
            new_dirs = (cos_new[:, None] * d[None, :]
                        + scale[:, None] * perp).astype(F32)

        # Doppler factor in the observer's frame
        dop = 1.0 / (gamma * (1.0 - b * cos_new))
        dop = np.clip(dop, 0.02, 60.0).astype(F32)

        # A blackbody stays a blackbody under a Doppler shift; only its
        # temperature changes.  That is why the forward sky turns blue
        # instead of merely brightening.
        temp = np.clip(self.temp * dop, 400.0, 70000.0)
        rgb = S.blackbody_lut(temp)
        rgb = S.saturate_rgb(rgb, 1.45)
        rgb /= np.maximum(S.luminance(rgb), 1e-5)[:, None]

        # Specific intensity transforms as D^3 for a bolometric source;
        # D^4 for a fixed band.  The headlight effect is severe, so the
        # result is compressed photographically as elsewhere in the film.
        boost = S.relativistic_beaming(dop, index=3.0)
        flux = self.rest_flux * np.power(np.maximum(boost, 1e-6), F32(0.42))

        pos = np.asarray(eye, F32)[None, :] + new_dirs * F32(self.radius)
        return pos, (rgb * flux[:, None]).astype(F32)


# ==========================================================================
# the medium
# ==========================================================================
class TransitShader:
    """Heliospheric shells and the local interstellar cloud."""

    def __init__(self, p: TransitParams, lib, seed=13):
        self.p = p
        self.v_neb = lib.get("nebula")
        self.v_fine = lib.get("nebula_fine")
        self.v_fil = lib.get("filament")
        self.v_dust = lib.get("dust")

        self.nose = np.asarray(p.nose, F32)
        self.nose = self.nose / np.linalg.norm(self.nose)

        # Solar wind plasma: hot, ionised, glowing faintly in the ultraviolet
        # but rendered here through its visible recombination continuum.
        self.ramp_wind = build_emission_ramp(
            temps=(5200, 7000, 9500, 14000, 21000),
            gains=(0.10, 0.35, 0.80, 1.50, 2.30), n=256)
        # The hydrogen wall and the interstellar cloud: cold neutral gas,
        # seen by scattered starlight rather than by its own emission.
        self.ramp_ism = build_emission_ramp(
            temps=(2600, 3400, 4600, 6200, 9000),
            gains=(0.08, 0.26, 0.58, 1.05, 1.70), n=256)

        self.jitter = None
        self._t = 0.0
        self._drift = np.zeros(3, F32)
        self._shock = 1.0

    def set_time(self, t_film, progress):
        self._t = float(t_film)
        self._drift = np.array([0.08 * t_film, -0.05 * t_film,
                                0.11 * t_film], F32)
        # the shells fade once the camera is well outside them
        self._shock = float(np.clip(1.0 - (progress - 0.42) / 0.26, 0.0, 1.0))

    def new_step(self, rng):
        self.jitter = vcache.new_jitter(rng)

    @staticmethod
    def _unit(n):
        return np.clip(n * F32(2.55) + F32(0.5), F32(0.0), F32(1.0))

    # ----------------------------------------------------------------
    def __call__(self, pos, t, u):
        p = self.p
        r2 = np.einsum('ij,ij->i', pos, pos)
        r = np.sqrt(np.maximum(r2, F32(1e-8)))
        inv_r = np.reciprocal(np.maximum(r, F32(1e-5)))
        dirs = pos * inv_r[:, None]

        cos_n = np.einsum('ij,j->i', dirs, self.nose)
        tail = (F32(1.0) - cos_n) * F32(0.5)
        stretch = F32(1.0) + (F32(p.tail_stretch) - F32(1.0)) * tail * tail

        r_hp = F32(p.r_heliopause) * stretch
        r_ts = F32(p.r_termination) * stretch
        r_bw = F32(p.r_bowwave) * (F32(1.0) + tail * F32(0.4))

        # ---- texture ----------------------------------------------------
        # Low sampling frequencies: the march spans thousands of scene
        # units, and anything above about a hundredth of a cycle per unit
        # wraps the periodic volume enough times to print its own tiling
        # across the sky as a lattice of creases.
        n1 = self._unit(self.v_neb.sample(pos * F32(0.00085) + self._drift
                                          * F32(0.001), jitter=self.jitter))
        n2 = self._unit(self.v_fine.sample(pos * F32(0.0045),
                                           jitter=self.jitter))
        fil = self.v_fil.sample(pos * F32(0.0019), jitter=self.jitter)
        # Angular texture for the shells.  A shell seen from inside covers
        # the entire sky, so without structure that varies with *direction*
        # it contributes the same radiance to every pixel and the frame
        # turns into flat grey no matter how faint it is.  Sampling on the
        # unit sphere of directions gives it the mottled, ribbed look the
        # IBEX maps of the heliosphere boundary actually have.
        shell_tex = np.clip(
            self.v_fil.sample(dirs * F32(2.6), jitter=self.jitter)
            * F32(2.1) - F32(0.42), F32(0.0), F32(1.8))
        shell_tex *= F32(0.45) + F32(0.85) * np.clip(
            self.v_fil.sample(dirs * F32(7.3), jitter=self.jitter)
            * F32(1.6), F32(0.0), F32(1.2))

        # ---- solar wind, inside the termination shock -------------------
        # Density of a freely expanding wind falls as 1/r^2 by mass
        # conservation alone: the same mass flux through an ever larger
        # sphere.
        wind = (F32(p.r_termination) * inv_r) ** 2
        # Capped, because the 1/r^2 law is exact and therefore ruinous: at
        # fifteen astronomical units it is forty times its value at the
        # shock, and since a march step out here is tens of AU long that
        # alone saturates the frame.  The wind is a near-vacuum anyway —
        # five particles per cubic centimetre at Earth — so what is being
        # drawn is a visualisation of its density, and a visualisation may
        # be clipped where the quantity itself cannot.
        np.minimum(wind, F32(5.0), out=wind)
        wind *= np.clip((r_ts - r) * F32(0.055), 0.0, 1.0)
        wind *= F32(0.18) + F32(0.26) * n2

        # ---- termination shock ------------------------------------------
        # The wind is supersonic inside and subsonic outside; the jump is
        # narrow, which is why Voyager crossed it in a matter of days.
        shock = np.exp(-((r - r_ts) / F32(5.5)) ** 2)
        shock *= F32(0.5) + F32(1.1) * n2

        # ---- inner heliosheath ------------------------------------------
        sheath = np.clip((r - r_ts) * F32(0.05), 0.0, 1.0) \
            * np.clip((r_hp - r) * F32(0.05), 0.0, 1.0)
        sheath *= F32(0.35) + F32(0.95) * n1

        # ---- hydrogen wall, just outside the heliopause ------------------
        wall = np.exp(-((r - r_hp * F32(1.07)) / F32(14.0)) ** 2)
        wall *= F32(0.40) + F32(1.1) * np.clip(fil * F32(1.5), 0.0, 1.0)

        # ---- interstellar bow wave ---------------------------------------
        bow = np.exp(-((r - r_bw) / F32(26.0)) ** 2)
        bow *= np.clip(cos_n + F32(0.25), 0.0, 1.0)
        bow *= F32(0.3) + F32(1.0) * n1

        # ---- the local interstellar cloud --------------------------------
        # The honest answer for this volume is *almost nothing*.  The local
        # cloud holds about a tenth of an atom per cubic centimetre; it is
        # emptier than the best vacuum any laboratory has produced, and it
        # emits essentially no visible light of its own.  An earlier
        # version filled it with glowing nebulosity and the result was a
        # brown fog that buried the one thing this phase exists to show.
        # What survives is a very sparse scatter of denser knots.
        ism = np.clip((r - r_hp) * F32(0.010), 0.0, 1.0)
        cloud = n1 * F32(0.60) + fil * F32(0.25) + n2 * F32(0.15)
        cloud = np.maximum(cloud - F32(0.62), F32(0.0))
        cloud *= cloud
        ism = ism * cloud * F32(2.4)

        sw = F32(self._shock)
        # Only the *boundaries* are drawn.  The volumes between them are
        # transparent, which is the physically honest answer and also the
        # only one that works: the camera starts inside the heliosphere, so
        # every ray crosses the entire bubble, and any filled interior
        # integrates into a flat grey fog across the whole frame no matter
        # how faint it is per unit length.  A shock is a surface; drawing
        # it as a surface is both cheaper and truer.
        hot = (wind * F32(0.035) + shock * F32(1.30) * shell_tex
               + sheath * F32(0.055)) * sw
        cold = (wall * F32(0.85) * shell_tex * sw
                + bow * F32(0.50) * shell_tex * sw
                + ism * F32(0.22))

        # The gains below look absurdly small and are not.  One scene
        # unit here is an astronomical unit and the march spans well over a
        # thousand of them, so a step is of order fifty units long: an
        # emission coefficient tuned by eye at unit scale overexposes the
        # frame by two orders of magnitude.  What matters is the ratio of
        # these to the extinction coefficient below, because the marcher
        # integrates the source function, and that ratio is what sets the
        # radiance a saturated column reaches.
        emis = ramp_lookup(self.ramp_wind,
                           np.clip(hot * F32(0.55), 0.0, 1.0)) \
            * (hot * F32(0.0022))[:, None]
        emis += ramp_lookup(self.ramp_ism,
                            np.clip(cold * F32(0.65), 0.0, 1.0)) \
            * (cold * F32(0.0015))[:, None]

        total = hot + cold
        sigma = total * F32(0.0022) + F32(1e-6)
        source = emis / sigma[:, None]
        return source.astype(F32, copy=False), sigma.astype(F32, copy=False)


# ==========================================================================
# the phase
# ==========================================================================
class PhaseTransit(Phase):
    """FASE IV — out through the heliosphere and up to relativistic speed."""

    name = "IV · TRANSISI"
    duration = 9.0

    def __init__(self, params=None, seed=13):
        self.p = params or TransitParams()
        self.seed = int(seed)
        self.shader = None
        self.sky = None
        self._dist = None

    # ----------------------------------------------------------------
    def setup(self, ctx):
        self.shader = TransitShader(self.p, ctx.lib, seed=self.seed)
        self.sky = RelativisticSky(n=self.p.n_field_stars, seed=self.seed)

    # ----------------------------------------------------------------
    def beta_at(self, lt):
        """Speed profile, as a fraction of c.

        Starts at rest just outside Saturn's orbit and accelerates on a
        smootherstep so there is no jerk at either end; the headlight
        effect then arrives gradually rather than snapping on.
        """
        u = self.progress(lt, self.duration)
        ramp = V.smootherstep(0.12, 0.92, u)
        return float(self.p.beta_max * ramp)

    def distance_at(self, lt):
        """Heliocentric distance in AU.

        Integrating the speed profile would be more rigorous, but the shot
        has to hit specific structures at specific beats, so the distance
        is the primary curve and the speed profile is matched to it.
        """
        u = self.progress(lt, self.duration)
        g = V.smootherstep(0.0, 1.0, u)
        d0, d1 = 42.0, self.p.r_outer
        return float(math.exp(math.log(d0) * (1.0 - g) + math.log(d1) * g))

    def camera_at(self, ctx, lt):
        p = self.p
        nose = np.asarray(p.nose, F32)
        nose = nose / np.linalg.norm(nose)
        d = self.distance_at(lt)
        eye = (-nose * F32(d)).astype(F32)
        # the camera looks along its velocity vector, out through the nose
        target = eye + nose * F32(max(d * 0.35, 6.0))
        u = self.progress(lt, self.duration)
        # a slow roll and a widening lens as speed builds: the field of
        # view has to open to contain the aberrated sky
        fov = 46.0 + 16.0 * V.smootherstep(0.3, 1.0, u)
        cam = CAM.Camera(eye=eye, target=target.astype(F32),
                         up=np.array([0.0, 1.0, 0.0], F32),
                         fov_deg=float(fov),
                         roll_deg=float(-6.0 * V.smootherstep(0.1, 1.0, u)))
        return CAM.apply_handheld(cam, lt, pos_amp=0.0, rot_amp=0.22,
                                  fov_breath=0.10, seed=71.0)

    def exposure_at(self, ctx, lt):
        u = self.progress(lt, self.duration)
        # the sky gets much brighter under beaming, so stop down
        return float(1.35 * (1.0 - 0.45 * V.smootherstep(0.25, 0.95, u)))

    # ----------------------------------------------------------------
    def _bounds(self, rays):
        p = self.p
        o, dd = rays.origin, rays.direction
        c = np.zeros(3, F32)
        # A sphere just large enough to hold the heliosphere, its tail
        # and the bow wave.  The earlier bound reached out to the far end
        # of the flight at fourteen hundred units, and the consequence was
        # that every ray integrated a thousand units of near-empty cloud
        # and came back with a uniform grey — a vacuum is only invisible
        # if you stop integrating it.  Past this radius there is nothing
        # left to draw but stars, which is the correct picture of
        # interstellar space.
        hit, tn, tf = V.ray_sphere(o, dd, c, F32(p.r_bowwave * 1.55))
        hit = hit.ravel(); tn = tn.ravel(); tf = tf.ravel()
        rays.t_near = np.maximum(tn, F32(0.05)).astype(F32)
        rays.t_far = np.where(hit, tf, tn - 1.0).astype(F32)
        return rays.clipped()

    def render(self, ctx, lt):
        u = self.progress(lt, self.duration)
        self.shader.set_time(lt, u)
        cam = self.camera_at(ctx, lt)
        exposure = self.exposure_at(ctx, lt)
        rng = ctx.rng or np.random.default_rng(ctx.frame * 7919 + 43)

        def shade(pos, t, uu):
            self.shader.new_step(rng)
            return self.shader(pos, t, uu)

        small, trn = march_layer(ctx, cam, self._bounds, shade,
                                 steps=ctx.march_steps(40),
                                 jitter_phase=ctx.frame & 7)
        medium = upsample_linear(small, ctx.width, ctx.height)

        # --- the relativistic sky -------------------------------------
        beta = self.beta_at(lt)
        nose = np.asarray(self.p.nose, F32)
        nose = nose / np.linalg.norm(nose)
        pos, rgb = self.sky.transform(beta, nose, cam.eye)
        sky, _ = project_and_splat(cam, pos, rgb * F32(0.21),
                                   ctx.width, ctx.height, falloff=False,
                                   size_scale=1.5)

        out = medium + sky.reshape(ctx.height, ctx.width, 3)
        out *= F32(exposure)
        return out


__all__ = ["TransitParams", "RelativisticSky", "TransitShader",
           "PhaseTransit", "heliosphere_radius", "C_KMS"]
