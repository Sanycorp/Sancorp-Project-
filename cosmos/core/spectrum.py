"""
cosmos.core.spectrum
====================

Physically-grounded colour: Planck blackbody radiation, CIE 1931 colour
matching, sRGB/Rec.709 primaries, and the relativistic frequency shifts the
black-hole phase needs.

Why bother with real physics here
---------------------------------
Every emitter in this film is thermal — stars, solar plasma, the accretion
disc.  Picking their colours by eye produces the familiar "orange and
teal space poster" look.  Deriving them from temperature instead gives the
sequence a single consistent colour logic for free: a 3,000 K red dwarf, a
5,772 K sun and a 40,000 K inner-disc annulus end up in correct relative
hue *and* relative brightness without any art direction at all.

The chain is:

    T  ->  Planck spectral radiance B(lambda, T)
       ->  integrate against CIE x-bar/y-bar/z-bar  ->  XYZ
       ->  linear sRGB matrix                       ->  linear RGB
       ->  (renderer keeps it linear; the display LUT tonemaps at the end)

Everything is cached into lookup tables at import time, so the per-sample
cost in the ray-marcher is a single array index.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32

# ---- physical constants (SI) ---------------------------------------------
H_PLANCK = 6.62607015e-34       # J s
C_LIGHT = 2.99792458e8          # m / s
K_BOLTZ = 1.380649e-23          # J / K
SIGMA_SB = 5.670374419e-8       # W m^-2 K^-4
WIEN_B = 2.897771955e-3         # m K

# ---- reference temperatures ----------------------------------------------
T_SUN_SURFACE = 5772.0
T_SUN_CORONA = 1.5e6
T_SUNSPOT = 4000.0
T_FLARE = 2.0e7

# Spectral sampling grid: 5 nm from 360 to 830 nm is the CIE-recommended
# resolution and is far finer than the error of the matching-function fit.
LAMBDA_NM = np.arange(360.0, 831.0, 5.0)
LAMBDA_M = LAMBDA_NM * 1e-9


# ==========================================================================
# CIE 1931 colour matching functions
# ==========================================================================
def _gauss_piecewise(x, mu, s1, s2):
    """Asymmetric Gaussian: sigma differs either side of the peak."""
    s = np.where(x < mu, s1, s2)
    t = (x - mu) / s
    return np.exp(-0.5 * t * t)


def cie_xyz_bar(lambda_nm):
    """Multi-lobe Gaussian fit to the CIE 1931 2-degree observer.

    Wyman, Sloan & Shirley (2013), "Simple Analytic Approximations to the
    CIE XYZ Color Matching Functions".  Maximum error is well under one
    percent of peak — invisible after tonemapping, and it saves shipping a
    tabulated dataset.
    """
    x = np.asarray(lambda_nm, np.float64)
    xb = (1.056 * _gauss_piecewise(x, 599.8, 37.9, 31.0)
          + 0.362 * _gauss_piecewise(x, 442.0, 16.0, 26.7)
          - 0.065 * _gauss_piecewise(x, 501.1, 20.4, 26.2))
    yb = (0.821 * _gauss_piecewise(x, 568.8, 46.9, 40.5)
          + 0.286 * _gauss_piecewise(x, 530.9, 16.3, 31.1))
    zb = (1.217 * _gauss_piecewise(x, 437.0, 11.8, 36.0)
          + 0.681 * _gauss_piecewise(x, 459.0, 26.0, 13.8))
    return xb, yb, zb


_XBAR, _YBAR, _ZBAR = cie_xyz_bar(LAMBDA_NM)

# XYZ -> linear sRGB / Rec.709, D65 white point.
XYZ_TO_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], np.float64)

RGB_TO_XYZ = np.linalg.inv(XYZ_TO_RGB)


# ==========================================================================
# Planck's law
# ==========================================================================
def planck_radiance(lambda_m, temperature):
    """Spectral radiance B(lambda, T) in W sr^-1 m^-3.

    Both arguments broadcast, so a ``(T, 1)`` temperature column against a
    ``(1, L)`` wavelength row yields the full matrix in one call.
    """
    lam = np.asarray(lambda_m, np.float64)
    t = np.asarray(temperature, np.float64)
    lam5 = lam ** 5
    expo = (H_PLANCK * C_LIGHT) / (lam * K_BOLTZ * np.maximum(t, 1.0))
    # expm1 is the numerically stable form; at the blue end of a cool
    # blackbody the exponent reaches ~50 and plain exp(x)-1 loses precision.
    return (2.0 * H_PLANCK * C_LIGHT ** 2) / (lam5 * np.expm1(np.minimum(expo, 700.0)))


def wien_peak_nm(temperature):
    """Wavelength of peak emission, nanometres."""
    return WIEN_B / np.maximum(np.asarray(temperature, np.float64), 1.0) * 1e9


def stefan_boltzmann(temperature):
    """Total radiant exitance, W m^-2."""
    return SIGMA_SB * np.asarray(temperature, np.float64) ** 4


# ==========================================================================
# temperature -> colour
# ==========================================================================
def blackbody_xyz(temperature):
    """CIE XYZ of a blackbody at ``temperature`` (chromaticity-normalised)."""
    t = np.atleast_1d(np.asarray(temperature, np.float64))[:, None]
    spec = planck_radiance(LAMBDA_M[None, :], t)
    x = (spec * _XBAR[None, :]).sum(axis=1)
    y = (spec * _YBAR[None, :]).sum(axis=1)
    z = (spec * _ZBAR[None, :]).sum(axis=1)
    norm = np.maximum(y, 1e-30)
    return np.stack([x / norm, y / norm, z / norm], axis=-1)


def blackbody_rgb(temperature, clip_negative=True):
    """Linear Rec.709 RGB of a blackbody, normalised to unit luminance.

    Negative components appear for saturated temperatures that fall outside
    the sRGB gamut.  We desaturate toward white rather than hard-clipping,
    which keeps very hot and very cool sources from flipping hue.
    """
    xyz = blackbody_xyz(temperature)
    rgb = xyz @ XYZ_TO_RGB.T
    if clip_negative:
        shortfall = np.maximum(0.0, -rgb.min(axis=1, keepdims=True))
        rgb = rgb + shortfall
    peak = np.maximum(rgb.max(axis=1, keepdims=True), 1e-12)
    rgb = rgb / peak
    return np.clip(rgb, 0.0, 1.0)


# --- cached lookup table ---------------------------------------------------
_LUT_T_MIN = 300.0
_LUT_T_MAX = 50000.0
_LUT_N = 1024
# Log spacing: perceptual colour change per kelvin is far larger at 1,000 K
# than at 40,000 K, so a linear table would waste nine tenths of its entries
# on the hot end where nothing changes.
_LUT_T = np.exp(np.linspace(np.log(_LUT_T_MIN), np.log(_LUT_T_MAX), _LUT_N))
_LUT_RGB = blackbody_rgb(_LUT_T).astype(F32)
_LOG_MIN = np.float32(np.log(_LUT_T_MIN))
_LOG_SCALE = np.float32((_LUT_N - 1) / (np.log(_LUT_T_MAX) - np.log(_LUT_T_MIN)))


def blackbody_lut(temperature):
    """Fast cached temperature -> linear RGB.  Accepts arrays of any shape."""
    t = np.asarray(temperature, F32)
    idx = (np.log(np.clip(t, _LUT_T_MIN, _LUT_T_MAX)) - _LOG_MIN) * _LOG_SCALE
    i0 = np.clip(idx.astype(np.int32), 0, _LUT_N - 2)
    f = (idx - i0).astype(F32)[..., None]
    return _LUT_RGB[i0] * (F32(1.0) - f) + _LUT_RGB[i0 + 1] * f


def blackbody_emission(temperature, scale=1.0, luminance_power=4.0,
                       t_reference=T_SUN_SURFACE):
    """Linear RGB scaled by Stefan-Boltzmann output.

    ``blackbody_lut`` returns hue only.  Multiplying by (T/T_ref)^4 restores
    the enormous dynamic range between a cool dust grain and the inner edge
    of an accretion disc, which is exactly what makes HDR bloom behave
    correctly without any manual per-object intensity tweaking.
    """
    t = np.asarray(temperature, F32)
    rgb = blackbody_lut(t)
    power = (t / F32(t_reference)) ** F32(luminance_power)
    return rgb * power[..., None] * F32(scale)


# ==========================================================================
# stellar populations
# ==========================================================================
# Harvard spectral classes with representative effective temperatures and
# the initial-mass-function weights that set how common each class is.
STELLAR_CLASSES = (
    ("O", 38000.0, 0.00003),
    ("B", 16500.0, 0.0013),
    ("A", 8700.0, 0.006),
    ("F", 6600.0, 0.030),
    ("G", 5700.0, 0.076),
    ("K", 4400.0, 0.121),
    ("M", 3200.0, 0.765),
)


def sample_stellar_temperatures(n, rng, hot_bias=0.0):
    """Draw ``n`` stellar temperatures from a realistic class distribution.

    ``hot_bias`` in [0, 1] tilts the draw toward early types; galactic spiral
    arms are where O and B stars actually live, so the galaxy phase raises
    the bias inside arms and drops it in the inter-arm regions.
    """
    temps = np.array([c[1] for c in STELLAR_CLASSES], np.float64)
    weights = np.array([c[2] for c in STELLAR_CLASSES], np.float64)
    if hot_bias > 0.0:
        weights = weights * np.exp(np.linspace(hot_bias * 6.0, 0.0, len(weights)))
    weights = weights / weights.sum()
    idx = rng.choice(len(temps), size=int(n), p=weights)
    # spread within each class so the field is not visibly quantised
    jitter = np.exp(rng.normal(0.0, 0.10, size=int(n)))
    return (temps[idx] * jitter).astype(F32)


def mass_from_temperature(t):
    """Crude main-sequence mass-temperature relation, in solar masses."""
    return np.clip((np.asarray(t, np.float64) / T_SUN_SURFACE) ** 1.7, 0.08, 120.0)


def luminosity_from_mass(m):
    """Main-sequence mass-luminosity relation, in solar luminosities."""
    m = np.asarray(m, np.float64)
    return np.where(m < 0.43, 0.23 * m ** 2.3,
                    np.where(m < 2.0, m ** 4.0,
                             np.where(m < 55.0, 1.4 * m ** 3.5, 32000.0 * m)))


# ==========================================================================
# relativistic shifts
# ==========================================================================
def doppler_factor(beta, cos_theta):
    """Relativistic Doppler factor D = 1 / (gamma * (1 - beta*cos(theta))).

    ``beta`` is v/c and ``cos_theta`` is the cosine of the angle between the
    velocity and the direction to the observer.  Material on the approaching
    side of an accretion disc has D > 1.
    """
    b = np.clip(np.asarray(beta, np.float64), 0.0, 0.999999)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - b * b, 1e-12))
    return 1.0 / np.maximum(gamma * (1.0 - b * np.asarray(cos_theta, np.float64)),
                            1e-9)


def gravitational_redshift(r, rs):
    """Schwarzschild redshift factor sqrt(1 - rs/r) for a static emitter.

    Returns the ratio of observed to emitted frequency; it falls to zero at
    the horizon, which is precisely why the event horizon renders black no
    matter how bright the gas just outside it is.
    """
    rr = np.maximum(np.asarray(r, np.float64), np.asarray(rs, np.float64) * 1.0000001)
    return np.sqrt(np.maximum(1.0 - np.asarray(rs, np.float64) / rr, 0.0))


def shift_temperature(t_rest, shift):
    """Apply a frequency shift to a blackbody.

    A blackbody stays a blackbody under a uniform frequency scaling — its
    temperature simply multiplies by the same factor.  That is why Doppler
    boosting on a disc can be rendered correctly as a temperature change
    rather than a per-wavelength spectral integration.
    """
    return np.asarray(t_rest, F32) * np.asarray(shift, F32)


def relativistic_beaming(shift, index=3.0):
    """Intensity boost I_obs / I_emit = D^(3+alpha).

    The exponent is 3 for a bolometric measurement of an isotropic emitter
    plus the spectral index of the source.  This single term is what makes
    one side of an accretion disc dramatically brighter than the other.
    """
    return np.asarray(shift, F32) ** F32(index)


# ==========================================================================
# convenience colour utilities
# ==========================================================================
def luminance(rgb):
    """Rec.709 relative luminance of linear RGB."""
    a = np.asarray(rgb, F32)
    return (a[..., 0] * F32(0.2126) + a[..., 1] * F32(0.7152)
            + a[..., 2] * F32(0.0722))


def saturate_rgb(rgb, amount):
    """Scale chroma about the luminance axis."""
    a = np.asarray(rgb, F32)
    l = luminance(a)[..., None]
    return l + (a - l) * F32(amount)


def kelvin_preview(temps=(1000, 2000, 3000, 4000, 5772, 8000, 12000, 20000, 40000)):
    """Human-readable table — used by the self-test, not by the renderer."""
    rows = []
    for t in temps:
        r, g, b = blackbody_rgb(float(t))[0]
        rows.append((float(t), float(wien_peak_nm(t)), float(r), float(g), float(b)))
    return rows


__all__ = [
    "planck_radiance", "wien_peak_nm", "stefan_boltzmann",
    "blackbody_xyz", "blackbody_rgb", "blackbody_lut", "blackbody_emission",
    "sample_stellar_temperatures", "mass_from_temperature",
    "luminosity_from_mass", "STELLAR_CLASSES",
    "doppler_factor", "gravitational_redshift", "shift_temperature",
    "relativistic_beaming",
    "luminance", "saturate_rgb", "kelvin_preview",
    "T_SUN_SURFACE", "T_SUN_CORONA", "T_SUNSPOT", "T_FLARE",
]
