"""
cosmos.post
===========

The display chain: everything that happens between the physically-computed
radiance of a frame and the eight-bit sRGB values that go into the encoder.

The renderers upstream produce open-ended linear radiance — a pixel on the
photosphere can be ten thousand times a pixel in a coronal hole — and none
of that survives contact with a display without being deliberately mapped.
This module does that mapping, in this order:

  1. **Glare.**  A bright-pass pyramid convolved toward the low frequencies,
     plus an anamorphic horizontal streak and a halation term biased to the
     red.  Real optics scatter light; a frame without any scattering reads
     as computer graphics no matter how good the physics behind it is.
  2. **Chromatic aberration.**  A radial scale difference between the red
     and blue channels, growing with distance from the centre, as a real
     lens does.
  3. **Vignette.**  Natural illumination falloff, roughly cos^4.
  4. **Tone mapping.**  The ACES filmic curve, applied through a twelve-bit
     lookup table so the per-pixel cost is a gather rather than a rational
     function.
  5. **Grain.**  Signal-dependent, as photographic grain is: strongest in
     the midtones, nearly absent in the deepest shadows and the highlights.

All of it runs on float32 arrays shaped ``(H, W, 3)``.  Nothing here uses
SciPy; the blurs are separable box filters applied with cumulative sums,
which for the radii involved are both faster and more predictable than an
FFT convolution would be.
"""
from __future__ import annotations

import math

import numpy as np

F32 = np.float32

# --- ACES filmic tone curve constants -----------------------------------
ACES_A = 2.51
ACES_B = 0.03
ACES_C = 2.43
ACES_D = 0.59
ACES_E = 0.14

#: resolution of the tone-mapping lookup table
LUT_N = 4096
#: radiance above which the lookup saturates
HDR_MAX = 12.0


# ==========================================================================
# primitives
# ==========================================================================
def box_blur_h(img, radius):
    """Horizontal box blur by prefix sums: O(1) per pixel in the radius."""
    if radius < 1:
        return img
    h, w = img.shape[:2]
    r = int(radius)
    pad = np.concatenate([
        np.repeat(img[:, :1], r + 1, axis=1),
        img,
        np.repeat(img[:, -1:], r, axis=1)], axis=1)
    cs = np.cumsum(pad, axis=1, dtype=np.float32)
    out = cs[:, 2 * r + 1:] - cs[:, :-(2 * r + 1)]
    out *= F32(1.0 / (2 * r + 1))
    return out[:, :w]


def box_blur_v(img, radius):
    """Vertical box blur, same method."""
    if radius < 1:
        return img
    h = img.shape[0]
    r = int(radius)
    pad = np.concatenate([
        np.repeat(img[:1], r + 1, axis=0),
        img,
        np.repeat(img[-1:], r, axis=0)], axis=0)
    cs = np.cumsum(pad, axis=0, dtype=np.float32)
    out = cs[2 * r + 1:] - cs[:-(2 * r + 1)]
    out *= F32(1.0 / (2 * r + 1))
    return out[:h]


def blur(img, radius, passes=2):
    """Separable box blur, repeated.

    Three passes of a box filter approximate a Gaussian to within a couple
    of per cent, which is far closer than anything downstream can resolve,
    and each pass costs two cumulative sums.
    """
    out = img
    for _ in range(passes):
        out = box_blur_h(out, radius)
        out = box_blur_v(out, radius)
    return out


def downsample2(img):
    """Exact 2x box downsample, cropping any odd row or column."""
    h, w = img.shape[:2]
    h2, w2 = h // 2 * 2, w // 2 * 2
    a = img[:h2, :w2]
    return (a[0::2, 0::2] + a[0::2, 1::2]
            + a[1::2, 0::2] + a[1::2, 1::2]) * F32(0.25)


def upsample2(img, shape):
    """Nearest-neighbour 2x upsample to an exact target shape.

    The result is immediately blurred by the caller, so bilinear
    interpolation here would be work that the blur undoes.
    """
    out = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    h, w = shape[:2]
    oh, ow = out.shape[:2]
    if oh < h or ow < w:
        out = np.pad(out, ((0, max(0, h - oh)), (0, max(0, w - ow)),
                           (0, 0)), mode="edge")
    return out[:h, :w]


# ==========================================================================
# glare
# ==========================================================================
def luma(img):
    """Rec. 709 luminance."""
    return (img[:, :, 0] * F32(0.2126)
            + img[:, :, 1] * F32(0.7152)
            + img[:, :, 2] * F32(0.0722))


def bright_pass(img, threshold, knee=0.55):
    """Soft-knee highlight extraction.

    A hard threshold makes bloom pop on and off between frames as a
    feature crosses it; the knee makes the transition continuous, which
    matters at sixty frames a second.
    """
    l = luma(img)
    soft = np.clip((l - F32(threshold * (1.0 - knee)))
                   / F32(max(threshold * knee * 2.0, 1e-4)), 0.0, 1.0)
    w = np.where(l > F32(threshold), F32(1.0), soft * soft)
    return img * w[:, :, None]


def glare(img, threshold=0.78, strength=0.30, levels=5, ana=0.34,
          halation=0.22):
    """Lens and sensor glare: a bloom pyramid, a streak, and halation.

    The pyramid is the standard construction — repeatedly halve, blur,
    then sum the levels back up with decreasing weight — which produces a
    glow whose profile falls off over several octaves of scale rather than
    the single Gaussian lobe a one-pass blur gives.

    Every part of it is computed on a half-resolution copy and upsampled
    once at the end.  Glare is by definition low-frequency, so the only
    thing full-resolution work buys is a quarter of the frame rate: at
    1920x1080 the naive version of this function cost 1.4 seconds a frame,
    which across the whole film was more than the galaxy phase.
    """
    bp = bright_pass(img, threshold)
    peak = float(bp.max())
    if peak <= 1e-6:
        return img

    half = downsample2(bp)
    shapes = [half.shape]
    acc = np.zeros_like(half)
    cur = half
    pyramid = []
    for i in range(levels):
        if min(cur.shape[0], cur.shape[1]) < 8:
            break
        cur = downsample2(cur)
        cur = blur(cur, 2, passes=2)
        pyramid.append(cur)
        shapes.append(cur.shape)

    # sum back up, coarse levels contributing a wider, weaker halo
    for i, level in enumerate(pyramid):
        up = level
        for j in range(i, -1, -1):
            up = upsample2(up, shapes[j])
            up = blur(up, 2, passes=1)
        acc += up * F32(0.72 ** i)

    acc *= F32(strength)

    # --- anamorphic streak -------------------------------------------
    # A cylindrical front element spreads a point into a horizontal line.
    if ana > 0.0:
        small = downsample2(downsample2(half))
        w = small.shape[1]
        streak = box_blur_h(small, max(4, w // 11))
        streak = box_blur_h(streak, max(4, w // 22))
        streak = box_blur_v(streak, 1)
        streak = upsample2(upsample2(streak, shapes[1]), shapes[0])
        tint = np.array([0.74, 0.86, 1.0], F32)
        acc += streak * tint[None, None, :] * F32(ana)

    # --- halation ------------------------------------------------------
    # In film, light that gets through the emulsion reflects off the base
    # and re-exposes it from behind.  Red light penetrates deepest, so the
    # halo around a highlight is warm.
    if halation > 0.0:
        small = downsample2(downsample2(half))
        hal = blur(small, max(3, small.shape[1] // 30), passes=2)
        hal = upsample2(upsample2(hal, shapes[1]), shapes[0])
        tint = np.array([1.0, 0.42, 0.22], F32)
        acc += hal * tint[None, None, :] * F32(halation)

    acc = upsample2(acc, img.shape)
    acc = blur(acc, 2, passes=1)
    return img + acc


# ==========================================================================
# aperture diffraction
# ==========================================================================
def directional_streak(img, angle, taps=7, decay=0.60, step=1.5):
    """Smear an image along one direction with geometrically spaced taps.

    A single streak of a diffraction star is a line whose intensity falls
    off roughly exponentially with distance from the source.  Sampling it
    at *geometrically* spaced offsets — one, two, four, eight pixels and
    so on — reproduces that profile with a handful of taps instead of the
    hundreds a uniform line integral would need, because the far part of
    the streak is both faint and smooth and does not deserve dense
    sampling.
    """
    ca = math.cos(angle)
    sa = math.sin(angle)
    acc = np.zeros_like(img)
    w = 1.0
    total = 0.0
    for k in range(taps):
        off = step * (2.0 ** k)
        dx = int(round(ca * off))
        dy = int(round(sa * off))
        for sgn in (1, -1):
            sx, sy = dx * sgn, dy * sgn
            shifted = np.roll(img, (sy, sx), axis=(0, 1))
            # kill the wrap-around, which would otherwise print a copy of
            # the opposite edge across the frame
            if sy > 0:
                shifted[:sy] = 0.0
            elif sy < 0:
                shifted[sy:] = 0.0
            if sx > 0:
                shifted[:, :sx] = 0.0
            elif sx < 0:
                shifted[:, sx:] = 0.0
            acc += shifted * F32(w)
            total += w
        w *= decay
    return acc * F32(1.0 / max(total, 1e-6))


def starburst(img, blades=6, strength=0.26, threshold=1.6, angle_deg=9.0,
              taps=7):
    """Diffraction star from a polygonal iris.

    An aperture with ``n`` straight blades diffracts a point source into
    ``n`` rays for even ``n`` and ``2n`` for odd ``n``, because opposite
    edges of an even-bladed iris are parallel and their two rays
    coincide.  This is the one optical artefact in the chain that is a
    genuine diffraction phenomenon rather than a scattering or reflection
    one, and it is the signature that reads instantly as "photographed
    through a real lens".

    Only the brightest part of the frame contributes: the threshold is
    well above the glare threshold, so the disc of the black hole and the
    photosphere of the sun get spikes and the diffuse nebulae do not.
    """
    if strength <= 0.0:
        return img
    bp = np.maximum(img - F32(threshold), F32(0.0))
    if float(bp.max()) <= 1e-6:
        return img
    small = downsample2(downsample2(bp))
    rays = blades if blades % 2 == 0 else 2 * blades
    half = rays // 2
    acc = np.zeros_like(small)
    base = math.radians(angle_deg)
    for i in range(half):
        acc += directional_streak(small, base + i * math.pi / half,
                                  taps=taps, decay=0.58)
    acc *= F32(1.0 / half)
    # a faint spectral split along the spikes: diffraction is dispersive,
    # so the outer part of each ray is coloured
    tint = np.array([1.06, 1.0, 1.12], F32)
    acc = acc * tint[None, None, :]
    acc = upsample2(upsample2(acc, (img.shape[0] // 2 + img.shape[0] % 2,
                                    img.shape[1] // 2 + img.shape[1] % 2,
                                    3)), img.shape)
    return img + acc * F32(strength)


# ==========================================================================
# internal reflections
# ==========================================================================
def scale_about_centre(img, s):
    """Resample an image by a signed factor about its centre.

    Negative factors flip it, which is what an internal reflection does:
    the ghost of a highlight appears on the opposite side of the optical
    axis.  Separable, so two 1-D gathers rather than a 2-D one.
    """
    h, w, _ = img.shape
    yy = (np.arange(h, dtype=np.float32) - (h - 1) * 0.5) / s + (h - 1) * 0.5
    xx = (np.arange(w, dtype=np.float32) - (w - 1) * 0.5) / s + (w - 1) * 0.5
    y0 = np.clip(np.floor(yy), 0, h - 1).astype(np.int32)
    x0 = np.clip(np.floor(xx), 0, w - 1).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = np.clip(yy - y0, 0.0, 1.0).astype(F32)[:, None, None]
    fx = np.clip(xx - x0, 0.0, 1.0).astype(F32)[None, :, None]
    rows = img[y0] * (1.0 - fy) + img[y1] * fy
    out = rows[:, x0] * (1.0 - fx) + rows[:, x1] * fx
    # anything that came from outside the source frame must be dark
    inside_y = ((yy >= 0) & (yy <= h - 1)).astype(F32)[:, None, None]
    inside_x = ((xx >= 0) & (xx <= w - 1)).astype(F32)[None, :, None]
    return out * inside_y * inside_x


#: Ghost specification: (scale, tint, weight).  Negative scales are
#: reflections through the optical axis, positive ones are the
#: same-side ghosts that a cemented doublet produces.  The values are
#: the usual photographic set — a tight warm ghost close in, a large
#: cool one spread across the opposite quadrant, and a faint ring.
GHOSTS = (
    (-0.42, (1.00, 0.72, 0.48), 0.55),
    (-0.86, (0.52, 0.78, 1.00), 0.38),
    (-1.55, (0.72, 1.00, 0.80), 0.22),
    (0.63, (1.00, 0.88, 0.66), 0.18),
)


def ghosts(img, strength=0.07, threshold=1.3, specs=GHOSTS):
    """Internal reflections between lens elements.

    Every air-glass surface reflects a fraction of a per cent of the
    light that reaches it, and with a dozen elements there are hundreds
    of two-surface paths.  Each one re-images the bright parts of the
    scene, usually defocused and almost always displaced through the
    optical axis.  A single very bright source in a dark field — which is
    every frame of this film — is the situation in which they are most
    visible.
    """
    if strength <= 0.0:
        return img
    bp = np.maximum(img - F32(threshold), F32(0.0))
    if float(bp.max()) <= 1e-6:
        return img
    small = downsample2(downsample2(downsample2(bp)))
    small = blur(small, 3, passes=2)
    acc = np.zeros_like(small)
    for s, tint, w in specs:
        g = scale_about_centre(small, s)
        # a ghost is an out-of-focus image of the aperture, never a sharp
        # one, and the defocus grows with how far the reflection has to
        # travel; scale it with the magnification
        g = blur(g, max(2, int(round(2.0 + 2.5 * abs(s)))), passes=2)
        acc += g * np.asarray(tint, F32)[None, None, :] * F32(w)
    sh = [(img.shape[0] + 3) // 4, (img.shape[1] + 3) // 4, 3]
    sh2 = [(img.shape[0] + 1) // 2, (img.shape[1] + 1) // 2, 3]
    acc = blur(acc, 2, passes=1)
    acc = blur(upsample2(acc, tuple(sh)), 2, passes=1)
    acc = blur(upsample2(acc, tuple(sh2)), 2, passes=1)
    acc = blur(upsample2(acc, img.shape), 2, passes=1)
    return img + acc * F32(strength)


# ==========================================================================
# highlight rendition
# ==========================================================================
def highlight_desaturate(img, start=2.2, hdr_max=HDR_MAX, amount=0.85):
    """Bleach saturated highlights toward white before the tone curve.

    Film and digital sensors both do this: once one channel saturates,
    further exposure can only raise the others, so every highlight walks
    toward white as it clips.  Without it a blue-white region such as the
    Doppler-boosted limb of the accretion disc clips channel by channel
    and lands on a flat, electric, obviously-digital blue instead of a
    hot white core with a blue fringe.
    """
    if amount <= 0.0:
        return img
    l = luma(img)[:, :, None]
    t = np.clip((l - F32(start)) / F32(max(hdr_max - start, 1e-3)),
                0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t) * F32(amount)
    return img * (1.0 - t) + l * t


# ==========================================================================
# geometric effects
# ==========================================================================
def _radial_scale(channel, amount):
    """Scale one channel about the image centre by ``1 + amount``.

    Separable: the rows are resampled, then the columns.  A 2-D bilinear
    gather needs four full-frame fancy-index reads and three full-frame
    multiply-adds; doing it as two 1-D passes needs two of each, and since
    a pure scale about the centre *is* separable the result is identical
    to the floating-point noise.
    """
    h, w = channel.shape
    if abs(amount) < 1e-7:
        return channel
    inv = 1.0 / (1.0 + amount)

    cx = (w - 1) * 0.5
    xs = (np.arange(w, dtype=np.float32) - cx) * inv + cx
    np.clip(xs, 0, w - 1.001, out=xs)
    x0 = xs.astype(np.int32)
    fx = (xs - x0).astype(F32)[None, :]
    x1 = np.minimum(x0 + 1, w - 1)
    tmp = channel[:, x0] * (F32(1.0) - fx) + channel[:, x1] * fx

    cy = (h - 1) * 0.5
    ys = (np.arange(h, dtype=np.float32) - cy) * inv + cy
    np.clip(ys, 0, h - 1.001, out=ys)
    y0 = ys.astype(np.int32)
    fy = (ys - y0).astype(F32)[:, None]
    y1 = np.minimum(y0 + 1, h - 1)
    return (tmp[y0] * (F32(1.0) - fy) + tmp[y1] * fy).astype(F32)


def chromatic(img, amount=0.0018):
    """Lateral chromatic aberration: the lens focuses colours differently."""
    if amount <= 0.0:
        return img
    out = np.empty_like(img)
    out[:, :, 0] = _radial_scale(img[:, :, 0], amount)
    out[:, :, 1] = img[:, :, 1]
    out[:, :, 2] = _radial_scale(img[:, :, 2], -amount)
    return out


def vignette_mask(h, w, strength=0.30, power=1.5):
    """Natural illumination falloff toward the corners."""
    yy = (np.arange(h, dtype=np.float32) - (h - 1) * 0.5) / (h * 0.5)
    xx = (np.arange(w, dtype=np.float32) - (w - 1) * 0.5) / (w * 0.5)
    r2 = yy[:, None] ** 2 * 1.0 + xx[None, :] ** 2 * 0.72
    v = 1.0 - strength * np.power(np.clip(r2, 0.0, 2.0) * 0.72, power)
    return np.clip(v, 0.0, 1.0).astype(F32)


# ==========================================================================
# tone mapping
# ==========================================================================
def _build_lut(n=LUT_N, hdr_max=HDR_MAX, gamma=2.2, lift=0.0,
               contrast=1.0):
    """Precompute radiance -> 8-bit sRGB.

    The ACES approximation plus the display transfer function collapse into
    a single one-dimensional table, so the per-pixel work at 1080p drops
    from two million rational evaluations and two million powers to two
    million gathers.
    """
    x = np.linspace(0.0, hdr_max, n, dtype=np.float64)
    y = (x * (ACES_A * x + ACES_B)) / (x * (ACES_C * x + ACES_D) + ACES_E)
    y = np.clip(y, 0.0, 1.0)
    if contrast != 1.0:
        y = np.clip((y - 0.5) * contrast + 0.5, 0.0, 1.0)
    if lift != 0.0:
        y = np.clip(y * (1.0 - lift) + lift, 0.0, 1.0)
    y = np.power(y, 1.0 / gamma)
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


_LUT_CACHE = {}


def tone_lut(gamma=2.2, lift=0.0, contrast=1.0, hdr_max=HDR_MAX):
    key = (round(gamma, 4), round(lift, 4), round(contrast, 4),
           round(hdr_max, 3))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = _build_lut(LUT_N, hdr_max, gamma, lift, contrast)
        _LUT_CACHE[key] = lut
    return lut


def apply_lut(img, lut, hdr_max=HDR_MAX):
    idx = np.clip(img * F32((LUT_N - 1) / hdr_max), 0,
                  LUT_N - 1).astype(np.uint16)
    return lut[idx]


# ==========================================================================
# colour
# ==========================================================================
def saturate(img, amount):
    if abs(amount - 1.0) < 1e-4:
        return img
    l = luma(img)[:, :, None]
    return np.maximum(l + (img - l) * F32(amount), F32(0.0))


def split_tone(img, shadow=(0.92, 0.97, 1.10), highlight=(1.05, 1.0, 0.94),
               pivot=0.45):
    """Cool the shadows, warm the highlights.

    Film stocks do this on their own because the three emulsion layers have
    different toes and shoulders.  It is the single cheapest thing that
    makes a synthetic image read as photographed.
    """
    l = np.clip(luma(img) / F32(pivot), 0.0, 1.0)[:, :, None]
    sh = np.asarray(shadow, F32)[None, None, :]
    hi = np.asarray(highlight, F32)[None, None, :]
    return img * (sh * (1.0 - l) + hi * l)


# ==========================================================================
# grain
# ==========================================================================
def grain(img_u8, rng, amount=3.4):
    """Signal-dependent monochrome grain, added in display space.

    Photographic grain is a property of the emulsion, not of the scene, so
    it belongs after the tone curve.  Its visibility peaks in the midtones:
    the shadows have too few developed crystals to show structure and the
    highlights are saturated.
    """
    if amount <= 0.0:
        return img_u8
    h, w = img_u8.shape[:2]
    # Generated at half resolution and doubled.  Real grain clumps — the
    # developed silver crystals are larger than a scanned pixel — so the
    # pairing is a feature, and drawing two million gaussians per frame
    # instead of half a million was costing more than the tone map.
    n = rng.standard_normal((h // 2 + 1, w // 2 + 1), dtype=np.float32)
    n = np.repeat(np.repeat(n, 2, axis=0), 2, axis=1)[:h, :w]
    l = img_u8[:, :, 1].astype(np.float32) * (1.0 / 255.0)
    weight = 4.0 * l * (1.0 - l)
    d = (n * weight * amount)[:, :, None]
    return np.clip(img_u8.astype(np.float32) + d, 0, 255).astype(np.uint8)


# ==========================================================================
# the chain
# ==========================================================================
class PostChain:
    """Holds the per-resolution precomputations and runs the whole chain."""

    def __init__(self, width, height, *, glare_threshold=0.78,
                 glare_strength=0.30, anamorphic=0.34, halation=0.22,
                 chroma=0.0019, vignette=0.30, saturation=1.12,
                 grain_amount=3.1, gamma=2.2, contrast=1.045,
                 hdr_max=HDR_MAX, star_strength=0.30, star_blades=6,
                 star_threshold=1.15, ghost_strength=0.026,
                 ghost_threshold=1.6, bleach_start=2.1, bleach=0.85):
        self.width = int(width)
        self.height = int(height)
        self.glare_threshold = float(glare_threshold)
        self.glare_strength = float(glare_strength)
        self.anamorphic = float(anamorphic)
        self.halation = float(halation)
        self.chroma = float(chroma)
        self.saturation = float(saturation)
        self.grain_amount = float(grain_amount)
        self.hdr_max = float(hdr_max)
        self.star_strength = float(star_strength)
        self.star_blades = int(star_blades)
        self.star_threshold = float(star_threshold)
        self.ghost_strength = float(ghost_strength)
        self.ghost_threshold = float(ghost_threshold)
        self.bleach_start = float(bleach_start)
        self.bleach = float(bleach)
        self._vig = vignette_mask(self.height, self.width, vignette)
        self._lut = tone_lut(gamma=gamma, contrast=contrast,
                             hdr_max=self.hdr_max)

    def __call__(self, img, rng, fade=1.0, extra_glare=0.0):
        out = np.ascontiguousarray(np.maximum(img, F32(0.0)), F32)
        # Order matters and follows the light path through a real
        # camera: diffraction happens at the iris, reflections happen
        # between the elements behind it, scattering and halation happen
        # at the sensor, and everything else is development.
        out = starburst(out, blades=self.star_blades,
                        strength=self.star_strength,
                        threshold=self.star_threshold)
        out = ghosts(out, strength=self.ghost_strength,
                     threshold=self.ghost_threshold)
        out = glare(out, threshold=self.glare_threshold,
                    strength=self.glare_strength + extra_glare,
                    ana=self.anamorphic, halation=self.halation)
        out = chromatic(out, self.chroma)
        out = highlight_desaturate(out, self.bleach_start, self.hdr_max,
                                   self.bleach)
        out = split_tone(out)
        out = saturate(out, self.saturation)
        out *= self._vig[:, :, None]
        if fade < 1.0:
            out *= F32(max(fade, 0.0))
        u8 = apply_lut(out, self._lut, self.hdr_max)
        return grain(u8, rng, self.grain_amount)


__all__ = ["PostChain", "glare", "chromatic", "vignette_mask", "tone_lut",
           "starburst", "ghosts", "directional_streak", "scale_about_centre",
           "highlight_desaturate",
           "apply_lut", "saturate", "split_tone", "grain", "blur",
           "box_blur_h", "box_blur_v", "downsample2", "upsample2",
           "bright_pass", "luma", "HDR_MAX", "LUT_N"]
