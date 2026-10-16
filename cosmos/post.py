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
                 hdr_max=HDR_MAX):
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
        self._vig = vignette_mask(self.height, self.width, vignette)
        self._lut = tone_lut(gamma=gamma, contrast=contrast,
                             hdr_max=self.hdr_max)

    def __call__(self, img, rng, fade=1.0, extra_glare=0.0):
        out = np.ascontiguousarray(np.maximum(img, F32(0.0)), F32)
        out = glare(out, threshold=self.glare_threshold,
                    strength=self.glare_strength + extra_glare,
                    ana=self.anamorphic, halation=self.halation)
        out = chromatic(out, self.chroma)
        out = split_tone(out)
        out = saturate(out, self.saturation)
        out *= self._vig[:, :, None]
        if fade < 1.0:
            out *= F32(max(fade, 0.0))
        u8 = apply_lut(out, self._lut, self.hdr_max)
        return grain(u8, rng, self.grain_amount)


__all__ = ["PostChain", "glare", "chromatic", "vignette_mask", "tone_lut",
           "apply_lut", "saturate", "split_tone", "grain", "blur",
           "box_blur_h", "box_blur_v", "downsample2", "upsample2",
           "bright_pass", "luma", "HDR_MAX", "LUT_N"]
