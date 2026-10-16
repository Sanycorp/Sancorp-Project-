"""
Core rendering engine for the SANCORP motion-graphics pipeline.

Everything is rendered natively at 1920x1080 / 60 fps.  A frame is a float32
numpy buffer (H, W, 3) in 0..1; vector art is drawn with PIL into RGBA layers
and composited with normal / additive blending.  A small post chain (bloom,
chromatic aberration, vignette, scanlines, grain) ties the look together.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------
# format
# --------------------------------------------------------------------------
W, H = 1920, 1080
FPS = 60
CX, CY = W // 2, H // 2

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------
BG = (7, 7, 9)
INK = (12, 12, 15)
WHITE = (255, 255, 255)
BONE = (236, 238, 243)
CRIMSON = (255, 36, 72)
CRIM_HOT = (255, 92, 118)
CRIM_DEEP = (142, 10, 32)
EMBER = (255, 146, 82)
LIME = (200, 255, 70)
STEEL = (124, 131, 148)
SLATE = (38, 41, 50)

ANTON = "Anton.ttf"
ARCHIVO = "ArchivoBlack.ttf"
BEBAS = "BebasNeue.ttf"
INTER = "Inter-Regular.ttf"
INTER_SB = "Inter-SemiBold.ttf"
INTER_B = "Inter-Bold.ttf"
INTER_BK = "Inter-Black.ttf"
MONO = "JetBrainsMono-Regular.ttf"
MONO_B = "JetBrainsMono-Bold.ttf"
MONO_XB = "JetBrainsMono-ExtraBold.ttf"


# ==========================================================================
# easing
# ==========================================================================
def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def linear(t):
    return t


def out_cubic(t):
    return 1.0 - (1.0 - t) ** 3


def in_cubic(t):
    return t * t * t


def out_quart(t):
    return 1.0 - (1.0 - t) ** 4


def out_quint(t):
    return 1.0 - (1.0 - t) ** 5


def in_quad(t):
    return t * t


def out_expo(t):
    return 1.0 if t >= 1.0 else 1.0 - 2 ** (-11 * t)


def in_expo(t):
    return 0.0 if t <= 0.0 else 2 ** (11 * (t - 1))


def in_out_cubic(t):
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def in_out_quart(t):
    return 8 * t**4 if t < 0.5 else 1 - (-2 * t + 2) ** 4 / 2


def in_out_expo(t):
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    if t < 0.5:
        return 2 ** (20 * t - 10) / 2
    return (2 - 2 ** (-20 * t + 10)) / 2


def out_back(t, s=1.70158):
    c3 = s + 1
    return 1 + c3 * (t - 1) ** 3 + s * (t - 1) ** 2


def out_elastic(t, p=0.34):
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi) / 3 / p * 0.34) + 1


def span(t, a, b, ease=linear):
    """Normalised, eased progress of `t` inside the window [a, b]."""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return ease(clamp((t - a) / (b - a)))


def pulse(t, a, b, ease_in=out_cubic, ease_out=in_cubic, hold=0.0):
    """0 -> 1 -> 0 envelope across [a, b] with an optional flat hold."""
    if t < a or t > b:
        return 0.0
    d = b - a
    rise = d * (1 - hold) * 0.5
    if t < a + rise:
        return ease_in(clamp((t - a) / rise))
    if t > b - rise:
        return 1.0 - ease_out(clamp((t - (b - rise)) / rise))
    return 1.0


def mix(a, b, t):
    return a + (b - a) * t


def mixc(c1, c2, t):
    t = clamp(t)
    return tuple(int(round(mix(c1[i], c2[i], t))) for i in range(3))


def shake(seed, amp, t, freq=37.0):
    """Deterministic 2-axis shake."""
    if amp <= 0:
        return 0.0, 0.0
    return (
        math.sin(t * freq + seed * 12.9898) * amp,
        math.cos(t * freq * 1.31 + seed * 78.233) * amp,
    )


# ==========================================================================
# fonts & text
# ==========================================================================
_font_cache: dict = {}
_text_cache: dict = {}


def font(name, size):
    key = (name, int(size))
    f = _font_cache.get(key)
    if f is None:
        f = ImageFont.truetype(os.path.join(FONT_DIR, name), int(size))
        _font_cache[key] = f
    return f


def text_img(
    text,
    fontname,
    size,
    fill=WHITE,
    tracking=0.0,
    stroke=0,
    stroke_fill=(0, 0, 0),
    pad=12,
):
    """Tight-cropped RGBA image of rendered text.  Cached aggressively."""
    key = (text, fontname, int(size), tuple(fill), round(tracking, 2), stroke, tuple(stroke_fill))
    hit = _text_cache.get(key)
    if hit is not None:
        return hit

    f = font(fontname, size)
    if len(fill) == 3:
        fill = fill + (255,)

    if tracking == 0.0:
        wpx = int(f.getlength(text) + 2 * pad + 2 * stroke + size * 0.6)
    else:
        wpx = int(sum(f.getlength(c) + tracking for c in text) + 2 * pad + 2 * stroke + size * 0.6)
    hpx = int(size * 2.1 + 2 * pad + 2 * stroke)
    img = Image.new("RGBA", (max(wpx, 2), max(hpx, 2)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if tracking == 0.0:
        d.text((pad + stroke, pad + stroke), text, font=f, fill=fill,
               stroke_width=stroke, stroke_fill=stroke_fill)
    else:
        x = float(pad + stroke)
        for ch in text:
            d.text((x, pad + stroke), ch, font=f, fill=fill,
                   stroke_width=stroke, stroke_fill=stroke_fill)
            x += f.getlength(ch) + tracking

    bb = img.getbbox()
    if bb:
        img = img.crop(bb)
    _text_cache[key] = img
    return img


def text_width(text, fontname, size, tracking=0.0):
    f = font(fontname, size)
    if tracking == 0.0:
        return f.getlength(text)
    return sum(f.getlength(c) + tracking for c in text) - tracking


# ==========================================================================
# layers & blitting
# ==========================================================================
def layer(size=None):
    return Image.new("RGBA", size or (W, H), (0, 0, 0, 0))


def blit(dst: Image.Image, src: Image.Image, x: int, y: int):
    """alpha_composite `src` onto `dst` at (x, y), clipped to dst bounds."""
    x, y = int(round(x)), int(round(y))
    sw, sh = src.size
    dw, dh = dst.size
    sx0 = max(0, -x)
    sy0 = max(0, -y)
    sx1 = min(sw, dw - x)
    sy1 = min(sh, dh - y)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    if (sx0, sy0, sx1, sy1) != (0, 0, sw, sh):
        src = src.crop((sx0, sy0, sx1, sy1))
    dst.alpha_composite(src, (x + sx0, y + sy0))


def place(
    dst: Image.Image,
    src: Image.Image,
    cx,
    cy,
    scale=1.0,
    rotate=0.0,
    alpha=1.0,
    anchor="mm",
    skew=0.0,
):
    """Scale / rotate / fade `src` and stamp it onto `dst`."""
    if alpha <= 0.002 or scale <= 0.0005:
        return
    img = src
    if scale != 1.0:
        nw = max(1, int(round(img.width * scale)))
        nh = max(1, int(round(img.height * scale)))
        img = img.resize((nw, nh), Image.LANCZOS if scale < 1.0 else Image.BICUBIC)
    if skew:
        nw = img.width + int(abs(skew) * img.height)
        img = img.transform(
            (nw, img.height), Image.AFFINE,
            (1, skew, -skew * img.height if skew > 0 else 0, 0, 1, 0),
            resample=Image.BICUBIC,
        )
    if rotate:
        img = img.rotate(rotate, resample=Image.BICUBIC, expand=True)
    if alpha < 1.0:
        a = img.getchannel("A").point(lambda v: int(v * alpha))
        img = img.copy()
        img.putalpha(a)

    ax = {"l": 0.0, "m": 0.5, "r": 1.0}[anchor[0]]
    ay = {"t": 0.0, "m": 0.5, "b": 1.0}[anchor[1]]
    blit(dst, img, cx - img.width * ax, cy - img.height * ay)


def place_blur(dst, src, cx, cy, scale=1.0, rotate=0.0, alpha=1.0, blur=0.0, anchor="mm"):
    img = src
    if blur > 0.3:
        pad = int(blur * 3)
        img = Image.new("RGBA", (src.width + 2 * pad, src.height + 2 * pad), (0, 0, 0, 0))
        img.paste(src, (pad, pad))
        img = img.filter(ImageFilter.GaussianBlur(blur))
    place(dst, img, cx, cy, scale, rotate, alpha, anchor)


def trail(dst, src, cx, cy, vx, vy, n=5, scale=1.0, alpha=1.0, rotate=0.0):
    """Cheap directional motion blur: stamp n decreasing-alpha copies."""
    if n <= 1 or (abs(vx) < 0.6 and abs(vy) < 0.6):
        place(dst, src, cx, cy, scale, rotate, alpha)
        return
    for i in range(n, 0, -1):
        f = i / n
        place(dst, src, cx - vx * f, cy - vy * f, scale, rotate,
              alpha * (1.0 - f) ** 1.5 * 0.85)
    place(dst, src, cx, cy, scale, rotate, alpha)


# ==========================================================================
# shapes
# ==========================================================================
def rrect(d: ImageDraw.ImageDraw, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def bezier_pts(p0, p1, p2, p3, n=48):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        x = u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]
        out.append((x, y))
    return out


def poly_at(pts, u):
    """Point at normalised arc position u along a polyline."""
    u = clamp(u)
    segs = []
    total = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        L = math.hypot(dx, dy)
        segs.append(L)
        total += L
    if total <= 0:
        return pts[0]
    target = u * total
    acc = 0.0
    for i, L in enumerate(segs):
        if acc + L >= target:
            f = (target - acc) / max(L, 1e-6)
            return (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f,
                    pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f)
        acc += L
    return pts[-1]


def draw_path(d, pts, color, width, u0=0.0, u1=1.0):
    """Draw a sub-range of a polyline (for line-draw-on animations)."""
    if u1 <= u0:
        return
    n = len(pts)
    i0 = int(clamp(u0) * (n - 1))
    i1 = int(math.ceil(clamp(u1) * (n - 1)))
    sub = pts[i0:i1 + 1]
    if len(sub) >= 2:
        d.line(sub, fill=color, width=width, joint="curve")


# ==========================================================================
# backgrounds
# ==========================================================================
_bg_cache: dict = {}


def dot_grid(step=48, r=1.6, color=(72, 78, 92), alpha=90):
    key = ("dots", step, r, color, alpha)
    hit = _bg_cache.get(key)
    if hit is not None:
        return hit
    img = Image.new("RGBA", (W + step, H + step), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y in range(0, H + step, step):
        for x in range(0, W + step, step):
            d.ellipse((x - r, y - r, x + r, y + r), fill=color + (alpha,))
    _bg_cache[key] = img
    return img


def line_grid(step=120, color=(44, 48, 60), alpha=70, width=1):
    key = ("lines", step, color, alpha, width)
    hit = _bg_cache.get(key)
    if hit is not None:
        return hit
    img = Image.new("RGBA", (W + step, H + step), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for x in range(0, W + step, step):
        d.line([(x, 0), (x, H + step)], fill=color + (alpha,), width=width)
    for y in range(0, H + step, step):
        d.line([(0, y), (W + step, y)], fill=color + (alpha,), width=width)
    _bg_cache[key] = img
    return img


def grid_field(step=60, r=1.8, color=(104, 112, 132), alpha=120, kind="dots",
               width=1):
    """Pre-multiplied float32 grid tile of shape (H+step, W+step, 3).

    `step` must divide W and H so that slicing the tile at any offset in
    [0, step) yields a seamless full-frame grid with zero copying.
    """
    key = ("gf", kind, step, r, color, alpha, width)
    hit = _bg_cache.get(key)
    if hit is not None:
        return hit
    img = Image.new("RGBA", (W + step, H + step), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if kind == "dots":
        for y in range(0, H + step + 1, step):
            for x in range(0, W + step + 1, step):
                d.ellipse((x - r, y - r, x + r, y + r), fill=color + (alpha,))
    else:
        for x in range(0, W + step + 1, step):
            d.line([(x, 0), (x, H + step)], fill=color + (alpha,), width=width)
        for y in range(0, H + step + 1, step):
            d.line([(0, y), (W + step, y)], fill=color + (alpha,), width=width)
    a = np.asarray(img, np.float32) / 255.0
    f = np.ascontiguousarray(a[..., :3] * a[..., 3:4], dtype=np.float32)
    _bg_cache[key] = f
    return f


def radial_wash(color, cx=CX, cy=CY, radius=900, strength=1.0, down=6):
    """Soft radial glow as a float32 (H,W,3) additive field (cached)."""
    key = ("wash", color, cx, cy, radius, round(strength, 3), down)
    hit = _bg_cache.get(key)
    if hit is not None:
        return hit
    hh, ww = H // down, W // down
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    dx = (xx - cx / down) / (radius / down)
    dy = (yy - cy / down) / (radius / down)
    g = np.exp(-(dx * dx + dy * dy) * 1.9).astype(np.float32)
    g = np.asarray(
        Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC),
        np.float32,
    ) / 255.0
    field = g[..., None] * (np.asarray(color, np.float32) / 255.0) * strength
    _bg_cache[key] = field
    return field


# ==========================================================================
# compositing
# ==========================================================================
def _bbox_view(canvas, lyr):
    """Crop a layer to its non-empty alpha box and return (rgba_f32, region)."""
    bb = lyr.getbbox()
    if bb is None:
        return None, None
    x0, y0, x1, y1 = bb
    sub = lyr if bb == (0, 0, lyr.width, lyr.height) else lyr.crop(bb)
    return np.asarray(sub, np.float32), canvas[y0:y1, x0:x1]


def over(canvas: np.ndarray, lyr: Image.Image, opacity=1.0):
    """Normal (source-over) blend, limited to the layer's bounding box."""
    l, region = _bbox_view(canvas, lyr)
    if l is None:
        return canvas
    a = l[..., 3:4] * np.float32(opacity / 255.0)
    rgb = l[..., :3] * np.float32(1.0 / 255.0)
    np.subtract(rgb, region, out=rgb)
    np.multiply(rgb, a, out=rgb)
    np.add(region, rgb, out=region)
    return canvas


def add(canvas: np.ndarray, lyr: Image.Image, gain=1.0):
    """Additive blend, limited to the layer's bounding box."""
    l, region = _bbox_view(canvas, lyr)
    if l is None:
        return canvas
    rgb = l[..., :3] * (l[..., 3:4] * np.float32(gain / (255.0 * 255.0)))
    np.add(region, rgb, out=region)
    return canvas


def screen(canvas: np.ndarray, lyr: Image.Image, gain=1.0):
    l, region = _bbox_view(canvas, lyr)
    if l is None:
        return canvas
    s = l[..., :3] * (l[..., 3:4] * np.float32(gain / (255.0 * 255.0)))
    np.subtract(s, region * s, out=s)
    np.add(region, s, out=region)
    return canvas


# ==========================================================================
# post chain
# ==========================================================================
_lens_cache: dict = {}


def lens(canvas, vig=0.62, scan=0.045, period=3.0):
    """Combined vignette + scanline multiply (single cached array)."""
    key = (round(vig, 3), round(scan, 4), round(period, 2))
    m = _lens_cache.get(key)
    if m is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx = (xx - CX) / (W * 0.62)
        dy = (yy - CY) / (H * 0.62)
        r = np.sqrt(dx * dx + dy * dy)
        v = np.clip(1.0 - (r**1.9) * 0.78, 0.0, 1.0)
        m = 1.0 - vig * (1.0 - v) ** 1.5
        if scan > 0:
            s = 0.5 + 0.5 * np.cos(np.arange(H, dtype=np.float32) * (2 * math.pi / period))
            m = m * (1.0 - scan * s[:, None])
        m = m.astype(np.float32)[..., None]
        _lens_cache[key] = m
    canvas *= m
    return canvas


def bloom(canvas, thresh=0.62, strength=0.52, radius=10.0, down=4):
    # 2-tap strided downsample: ~35x cheaper than a full box mean and the
    # result is blurred to oblivion anyway.
    small = (canvas[::down, ::down] + canvas[down // 2::down, down // 2::down]) * 0.5
    lum = small.max(axis=2, keepdims=True)
    k = np.clip((lum - thresh) / (1.0 - thresh + 1e-6), 0.0, 1.0)
    bright = np.clip(small * k, 0.0, 4.0)
    im = Image.fromarray(np.clip(bright * 255.0, 0, 255).astype(np.uint8))
    b1 = im.filter(ImageFilter.GaussianBlur(radius))
    b2 = b1.filter(ImageFilter.GaussianBlur(radius * 2.6))
    a1 = np.asarray(b1, np.float32)
    a2 = np.asarray(b2, np.float32)
    glow = np.clip(a1 * 0.62 + a2 * 0.72, 0, 255).astype(np.uint8)
    up = Image.fromarray(glow).resize((W, H), Image.BILINEAR)
    g = np.asarray(up, np.float32)
    np.add(canvas, g * np.float32(strength / 255.0), out=canvas)
    return canvas


def chroma(canvas, amount):
    """Radial chromatic aberration; `amount` is roughly pixels at the edge."""
    if amount < 0.25:
        return canvas
    s = 1.0 + amount / 900.0
    base = Image.fromarray(np.clip(canvas * 255.0, 0, 255).astype(np.uint8))
    bw, bh = int(W * s), int(H * s)
    big = base.resize((bw, bh), Image.BILINEAR)
    ox, oy = (bw - W) // 2, (bh - H) // 2
    r = np.asarray(big.crop((ox, oy, ox + W, oy + H)), np.float32)[..., 0]
    sw, sh = int(W / s), int(H / s)
    sml = base.resize((sw, sh), Image.BILINEAR).resize((W, H), Image.BILINEAR)
    b = np.asarray(sml, np.float32)[..., 2]
    canvas[..., 0] = r / 255.0
    canvas[..., 2] = b / 255.0
    return canvas


def grain(canvas, amount, rng, down=3):
    if amount <= 0.0005:
        return canvas
    h, w = H // down, W // down
    n = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    up = np.asarray(Image.fromarray(n).resize((W, H), Image.BILINEAR), np.float32)
    canvas += (up / 255.0 - 0.5)[..., None] * amount
    return canvas


def exposure(canvas, ev=0.0, contrast=1.0, lift=0.0):
    if ev:
        canvas *= 2.0**ev
    if contrast != 1.0:
        canvas = (canvas - 0.5) * contrast + 0.5
    if lift:
        canvas += lift
    return canvas


def tonemap(canvas, shoulder=0.95):
    """Soft filmic roll-off (Reinhard-Jodie style) so bloom never clips hard."""
    np.maximum(canvas, 0.0, out=canvas)
    num = canvas * np.float32(1.0 / (shoulder * shoulder))
    num += 1.0
    num *= canvas
    den = canvas + 1.0
    np.divide(num, den, out=canvas)
    return canvas


def to_u8(canvas):
    return np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8)
