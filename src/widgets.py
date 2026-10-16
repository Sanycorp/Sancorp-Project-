"""Reusable UI / motion-design widgets built on top of engine primitives."""
from __future__ import annotations

import math

from PIL import Image, ImageChops, ImageDraw, ImageFilter

import engine as E
from engine import (ANTON, ARCHIVO, BEBAS, INTER, INTER_B, INTER_BK, INTER_SB,
                    MONO, MONO_B, MONO_XB)

_cache: dict = {}


# --------------------------------------------------------------------------
def checker(w, h, cell=22, a=(44, 46, 52), b=(26, 27, 32)):
    key = ("chk", w, h, cell, a, b)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    img = Image.new("RGBA", (w, h), a + (255,))
    d = ImageDraw.Draw(img)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2:
                d.rectangle((x, y, x + cell - 1, y + cell - 1), fill=b + (255,))
    _cache[key] = img
    return img


# --------------------------------------------------------------------------
def glass_card(w, h, accent=E.CRIMSON, radius=22, fill=(18, 19, 24), fill_a=232,
               border_a=46, glow=0.0):
    """Frosted dark card with a hairline border and a top sheen."""
    key = ("card", w, h, accent, radius, fill, fill_a, border_a, round(glow, 2))
    hit = _cache.get(key)
    if hit is not None:
        return hit

    pad = 30
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    box = (pad, pad, pad + w, pad + h)

    if glow > 0:
        g = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(g)
        gd.rounded_rectangle(box, radius=radius, fill=accent + (int(150 * glow),))
        g = g.filter(ImageFilter.GaussianBlur(16))
        img.alpha_composite(g)

    d.rounded_rectangle(box, radius=radius, fill=fill + (fill_a,))

    # top sheen — smooth vertical falloff (a hard-edged rect reads as a seam)
    gh = max(2, int(h * 0.8))
    col = Image.new("L", (1, gh))
    col.putdata([int(22 * (1.0 - yy / gh) ** 1.7) for yy in range(gh)])
    sh_rgb = Image.new("RGBA", (w, gh), (255, 255, 255, 255))
    sh_rgb.putalpha(col.resize((w, gh)))
    sheen = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sheen.paste(sh_rgb, (pad, pad))
    clip = Image.new("L", img.size, 0)
    ImageDraw.Draw(clip).rounded_rectangle(box, radius=radius, fill=255)
    sheen.putalpha(ImageChops.multiply(sheen.getchannel("A"), clip))
    img.alpha_composite(sheen)
    d.rounded_rectangle(box, radius=radius, outline=(255, 255, 255, border_a), width=2)
    _cache[key] = img
    return img


def node_card(label, title, accent=E.CRIMSON, w=340, h=160, icon=None,
              title_font=INTER_BK, title_size=44, hot=False):
    """Card used in the protocol diagram: tiny label + bold title + icon tile.

    `title` may contain a newline for a stacked two-line lockup.
    """
    key = ("node", label, title, accent, w, h, icon, title_font, title_size, hot)
    hit = _cache.get(key)
    if hit is not None:
        return hit

    base = glass_card(w, h, accent, radius=20,
                      fill=(28, 13, 18) if hot else (18, 19, 24),
                      fill_a=242, border_a=78 if hot else 42,
                      glow=0.9 if hot else 0.0)
    pad = (base.width - w) // 2
    d = ImageDraw.Draw(base)

    isz = 52
    ix, iy = pad + 22, pad + 22
    d.rounded_rectangle((ix, iy, ix + isz, iy + isz), radius=13,
                        fill=(accent + (78,)) if hot else (255, 255, 255, 20),
                        outline=accent + (180,) if hot else (255, 255, 255, 38), width=2)
    if icon:
        gi = E.text_img(icon, INTER_BK if len(icon) <= 2 else MONO_B, 26,
                        accent if hot else (214, 218, 228))
        E.place(base, gi, ix + isz / 2, iy + isz / 2)

    tx = ix + isz + 18
    lb = E.text_img(label.upper(), MONO_B, 16, accent if hot else E.STEEL, tracking=3.0)
    E.place(base, lb, tx, iy + 13, anchor="lm")

    lines = title.split("\n")
    if len(lines) == 1:
        tt = E.text_img(lines[0], title_font, title_size, E.WHITE if hot else E.BONE)
        E.place(base, tt, tx - 2, iy + 50, anchor="lm")
    else:
        ls = int(title_size * 0.78)
        y = iy + 46
        for ln in lines:
            tt = E.text_img(ln, title_font, ls, E.WHITE if hot else E.BONE)
            E.place(base, tt, tx - 2, y, anchor="lm")
            y += ls * 1.02

    _cache[key] = base
    return base


# --------------------------------------------------------------------------
def chip(text, fg=E.WHITE, bg=(22, 23, 28), accent=None, size=22, padx=18, pady=10,
         fontname=MONO_B, dot=False, tracking=1.4):
    key = ("chip", text, fg, bg, accent, size, padx, pady, fontname, dot, tracking)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    t = E.text_img(text, fontname, size, fg, tracking=tracking)
    dw = (size + 10) if dot else 0
    w = t.width + 2 * padx + dw
    h = t.height + 2 * pady
    pad = 16
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    box = (pad, pad, pad + w, pad + h)
    d.rounded_rectangle(box, radius=h // 2, fill=bg + (236,),
                        outline=(accent + (150,)) if accent else (255, 255, 255, 34), width=2)
    x = pad + padx
    if dot:
        cy = pad + h / 2
        r = size * 0.21
        c = accent or E.CRIMSON
        d.ellipse((x - 2, cy - r, x - 2 + 2 * r, cy + r), fill=c + (255,))
        x += dw
    E.place(img, t, x, pad + h / 2, anchor="lm")
    _cache[key] = img
    return img


# --------------------------------------------------------------------------
def kicker(text, accent=E.CRIMSON, size=20, tracking=5.0, bar=True):
    """Small uppercase section label with a leading accent bar."""
    key = ("kick", text, accent, size, tracking, bar)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    t = E.text_img(text.upper(), MONO_B, size, E.BONE, tracking=tracking)
    bw = 46 if bar else 0
    img = Image.new("RGBA", (t.width + bw + 24, max(t.height, size) + 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cy = img.height / 2
    if bar:
        d.rectangle((0, cy - 2, 34, cy + 2), fill=accent + (255,))
    E.place(img, t, bw, cy, anchor="lm")
    _cache[key] = img
    return img


# --------------------------------------------------------------------------
def progress_bar(w, h, frac, accent=E.CRIMSON, track=(40, 42, 50), glow=True):
    pad = 22
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = h / 2
    d.rounded_rectangle((pad, pad, pad + w, pad + h), radius=r, fill=track + (190,))
    fw = max(0.0, min(1.0, frac)) * w
    if fw > 2:
        if glow:
            g = Image.new("RGBA", img.size, (0, 0, 0, 0))
            gd = ImageDraw.Draw(g)
            gd.rounded_rectangle((pad, pad, pad + fw, pad + h), radius=r, fill=accent + (190,))
            g = g.filter(ImageFilter.GaussianBlur(10))
            img.alpha_composite(g)
        d.rounded_rectangle((pad, pad, pad + fw, pad + h), radius=r, fill=accent + (255,))
        d.rounded_rectangle((pad, pad, pad + fw, pad + h * 0.5), radius=r,
                            fill=(255, 255, 255, 48))
    return img


# --------------------------------------------------------------------------
def window_panel(w, h, title="sancorp — mcp bridge", accent=E.CRIMSON, radius=18):
    """macOS-ish window chrome used for the terminal scene."""
    key = ("win", w, h, title, accent, radius)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    pad = 34
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((pad, pad + 10, pad + w, pad + h + 10),
                                         radius=radius, fill=(0, 0, 0, 190))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(22)))

    box = (pad, pad, pad + w, pad + h)
    d.rounded_rectangle(box, radius=radius, fill=(13, 14, 18, 250))
    d.rounded_rectangle(box, radius=radius, outline=(255, 255, 255, 40), width=2)

    bar_h = 54
    d.rounded_rectangle((pad, pad, pad + w, pad + bar_h + radius), radius=radius,
                        fill=(24, 25, 31, 255))
    d.rectangle((pad, pad + bar_h - 1, pad + w, pad + bar_h + 1), fill=(255, 255, 255, 26))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = pad + 28 + i * 26
        d.ellipse((cx - 7, pad + bar_h / 2 - 7, cx + 7, pad + bar_h / 2 + 7), fill=c + (255,))
    tt = E.text_img(title, MONO, 20, (150, 156, 170))
    E.place(img, tt, pad + w / 2, pad + bar_h / 2)
    _cache[key] = img
    return img


# --------------------------------------------------------------------------
def hud(t, total, accent=E.CRIMSON, alpha=1.0, label="SANCORP // MOTION SYSTEM"):
    """Persistent corner HUD: ticks, timecode, format stamp."""
    img = E.layer()
    d = ImageDraw.Draw(img)
    m = 56
    L = 34
    col = (255, 255, 255, int(52 * alpha))
    for (x, y, dx, dy) in ((m, m, 1, 1), (E.W - m, m, -1, 1),
                           (m, E.H - m, 1, -1), (E.W - m, E.H - m, -1, -1)):
        d.line([(x, y), (x + dx * L, y)], fill=col, width=2)
        d.line([(x, y), (x, y + dy * L)], fill=col, width=2)

    f = int(round(t * E.FPS))
    tc = "%02d:%02d:%02d" % (int(t) // 60, int(t) % 60, f % E.FPS)
    a = int(150 * alpha)
    E.place(img, E.text_img(tc, MONO_B, 22, (190, 196, 208), tracking=2.0),
            m + 6, E.H - m - 26, anchor="lb", alpha=alpha * 0.9)
    E.place(img, E.text_img("1920x1080  60FPS", MONO, 20, (118, 124, 138), tracking=2.4),
            E.W - m - 6, E.H - m - 26, anchor="rb", alpha=alpha * 0.9)
    E.place(img, E.text_img(label, MONO_B, 20, (118, 124, 138), tracking=3.4),
            m + 6, m + 26, anchor="lt", alpha=alpha * 0.9)

    # rec dot
    blink = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(t * 5.0))
    rx, ry = E.W - m - 8, m + 34
    d.ellipse((rx - 7, ry - 7, rx + 7, ry + 7), fill=accent + (int(255 * blink * alpha),))
    E.place(img, E.text_img("LIVE", MONO_B, 20, (200, 206, 218), tracking=3.0),
            rx - 22, ry, anchor="rm", alpha=alpha * 0.9)

    # progress hairline
    d.rectangle((m, E.H - m - 2, E.W - m, E.H - m), fill=(255, 255, 255, int(22 * alpha)))
    d.rectangle((m, E.H - m - 2, m + (E.W - 2 * m) * E.clamp(t / total), E.H - m),
                fill=accent + (int(190 * alpha),))
    return img


# --------------------------------------------------------------------------
def scanner_line(y, width=E.W, color=E.CRIMSON, alpha=1.0, thick=3, spread=90):
    """Horizontal energy scan line with falloff."""
    img = E.layer()
    d = ImageDraw.Draw(img)
    for i in range(6, 0, -1):
        a = int(alpha * 26 * (i / 6.0) ** 0.6)
        hgt = spread * (i / 6.0)
        d.rectangle((0, y - hgt, width, y + hgt), fill=color + (a,))
    d.rectangle((0, y - thick / 2, width, y + thick / 2),
                fill=(255, 255, 255, int(235 * alpha)))
    return img


def streak(x, y, w, h, angle, color, alpha=1.0, blur=24):
    """Diagonal light streak / lens flare bar."""
    s = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    ImageDraw.Draw(s).rectangle((0, 0, w, h), fill=color + (int(255 * alpha),))
    s = s.filter(ImageFilter.GaussianBlur(blur))
    s = s.rotate(angle, resample=Image.BICUBIC, expand=True)
    return s


def glyph_check(size=24, color=E.CRIMSON, width=4):
    """Vector check mark (latin-subset fonts have no U+2713)."""
    key = ("chk_g", size, color, width)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.line([(s * 0.16, s * 0.54), (s * 0.40, s * 0.78), (s * 0.86, s * 0.20)],
           fill=color + (255,), width=width, joint="curve")
    _cache[key] = img
    return img


def glyph_arrow(size=24, color=E.CRIMSON, width=3):
    """Vector right arrow."""
    key = ("arr_g", size, color, width)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    s = size
    img = Image.new("RGBA", (int(s * 1.25), s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = s * 0.5
    d.line([(s * 0.08, y), (s * 1.08, y)], fill=color + (255,), width=width)
    d.line([(s * 0.70, y - s * 0.26), (s * 1.10, y), (s * 0.70, y + s * 0.26)],
           fill=color + (255,), width=width, joint="curve")
    _cache[key] = img
    return img


def glyph_spark(size=28, color=E.CRIMSON, width=3):
    """Small 4-point spark / asterisk mark."""
    key = ("spk_g", size, color, width)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = s / 2
    for a in range(0, 360, 45):
        r = c * (0.95 if a % 90 == 0 else 0.62)
        x = c + math.cos(math.radians(a)) * r
        y = c + math.sin(math.radians(a)) * r
        d.line([(c, c), (x, y)], fill=color + (255,), width=width)
    _cache[key] = img
    return img


def ring(r, color, width=3, alpha=255, glow=0.0):
    """Circle outline with optional bloom halo."""
    pad = int(width * 2 + 24 + glow * 30)
    size = int(r * 2 + pad * 2)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    box = (pad, pad, pad + 2 * r, pad + 2 * r)
    if glow > 0:
        g = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).ellipse(box, outline=color + (int(190 * glow),),
                                  width=int(width * 2.2))
        img.alpha_composite(g.filter(ImageFilter.GaussianBlur(12)))
    d.ellipse(box, outline=color + (alpha,), width=width)
    return img


def grid_panels(cols, rows, gap, w, h):
    """Return list of (x, y, w, h) tiles for a montage grid centred on screen."""
    tw = (w - gap * (cols - 1)) / cols
    th = (h - gap * (rows - 1)) / rows
    x0 = E.CX - w / 2
    y0 = E.CY - h / 2
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append((x0 + c * (tw + gap), y0 + r * (th + gap), tw, th))
    return out
