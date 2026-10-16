"""
Scene graph for "NOT JUST AN MCP" — a 1920x1080 / 60 fps title piece.

Every scene is a function (canvas, lt, t, fx, rng) that composites directly
into the float32 canvas.  `lt` is scene-local time, `t` is global time and
`fx` is a mutable dict the post stage reads (flash / chroma / glitch / zoom).

Timing is locked to 120 BPM: beat = 0.5 s, bar = 2.0 s.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

import engine as E
import widgets as Wg
from engine import (ANTON, ARCHIVO, BEBAS, INTER, INTER_B, INTER_BK, INTER_SB,
                    MONO, MONO_B, MONO_XB, CX, CY, H, W, clamp, in_cubic,
                    in_out_cubic, in_out_expo, in_quad, linear, mix, mixc,
                    out_back, out_cubic, out_expo, out_quart, out_quint, pulse,
                    span)

BEAT = 0.5
BAR = 2.0
ACC = E.CRIMSON

_lay: dict = {}


# ==========================================================================
# shared helpers
# ==========================================================================
def impact(lt, times, dur=0.22):
    """Decaying 1 -> 0 envelope triggered at each time in `times`."""
    v = 0.0
    for tt in times:
        if lt >= tt:
            v = max(v, 1.0 - in_cubic(clamp((lt - tt) / dur)))
    return v


def slam(lt, t0, dur=0.42, s0=2.7):
    """Scale / alpha / progress for a punch-in word."""
    p = span(lt, t0, t0 + dur, out_expo)
    return mix(s0, 1.0, p), span(lt, t0, t0 + dur * 0.26), p


def backdrop(canvas, t, accent=ACC, wash=0.26, grid_a=0.85, drift=1.0,
             step=60, wash_y=None, kind="dots"):
    canvas[:] = np.asarray(E.BG, np.float32) / 255.0
    if wash > 0:
        canvas += E.radial_wash(accent, CX, wash_y if wash_y is not None else CY + 210,
                                1040, 1.0) * wash
    if grid_a > 0:
        if kind == "dots":
            g = E.grid_field(step, 1.8, (104, 112, 132), 120, "dots")
            sp = step
        else:
            sp = step * 2
            g = E.grid_field(sp, 0, (62, 68, 84), 95, "lines", 1)
        ox = int((t * 9.0 * drift) % sp)
        oy = int((t * 6.0 * drift) % sp)
        view = g[oy:oy + H, ox:ox + W]
        np.add(canvas, view * np.float32(grid_a), out=canvas)
    return canvas


def glitch(canvas, amt, rng, slices=16):
    if amt <= 0.012:
        return
    n = int(slices * amt) + 1
    for _ in range(n):
        y0 = int(rng.integers(0, H - 10))
        y1 = min(H, y0 + int(rng.integers(8, 86)))
        sh = int(rng.integers(-110, 110) * amt)
        if sh:
            canvas[y0:y1] = np.roll(canvas[y0:y1], sh, axis=1)


def knockout_disc(canvas, wordlayer, cx, cy, r, fill=ACC, ink=(14, 7, 10)):
    """Signature move: a solid accent disc with the type knocked out of it.

    The disc is built only at its own bounding box, so cost scales with the
    radius rather than the frame.
    """
    if r <= 2:
        return
    size = int(r * 2)
    x0, y0 = int(cx - r), int(cy - r)
    disc = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dd = ImageDraw.Draw(disc)
    dd.ellipse((0, 0, size - 1, size - 1), fill=fill + (255,))

    crop = wordlayer.crop((x0, y0, x0 + size, y0 + size))
    dark = Image.new("RGBA", (size, size), ink + (255,))
    dark.putalpha(crop.getchannel("A"))
    disc.alpha_composite(dark)

    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).ellipse((0, 0, size - 1, size - 1), fill=255)
    disc.putalpha(ImageChops.multiply(disc.getchannel("A"), m))

    lyr = E.layer()
    E.blit(lyr, disc, x0, y0)
    E.over(canvas, lyr)


def marquee(layer, text, fontname, size, y, offset, color, alpha=255,
            tracking=10.0, gap=90):
    """Seamless horizontal scrolling text band."""
    key = ("mq", text, fontname, size, color, tracking, gap, alpha)
    tile = _lay.get(key)
    if tile is None:
        one = E.text_img(text, fontname, size, color, tracking=tracking)
        tw = one.width + gap
        reps = int(W / tw) + 3
        tile = Image.new("RGBA", (tw * reps, one.height), (0, 0, 0, 0))
        for i in range(reps):
            tile.alpha_composite(one, (i * tw, 0))
        if alpha < 255:
            tile.putalpha(tile.getchannel("A").point(lambda v: v * alpha // 255))
        _lay[key] = tile
    period = tile.width // 3 if tile.width > 3 * W else tile.width
    ox = -int(offset % period)
    E.blit(layer, tile, ox, int(y - tile.height / 2))
    E.blit(layer, tile, ox + period, int(y - tile.height / 2))


def sweep_streak(add_layer, prog, color=(255, 255, 255), angle=22, width=260,
                 alpha=0.55):
    """Diagonal light sweep across the frame."""
    if prog <= 0 or prog >= 1:
        return
    x = mix(-500, W + 500, prog)
    s = Wg.streak(0, 0, width, H * 1.9, angle, color, alpha * math.sin(prog * math.pi), 40)
    E.place(add_layer, s, x, CY)


# ==========================================================================
# S1 — HOOK  (0.0 - 8.0)
# ==========================================================================
_HOOK_W = ["THIS", "ISN'T", "AN"]


def _hook_layout():
    hit = _lay.get("hookA")
    if hit:
        return hit
    size, gap = 166, 40
    widths = [E.text_width(w, ANTON, size) for w in _HOOK_W]
    total = sum(widths) + gap * (len(widths) - 1)
    x = CX - total / 2
    pos = []
    for wd in widths:
        pos.append(x + wd / 2)
        x += wd + gap
    _lay["hookA"] = (size, pos, widths)
    return _lay["hookA"]


def scene_hook(canvas, lt, t, fx, rng):
    partB = lt >= 4.0
    hits = [0.5, 1.0, 1.5, 2.0] if not partB else [4.5, 5.0, 5.5]
    imp = impact(lt, hits, 0.20)
    fx["chroma"] = max(fx["chroma"], imp * 9.0)
    fx["shake"] = max(fx["shake"], imp * 7.0)

    backdrop(canvas, t, wash=0.15 + imp * 0.26, grid_a=span(lt, 0.15, 0.9) * 0.8,
             wash_y=CY + 120)

    base = E.layer()
    glow = E.layer()

    if not partB:
        size, pos, widths = _hook_layout()
        y1, y2 = 392, 648

        for i, wtxt in enumerate(_HOOK_W):
            t0 = 0.5 + i * 0.5
            if lt < t0:
                continue
            sc, al, p = slam(lt, t0, 0.40, 2.5)
            img = E.text_img(wtxt, ANTON, size, E.WHITE)
            b = (1.0 - p) * 16
            E.place_blur(base, img, pos[i], y1, sc, 0, al, b)

        # --- MCP lockup with box wipe
        if lt >= 2.0:
            sc, al, p = slam(lt, 2.0, 0.44, 2.3)
            mcp = E.text_img("MCP", ANTON, 286, E.WHITE)
            bw = mcp.width + 92
            bh = mcp.height + 52
            wipe = span(lt, 2.06, 2.42, out_expo)
            box = Image.new("RGBA", (int(bw), int(bh)), (0, 0, 0, 0))
            bd = ImageDraw.Draw(box)
            bd.rectangle((0, 0, bw * wipe, bh), fill=ACC + (255,))
            if wipe < 1.0:
                bd.rectangle((max(0, bw * wipe - 5), 0, bw * wipe, bh),
                             fill=(255, 255, 255, 255))
            E.place(base, box, CX, y2, sc, 0, al)
            E.place_blur(base, mcp, CX, y2 + 2, sc, 0, al, (1.0 - p) * 14)
            if lt < 2.5:
                E.place(glow, box, CX, y2, sc, 0, al * span(2.5, lt, 2.5) * 0.0 + al * 0.35)

        # --- subtitle
        if lt >= 2.5:
            full = "MODEL  CONTEXT  PROTOCOL"
            n = int(span(lt, 2.5, 3.25, linear) * len(full))
            sub = full[:n]
            if sub:
                si = E.text_img(sub, MONO_B, 34, (196, 202, 214), tracking=7.0)
                E.place(base, si, CX - E.text_width(full, MONO_B, 34, 7.0) / 2, 848,
                        1.0, 0, 1.0, anchor="lm")
            if (lt * 2.4) % 1 < 0.55 and lt < 3.6:
                cw = E.text_width(full[:n], MONO_B, 34, 7.0)
                cx0 = CX - E.text_width(full, MONO_B, 34, 7.0) / 2 + cw
                cd = ImageDraw.Draw(base)
                cd.rectangle((cx0, 848 - 20, cx0 + 16, 848 + 20), fill=ACC + (255,))

        # --- scanner intro
        if lt < 0.52:
            p = span(lt, 0.0, 0.5, out_cubic)
            E.add(canvas, Wg.scanner_line(mix(-80, H + 80, p), color=ACC,
                                          alpha=1.0 - p * 0.3, spread=120), 0.9)

        sweep_streak(glow, span(lt, 3.0, 3.75, in_out_cubic))
        fx["glitch"] = max(fx["glitch"], pulse(lt, 3.66, 3.9, hold=0.2) * 0.85)
        fx["chroma"] = max(fx["chroma"], pulse(lt, 3.66, 3.9, hold=0.2) * 16)

    else:
        # ---------------- part B ----------------
        lines = [("IT'S A", 104, 356, E.BONE, 4.5),
                 ("MOTION", 232, 512, E.WHITE, 5.0),
                 ("DESIGNER.", 232, 724, ACC, 5.5)]
        for txt, sz, y, col, t0 in lines:
            if lt < t0:
                continue
            sc, al, p = slam(lt, t0, 0.40, 2.2 if sz > 150 else 1.8)
            img = E.text_img(txt, ANTON, sz, col)
            E.place_blur(base, img, CX, y, sc, 0, al, (1.0 - p) * 15)
            if col == ACC and p > 0.3:
                E.place(glow, img, CX, y, sc, 0, 0.5)

        if lt >= 6.0:
            u = span(lt, 6.0, 6.5, out_expo)
            lw = 760 * u
            d = ImageDraw.Draw(base)
            d.rectangle((CX - lw / 2, 836, CX + lw / 2, 842), fill=ACC + (255,))
            k = E.text_img("POWERED BY CLAUDE  //  MCP", MONO_B, 28,
                           (178, 184, 198), tracking=6.5)
            E.place(base, k, CX, 896, 1.0, 0, span(lt, 6.3, 6.9, out_cubic))

        sweep_streak(glow, span(lt, 6.6, 7.3, in_out_cubic), alpha=0.42)
        if lt >= 7.4:
            fx["zoom"] = mix(1.0, 1.10, span(lt, 7.4, 8.0, in_quad))

    E.over(canvas, base)
    E.add(canvas, glow, 0.85)
    if imp > 0:
        canvas += imp * 0.06


# ==========================================================================
# S2 — THE BRIDGE / NODE DIAGRAM  (8.0 - 18.0)
# ==========================================================================
_LOG = [
    ("comp.create", "1920x1080  60fps"),
    ("text.add", '"MCP"  anton 286'),
    ("key.scale", "0 -> 100  ease:expo"),
    ("fx.glow", "crimson  18px  screen"),
    ("audio.sync", "120 bpm  beat-locked"),
    ("render.queue", "h264  crf 18"),
]


def scene_bridge(canvas, lt, t, fx, rng):
    imp = impact(lt, [0.3, 0.55, 1.0, 7.5], 0.22)
    fx["chroma"] = max(fx["chroma"], imp * 6.0)
    push = span(lt, 0.0, 10.0, linear) * 0.055

    backdrop(canvas, t, wash=0.16 + impact(lt, [1.0, 7.5], 0.5) * 0.20,
             grid_a=0.9, wash_y=CY + 60)

    base = E.layer()
    glow = E.layer()
    d = ImageDraw.Draw(base)

    yC = CY - 52
    xl, xr = CX - 548, CX + 548

    # ---- connection paths (drawn before the cards)
    pL = E.bezier_pts((xl + 172, yC), (xl + 300, yC + 74),
                      (CX - 290, yC - 74), (CX - 156, yC), 46)
    pR = E.bezier_pts((CX + 156, yC), (CX + 290, yC + 74),
                      (xr - 300, yC - 74), (xr - 172, yC), 46)

    uL = span(lt, 1.15, 2.05, out_cubic)
    uR = span(lt, 1.55, 2.45, out_cubic)
    for pts, u in ((pL, uL), (pR, uR)):
        if u > 0:
            E.draw_path(d, pts, (158, 166, 188, 205), 4, 0, u)

    # travelling energy pulses
    if lt > 2.2:
        gd = ImageDraw.Draw(glow)
        for pts, u in ((pL, uL), (pR, uR)):
            if u < 0.99:
                continue
            for k in range(3):
                up = ((lt - 2.2) * 0.42 + k / 3.0) % 1.0
                px, py = E.poly_at(pts, up)
                fade = math.sin(up * math.pi) ** 0.5
                for rr, aa in ((16, 50), (9, 120), (4.5, 255)):
                    gd.ellipse((px - rr, py - rr, px + rr, py + rr),
                               fill=ACC + (int(aa * fade),))
                tail = E.poly_at(pts, max(0.0, up - 0.07))
                gd.line([tail, (px, py)], fill=ACC + (int(120 * fade),), width=5)

    # ---- cards
    cards = [
        (xl, yC, "AI MODEL", "Claude", "C", False, 1.10, 0.30, 340, 160),
        (xr, yC, "APP", "After\nEffects", "Ae", False, 1.10, 0.55, 340, 170),
        (CX, yC, "PROTOCOL", "MCP", "<>", True, 1.0, 0.95, 312, 182),
    ]
    for (cx, cy, lab, title, icon, hot, _s, t0, cw, ch) in cards:
        if lt < t0:
            continue
        p = span(lt, t0, t0 + 0.55, out_back if not hot else out_expo)
        al = span(lt, t0, t0 + 0.22)
        img = Wg.node_card(lab, title, ACC, cw, ch, icon, hot=hot)
        if hot:
            sc = mix(1.9, 1.0, p) * (1 + push)
            beat = 1.0 + 0.022 * math.sin((lt - t0) * math.pi * 4.0)
            sc *= beat
        else:
            sc = (1 + push)
            dx = (-230 if cx < CX else 230) * (1 - p)
            cx = cx + dx
        E.place(base, img, cx, cy, sc, 0, al)
        if hot and p > 0.2:
            E.place(glow, img, cx, cy, sc, 0, 0.30 + 0.12 * math.sin(lt * 5))

    # impact ring on the protocol card
    if 0.95 <= lt < 1.9:
        rp = span(lt, 0.95, 1.8, out_expo)
        E.place(glow, Wg.ring(int(mix(40, 420, rp)), ACC, 4,
                              int(220 * (1 - rp)), glow=0.8), CX, yC)

    # ---- chips
    if lt >= 2.6:
        E.place(base, Wg.chip("tool calls", E.BONE, (20, 21, 26), ACC, 24, dot=True),
                CX - 350, yC + 108, mix(1.3, 1.0, span(lt, 2.6, 2.9, out_back)), 0,
                span(lt, 2.6, 2.85))
    if lt >= 2.9:
        E.place(base, Wg.chip("controlling", E.BONE, (20, 21, 26), ACC, 24, dot=True),
                CX + 350, yC + 108, mix(1.3, 1.0, span(lt, 2.9, 3.2, out_back)), 0,
                span(lt, 2.9, 3.15))

    # ---- kicker
    if lt >= 0.15:
        E.place(base, Wg.kicker("01 — THE BRIDGE", ACC, 24, 6.0),
                150, 168, 1.0, 0, span(lt, 0.15, 0.6, out_cubic), anchor="lm")

    # headline — swaps to the pay-off line near the end of the scene
    swap = span(lt, 7.5, 7.9, in_out_cubic)
    if lt >= 0.45 and swap < 1.0:
        a = span(lt, 0.45, 1.0, out_cubic) * (1 - swap)
        h1 = E.text_img("CLAUDE DRIVES AFTER EFFECTS.", ANTON, 82, E.WHITE)
        E.place(base, h1, CX, 288 - swap * 70,
                mix(1.1, 1.0, span(lt, 0.45, 0.95, out_expo)), 0, a)

    # ---- streaming log panel
    if lt >= 3.4:
        pw, ph = 1180, 246
        px, py = CX - pw / 2, 706
        pa = span(lt, 3.4, 3.8, out_cubic)
        pan = Wg.glass_card(pw, ph, ACC, 18, (14, 15, 19), 225, 34)
        E.place(base, pan, CX, py + ph / 2, 1.0, 0, pa)
        hdr = E.text_img("tool://after_effects", MONO_B, 22, (130, 136, 152), tracking=2.0)
        E.place(base, hdr, px + 34, py + 34, 1.0, 0, pa, anchor="lm")
        d.rectangle((px + 34, py + 52, px + pw - 34, py + 53), fill=(255, 255, 255, 30))

        for i, (fn, arg) in enumerate(_LOG):
            lt0 = 3.9 + i * 0.42
            if lt < lt0:
                continue
            a = span(lt, lt0, lt0 + 0.22)
            sx = px + 36 + (1 - a) * 26
            yy = py + 86 + i * 25
            if yy > py + ph - 20:
                continue
            E.place(base, Wg.glyph_arrow(18, ACC, 3), sx, yy, 1, 0, a, anchor="lm")
            E.place(base, E.text_img(fn, MONO_B, 21, E.BONE), sx + 32, yy, 1, 0, a,
                    anchor="lm")
            E.place(base, E.text_img(arg, MONO, 21, (126, 132, 148)), sx + 232, yy,
                    1, 0, a * 0.95, anchor="lm")
            if lt - lt0 < 0.3:
                E.place(glow, E.text_img(fn, MONO_B, 21, E.WHITE), sx + 32, yy,
                        1, 0, 1 - (lt - lt0) / 0.3, anchor="lm")

    # ---- closing caption
    if lt >= 7.8:
        a = span(lt, 7.8, 8.2, out_cubic)
        cap = E.text_img("ONE PROTOCOL. FULL CONTROL.", ANTON, 88, E.WHITE)
        E.place(base, cap, CX, 288, mix(1.16, 1.0, span(lt, 7.8, 8.25, out_expo)), 0, a)
        E.place(glow, cap, CX, 288, 1.0, 0, a * 0.22)
        fx["chroma"] = max(fx["chroma"], impact(lt, [7.8], 0.25) * 10)

    sweep_streak(glow, span(lt, 6.4, 7.1, in_out_cubic), alpha=0.32)
    if lt >= 9.5:
        fx["zoom"] = mix(1.0, 1.08, span(lt, 9.5, 10.0, in_quad))

    E.over(canvas, base)
    E.add(canvas, glow, 0.9)


# ==========================================================================
# S3 — THE PROMPT  (18.0 - 26.0)
# ==========================================================================
_PROMPT = 'make a 60fps title sequence - crimson, brutalist, 1920x1080'
_RESULT = [
    ("comp created", "1920x1080 @ 60fps", "0.4s"),
    ("14 layers", "typography / grid / glow", "1.1s"),
    ("212 keyframes", "eased, beat-locked", "2.0s"),
    ("render queued", "h264 - crf 18", "2.6s"),
]


def scene_prompt(canvas, lt, t, fx, rng):
    backdrop(canvas, t, wash=0.17, grid_a=0.7, wash_y=CY + 40, kind="lines")
    base = E.layer()
    glow = E.layer()

    pw, ph = 1296, 536
    p_in = span(lt, 0.0, 0.62, out_expo)
    sc = mix(1.22, 1.0, p_in) * (1 + span(lt, 0, 8, linear) * 0.04)
    panel = Wg.window_panel(pw, ph, "sancorp — claude x after effects", ACC)
    pan_layer = E.layer()
    E.place(pan_layer, panel, CX, CY - 8, sc, 0, span(lt, 0.0, 0.25))

    # content drawn in panel space then transformed with the same scale
    inner = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    di = ImageDraw.Draw(inner)
    top = 54

    # prompt line
    E.place(inner, E.text_img(">", MONO_B, 30, ACC), 42, top + 56, anchor="lm")
    n = int(span(lt, 0.55, 2.95, linear) * len(_PROMPT))
    typed = _PROMPT[:n]
    if typed:
        E.place(inner, E.text_img(typed, MONO, 30, E.BONE), 78, top + 56, anchor="lm")
    cx0 = 78 + E.text_width(typed, MONO, 30)
    if lt < 3.0 and (lt * 2.6) % 1 < 0.6:
        di.rectangle((cx0 + 4, top + 38, cx0 + 20, top + 74), fill=ACC + (255,))

    di.rectangle((42, top + 96, pw - 42, top + 97), fill=(255, 255, 255, 26))

    # thinking
    if 3.0 <= lt < 3.7:
        a = span(lt, 3.0, 3.15) * (1 - span(lt, 3.5, 3.7))
        dots = "." * (1 + int((lt * 6) % 3))
        E.place(inner, E.text_img("claude is composing" + dots, MONO, 26,
                                  (138, 144, 160)), 42, top + 140, anchor="lm")

    # result rows
    for i, (title, sub, tm) in enumerate(_RESULT):
        rt = 3.7 + i * 0.5
        if lt < rt:
            continue
        a = span(lt, rt, rt + 0.24)
        yy = top + 150 + i * 70
        xo = (1 - a) * 30
        E.place(inner, Wg.glyph_check(26, ACC, 4), 44 + xo, yy, 1, 0, a, anchor="lm")
        E.place(inner, E.text_img(title, MONO_B, 29, E.WHITE), 90 + xo, yy, 1, 0, a,
                anchor="lm")
        E.place(inner, E.text_img(sub, MONO, 25, (132, 138, 154)), 400 + xo, yy, 1, 0,
                a * 0.95, anchor="lm")
        E.place(inner, E.text_img(tm, MONO_B, 24, ACC), pw - 46, yy, 1, 0, a,
                anchor="rm")

    # bottom status strip
    if lt >= 5.8:
        a = span(lt, 5.8, 6.1)
        di.rectangle((42, ph - 74, pw - 42, ph - 73), fill=(255, 255, 255, 26))
        E.place(inner, E.text_img("BUILD COMPLETE", MONO_B, 24, ACC, tracking=4.0),
                42, ph - 44, 1, 0, a, anchor="lm")
        E.place(inner, E.text_img("3.1s total", MONO, 24, (138, 144, 160)),
                pw - 46, ph - 44, 1, 0, a, anchor="rm")

    E.place(pan_layer, inner, CX, CY - 8 + 6 * sc, sc, 0, 1.0)
    E.over(canvas, pan_layer)

    # enter flash
    fx["flash"] = max(fx["flash"], pulse(lt, 2.98, 3.26, hold=0.1) * 0.30)
    fx["chroma"] = max(fx["chroma"], impact(lt, [3.0, 5.8], 0.25) * 8)

    if lt >= 0.2:
        E.place(base, Wg.kicker("02 — THE PROMPT", ACC, 24, 6.0), 150, 180, 1.0, 0,
                span(lt, 0.2, 0.65, out_cubic), anchor="lm")
    cap = "NO TIMELINE. NO KEYFRAMING BY HAND."
    if lt >= 6.4:
        E.place(base, E.text_img(cap, MONO_B, 30, (196, 202, 214), tracking=4.5),
                CX, H - 136, 1.0, 0, span(lt, 6.4, 6.9, out_cubic))

    sweep_streak(glow, span(lt, 6.0, 6.7, in_out_cubic), alpha=0.3)
    if lt >= 7.4:
        fx["zoom"] = mix(1.0, 1.12, span(lt, 7.4, 8.0, in_quad))
        fx["flash"] = max(fx["flash"], span(lt, 7.82, 8.0, in_quad) * 0.5)

    E.over(canvas, base)
    E.add(canvas, glow, 0.85)


# ==========================================================================
# S4 — KINETIC TYPE  (26.0 - 34.0)
# ==========================================================================
def scene_kinetic(canvas, lt, t, fx, rng):
    partB = lt >= 4.5
    canvas[:] = np.asarray((5, 5, 7), np.float32) / 255.0

    base = E.layer()
    glow = E.layer()

    if not partB:
        canvas += E.radial_wash(E.CRIM_DEEP, CX, CY, 1150, 1.0) * 0.40

        # background marquees
        marquee(base, "MOTION  SYSTEM  ", ANTON, 96, 130, lt * 150,
                (46, 48, 58), 255, 14, 60)
        marquee(base, "RENDER  FREE  ", ANTON, 96, H - 130, -lt * 150,
                (46, 48, 58), 255, 14, 60)

        # hero words counter-sliding (kept on their own layer so the disc can
        # knock them out)
        words = E.layer()
        a1 = span(lt, 0.05, 0.75, out_expo)
        a2 = span(lt, 0.25, 0.95, out_expo)
        w1 = E.text_img("MONOSPACE", ANTON, 196, E.WHITE)
        w2 = E.text_img("TYPE", ANTON, 290, E.WHITE)
        drift = (lt - 0.5) * 26
        E.place(words, w1, mix(-760, CX, a1) - drift * 0.5, 426, 1.0, 0, a1)
        E.place(words, w2, mix(W + 760, CX, a2) + drift * 0.5, 676, 1.0, 0, a2)

        E.over(canvas, base)
        E.over(canvas, words)

        # accent disc with knocked-out type
        dcx, dcy = CX + 272, CY + 14
        if lt >= 1.05:
            rr = mix(0, 430, span(lt, 1.05, 1.8, out_expo))
            if lt > 3.25:
                rr = mix(430, 0, span(lt, 3.25, 3.9, in_out_cubic))
            knockout_disc(canvas, words, dcx, dcy, rr)
            if rr > 6:
                E.place(glow, Wg.ring(int(rr), E.CRIM_HOT, 3, 150, 0.7), dcx, dcy)
            if 1.05 <= lt < 2.0:
                E.place(glow, Wg.ring(int(mix(40, 700, span(lt, 1.05, 2.0, out_expo))),
                                      E.WHITE, 3,
                                      int(190 * (1 - span(lt, 1.05, 2.0))), 0.5),
                        dcx, dcy)

        # centre hairline sweep
        if 1.6 <= lt < 2.6:
            p = span(lt, 1.6, 2.5, in_out_expo)
            gd = ImageDraw.Draw(glow)
            gd.line([(mix(-200, W, p) - 700, dcy), (mix(-200, W, p), dcy)],
                    fill=(255, 255, 255, 230), width=3)

        fx["chroma"] = max(fx["chroma"], impact(lt, [1.05, 2.0, 3.25], 0.22) * 12)
        if lt >= 3.95:
            fx["flash"] = max(fx["flash"], pulse(lt, 3.95, 4.35, hold=0.15) * 0.55)

    else:
        l2 = lt - 4.5
        canvas += E.radial_wash(ACC, CX, CY, 1000, 1.0) * (0.22 + impact(l2, [0.0], 0.5) * 0.3)
        marquee(base, "60 FPS  ", ANTON, 70, 128, l2 * 240, (54, 24, 32), 255, 12, 50)
        marquee(base, "60 FPS  ", ANTON, 70, H - 128, -l2 * 240, (54, 24, 32), 255, 12, 50)

        sc, al, p = slam(l2, 0.05, 0.5, 3.0)
        big = E.text_img("60 FPS", ANTON, 330, E.WHITE)
        for k in range(4, 0, -1):
            E.place(base, big, CX, 470, sc * (1 + k * 0.055), 0, al * 0.10 * k / 4)
        E.place_blur(base, big, CX, 470, sc, 0, al, (1 - p) * 20)

        if l2 >= 0.6:
            sc2, al2, p2 = slam(l2, 0.6, 0.42, 1.9)
            sub = E.text_img("ZERO RENDER QUEUE", ANTON, 120, ACC)
            E.place_blur(base, sub, CX, 720, sc2, 0, al2, (1 - p2) * 12)
            E.place(glow, sub, CX, 720, sc2, 0, al2 * 0.55)

        if l2 >= 1.25:
            a = span(l2, 1.25, 1.7, out_cubic)
            chips = ["PROGRESSIVE", "REC.709", "CRF 18", "NATIVE 1080p"]
            tot = 0
            imgs = []
            for c in chips:
                im = Wg.chip(c, (206, 212, 224), (18, 19, 24), None, 22, dot=False)
                imgs.append(im)
                tot += im.width - 24
            x = CX - tot / 2
            for i, im in enumerate(imgs):
                aa = span(l2, 1.25 + i * 0.08, 1.6 + i * 0.08, out_cubic)
                E.place(base, im, x + im.width / 2, 866, 1.0, 0, aa)
                x += im.width - 24

        fx["chroma"] = max(fx["chroma"], impact(l2, [0.05, 0.6], 0.24) * 11)
        fx["shake"] = max(fx["shake"], impact(l2, [0.05], 0.2) * 8)
        E.over(canvas, base)
        if l2 >= 2.9:
            fx["zoom"] = mix(1.0, 1.1, span(l2, 2.9, 3.5, in_quad))

    E.add(canvas, glow, 0.9)


# ==========================================================================
# S5 — RENDER  (34.0 - 42.0)
# ==========================================================================
def _render_word():
    hit = _lay.get("renderword")
    if hit:
        return hit
    txt = E.text_img("RENDER", ARCHIVO, 250, E.WHITE)
    a = txt.getchannel("A")
    inner = a.filter(ImageFilter.MinFilter(9))
    chk = Wg.checker(txt.width, txt.height, 24, (108, 112, 124), (62, 65, 74)).copy()
    chk.putalpha(inner)
    out = txt.copy()
    out.alpha_composite(chk)
    _lay["renderword"] = out
    return out


def scene_render(canvas, lt, t, fx, rng):
    done = lt >= 4.6
    backdrop(canvas, t, wash=0.14 if not done else 0.34, grid_a=0.55,
             wash_y=CY + 160, kind="lines")

    base = E.layer()
    glow = E.layer()
    d = ImageDraw.Draw(base)

    word = _render_word()
    wa = span(lt, 0.05, 0.5, out_expo)
    E.place(base, word, CX, 340, mix(1.14, 1.0, wa), 0, span(lt, 0.05, 0.3))

    # progress maths — a slow crawl that suddenly snaps to 100 %
    if lt < 2.4:
        frac = span(lt, 0.6, 2.4, linear) * 0.07
    else:
        frac = 0.07 + span(lt, 2.4, 4.5, out_expo) * 0.93
    frac = clamp(frac)
    pct = int(round(frac * 100))

    bx, bw = CX - 740, 1480
    by = 560
    if lt >= 0.5:
        a = span(lt, 0.5, 0.8, out_cubic)
        lbl = "RENDERING" + "." * (1 + int((lt * 4) % 3)) if not done else "COMPLETE"
        E.place(base, E.text_img(lbl, MONO_B, 36, E.WHITE if not done else ACC,
                                 tracking=3.0), bx, by - 46, 1, 0, a, anchor="lm")
        E.place(base, E.text_img(f"{pct}%", MONO_XB, 40, ACC if not done else E.WHITE),
                bx + bw, by - 46, 1, 0, a, anchor="rm")
        E.place(base, Wg.progress_bar(bw, 22, frac, ACC), bx + bw / 2, by + 10, 1, 0, a)

    # ETA readout
    if lt >= 1.0:
        a = span(lt, 1.0, 1.3, out_cubic)
        if lt < 2.4:
            secs = 10829 - int(span(lt, 1.0, 2.4, linear) * 40)
        else:
            secs = int(mix(10789, 0, span(lt, 2.4, 4.4, out_expo)))
        hh, mm, ss = secs // 3600, (secs % 3600) // 60, secs % 60
        E.place(base, E.text_img("ETA", MONO_B, 28, (128, 134, 150), tracking=5.0),
                bx + 4, by + 118, 1, 0, a, anchor="lm")
        tc = "%02d:%02d:%02d" % (hh, mm, ss)
        col = E.WHITE if not done else ACC
        E.place(base, E.text_img(tc, MONO_XB, 128, col), bx, by + 212, 1, 0, a,
                anchor="lm")
        if done:
            E.place(glow, E.text_img(tc, MONO_XB, 128, col), bx, by + 212, 1, 0, 0.5,
                    anchor="lm")

        # right-hand spec stack
        specs = [("CODEC", "H.264 HIGH"), ("FRAMES", "3 360"), ("SIZE", "1920x1080"),
                 ("RATE", "60.000 FPS")]
        for i, (k, v) in enumerate(specs):
            aa = span(lt, 1.2 + i * 0.12, 1.5 + i * 0.12, out_cubic)
            yy = by + 128 + i * 52
            E.place(base, E.text_img(k, MONO, 24, (110, 116, 132), tracking=3.0),
                    bx + bw - 330, yy, 1, 0, aa, anchor="lm")
            E.place(base, E.text_img(v, MONO_B, 26, E.BONE), bx + bw, yy, 1, 0, aa,
                    anchor="rm")

    # completion burst
    if done:
        bp = span(lt, 4.6, 5.5, out_expo)
        E.place(glow, Wg.ring(int(mix(60, 900, bp)), ACC, 6, int(230 * (1 - bp)), 0.9),
                CX, CY)
        E.place(glow, Wg.ring(int(mix(20, 560, span(lt, 4.72, 5.6, out_expo))), E.WHITE,
                              3, int(170 * (1 - span(lt, 4.72, 5.6))), 0.5), CX, CY)
        gd = ImageDraw.Draw(glow)
        for i in range(18):
            ang = i * 20 + 7
            r0 = mix(70, 520, bp)
            r1 = r0 + mix(0, 190, bp)
            aa = int(200 * (1 - bp))
            gd.line([(CX + math.cos(math.radians(ang)) * r0,
                      CY + math.sin(math.radians(ang)) * r0),
                     (CX + math.cos(math.radians(ang)) * r1,
                      CY + math.sin(math.radians(ang)) * r1)],
                    fill=ACC + (aa,), width=4)
        fx["flash"] = max(fx["flash"], pulse(lt, 4.58, 4.92, hold=0.1) * 0.62)
        fx["chroma"] = max(fx["chroma"], impact(lt, [4.6], 0.4) * 18)
        fx["shake"] = max(fx["shake"], impact(lt, [4.6], 0.3) * 10)

        if lt >= 5.1:
            a = span(lt, 5.1, 5.5, out_cubic)
            E.place(base, Wg.chip("RENDERED IN 4.2s", E.WHITE, (26, 12, 17), ACC, 30,
                                  dot=True), CX, H - 150,
                    mix(1.25, 1.0, span(lt, 5.1, 5.45, out_back)), 0, a)

    if lt >= 0.15:
        E.place(base, Wg.kicker("03 — THE PUNCHLINE", ACC, 24, 6.0), 150, 180, 1.0, 0,
                span(lt, 0.15, 0.6, out_cubic), anchor="lm")

    if lt >= 7.4:
        fx["zoom"] = mix(1.0, 1.09, span(lt, 7.4, 8.0, in_quad))

    E.over(canvas, base)
    E.add(canvas, glow, 0.95)


# ==========================================================================
# S6 — OUTPUT MONTAGE  (42.0 - 48.0)
# ==========================================================================
def _tile_bars(tile, u, lt, i):
    d = ImageDraw.Draw(tile)
    w, h = tile.size
    n = 7
    for k in range(n):
        ph = (math.sin(lt * 3.4 + k * 0.8) * 0.5 + 0.5) * 0.78 + 0.12
        bh = (h - 64) * ph * u
        x = 34 + k * ((w - 68) / n)
        bw = (w - 68) / n - 10
        d.rectangle((x, h - 32 - bh, x + bw, h - 32),
                    fill=(ACC if k % 2 else (236, 238, 243)) + (235,))


def _tile_rings(tile, u, lt, i):
    d = ImageDraw.Draw(tile)
    w, h = tile.size
    for k in range(5):
        p = ((lt * 0.62 + k * 0.2) % 1.0)
        r = p * min(w, h) * 0.46 * u
        a = int(220 * (1 - p))
        d.ellipse((w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r),
                  outline=ACC + (a,), width=3)


def _tile_wave(tile, u, lt, i):
    d = ImageDraw.Draw(tile)
    w, h = tile.size
    pts = []
    for x in range(0, w, 6):
        yy = h / 2 + math.sin(x * 0.018 + lt * 5.2) * h * 0.26 * u \
            * (0.5 + 0.5 * math.sin(x * 0.006 - lt * 2))
        pts.append((x, yy))
    if len(pts) > 1:
        d.line(pts, fill=(236, 238, 243, 240), width=4, joint="curve")


def _tile_type(tile, u, lt, i):
    w, h = tile.size
    s = 0.86 + 0.14 * math.sin(lt * 4 + i)
    E.place(tile, E.text_img("Ae", ARCHIVO, 104, E.WHITE), w / 2, h / 2, s * u)


def _tile_square(tile, u, lt, i):
    d = ImageDraw.Draw(tile)
    w, h = tile.size
    sq = Image.new("RGBA", (int(min(w, h) * 0.62), int(min(w, h) * 0.62)), (0, 0, 0, 0))
    ImageDraw.Draw(sq).rectangle((2, 2, sq.width - 3, sq.height - 3),
                                 outline=ACC + (240,), width=5)
    E.place(tile, sq, w / 2, h / 2, u, lt * 46)


def _tile_count(tile, u, lt, i):
    w, h = tile.size
    v = int((lt * 180) % 1000)
    E.place(tile, E.text_img("%03d" % v, MONO_XB, 86, E.WHITE), w / 2, h / 2 - 10, u)
    E.place(tile, E.text_img("FRAMES OUT", MONO_B, 18, E.STEEL, tracking=3.0),
            w / 2, h / 2 + 54, u)


_TILES = [_tile_bars, _tile_rings, _tile_wave, _tile_type, _tile_square, _tile_count]


def scene_montage(canvas, lt, t, fx, rng):
    backdrop(canvas, t, wash=0.20, grid_a=0.7, wash_y=CY)
    base = E.layer()
    glow = E.layer()

    panels = Wg.grid_panels(3, 2, 34, 1480, 720)
    merge = span(lt, 4.35, 5.55, in_out_expo)

    for i, (x, y, w, h) in enumerate(panels):
        t0 = 0.25 + i * 0.25
        if lt < t0:
            continue
        p = span(lt, t0, t0 + 0.42, out_back)
        a = span(lt, t0, t0 + 0.18)
        u = span(lt, t0 + 0.1, t0 + 0.7, out_cubic)

        card = Wg.glass_card(int(w), int(h), ACC, 18, (16, 17, 21), 235, 40)
        tile = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
        _TILES[i](tile, u, lt, i)
        lab = ["MOTION", "PULSE", "SIGNAL", "TYPE", "FRAME", "RATE"][i]
        E.place(tile, E.text_img(lab, MONO_B, 19, E.STEEL, tracking=3.4), 26,
                h - 26, 1, 0, u, anchor="lm")
        E.place(tile, E.text_img("0%d" % (i + 1), MONO_B, 19, ACC), w - 26, h - 26,
                1, 0, u, anchor="rm")

        composite = card.copy()
        E.place(composite, tile, composite.width / 2, composite.height / 2)

        # converge toward centre at the end
        cxp = x + w / 2
        cyp = y + h / 2
        cxp = mix(cxp, CX, merge)
        cyp = mix(cyp, CY, merge)
        sc = p * mix(1.0, 0.25, merge)
        E.place(base, composite, cxp, cyp, sc, 0, a * (1 - merge * 0.8))

    if lt >= 0.1:
        E.place(base, Wg.kicker("04 — OUTPUT", ACC, 24, 6.0), 150, 150, 1.0, 0,
                span(lt, 0.1, 0.5, out_cubic), anchor="lm")

    fx["chroma"] = max(fx["chroma"],
                       impact(lt, [0.25 + i * 0.25 for i in range(6)], 0.16) * 6)
    if lt >= 5.2:
        fx["flash"] = max(fx["flash"], span(lt, 5.5, 6.0, in_quad) * 0.62)
        fx["zoom"] = mix(1.0, 1.28, span(lt, 5.2, 6.0, in_quad))

    E.over(canvas, base)
    E.add(canvas, glow, 0.9)


# ==========================================================================
# S7 — OUTRO  (48.0 - 56.0)
# ==========================================================================
_SPECS = ["1920 x 1080", "60 FPS", "H.264", "MCP DRIVEN"]


def scene_outro(canvas, lt, t, fx, rng):
    backdrop(canvas, t, wash=0.16 + pulse(lt, 0.4, 1.4, hold=0.3) * 0.22,
             grid_a=span(lt, 0.6, 1.6) * 0.7, wash_y=CY + 40, drift=0.5)

    base = E.layer()
    glow = E.layer()
    d = ImageDraw.Draw(base)

    # opening light seam
    if lt < 1.0:
        p = span(lt, 0.0, 0.55, out_expo)
        lw = mix(10, W * 0.8, p)
        lh = mix(10, 3, p)
        gd = ImageDraw.Draw(glow)
        gd.rectangle((CX - lw / 2, CY - lh / 2, CX + lw / 2, CY + lh / 2),
                     fill=(255, 255, 255, int(255 * (1 - span(lt, 0.45, 1.0)))))

    # wordmark builds letter by letter
    word = "SANCORP"
    size = 214
    tr = 10.0
    total = E.text_width(word, ANTON, size, tr)
    x = CX - total / 2
    for i, ch in enumerate(word):
        t0 = 0.55 + i * 0.075
        cw = E.text_width(ch, ANTON, size)
        if lt >= t0:
            p = span(lt, t0, t0 + 0.42, out_expo)
            a = span(lt, t0, t0 + 0.2)
            img = E.text_img(ch, ANTON, size, E.WHITE)
            E.place_blur(base, img, x + cw / 2, mix(CY - 150, CY - 70, p), 1.0, 0, a,
                         (1 - p) * 13)
        x += cw + tr

    # underline sweep
    if lt >= 1.35:
        u = span(lt, 1.35, 1.95, out_expo)
        lw = total * u
        d.rectangle((CX - total / 2, CY + 56, CX - total / 2 + lw, CY + 64),
                    fill=ACC + (255,))
        if u < 1.0:
            d.rectangle((CX - total / 2 + lw - 8, CY + 50, CX - total / 2 + lw, CY + 70),
                        fill=(255, 255, 255, 255))

    # subtitle
    if lt >= 1.8:
        E.place(base, E.text_img("MOTION SYSTEMS", MONO_B, 34, (196, 202, 214),
                                 tracking=11.0),
                CX, CY + 128, 1.0, 0, span(lt, 1.8, 2.3, out_cubic))

    # spec chips
    if lt >= 2.35:
        imgs = [Wg.chip(s, (210, 216, 228), (18, 19, 24), None, 23) for s in _SPECS]
        tot = sum(im.width - 26 for im in imgs)
        x = CX - tot / 2
        for i, im in enumerate(imgs):
            a = span(lt, 2.35 + i * 0.1, 2.75 + i * 0.1, out_cubic)
            E.place(base, im, x + im.width / 2, CY + 232,
                    mix(1.18, 1.0, span(lt, 2.35 + i * 0.1, 2.8 + i * 0.1, out_back)),
                    0, a)
            x += im.width - 26

    # spark marks
    if lt >= 1.1:
        for sx, sy, dl in ((CX - total / 2 - 70, CY - 130, 1.1),
                           (CX + total / 2 + 70, CY - 130, 1.25)):
            a = span(lt, dl, dl + 0.35, out_cubic)
            E.place(glow, Wg.glyph_spark(44, ACC, 3), sx, sy,
                    0.7 + 0.3 * math.sin(lt * 4), lt * 30, a)

    # repo line
    if lt >= 3.4:
        E.place(base, E.text_img("github.com/Sanycorp/Sancorp-Project-", MONO, 24,
                                 (120, 126, 142), tracking=2.0),
                CX, H - 150, 1.0, 0, span(lt, 3.4, 3.9, out_cubic) * (1 - span(lt, 7.0, 7.8)))

    sweep_streak(glow, span(lt, 3.0, 3.9, in_out_cubic), alpha=0.4)
    fx["chroma"] = max(fx["chroma"], impact(lt, [0.55, 1.35], 0.3) * 8)

    # fade out
    if lt >= 6.9:
        f = span(lt, 6.9, 7.9, in_out_cubic)
        E.over(canvas, base, 1.0)
        E.add(canvas, glow, 0.9)
        canvas *= (1.0 - f)
        return

    E.over(canvas, base)
    E.add(canvas, glow, 0.9)


# ==========================================================================
# timeline
# ==========================================================================
TIMELINE = [
    (0.0, 8.0, scene_hook, True),
    (8.0, 18.0, scene_bridge, True),
    (18.0, 26.0, scene_prompt, True),
    (26.0, 34.0, scene_kinetic, False),
    (34.0, 42.0, scene_render, True),
    (42.0, 48.0, scene_montage, True),
    (48.0, 56.0, scene_outro, False),
]
TOTAL = TIMELINE[-1][1]
