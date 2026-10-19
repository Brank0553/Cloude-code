#!/usr/bin/env python3
"""Animacija "Šta je API?" — renderuje video 854x480 (480p), 30 fps.

Pokretanje:
    python3 render.py                 # renderuje kompletan video
    python3 render.py --preview 5 30  # snima PNG sličice za zadane sekunde

Potrebno: Python 3, Pillow, NumPy i ffmpeg (sa libx264).
"""
import math
import os
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 854, 480          # 480p, 16:9
S = 2                    # supersampling radi glatkih ivica
FPS = 30
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "api_animacija_480p.mp4")

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
FONT_FILES = {
    "sans": "DejaVuSans.ttf",
    "bold": "DejaVuSans-Bold.ttf",
    "mono": "DejaVuSansMono.ttf",
    "monob": "DejaVuSansMono-Bold.ttf",
}
_font_cache = {}


def font(kind, size):
    key = (kind, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT_DIR + FONT_FILES[kind], max(1, int(round(size * S))))
    return _font_cache[key]


# ---------------------------------------------------------------- boje
BG_TOP = (10, 15, 35)
BG_BOT = (24, 32, 66)
TXT = (226, 232, 240)
MUTED = (148, 163, 184)
DARK = (15, 23, 42)
PANEL = (30, 41, 59)
PANEL2 = (51, 65, 85)
BLUE = (56, 189, 248)
GREEN = (74, 222, 128)
ORANGE = (251, 146, 60)
PINK = (244, 114, 182)
YELLOW = (250, 204, 21)
PURPLE = (167, 139, 250)
RED = (248, 113, 113)
SKIN = (253, 214, 170)
HAIR = (68, 46, 30)


def A(c, a=1.0):
    return (c[0], c[1], c[2], int(255 * max(0.0, min(1.0, a))))


# ---------------------------------------------------------------- easing
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def prog(t, a, b):
    if b <= a:
        return 1.0 if t >= a else 0.0
    return clamp((t - a) / (b - a))


def eo(p):
    return 1 - (1 - p) ** 3


def eio(p):
    return p * p * (3 - 2 * p)


def back(p):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


def lerp(a, b, p):
    return a + (b - a) * p


def fade(t, start, dur=0.5):
    return eo(prog(t, start, start + dur))


def window(t, start, end, dur=0.4):
    """Alfa koja se pojavi u `start` i nestane u `end`."""
    return fade(t, start, dur) * (1 - prog(t, end, end + dur))


# ---------------------------------------------------------------- platno
class Cv:
    """Tanki omotač oko ImageDraw koji radi u 854x480 koordinatama."""

    def __init__(self, img):
        self.img = img
        self.d = ImageDraw.Draw(img, "RGBA")

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, w=1, r=0):
        if x1 - x0 < 0.5 or y1 - y0 < 0.5:
            return
        r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S, fill=fill,
                                 outline=outline, width=max(1, int(round(w * S))) if outline else 0)

    def ellipse(self, x0, y0, x1, y1, fill=None, outline=None, w=1):
        if x1 - x0 < 0.5 or y1 - y0 < 0.5:
            return
        self.d.ellipse([x0 * S, y0 * S, x1 * S, y1 * S], fill=fill, outline=outline,
                       width=max(1, int(round(w * S))) if outline else 0)

    def circle(self, x, y, r, fill=None, outline=None, w=1):
        self.ellipse(x - r, y - r, x + r, y + r, fill, outline, w)

    def chord(self, x, y, r, start, end, fill):
        if r <= 0.3:
            return
        self.d.chord([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], start, end, fill=fill)

    def arc(self, x, y, r, start, end, fill, w=2):
        if r <= 0.3:
            return
        self.d.arc([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], start, end, fill=fill,
                   width=max(1, int(round(w * S))))

    def line(self, pts, fill, w=2):
        self.d.line([(x * S, y * S) for x, y in pts], fill=fill, width=max(1, int(round(w * S))),
                    joint="curve")

    def poly(self, pts, fill=None, outline=None):
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill, outline=outline)

    def text(self, x, y, s, size, fill, anchor="mm", kind="sans"):
        if size < 1 or not s:
            return
        alpha = fill[3] if len(fill) == 4 else 255
        if alpha <= 0:
            return
        f = font(kind, size)
        if alpha >= 255:
            self.d.text((x * S, y * S), s, font=f, fill=fill[:3], anchor=anchor)
            return
        # ImageDraw ne poštuje alfu kod teksta, pa ga crtamo kroz masku
        x0, y0, x1, y1 = self.d.textbbox((x * S, y * S), s, font=f, anchor=anchor)
        x0, y0, x1, y1 = int(math.floor(x0)) - 1, int(math.floor(y0)) - 1, int(math.ceil(x1)) + 1, int(math.ceil(y1)) + 1
        if x1 <= x0 or y1 <= y0:
            return
        mask = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(mask).text((x * S - x0, y * S - y0), s, font=f, fill=alpha, anchor=anchor)
        self.img.paste(fill[:3], (x0, y0, x1, y1), mask)

    def tlen(self, s, size, kind="sans"):
        return font(kind, size).getlength(s) / S

    def dashed(self, x0, y0, x1, y1, fill, w=2, dash=8, gap=6, offset=0.0):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            return
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        pos = -(offset % (dash + gap))
        while pos < length:
            a = max(0, pos)
            b = min(length, pos + dash)
            if b > a:
                self.line([(x0 + ux * a, y0 + uy * a), (x0 + ux * b, y0 + uy * b)], fill, w)
            pos += dash + gap

    def arrow_head(self, x, y, ang, size, fill):
        p1 = (x, y)
        p2 = (x - size * math.cos(ang - 0.45), y - size * math.sin(ang - 0.45))
        p3 = (x - size * math.cos(ang + 0.45), y - size * math.sin(ang + 0.45))
        self.poly([p1, p2, p3], fill=fill)


def wrap(c, text, size, maxw, kind="sans"):
    lines, cur = [], ""
    for word in text.split():
        cand = (cur + " " + word).strip()
        if c.tlen(cand, size, kind) <= maxw or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------- zajednički elementi
def make_background():
    w, h = W * S, H * S
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    top, bot = np.array(BG_TOP, float), np.array(BG_BOT, float)
    img = top[None, None, :] * (1 - y[..., None]) + bot[None, None, :] * y[..., None]
    img = np.broadcast_to(img, (h, w, 3)).copy()
    # blagi sjaj u uglovima
    for cx, cy, col, rad in ((0.15, 0.2, BLUE, 0.45), (0.9, 0.85, PURPLE, 0.5)):
        dist = np.sqrt(((x - cx) * 16 / 9) ** 2 + (y - cy) ** 2)
        g = np.clip(1 - dist / rad, 0, 1) ** 2 * 0.10
        img += g[..., None] * (np.array(col, float)[None, None, :] - img)
    bg = Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(bg, "RGBA")
    for gx in range(0, W + 1, 32):
        for gy in range(0, H + 1, 32):
            d.ellipse([gx * S - 1, gy * S - 1, gx * S + 1, gy * S + 1], fill=(255, 255, 255, 14))
    return bg


BG = make_background()


def header(c, t, title, idx, color=BLUE):
    a = fade(t, 0.05, 0.5)
    c.text(30, 30, title, 22, A(TXT, a), anchor="lm", kind="bold")
    w = c.tlen(title, 22, "bold")
    c.rect(30, 45, 30 + w * eo(prog(t, 0.25, 1.0)), 48, fill=A(color, a), r=1.5)
    c.text(W - 30, 30, f"{idx:02d} / {len(SCENES):02d}", 12, A(MUTED, a), anchor="rm", kind="mono")


def caption(c, t, start, text, y=430, size=15, end=None, color=TXT, maxw=740):
    a = fade(t, start, 0.6)
    if end is not None:
        a *= 1 - prog(t, end, end + 0.4)
    if a <= 0:
        return
    lines = wrap(c, text, size, maxw)
    lh = size * 1.4
    bw = max(c.tlen(l, size) for l in lines) + 36
    bh = len(lines) * lh + 16
    dy = (1 - a) * 8
    c.rect(W / 2 - bw / 2, y - bh / 2 + dy, W / 2 + bw / 2, y + bh / 2 + dy,
           fill=A(PANEL, 0.88 * a), outline=A(PANEL2, a), w=1, r=10)
    for i, l in enumerate(lines):
        c.text(W / 2, y - (len(lines) - 1) * lh / 2 + i * lh + dy, l, size, A(color, a))


def pill(c, x, y, text, size, fg, bg, a=1.0, kind="bold", padx=10, pady=5):
    w = c.tlen(text, size, kind) + padx * 2
    h = size + pady * 2
    c.rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2, fill=A(bg, a), r=h / 2)
    c.text(x, y, text, size, A(fg, a), kind=kind)
    return w


def bubble(c, x, y, text, size, a, tail_x=None, color=TXT):
    """Oblačić za govor; (x, y) je centar, rep pokazuje prema dolje."""
    if a <= 0:
        return
    w = c.tlen(text, size) + 24
    h = size + 18
    tx = x if tail_x is None else tail_x
    c.poly([(tx - 7, y + h / 2 - 1), (tx + 7, y + h / 2 - 1), (tx, y + h / 2 + 10)], fill=A(color, a))
    c.rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2, fill=A(color, a), r=10)
    c.text(x, y, text, size, A(DARK, a), kind="bold")


# ---------------------------------------------------------------- likovi i ikone
def person(c, x, y, sc, body, a, hair=HAIR):
    c.rect(x - 26 * sc, y - 18 * sc, x + 26 * sc, y + 42 * sc, fill=A(body, a), r=16 * sc)
    c.circle(x, y - 38 * sc, 16 * sc, fill=A(SKIN, a))
    c.chord(x, y - 40 * sc, 16.5 * sc, 180, 360, A(hair, a))
    c.circle(x - 5 * sc, y - 36 * sc, 1.8 * sc, fill=A(DARK, a))
    c.circle(x + 5 * sc, y - 36 * sc, 1.8 * sc, fill=A(DARK, a))
    c.arc(x, y - 33 * sc, 6 * sc, 30, 150, A(DARK, a), 1.4 * sc)


def guest(c, x, y, sc, a):
    person(c, x, y, sc, BLUE, a)
    # sto
    c.rect(x + 22 * sc, y + 14 * sc, x + 100 * sc, y + 20 * sc, fill=A((180, 120, 70), a), r=2 * sc)
    c.rect(x + 58 * sc, y + 20 * sc, x + 64 * sc, y + 52 * sc, fill=A((150, 95, 55), a))
    c.rect(x + 44 * sc, y + 50 * sc, x + 78 * sc, y + 54 * sc, fill=A((150, 95, 55), a), r=2 * sc)


def waiter(c, x, y, sc, a):
    person(c, x, y, sc, (35, 38, 52), a, hair=(30, 30, 30))
    c.poly([(x - 11 * sc, y - 18 * sc), (x + 11 * sc, y - 18 * sc), (x, y + 8 * sc)], fill=A((241, 245, 249), a))
    c.poly([(x, y - 12 * sc), (x - 9 * sc, y - 17 * sc), (x - 9 * sc, y - 7 * sc)], fill=A(RED, a))
    c.poly([(x, y - 12 * sc), (x + 9 * sc, y - 17 * sc), (x + 9 * sc, y - 7 * sc)], fill=A(RED, a))
    c.circle(x, y - 12 * sc, 2.5 * sc, fill=A(RED, a))
    # ubrus preko ruke
    c.rect(x + 18 * sc, y + 4 * sc, x + 30 * sc, y + 26 * sc, fill=A((241, 245, 249), a), r=2 * sc)


def kitchen(c, x, y, sc, a, t, cooking):
    c.rect(x - 78 * sc, y - 62 * sc, x + 78 * sc, y + 54 * sc, fill=A(PANEL2, a), outline=A(MUTED, a * 0.6), w=1.5, r=12 * sc)
    c.rect(x - 50 * sc, y - 76 * sc, x + 50 * sc, y - 54 * sc, fill=A(ORANGE, a), r=8 * sc)
    c.text(x, y - 65 * sc, "KUHINJA", 11 * sc, A(DARK, a), kind="bold")
    # šporet
    c.rect(x - 52 * sc, y + 18 * sc, x + 52 * sc, y + 50 * sc, fill=A(PANEL, a), r=4 * sc)
    c.circle(x - 30 * sc, y + 34 * sc, 4 * sc, fill=A(MUTED, a))
    c.circle(x + 30 * sc, y + 34 * sc, 4 * sc, fill=A(MUTED, a))
    # plamen
    if cooking > 0:
        for i, dx in enumerate((-14, -4, 6, 16)):
            hgt = (7 + 4 * math.sin(t * 18 + i * 1.7)) * sc
            c.poly([(x + (dx - 5) * sc, y + 18 * sc), (x + (dx + 5) * sc, y + 18 * sc), (x + dx * sc, y + 18 * sc - hgt)],
                   fill=A(ORANGE, a * cooking))
    # lonac
    c.rect(x - 30 * sc, y - 16 * sc, x + 30 * sc, y + 12 * sc, fill=A((148, 163, 184), a), r=5 * sc)
    c.rect(x - 36 * sc, y - 18 * sc, x + 36 * sc, y - 12 * sc, fill=A((203, 213, 225), a), r=3 * sc)
    c.rect(x - 42 * sc, y - 8 * sc, x - 30 * sc, y - 3 * sc, fill=A((100, 116, 139), a), r=2 * sc)
    c.rect(x + 30 * sc, y - 8 * sc, x + 42 * sc, y - 3 * sc, fill=A((100, 116, 139), a), r=2 * sc)
    # para
    if cooking > 0:
        for i, dx in enumerate((-14, 0, 14)):
            pts = []
            for k in range(12):
                yy = y - 22 * sc - k * 2.6 * sc
                xx = x + dx * sc + math.sin(k * 0.7 + t * 5 + i) * 3 * sc
                pts.append((xx, yy))
            c.line(pts, A(TXT, a * cooking * 0.55), 2 * sc)


def note(c, x, y, a):
    c.rect(x - 13, y - 16, x + 13, y + 16, fill=A((248, 250, 252), a), r=2)
    for i in range(4):
        c.line([(x - 8, y - 8 + i * 6), (x + (8 if i < 3 else 2), y - 8 + i * 6)], A(MUTED, a), 1.5)


def pizza(c, x, y, a, r=19):
    c.circle(x, y, r, fill=A((214, 150, 72), a))
    c.circle(x, y, r - 4, fill=A((250, 200, 90), a))
    for dx, dy in ((-6, -6), (6, -4), (-2, 6), (8, 7), (-9, 3)):
        c.circle(x + dx * r / 19, y + dy * r / 19, 3 * r / 19, fill=A((200, 50, 50), a))
    for k in range(3):
        ang = k * math.pi / 3
        c.line([(x - math.cos(ang) * (r - 4), y - math.sin(ang) * (r - 4)),
                (x + math.cos(ang) * (r - 4), y + math.sin(ang) * (r - 4))], A((214, 150, 72), a * 0.7), 1)


def sun(c, x, y, r, a, t=0.0):
    for i in range(8):
        ang = i * math.pi / 4 + t * 0.6
        c.line([(x + math.cos(ang) * r * 1.35, y + math.sin(ang) * r * 1.35),
                (x + math.cos(ang) * r * 1.8, y + math.sin(ang) * r * 1.8)], A(YELLOW, a), max(1.2, r * 0.22))
    c.circle(x, y, r, fill=A(YELLOW, a))


def phone(c, x, y, a, updated, t, tap=None):
    c.rect(x - 46, y - 86, x + 46, y + 86, fill=A(DARK, a), outline=A(MUTED, a * 0.8), w=2.5, r=14)
    c.rect(x - 38, y - 72, x + 38, y + 72, fill=A((23, 37, 84), a), r=6)
    c.rect(x - 12, y - 80, x + 12, y - 76, fill=A(PANEL2, a), r=2)
    c.text(x, y - 58, "Vrijeme", 10, A(MUTED, a), kind="bold")
    c.text(x, y - 40, "Sarajevo", 11, A(TXT, a))
    u = fade(t, updated, 0.6) if updated is not None else 0
    if u < 1:
        c.text(x, y - 5, "--°C", 22, A(MUTED, a * (1 - u)), kind="bold")
    if u > 0:
        sun(c, x, y - 12 + (1 - u) * 6, 8, a * u, t)
        c.text(x, y + 14 + (1 - u) * 6, "18°C", 20, A(TXT, a * u), kind="bold")
        c.text(x, y + 34, "Sunčano", 10, A(YELLOW, a * u))
    c.rect(x - 28, y + 48, x + 28, y + 64, fill=A(BLUE, a), r=8)
    c.text(x, y + 56, "Osvježi", 9, A(DARK, a), kind="bold")
    if tap is not None:
        p = prog(t, tap, tap + 0.8)
        if 0 < p < 1:
            c.circle(x + 6, y + 56, 6 + p * 18, outline=A(TXT, a * (1 - p)), w=2)
            c.circle(x + 6, y + 56, 6, fill=A(TXT, a * (1 - p) * 0.8))


def server(c, x, y, a, t, busy):
    for i in range(3):
        top = y + i * 34
        c.rect(x - 56, top, x + 56, top + 28, fill=A(PANEL2, a), outline=A(MUTED, a * 0.5), w=1, r=5)
        for k in range(3):
            c.line([(x - 44, top + 8 + k * 6), (x + 6, top + 8 + k * 6)], A(PANEL, a), 2)
        for k in range(3):
            on = 1.0
            if busy > 0:
                on = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * 14 + i * 2.1 + k * 1.3))
            col = GREEN if k < 2 else ORANGE
            c.circle(x + 22 + k * 11, top + 14, 3.2, fill=A(col, a * on))


def database(c, x, y, a, w=46, h=46):
    c.rect(x - w / 2, y, x + w / 2, y + h, fill=A(PURPLE, a))
    c.ellipse(x - w / 2, y + h - 8, x + w / 2, y + h + 8, fill=A(PURPLE, a))
    for k in (1, 2):
        c.arc(x, y + k * h / 3 - 1, w / 2, 20, 160, A((120, 95, 210), a), 1.5)
    c.ellipse(x - w / 2, y - 8, x + w / 2, y + 8, fill=A((196, 181, 253), a))


def icon(c, kind, x, y, col, a):
    k = A(col, a)
    bg = A(PANEL, a)
    if kind == "weather":
        sun(c, x - 5, y - 5, 6, a)
        c.circle(x + 1, y + 6, 6, fill=A(TXT, a))
        c.circle(x + 9, y + 3, 7.5, fill=A(TXT, a))
        c.rect(x - 4, y + 6, x + 16, y + 12, fill=A(TXT, a), r=3)
    elif kind == "card":
        c.rect(x - 15, y - 10, x + 15, y + 10, fill=k, r=3)
        c.rect(x - 15, y - 5, x + 15, y - 1, fill=bg)
        c.rect(x - 11, y + 3, x - 4, y + 7, fill=A(YELLOW, a), r=1)
    elif kind == "lock":
        c.arc(x, y - 4, 8, 180, 360, k, 3)
        c.line([(x - 8, y - 4), (x - 8, y)], k, 3)
        c.line([(x + 8, y - 4), (x + 8, y)], k, 3)
        c.rect(x - 11, y - 1, x + 11, y + 13, fill=k, r=3)
        c.circle(x, y + 5, 2.5, fill=bg)
        c.line([(x, y + 5), (x, y + 10)], bg, 2)
    elif kind == "pin":
        c.circle(x, y - 5, 9, fill=k)
        c.poly([(x - 8, y - 1), (x + 8, y - 1), (x, y + 13)], fill=k)
        c.circle(x, y - 5, 3.5, fill=bg)
    elif kind == "social":
        c.rect(x - 14, y - 11, x + 14, y + 6, fill=k, r=6)
        c.poly([(x - 8, y + 4), (x - 1, y + 4), (x - 10, y + 12)], fill=k)
        for dx in (-7, 0, 7):
            c.circle(x + dx, y - 2.5, 2.2, fill=bg)
    elif kind == "ai":
        c.rect(x - 11, y - 11, x + 11, y + 11, outline=k, w=2, r=3)
        for dd in (-5, 0, 5):
            c.line([(x + dd, y - 15), (x + dd, y - 11)], k, 1.6)
            c.line([(x + dd, y + 11), (x + dd, y + 15)], k, 1.6)
            c.line([(x - 15, y + dd), (x - 11, y + dd)], k, 1.6)
            c.line([(x + 11, y + dd), (x + 15, y + dd)], k, 1.6)
        c.text(x, y, "AI", 9, k, kind="bold")
    elif kind == "home":
        c.poly([(x - 15, y), (x, y - 13), (x + 15, y)], fill=k)
        c.rect(x - 10, y - 1, x + 10, y + 13, fill=k)
        c.rect(x - 3, y + 4, x + 3, y + 13, fill=bg)
    elif kind == "calendar":
        c.rect(x - 13, y - 10, x + 13, y + 13, outline=k, w=2, r=3)
        c.rect(x - 13, y - 10, x + 13, y - 4, fill=k, r=2)
        c.line([(x - 6, y - 14), (x - 6, y - 8)], k, 2)
        c.line([(x + 6, y - 14), (x + 6, y - 8)], k, 2)
        for gx in (-7, 0, 7):
            for gy in (1, 7):
                c.rect(x + gx - 2, y + gy - 2, x + gx + 2, y + gy + 2, fill=k)


def check(c, x, y, a, col=GREEN, r=11):
    c.circle(x, y, r, fill=A(col, a))
    c.line([(x - r * 0.45, y), (x - r * 0.1, y + r * 0.35), (x + r * 0.5, y - r * 0.35)], A(DARK, a), r * 0.22)


def move_arc(p, x0, y0, x1, y1, lift=40):
    return lerp(x0, x1, p), lerp(y0, y1, p) - math.sin(math.pi * p) * lift


# ================================================================= SCENE
def s_intro(c, t):
    cols = [BLUE, GREEN, ORANGE]
    words = ["Application", "Programming", "Interface"]
    up = eio(prog(t, 2.0, 2.8))
    for i, ch in enumerate("API"):
        p = prog(t, 0.2 + i * 0.25, 0.9 + i * 0.25)
        if p <= 0:
            continue
        size = int(110 * back(p))
        x = 427 + (i - 1) * 92
        y = lerp(205, 150, up)
        c.text(x, y, ch, size, A(cols[i], min(1, p * 2)), kind="bold")
    for i, wd in enumerate(words):
        a = fade(t, 2.6 + i * 0.35, 0.5)
        if a <= 0:
            continue
        lx = 427 + (i - 1) * 92
        wx = 427 + (i - 1) * 200
        lp = eo(prog(t, 2.5 + i * 0.35, 3.0 + i * 0.35))
        c.line([(lx, 200), (lerp(lx, wx, lp), lerp(200, 226, lp))], A(cols[i], 0.6 * a), 1.5)
        size = 23
        total = c.tlen(wd, size, "bold")
        first = c.tlen(wd[0], size, "bold")
        x0 = wx - total / 2
        dy = (1 - a) * 10
        c.text(x0, 245 + dy, wd[0], size, A(cols[i], a), anchor="lm", kind="bold")
        c.text(x0 + first, 245 + dy, wd[1:], size, A(TXT, a), anchor="lm", kind="bold")
    a = fade(t, 4.0, 0.6)
    c.text(427, 280, "(programski interfejs aplikacije)", 15, A(MUTED, a))

    # dvije aplikacije koje razgovaraju
    a = fade(t, 5.2, 0.6)
    if a > 0:
        for x, name, col in ((215, "Aplikacija A", BLUE), (639, "Aplikacija B", GREEN)):
            c.rect(x - 75, 330, x + 75, 376, fill=A(PANEL, a), outline=A(col, a), w=2, r=10)
            c.text(x, 353, name, 14, A(TXT, a), kind="bold")
        lp = eo(prog(t, 5.6, 6.4))
        c.dashed(290, 353, lerp(290, 564, lp), 353, A(MUTED, a * 0.7), 2, offset=t * 30)
        if t > 6.4:
            pill(c, 427, 353, "API", 13, DARK, YELLOW, fade(t, 6.4, 0.4))
            for k in range(2):
                ph = ((t - 6.4) * 0.45 + k * 0.5) % 1
                xs = lerp(295, 560, ph) if k == 0 else lerp(560, 295, ph)
                if abs(xs - 427) > 34:
                    c.circle(xs, 353, 5, fill=A(BLUE if k == 0 else GREEN, a))
    caption(c, t, 7.0, "API je skup pravila pomoću kojih programi međusobno razgovaraju.", y=428, size=16)


def s_restaurant(c, t):
    header(c, t, "Analogija: restoran", 2, ORANGE)
    gx, wx, kx, fy = 125, 427, 712, 238
    p = prog(t, 0.3, 0.9)
    if p > 0:
        guest(c, gx, fy, back(p), min(1, p * 2))
    p = prog(t, 0.8, 1.4)
    if p > 0:
        waiter(c, wx, fy, back(p), min(1, p * 2))
    p = prog(t, 1.3, 1.9)
    cooking = window(t, 8.6, 11.2, 0.4)
    if p > 0:
        kitchen(c, kx, fy, back(p), min(1, p * 2), t, cooking)
    labels = [(gx + 30, "GOST", "= klijent (tvoja aplikacija)", BLUE),
              (wx, "KONOBAR", "= API", YELLOW),
              (kx, "KUHINJA", "= server (sistem s podacima)", ORANGE)]
    for i, (x, title, sub, col) in enumerate(labels):
        a = fade(t, 1.9 + i * 0.3, 0.5)
        c.text(x, 316, title, 15, A(col, a), kind="bold")
        c.text(x, 337, sub, 12, A(MUTED, a))

    # zahtjev: gost -> konobar -> kuhinja
    bubble(c, gx + 5, 150, "Jednu pizzu, molim!", 12, window(t, 3.0, 5.6), tail_x=gx)
    if 4.4 < t < 8.8:
        p = eio(prog(t, 4.6, 6.4))
        x, y = move_arc(p, gx + 20, 205, wx - 30, 205, 45)
        if t > 6.4:
            p2 = eio(prog(t, 6.6, 8.4))
            x, y = move_arc(p2, wx - 30, 205, kx - 50, 205, 45)
        note(c, x, y, fade(t, 4.4, 0.3) * (1 - prog(t, 8.4, 8.8)))
    a = window(t, 4.6, 14.5, 0.5)
    if a > 0:
        c.text(276, 118, "zahtjev  →", 13, A(BLUE, a * fade(t, 4.6)), kind="bold")
        c.text(570, 118, "zahtjev  →", 13, A(BLUE, a * fade(t, 6.6)), kind="bold")
    if cooking > 0:
        dots = "." * (int(t * 3) % 4)
        c.text(kx, 140, "kuhanje" + dots, 13, A(TXT, cooking), kind="bold")

    # odgovor: kuhinja -> konobar -> gost
    if 11.0 < t < 14.8:
        p = eio(prog(t, 11.2, 12.7))
        x, y = move_arc(p, kx - 40, 225, wx + 30, 228, 30)
        if t > 12.7:
            p2 = eio(prog(t, 12.9, 14.4))
            x, y = move_arc(p2, wx + 30, 228, gx + 72, 236, 45)
        pizza(c, x, y, fade(t, 11.0, 0.3))
    if t > 14.4:
        pizza(c, gx + 72, 236, 1.0)
    a = window(t, 11.2, 14.5, 0.5)
    if a > 0:
        c.text(570, 284, "←  odgovor", 13, A(GREEN, a * fade(t, 11.2)), kind="bold")
        c.text(276, 284, "←  odgovor", 13, A(GREEN, a * fade(t, 12.9)), kind="bold")
    bubble(c, gx + 5, 150, "Hvala!", 12, window(t, 14.4, 16.4), tail_x=gx)

    caption(c, t, 15.0, "Gost ne mora znati KAKO kuhinja radi — dovoljno je da zna ŠTA može "
                        "naručiti iz menija. Taj „meni” je dokumentacija API-ja.", y=418, size=15, maxw=700)


def s_how(c, t):
    header(c, t, "Kako API radi?", 3, BLUE)
    px, py = 125, 248
    gate_x = 612
    sx, sy = 735, 162
    a = fade(t, 0.3, 0.6)
    tap = 2.4
    phone(c, px, py + (1 - a) * 15, a, 14.6 if t > 14.6 else None, t, tap)
    c.text(px, 145, "KLIJENT", 13, A(BLUE, fade(t, 1.2)), kind="bold")
    c.text(px, 349, "aplikacija na telefonu", 11, A(MUTED, fade(t, 1.4)))
    a = fade(t, 0.7, 0.6)
    server(c, sx, sy + (1 - a) * 15, a, t, window(t, 7.0, 11.0))
    c.text(sx, 145, "SERVER", 13, A(ORANGE, fade(t, 1.4)), kind="bold")

    # API kapija
    a = fade(t, 1.6, 0.5)
    if a > 0:
        c.rect(gate_x - 18, 168, gate_x + 18, 322, fill=A(YELLOW, a), r=18)
        for i, ch in enumerate("API"):
            c.text(gate_x, 215 + i * 30, ch, 18, A(DARK, a), kind="bold")
        c.line([(gate_x + 18, 210), (sx - 56, 210)], A(YELLOW, a * 0.7), 3)

    # trake
    a = fade(t, 2.6, 0.5)
    c.dashed(182, 200, 590, 200, A(BLUE, a * 0.5), 2, offset=-t * 25)
    c.arrow_head(594, 200, 0, 10, A(BLUE, a * 0.8))
    c.text(392, 176, "ZAHTJEV (request)", 12, A(BLUE, fade(t, 3.0)), kind="bold")
    a2 = fade(t, 10.8, 0.5)
    c.dashed(182, 292, 590, 292, A(GREEN, a2 * 0.5), 2, offset=t * 25)
    c.arrow_head(178, 292, math.pi, 10, A(GREEN, a2 * 0.8))
    c.text(392, 360, "ODGOVOR (response)", 12, A(GREEN, fade(t, 11.0)), kind="bold")

    # paket zahtjeva
    if 3.0 < t < 7.6:
        p = eio(prog(t, 3.2, 6.6))
        x = lerp(290, 497, p)
        aa = fade(t, 3.0, 0.4) * (1 - prog(t, 6.8, 7.4))
        txt = "GET /vrijeme?grad=Sarajevo"
        w = c.tlen(txt, 11, "mono") + 18
        c.rect(x - w / 2, 187, x + w / 2, 213, fill=A(DARK, aa), outline=A(BLUE, aa), w=1.5, r=6)
        c.text(x, 200, txt, 11, A(BLUE, aa), kind="mono")
    if 6.6 < t < 7.3:
        p = prog(t, 6.6, 7.3)
        c.circle(lerp(gate_x + 18, sx - 56, p), 210, 5, fill=A(BLUE, 1))

    # obrada na serveru + baza
    busy = window(t, 7.0, 11.0)
    da = fade(t, 7.3, 0.5)
    if da > 0:
        c.line([(sx, sy + 96), (sx, 300)], A(PURPLE, da * 0.6), 2)
        database(c, sx, 302, da)
        c.text(sx, 372, "baza podataka", 11, A(MUTED, da))
    if busy > 0:
        for k in range(2):
            ph = ((t - 7.5) * 0.8 + k * 0.5) % 1
            if t > 7.5:
                yy = lerp(sy + 98, 296, ph) if k == 0 else lerp(296, sy + 98, ph)
                c.circle(sx + (-8 if k == 0 else 8), yy, 3.5, fill=A(PURPLE if k == 0 else GREEN, busy))
        dots = "." * (int(t * 3) % 4)
        # spinner pored servera
        for k in range(8):
            ang = k * math.pi / 4 + t * 6
            c.circle(sx - 80 + math.cos(ang) * 9, 120 + math.sin(ang) * 9, 1.8 + k * 0.25,
                     fill=A(TXT, busy * (0.25 + k * 0.09)))
        c.text(sx - 64, 120, "obrada" + dots, 12, A(TXT, busy), anchor="lm", kind="bold")

    # paket odgovora (JSON)
    if 10.6 < t < 14.9:
        if t < 11.3:
            p = prog(t, 10.6, 11.3)
            c.circle(lerp(sx - 56, gate_x + 18, p), 285, 5, fill=A(GREEN, 1))
        p = eio(prog(t, 11.2, 14.4))
        x = lerp(500, 285, p)
        aa = fade(t, 11.1, 0.4) * (1 - prog(t, 14.3, 14.8))
        lines = ["{", '  "grad": "Sarajevo",', '  "temp": 18,', '  "opis": "sunčano"', "}"]
        w = max(c.tlen(l, 11, "mono") for l in lines) + 22
        c.rect(x - w / 2, 252, x + w / 2, 334, fill=A(DARK, aa), outline=A(GREEN, aa), w=1.5, r=8)
        for i, l in enumerate(lines):
            c.text(x - w / 2 + 11, 263 + i * 15, l, 11, A(GREEN if i in (0, 4) else TXT, aa),
                   anchor="lm", kind="mono")
        c.text(x + w / 2 - 6, 247, "JSON", 9, A(GREEN, aa), anchor="rb", kind="bold")

    caption(c, t, 15.6, "1. Klijent šalje zahtjev  →  2. API ga predaje serveru  →  3. Server vraća odgovor",
            y=410, size=14, maxw=780)
    caption(c, t, 17.0, "Odgovor najčešće stiže u JSON formatu — jednostavnom tekstu koji svaki program razumije.",
            y=448, size=13, color=MUTED, maxw=780)


URL_PARTS = [("https://", PURPLE, "protokol"),
             ("api.primjer.ba", BLUE, "adresa servera"),
             ("/v1", YELLOW, "verzija"),
             ("/korisnici", GREEN, "resurs (endpoint)"),
             ("/42", ORANGE, "ID")]


def s_anatomy(c, t):
    header(c, t, "Dijelovi API poziva", 4, GREEN)
    full = "".join(p for p, _, _ in URL_PARTS)
    size = 22
    cw = c.tlen("M", size, "mono")
    x0 = 427 - cw * len(full) / 2
    shown = int(len(full) * prog(t, 0.4, 2.4))
    pos = 0
    for i, (part, col, label) in enumerate(URL_PARTS):
        vis = part[:max(0, shown - pos)]
        hl = fade(t, 3.0 + i * 0.7, 0.5)
        colr = tuple(int(lerp(TXT[k], col[k], hl)) for k in range(3))
        if vis:
            c.text(x0 + pos * cw, 100, vis, size, A(colr), anchor="lm", kind="mono")
        if hl > 0:
            xa, xb = x0 + pos * cw, x0 + (pos + len(part)) * cw
            c.rect(xa + 1, 116, lerp(xa + 1, xb - 1, eo(hl)), 119, fill=A(col, hl), r=1.5)
            ly = 140 if i % 2 == 0 else 160
            mid = (xa + xb) / 2
            c.line([(mid, 120), (mid, ly - 8)], A(col, hl * 0.6), 1)
            c.text(mid, ly, label, 12, A(col, hl), kind="bold")
        pos += len(part)
    if shown < len(full) and int(t * 2.5) % 2 == 0:
        c.rect(x0 + shown * cw + 1, 88, x0 + shown * cw + 3, 112, fill=A(TXT))

    # HTTP metode
    a = fade(t, 7.0, 0.5)
    c.text(40, 205, "HTTP metode — ŠTA želiš uraditi", 15, A(TXT, a), anchor="lm", kind="bold")
    methods = [("GET", GREEN, "čitanje podataka"),
               ("POST", BLUE, "slanje / kreiranje novih"),
               ("PUT", ORANGE, "izmjena postojećih"),
               ("DELETE", RED, "brisanje podataka")]
    for i, (m, col, desc) in enumerate(methods):
        a = fade(t, 7.6 + i * 0.6, 0.5)
        if a <= 0:
            continue
        y = 244 + i * 45
        dx = (1 - a) * -20
        c.rect(40 + dx, y - 18, 400 + dx, y + 18, fill=A(PANEL, a * 0.9), r=8)
        c.rect(48 + dx, y - 11, 122 + dx, y + 11, fill=A(col, a), r=11)
        c.text(85 + dx, y, m, 12, A(DARK, a), kind="monob")
        c.text(136 + dx, y, desc, 13, A(TXT, a), anchor="lm")

    # statusni kodovi
    a = fade(t, 12.0, 0.5)
    c.text(440, 205, "Statusni kodovi — KAKO je prošlo", 15, A(TXT, a), anchor="lm", kind="bold")
    codes = [("200", GREEN, "OK — uspješno"),
             ("201", GREEN, "Created — kreirano"),
             ("401", YELLOW, "nemaš dozvolu (ključ?)"),
             ("404", ORANGE, "nije pronađeno"),
             ("500", RED, "greška na serveru")]
    for i, (code, col, desc) in enumerate(codes):
        a = fade(t, 12.6 + i * 0.6, 0.5)
        if a <= 0:
            continue
        y = 240 + i * 37
        dx = (1 - a) * 20
        c.rect(440 + dx, y - 15, 815 + dx, y + 15, fill=A(PANEL, a * 0.9), r=8)
        c.text(452 + dx, y, code, 15, A(col, a), anchor="lm", kind="monob")
        c.text(500 + dx, y, desc, 13, A(TXT, a), anchor="lm")

    caption(c, t, 17.0, "Metoda + adresa = zahtjev.   Statusni kod + podaci = odgovor.", y=443, size=14)


CODE = [
    "import requests",
    "",
    'URL = "https://api.primjer.ba/vrijeme"',
    'kljuc = "TVOJ_API_KLJUC"',
    "",
    "odgovor = requests.get(",
    "    URL,",
    '    params={"grad": "Sarajevo"},',
    '    headers={"Authorization": kljuc},',
    ")",
    "",
    "podaci = odgovor.json()",
    'print(podaci["temp"], "°C")',
]
CODE_START, CODE_CPS = 1.2, 32.0


def highlight(line):
    """Vrlo jednostavno bojenje Python sintakse -> lista (tekst, boja)."""
    segs = []
    i = 0
    keywords = {"import": PINK}
    builtins = {"print": BLUE}
    while i < len(line):
        ch = line[i]
        if ch == '"':
            j = line.index('"', i + 1) + 1
            segs.append((line[i:j], GREEN))
            i = j
        elif ch.isalpha() or ch == "_":
            j = i
            while j < len(line) and (line[j].isalnum() or line[j] == "_"):
                j += 1
            word = line[i:j]
            col = keywords.get(word) or builtins.get(word)
            if col is None:
                col = YELLOW if j < len(line) and line[j] == "(" else (PURPLE if word.isupper() else TXT)
            segs.append((word, col))
            i = j
        else:
            segs.append((ch, MUTED if ch in "(){}[],.:=" else TXT))
            i += 1
    return segs


def code_time(line_idx):
    """Trenutak u kojem je red `line_idx` u potpunosti otkucan."""
    n = sum(len(l) + 1 for l in CODE[:line_idx + 1])
    return CODE_START + n / CODE_CPS


def s_code(c, t):
    header(c, t, "Kako se API koristi u kodu?", 5, PURPLE)
    a = fade(t, 0.3, 0.5)
    ex0, ey0, ex1, ey1 = 26, 70, 488, 400
    c.rect(ex0, ey0, ex1, ey1, fill=A((13, 18, 34), a), outline=A(PANEL2, a), w=1.5, r=10)
    c.rect(ex0, ey0, ex1, ey0 + 26, fill=A(PANEL, a), r=10)
    c.rect(ex0, ey0 + 16, ex1, ey0 + 26, fill=A(PANEL, a))
    for i, col in enumerate((RED, YELLOW, GREEN)):
        c.circle(ex0 + 16 + i * 14, ey0 + 13, 4.5, fill=A(col, a))
    c.text((ex0 + ex1) / 2, ey0 + 13, "vrijeme.py", 11, A(MUTED, a), kind="mono")

    size = 12.5
    cw = c.tlen("M", size, "mono")
    lh = 22
    budget = max(0, int((t - CODE_START) * CODE_CPS))
    cx, cy = ex0 + 44, ey0 + 48
    caret = None
    for li, line in enumerate(CODE):
        y = ey0 + 48 + li * lh
        if a > 0:
            c.text(ex0 + 30, y, str(li + 1), 10, A(PANEL2, a), anchor="rm", kind="mono")
        if budget <= 0:
            if caret is None:
                caret = (ex0 + 44, y)
            continue
        take = min(len(line), budget)
        budget -= len(line) + 1
        col_i = 0
        for seg, col in highlight(line):
            if col_i >= take:
                break
            part = seg[:take - col_i]
            c.text(ex0 + 44 + col_i * cw, y, part, size, A(col, a), anchor="lm", kind="mono")
            col_i += len(seg)
        if take < len(line) or budget < 0:
            caret = (ex0 + 44 + take * cw, y)
        cx, cy = ex0 + 44 + take * cw, y
    if caret is None:
        caret = (cx, cy)
    if int(t * 2.5) % 2 == 0 and a > 0:
        c.rect(caret[0] + 1, caret[1] - 8, caret[0] + 2.5, caret[1] + 8, fill=A(TXT, a))

    # koraci desno, sinhronizovani s kucanjem
    steps = [(0.8, "Pročitaj dokumentaciju API-ja", "šta nudi i kako se poziva", BLUE),
             (code_time(3), "Nabavi API ključ", "dokaz da smiješ koristiti API", YELLOW),
             (code_time(9), "Pošalji HTTP zahtjev", "metoda, adresa, parametri", GREEN),
             (code_time(12), "Obradi JSON odgovor", "i prikaži podatke korisniku", ORANGE)]
    for i, (st, title, sub, col) in enumerate(steps):
        sa = fade(t, st, 0.5)
        if sa <= 0:
            continue
        y = 92 + i * 58
        dx = (1 - sa) * 20
        c.rect(506 + dx, y - 22, 830 + dx, y + 22, fill=A(PANEL, sa * 0.9), r=10)
        c.circle(528 + dx, y, 12, fill=A(col, sa))
        c.text(528 + dx, y, str(i + 1), 12, A(DARK, sa), kind="bold")
        c.text(548 + dx, y - 8, title, 13, A(TXT, sa), anchor="lm", kind="bold")
        c.text(548 + dx, y + 10, sub, 11, A(MUTED, sa), anchor="lm")

    # terminal
    tt = code_time(len(CODE) - 1) + 0.6
    ta = fade(t, tt, 0.5)
    if ta > 0:
        c.rect(506, 330, 830, 400, fill=A((5, 8, 18), ta), outline=A(PANEL2, ta), w=1.5, r=8)
        cmd = "$ python vrijeme.py"
        n = int(len(cmd) * prog(t, tt + 0.2, tt + 1.0))
        c.text(520, 350, cmd[:n], 12, A(TXT, ta), anchor="lm", kind="mono")
        if t > tt + 1.4:
            c.text(520, 374, "18 °C", 13, A(GREEN, fade(t, tt + 1.4, 0.3)), anchor="lm", kind="monob")
            pill(c, 790, 374, "200 OK", 10, DARK, GREEN, fade(t, tt + 1.6, 0.3), kind="bold", padx=8, pady=3)

    caption(c, t, 16.5, "Nekoliko linija koda — i tvoja aplikacija dobija podatke s drugog kraja svijeta.",
            y=440, size=14)


APPS = [("weather", "Vremenska\nprognoza", YELLOW),
        ("card", "Online\nplaćanje", GREEN),
        ("lock", "Prijava preko\nGoogle / Apple", BLUE),
        ("pin", "Mape i\ndostava", RED),
        ("social", "Društvene\nmreže", PINK),
        ("ai", "Vještačka\ninteligencija", PURPLE),
        ("home", "Pametna kuća\n(IoT)", ORANGE),
        ("calendar", "Rezervacije\ni putovanja", BLUE)]


def s_uses(c, t):
    header(c, t, "Gdje se API koristi danas?", 6, PINK)
    hx, hy = 427, 236
    # gornji red ima natpise iznad, donji ispod, da ih linije ne presijecaju
    nodes = [(544, 140), (724, 140), (724, 330), (544, 330),
             (310, 330), (130, 330), (130, 140), (310, 140)]
    # veze i paketi
    for i, ((kind, label, col), (nx, ny)) in enumerate(zip(APPS, nodes)):
        st = 1.0 + i * 1.5
        lp = eo(prog(t, st, st + 0.5))
        if lp <= 0:
            continue
        dx, dy = nx - hx, ny - hy
        dist = math.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        sx_, sy_ = hx + ux * 44, hy + uy * 44
        ex_, ey_ = nx - ux * 30, ny - uy * 30
        c.line([(sx_, sy_), (lerp(sx_, ex_, lp), lerp(sy_, ey_, lp))], A(col, 0.35), 2)
        if t > st + 0.6:
            for k, colp in ((0, BLUE), (1, col)):
                ph = ((t - st) * 0.55 + k * 0.5 + i * 0.11) % 1
                if k == 0:
                    px, py = lerp(sx_, ex_, ph), lerp(sy_, ey_, ph)
                else:
                    px, py = lerp(ex_, sx_, ph), lerp(ey_, sy_, ph)
                c.circle(px, py, 3.5, fill=A(colp, 0.9))
    # hub
    pulse = (t * 0.7) % 1
    c.circle(hx, hy, 42 + pulse * 40, outline=A(YELLOW, 0.5 * (1 - pulse) * fade(t, 0.3)), w=2)
    ha = fade(t, 0.2, 0.6)
    hr = 42 * back(prog(t, 0.2, 0.8))
    c.circle(hx, hy, hr, fill=A(YELLOW, ha))
    c.text(hx, hy, "API", 24 * hr / 42, A(DARK, ha), kind="bold")
    # čvorovi
    for i, ((kind, label, col), (nx, ny)) in enumerate(zip(APPS, nodes)):
        st = 1.0 + i * 1.5
        p = prog(t, st + 0.3, st + 0.9)
        if p <= 0:
            continue
        r = 28 * back(p)
        na = min(1, p * 2)
        c.circle(nx, ny, r, fill=A(PANEL, na), outline=A(col, na), w=2)
        if p > 0.5:
            icon(c, kind, nx, ny, col, fade(t, st + 0.6, 0.4))
        la = fade(t, st + 0.6, 0.5)
        base = ny + 42 if ny > hy else ny - 57
        for k, l in enumerate(label.split("\n")):
            c.text(nx, base + k * 15, l, 12, A(TXT, la), kind="bold" if k == 0 else "sans")

    caption(c, t, 14.5, "Gotovo svaka moderna aplikacija koristi više API-ja istovremeno — "
                        "često a da toga nismo ni svjesni.", y=444, size=14, maxw=760)


def s_summary(c, t):
    header(c, t, "Ukratko", 7, GREEN)
    items = ["API je skup pravila po kojima programi razgovaraju",
             "Klijent šalje zahtjev, server vraća odgovor",
             "Najčešće se koristi HTTP i JSON format",
             "Pokreće plaćanja, mape, prijave, AI i još mnogo toga"]
    for i, it in enumerate(items):
        a = fade(t, 0.6 + i * 1.3, 0.5)
        if a <= 0:
            continue
        y = 100 + i * 48
        dx = (1 - a) * 25
        check(c, 70 + dx, y, a)
        c.text(95 + dx, y, it, 18, A(TXT, a), anchor="lm")
    a = fade(t, 7.5, 0.8)
    if a > 0:
        glow = 0.5 + 0.5 * math.sin(t * 2.5)
        c.rect(147, 316, 707, 380, fill=A(PANEL, a * 0.9), outline=A(YELLOW, a * (0.5 + 0.5 * glow)), w=2, r=16)
        c.text(427, 348, "API = most između aplikacija", 26, A(YELLOW, a), kind="bold")
    a = fade(t, 9.0, 0.8)
    c.text(427, 420, "Hvala na pažnji!", 16, A(MUTED, a), kind="bold")


SCENES = [
    (s_intro, 10.0),
    (s_restaurant, 21.0),
    (s_how, 22.0),
    (s_anatomy, 21.0),
    (s_code, 21.0),
    (s_uses, 22.0),
    (s_summary, 13.0),
]
TOTAL = sum(d for _, d in SCENES)
N_FRAMES = int(round(TOTAL * FPS))
FADE = 0.4


def render(T):
    start = 0.0
    for fn, dur in SCENES:
        if T < start + dur or fn is SCENES[-1][0]:
            break
        start += dur
    t = T - start
    img = BG.copy()
    c = Cv(img)
    fn(c, t)
    f = min(prog(t, 0, FADE), 1 - prog(t, dur - FADE, dur))
    if f < 1:
        img = Image.blend(BG, img, max(0.0, f))
    img = img.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, H - 3, int(W * T / TOTAL), H], fill=(56, 189, 248, 200))
    return img


def render_bytes(i):
    return render(i / FPS).tobytes()


# ---------------------------------------------------------------- muzika
def make_audio(path, dur, sr=44100):
    n = int(sr * dur)
    out = np.zeros(n)

    def hz(m):
        return 440.0 * 2 ** ((m - 69) / 12)

    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]   # Am F C G
    bass = [45, 41, 48, 43]
    seg = 4.0
    k, start = 0, 0.0
    while start < dur:
        ch, b = chords[k % 4], bass[k % 4]
        s0 = int(start * sr)
        s1 = min(n, int((start + seg + 1.2) * sr))
        tt = np.arange(s1 - s0) / sr
        env = np.minimum(1, tt / 1.2) * np.clip((seg + 1.2 - tt) / 1.2, 0, 1)
        sig = np.zeros_like(tt)
        for m in ch:
            f = hz(m)
            sig += np.sin(2 * np.pi * f * tt) + 0.7 * np.sin(2 * np.pi * f * 1.004 * tt) \
                + 0.15 * np.sin(2 * np.pi * 2 * f * tt)
        sig += 1.2 * np.sin(2 * np.pi * hz(b) * tt)
        out[s0:s1] += sig * env * 0.045
        arp = ch + [ch[0] + 12]
        for i, idx in enumerate([0, 1, 2, 3, 2, 1, 2, 3]):
            ts = start + i * 0.5
            p0 = int(ts * sr)
            if p0 >= n:
                break
            p1 = min(n, p0 + int(1.2 * sr))
            tn = np.arange(p1 - p0) / sr
            f = hz(arp[idx] + 12)
            note_sig = (np.sin(2 * np.pi * f * tn) + 0.3 * np.sin(2 * np.pi * 2 * f * tn)) \
                * np.exp(-tn * 3.5) * np.minimum(1, tn / 0.01)
            out[p0:p1] += note_sig * 0.06
        start += seg
        k += 1
    fi, fo = int(2 * sr), int(3 * sr)
    out[:fi] *= np.linspace(0, 1, fi)
    out[-fo:] *= np.linspace(1, 0, fo)
    out = out / np.max(np.abs(out)) * 0.3
    pcm = (out * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        outdir = os.path.join(HERE, "preview")
        os.makedirs(outdir, exist_ok=True)
        for s in sys.argv[2:]:
            render(float(s)).save(os.path.join(outdir, f"frame_{float(s):06.1f}.png"))
        return
    wav = os.path.join(HERE, "_muzika.wav")
    make_audio(wav, TOTAL)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav,
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool() as pool:
        for i, frame in enumerate(pool.imap(render_bytes, range(N_FRAMES), chunksize=8)):
            proc.stdin.write(frame)
            if i % (FPS * 10) == 0:
                print(f"  {i / FPS:5.1f}s / {TOTAL:.0f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    os.remove(wav)
    print(f"Gotovo: {OUT}  ({TOTAL:.0f} s, {W}x{H}, {FPS} fps)")


if __name__ == "__main__":
    main()
