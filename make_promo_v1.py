#!/usr/bin/env python3
"""CADmind promo v2 — ~35s 1920x1080 30fps.
Official CADmind renders + CC BY 3D-printing b-roll (credits on screen).
No copyrighted third-party music; audio is fully synthesized.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi

ROOT = Path("/workspace/cadmind_video")
RENDERS = ROOT / "renders"
LOGO = ROOT / "logo" / "cadmind-mark-1024-on-dark.png"
KEYED = Path("/tmp/cm-keyed-v2")
PREVIEW = Path("/tmp/cm-promo-v2")
BROLL_CUTS = Path("/tmp/cm-broll-cuts")
OUT_MP4 = ROOT / "cadmind_promo_v2.mp4"
OUT_CONTACT = ROOT / "cadmind_promo_v2_contact.png"
AUDIO = Path("/tmp/cm-promo-v2/promo_audio.wav")

W, H, FPS = 1920, 1080, 30
DURATION = 35.0
NFRAMES = int(round(DURATION * FPS))  # 1050
NAVY = (16, 20, 28, 255)
CREAM = (248, 247, 245, 255)
ORANGE = (255, 79, 18, 255)
SURFACE = (24, 30, 42, 255)
GRID_C = (70, 82, 102, 255)
MUTED = (193, 194, 189, 255)  # #C1C2BD from the site

INTER = "/usr/share/fonts/truetype/sand-box/google/Inter/Inter-VariableFont_opsz,wght.ttf"
PLEX = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Sans/IBMPlexSans-VariableFont_wdth,wght.ttf"
MONO_R = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Regular.ttf"
MONO_M = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Medium.ttf"
MONO_B = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Bold.ttf"

PROMPT = "design a case for my bench power supply"


def font_inter(size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(INTER, size)
    opsz = float(max(14, min(32, int(size * 0.35))))
    f.set_variation_by_axes([opsz, float(weight)])
    return f


def font_plex(size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(PLEX, size)
    f.set_variation_by_axes([float(weight), 100.0])
    return f


def font_mono(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    path = {"regular": MONO_R, "medium": MONO_M, "bold": MONO_B}[weight]
    return ImageFont.truetype(path, size)


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_in_out(t: float) -> float:
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) * (1 - t)


def lerp(a, b, t):
    return a + (b - a) * t


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


# ---------------------------------------------------------------------------
# Keying
# ---------------------------------------------------------------------------
def axis_mask(a: np.ndarray) -> np.ndarray:
    r = (a[:, :, 0] > 160) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)
    g = (a[:, :, 1] > 160) & (a[:, :, 0] < 90) & (a[:, :, 2] < 90)
    b = (a[:, :, 2] > 160) & (a[:, :, 0] < 90) & (a[:, :, 1] < 100)
    return r | g | b


def inpaint_labels(rgb: np.ndarray) -> np.ndarray:
    """Fill small white part-number tags with nearby model colour."""
    h, w = rgb.shape[:2]
    # Tags are tiny white chips with black numerals; catch AA fringes too.
    white = (rgb[:, :, 0] >= 248) & (rgb[:, :, 1] >= 248) & (rgb[:, :, 2] >= 248)
    # also black/dark text that sits inside those chips
    dark = (rgb.max(axis=2) < 40)
    if not white.any():
        return rgb
    am = axis_mask(rgb)
    lab, n = ndi.label(white)
    kill = np.zeros((h, w), dtype=bool)
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) < 8:
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if bw > 80 or bh > 80:
            continue
        if am[y0 : y1 + 1, x0 : x1 + 1].any():
            continue
        pad = 6
        y0p, y1p = max(0, y0 - pad), min(h, y1 + 1 + pad)
        x0p, x1p = max(0, x0 - pad), min(w, x1 + 1 + pad)
        kill[y0p:y1p, x0p:x1p] = True
    if not kill.any():
        return rgb
    # distance_transform_edt: distance to nearest ZERO. Pass `kill` so
    # indices point at the closest non-kill (keep) pixel.
    _dist, inds = ndi.distance_transform_edt(kill, return_indices=True)
    out = rgb.copy()
    out[kill] = rgb[inds[0][kill], inds[1][kill]]
    return out


def key_render(src: Path, dest: Path) -> Image.Image:
    """Flood-fill key from the borders against the studio backdrop (~#F4F4F4),
    drop the XYZ gizmo, zero RGB on transparent pixels, and crop to the
    object bbox so a later Lanczos resize cannot leave a plate rectangle.
    """
    im = Image.open(src).convert("RGB")
    rgb = inpaint_labels(np.asarray(im).copy())
    h, w = rgb.shape[:2]
    # detect backdrop from the corners (official renders are #F4F4F4, but
    # don't assume a single exact value)
    corners = np.concatenate(
        [
            rgb[:12, :12].reshape(-1, 3),
            rgb[:12, -12:].reshape(-1, 3),
            rgb[-12:, :12].reshape(-1, 3),
            rgb[-12:, -12:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0).astype(np.int16)
    diff = np.abs(rgb.astype(np.int16) - bg).max(axis=2)
    # pixels similar enough to the backdrop to be flood-filled
    floodable = diff <= 30
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    lab, _n = ndi.label(floodable)
    seeds = set(lab[border & floodable].tolist())
    seeds.discard(0)
    bg_mask = np.isin(lab, list(seeds)) if seeds else floodable
    # also treat any isolated near-exact backdrop leftovers as background
    bg_mask = bg_mask | (diff <= 8)

    # drop the XYZ gizmo (axis-colored, almost always bottom-left)
    am = axis_mask(rgb)
    if am.any():
        am = ndi.binary_dilation(am, iterations=3)
        ys, xs = np.where(am)
        x0, x1 = int(xs.min()) - 22, int(xs.max()) + 22
        y0, y1 = int(ys.min()) - 22, int(ys.max()) + 22
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        cornerish = (x0 < w * 0.32 and y1 > h * 0.62) or (x0 < w * 0.32 and y0 > h * 0.55)
        if cornerish:
            bg_mask[y0:y1, x0:x1] = True

    # soft edge: 0 on backdrop, ramp over ~3px into the object
    dist = ndi.distance_transform_edt(~bg_mask)
    feather = 3.0
    alpha = np.clip(dist / feather * 255.0, 0, 255).astype(np.uint8)

    # transparent pixels must not keep the #F4F4F4 RGB — Lanczos would
    # otherwise bleed a light rectangular halo the size of the plate
    out_rgb = rgb.copy()
    out_rgb[alpha == 0] = 0
    # premultiply partial-alpha edge toward 0 so fringes stay dark, not gray
    fringe = (alpha > 0) & (alpha < 255)
    if fringe.any():
        a = (alpha[fringe].astype(np.float32) / 255.0)[..., None]
        out_rgb[fringe] = np.clip(out_rgb[fringe].astype(np.float32) * a, 0, 255).astype(np.uint8)

    img = Image.fromarray(np.dstack([out_rgb, alpha]), "RGBA")
    ys, xs = np.where(alpha > 0)
    if len(xs):
        pad = 6
        x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
        y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
        img = img.crop((x0, y0, x1, y1))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    return img


NEEDED = {
    "bench_psu_2": RENDERS / "bench_psu_2.png",
    "mini_itx_2": RENDERS / "mini_itx_2.png",
    "mini_itx_3": RENDERS / "mini_itx_3.png",
    "pixel10_1": RENDERS / "pixel10_1.png",
    "board_enclosure_1": RENDERS / "board_enclosure_1.png",
    "board_enclosure_4": RENDERS / "board_enclosure_4.png",  # lid section flash
    "zoom_h5_1": RENDERS / "zoom_h5_1.png",
    "epaper_frame_2": RENDERS / "epaper_frame_2.png",
    "macbook_stand_1": RENDERS / "macbook_stand_1.png",
    "headphone_hook_1": RENDERS / "headphone_hook_1.png",
    "desk_organizer_1": RENDERS / "desk_organizer_1.png",
    "tao_keychain_1": RENDERS / "tao_keychain_1.png",
    "toothbrush_stand_2": RENDERS / "toothbrush_stand_2.png",
}


def preprocess() -> dict[str, Image.Image]:
    keyed: dict[str, Image.Image] = {}
    for name, src in NEEDED.items():
        dest = KEYED / f"{name}.png"
        if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
            keyed[name] = Image.open(dest).convert("RGBA")
        else:
            print("keying", name, flush=True)
            keyed[name] = key_render(src, dest)
    return keyed


# ---------------------------------------------------------------------------
# Backdrop
# ---------------------------------------------------------------------------
def make_grid(alpha_scale: float = 1.0) -> Image.Image:
    """Faint blueprint grid on navy, cached at full opacity then multiplied."""
    img = Image.new("RGBA", (W, H), NAVY)
    d = ImageDraw.Draw(img)
    minor, major = 40, 200
    # minor
    a = int(28 * alpha_scale)
    col = (70, 82, 102, a)
    for x in range(0, W + 1, minor):
        d.line([(x, 0), (x, H)], fill=col, width=1)
    for y in range(0, H + 1, minor):
        d.line([(0, y), (W, y)], fill=col, width=1)
    # major
    a2 = int(55 * alpha_scale)
    col2 = (90, 104, 126, a2)
    for x in range(0, W + 1, major):
        d.line([(x, 0), (x, H)], fill=col2, width=1)
    for y in range(0, H + 1, major):
        d.line([(0, y), (W, y)], fill=col2, width=1)
    return img


GRID_FULL = None


def backdrop(grid_alpha: float = 1.0) -> Image.Image:
    global GRID_FULL
    if grid_alpha <= 0.001:
        return Image.new("RGBA", (W, H), NAVY)
    if GRID_FULL is None:
        GRID_FULL = make_grid(1.0)
    if grid_alpha >= 0.999:
        return GRID_FULL.copy()
    # fade grid toward plain navy
    base = Image.new("RGBA", (W, H), NAVY)
    g = GRID_FULL.copy()
    # mix only the grid lines: composite with reduced alpha
    arr = np.asarray(g).copy()
    # keep navy pixels, scale non-navy (the lines)
    navy = np.array(NAVY, dtype=np.int16)
    is_line = np.abs(arr.astype(np.int16) - navy).sum(axis=2) > 8
    arr[:, :, 3] = np.where(is_line, (arr[:, :, 3] * grid_alpha).astype(np.uint8), 255)
    # actually the grid image is opaque navy + opaque lines. Better rebuild cheaply
    return make_grid(grid_alpha)


# Prebuild a few grid strengths
_GRID_CACHE: dict[int, Image.Image] = {}


def grid(alpha: float) -> Image.Image:
    key = int(round(clamp01(alpha) * 20))
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = make_grid(key / 20.0)
    return _GRID_CACHE[key].copy()


def paste_centered(dst: Image.Image, src: Image.Image, cx: int, cy: int, scale: float) -> None:
    if scale <= 0:
        return
    nw, nh = max(1, int(src.width * scale)), max(1, int(src.height * scale))
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    dst.alpha_composite(resized, (int(cx - nw / 2), int(cy - nh / 2)))


def fit_scale(src: Image.Image, box_w: int, box_h: int) -> float:
    return min(box_w / src.width, box_h / src.height)


def ken_burns_paste(
    canvas: Image.Image,
    src: Image.Image,
    t: float,
    box=(120, 20, 1800, 960),
    zoom_from=1.0,
    zoom_to=1.08,
    dx=0.0,
    dy=0.0,
):
    """t in [0,1] within the shot. Place the keyed render in box with slow zoom/slide."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    base = fit_scale(src, bw, bh)
    z = lerp(zoom_from, zoom_to, ease_in_out(t))
    scale = base * z
    cx = (x0 + x1) / 2 + dx * t
    cy = (y0 + y1) / 2 + dy * t
    paste_centered(canvas, src, int(cx), int(cy), scale)


def draw_caption(canvas: Image.Image, text: str, alpha: float = 1.0) -> None:
    if alpha <= 0.01 or not text:
        return
    d = ImageDraw.Draw(canvas)
    font = font_mono(28, "medium")
    # orange tick + cream caption, bottom-left
    x, y = 80, 1000
    a = int(255 * clamp01(alpha))
    cream = (CREAM[0], CREAM[1], CREAM[2], a)
    orange = (ORANGE[0], ORANGE[1], ORANGE[2], a)
    d.rectangle([x, y + 8, x + 18, y + 12], fill=orange)
    d.text((x + 32, y), text, font=font, fill=cream)


def rounded_rect(draw, xy, r, fill=None, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------
def scene_hook(t: float, keyed) -> Image.Image:
    """0-5s. t is local time."""
    ga = ease_out(t / 1.25)
    canvas = grid(ga)
    d = ImageDraw.Draw(canvas)

    # Chat box fades in 0.45-1.0
    box_a = ease_out((t - 0.45) / 0.5)
    if box_a > 0.01:
        # typing
        start_type, end_type = 1.05, 3.85
        if t <= start_type:
            nchars = 0
        elif t >= end_type:
            nchars = len(PROMPT)
        else:
            nchars = int(round((t - start_type) / (end_type - start_type) * len(PROMPT)))
            nchars = max(0, min(len(PROMPT), nchars))
        shown = PROMPT[:nchars]
        mono = font_mono(34, "regular")
        label_f = font_mono(22, "bold")
        label = "Prompt"
        # measure
        tw, th = text_size(d, PROMPT, mono)
        lw, lh = text_size(d, label, label_f)
        pad_x, pad_y = 36, 28
        box_w, box_h = tw + pad_x * 2 + 18, 148
        bx = (W - box_w) // 2
        by = (H - box_h) // 2 - 20
        a = int(255 * box_a)
        surface = (SURFACE[0], SURFACE[1], SURFACE[2], a)
        border = (90, 100, 118, a)
        rounded_rect(d, [bx, by, bx + box_w, by + box_h], 16, fill=surface, outline=border, width=2)
        # accent bar on left
        d.rectangle([bx, by + 16, bx + 4, by + box_h - 16], fill=(ORANGE[0], ORANGE[1], ORANGE[2], a))
        # draw the label on its own layer so it cannot be clipped or painted over
        lab = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lab)
        ld.text((bx + pad_x, by + 16), label, font=label_f, fill=(ORANGE[0], ORANGE[1], ORANGE[2], a))
        canvas.alpha_composite(lab)
        prompt_y = by + 16 + lh + 10
        d.text((bx + pad_x, prompt_y), shown, font=mono, fill=(CREAM[0], CREAM[1], CREAM[2], a))
        # cursor
        if t >= start_type:
            on = int((t * 2) % 2) == 0  # 2 Hz blink
            if on or nchars < len(PROMPT):
                cw = text_size(d, shown, mono)[0]
                cx = bx + pad_x + cw + 4
                cy0, cy1 = prompt_y, prompt_y + 34
                d.rectangle([cx, cy0, cx + 3, cy1], fill=(ORANGE[0], ORANGE[1], ORANGE[2], a))
    return canvas


def scene_resolve(t: float, keyed) -> Image.Image:
    """5-8s, t local. Prompt resolves into bench_psu_2."""
    canvas = grid(1.0)
    # model fades/scales in over 0-1.3s
    appear = ease_out(t / 1.10)
    zoom = lerp(0.86, 1.0, ease_out(t / 2.4))
    src = keyed["bench_psu_2"]
    if appear > 0.01:
        # temporary alpha
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ken_burns_paste(
            layer,
            src,
            clamp01(t / 2.5),
            box=(200, 40, 1720, 920),
            zoom_from=zoom,
            zoom_to=zoom * 1.04,
        )
        if appear < 0.999:
            a = layer.split()[3].point(lambda p: int(p * appear))
            layer.putalpha(a)
        canvas.alpha_composite(layer)

    # chat box shrinks/fades 0-0.9s
    fade = 1.0 - ease_in_out(t / 0.85)
    if fade > 0.02:
        d = ImageDraw.Draw(canvas)
        mono = font_mono(34, "regular")
        a = int(255 * fade)
        tw, _ = text_size(d, PROMPT, mono)
        box_w, box_h = tw + 72 + 18, 132
        # slide down slightly as it fades
        bx = (W - box_w) // 2
        by = (H - box_h) // 2 - 20 + int(40 * (1 - fade))
        rounded_rect(
            d,
            [bx, by, bx + box_w, by + box_h],
            16,
            fill=(SURFACE[0], SURFACE[1], SURFACE[2], a),
            outline=(90, 100, 118, a),
            width=2,
        )

    # tag — drawn on its own layer because PIL treats RGBA fill as opaque
    tag_a = ease_out((t - 1.00) / 0.45)
    if tag_a > 0.01:
        f = font_mono(22, "medium")
        label = "delivered  ·  7/7 checks"
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        tw, th = text_size(ld, label, f)
        px, py = 80, 72
        pad_x, pad_y = 16, 10
        rounded_rect(
            ld,
            [px, py, px + tw + pad_x * 2, py + th + pad_y * 2],
            8,
            fill=(24, 30, 42, 230),
            outline=ORANGE,
            width=2,
        )
        ld.text((px + pad_x, py + pad_y - 1), label, font=f, fill=CREAM)
        if tag_a < 0.999:
            la = layer.split()[3].point(lambda p, a=tag_a: int(p * a))
            layer.putalpha(la)
        canvas.alpha_composite(layer)
    return canvas


# Mixed render / CC BY b-roll timeline (7.50–26.50). Cuts land on a 120 BPM grid.
# b-roll clips are category pairings, not the same CADmind meshes.
SHOTS = [
    # (kind, start, end, ...)
    # render: key, caption, zoom_from, zoom_to, dx, dy
    # broll:  clip_id, caption, credit
    ("broll", 7.50, 9.00, "duncan_hero", "bench power supply case", "Footage: Uncle Duncan's Electronics Shack  ·  CC BY"),
    ("render", 9.00, 10.00, "mini_itx_2", "Mini-ITX PC case", 1.00, 1.07, 12, -8),
    ("render", 10.00, 11.00, "mini_itx_3", "Mini-ITX PC case", 1.06, 1.00, -10, 6),
    ("render", 11.00, 11.50, "pixel10_1", "Pixel 10 case", 1.00, 1.08, 8, 4),
    ("render", 11.50, 12.00, "board_enclosure_4", "board enclosure with USB-C cut-out", 1.02, 1.08, 0, 0),
    ("render", 12.00, 13.00, "board_enclosure_1", "board enclosure with USB-C cut-out", 1.00, 1.07, -8, 6),
    ("broll", 13.00, 14.50, "adafruit_ports", "printed enclosure", "Footage: Adafruit Industries  ·  CC BY"),
    ("render", 14.50, 15.50, "zoom_h5_1", "Zoom H5 mic adapter", 1.00, 1.08, 6, -4),
    ("render", 15.50, 16.50, "epaper_frame_2", "e-paper calendar frame", 1.05, 1.00, -6, 8),
    ("render", 16.50, 17.50, "macbook_stand_1", "MacBook Air stand", 1.00, 1.07, 10, 0),
    ("broll", 17.50, 19.00, "kprint_print", "laptop stand", "Footage: KPrint  ·  CC BY"),
    ("broll", 19.00, 20.50, "kprint_offplate", "laptop stand", "Footage: KPrint  ·  CC BY"),
    ("render", 20.50, 21.50, "headphone_hook_1", "headphone hook", 1.00, 1.08, -8, 6),
    ("broll", 21.50, 23.00, "yo2btw_print", "headphone stand", "Footage: YO2BTW  ·  CC BY"),
    ("broll", 23.00, 24.50, "yo2btw_use", "headphone stand", "Footage: YO2BTW  ·  CC BY"),
    ("render", 24.50, 25.50, "desk_organizer_1", "desk organizer", 1.00, 1.07, 8, -6),
    ("render", 25.50, 26.00, "tao_keychain_1", "TAO keychain", 1.00, 1.10, 0, 4),
    ("render", 26.00, 27.00, "toothbrush_stand_2", "toothbrush stand", 1.00, 1.08, -6, -4),
]

SHOTS_END = 27.00
INTERRUPT_END = 29.50
NETWORK_END = 32.00
# end card 32.00–35.00


def fill_crop(im: Image.Image) -> Image.Image:
    """Crop near-black pillar/letterbox, then scale-to-cover 1920x1080 (bias up)."""
    a = np.array(im.convert("RGB"))
    lum = a.max(axis=2)
    colmax = lum.max(axis=0)
    rowmax = lum.max(axis=1)
    # near-black threshold: bars can be slightly above 0
    left = int(np.argmax(colmax > 12))
    right = int(len(colmax) - np.argmax(colmax[::-1] > 12))
    top = int(np.argmax(rowmax > 12))
    bot = int(len(rowmax) - np.argmax(rowmax[::-1] > 12))
    # only crop if bars are meaningful
    if left > 40 or (W - right) > 40 or top > 40 or (H - bot) > 40:
        left = max(0, left)
        top = max(0, top)
        right = min(W, right)
        bot = min(H, bot)
        if right - left >= 200 and bot - top >= 200:
            im = im.crop((left, top, right, bot))
    # scale to cover
    sw, sh = im.size
    scale = max(W / sw, H / sh)
    nw, nh = int(round(sw * scale)), int(round(sh * scale))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    # bias crop upward so heads/products near the top survive
    x0 = max(0, (nw - W) // 2)
    y0 = max(0, int((nh - H) * 0.22))
    x0 = min(x0, max(0, nw - W))
    y0 = min(y0, max(0, nh - H))
    return im.crop((x0, y0, x0 + W, y0 + H))


def grade_broll(rgb: np.ndarray, style: str = "default") -> np.ndarray:
    """Mild navy-palette grade: contrast, slight desat, cool shadows, vignette."""
    x = rgb.astype(np.float32) / 255.0
    contrast = 1.10 if style == "hot" else 1.08
    x = (x - 0.5) * contrast + (0.44 if style == "hot" else 0.48)
    x *= 0.84 if style == "hot" else 0.92
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    sat = 0.62 if style == "hot" else 0.82
    x = x * sat + y[..., None] * (1.0 - sat)
    navy = np.array([16.0, 20.0, 28.0], dtype=np.float32) / 255.0
    shadow = np.clip(1.0 - y * 1.8, 0.0, 1.0)[..., None]
    sh = 0.30 if style == "hot" else 0.24
    x = x * (1.0 - sh * shadow) + navy * (sh * shadow)
    mid = np.clip(1.0 - np.abs(y - 0.45) * 2.2, 0.0, 1.0)
    x[..., 2] = x[..., 2] + 0.04 * mid
    x[..., 0] = x[..., 0] - 0.025 * mid
    # pull hot yellow (Adafruit backdrop) toward a dimmer olive-navy
    yellow = np.clip((x[..., 0] + x[..., 1]) * 0.5 - x[..., 2] - 0.08, 0.0, 1.0)
    ystr = 0.55 if style == "hot" else 0.18
    x[..., 0] = x[..., 0] - (0.22 * ystr / 0.18) * yellow
    x[..., 1] = x[..., 1] - (0.28 * ystr / 0.18) * yellow
    x[..., 2] = x[..., 2] + (0.06 * ystr / 0.18) * yellow
    if style == "hot":
        # further crush bright backdrop toward a muted khaki-navy
        bg = np.clip((y - 0.42) / 0.40, 0.0, 1.0) * yellow
        target = navy * 0.55 + np.array([0.22, 0.22, 0.16])
        x = x * (1.0 - 0.55 * bg[..., None]) + target * (0.55 * bg[..., None])
        x *= 0.92
    h, w = y.shape
    yy, xx = np.ogrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w * 0.78)) ** 2 + ((yy - h / 2) / (h * 0.78)) ** 2)
    vig_amt = 0.36 if style == "hot" else 0.28
    vig = np.clip(1.0 - vig_amt * np.clip(r - 0.32, 0.0, 1.0) ** 1.35, 0.58, 1.0)
    x = x * vig[..., None]
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def load_broll() -> dict[str, list[Image.Image]]:
    """Load pre-extracted 30fps JPEG cuts, apply grade once, cache in memory."""
    clips: dict[str, list[Image.Image]] = {}
    ids = [
        "duncan_hero",
        "adafruit_ports",
        "kprint_print",
        "kprint_offplate",
        "yo2btw_print",
        "yo2btw_use",
    ]
    # extra zoom to drop a faint YouTube badge in the Adafruit lower-right
    extra_zoom = {"adafruit_ports": 1.16}
    for cid in ids:
        folder = BROLL_CUTS / cid
        frames = []
        paths = sorted(folder.glob("f*.jpg"))
        if not paths:
            raise FileNotFoundError(f"missing b-roll cut frames: {folder}")
        style = "hot" if cid == "adafruit_ports" else "default"
        z = extra_zoom.get(cid, 1.0)
        for fp in paths:
            im = Image.open(fp).convert("RGB")
            if im.size != (W, H):
                im = im.resize((W, H), Image.Resampling.LANCZOS)
            im = fill_crop(im)
            if im.size != (W, H):
                im = im.resize((W, H), Image.Resampling.LANCZOS)
            if z > 1.001:
                nw, nh = int(round(W * z)), int(round(H * z))
                im = im.resize((nw, nh), Image.Resampling.LANCZOS)
                # bias the crop toward the top-left so the lower-right badge is cut
                x0 = int((nw - W) * 0.22)
                y0 = int((nh - H) * 0.18)
                im = im.crop((x0, y0, x0 + W, y0 + H))
            graded = grade_broll(np.array(im), style=style)
            frames.append(Image.fromarray(graded, "RGB"))
        clips[cid] = frames
        print(f"broll {cid}: {len(frames)} frames", flush=True)
    return clips


def draw_credit(canvas: Image.Image, text: str, alpha: float = 1.0) -> None:
    """Small bottom-right corner credit on a dark pill."""
    if alpha <= 0.01 or not text:
        return
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    font = font_mono(22, "bold")
    tw, th = text_size(ld, text, font)
    pad_x, pad_y = 16, 10
    box_w, box_h = tw + pad_x * 2, th + pad_y * 2
    x = W - 64 - box_w
    y = H - 48 - box_h
    rounded_rect(
        ld,
        [x, y, x + box_w, y + box_h],
        6,
        fill=(10, 12, 18, 200),
        outline=(70, 82, 102, 180),
        width=1,
    )
    ld.text((x + pad_x, y + pad_y - 1), text, font=font, fill=CREAM)
    if alpha < 0.999:
        la = layer.split()[3].point(lambda p, a=alpha: int(p * a))
        layer.putalpha(la)
    canvas.alpha_composite(layer)


def scene_shots(t_abs: float, keyed, broll) -> Image.Image:
    shot = None
    for s in SHOTS:
        if s[1] <= t_abs < s[2]:
            shot = s
            break
    if shot is None and t_abs >= SHOTS[-1][1] and t_abs <= SHOTS[-1][2] + 1e-6:
        shot = SHOTS[-1]
    if shot is None:
        # never emit a blank/black frame; hold the nearest shot
        shot = SHOTS[0]
        for s in SHOTS:
            if t_abs >= s[1]:
                shot = s
    kind = shot[0]
    start, end = shot[1], shot[2]
    local = (t_abs - start) / max(1e-6, end - start)
    ca = ease_out(min(1.0, local / 0.15))
    if kind == "render":
        canvas = grid(1.0)
        _k, _s, _e, key, caption, z0, z1, dx, dy = shot
        ken_burns_paste(canvas, keyed[key], local, zoom_from=z0, zoom_to=z1, dx=dx, dy=dy)
        draw_caption(canvas, caption, ca)
        return canvas
    # b-roll
    canvas = Image.new("RGBA", (W, H), NAVY)
    _k, _s, _e, cid, caption, credit = shot
    frames = broll[cid]
    idx = int(clamp01(local) * (len(frames) - 1e-6))
    idx = max(0, min(len(frames) - 1, idx))
    fr = frames[idx].convert("RGBA")
    canvas.alpha_composite(fr, (0, 0))
    draw_caption(canvas, caption, ca)
    draw_credit(canvas, credit, min(1.0, ca + 0.15))
    return canvas


def scene_interrupt(t: float, keyed) -> Image.Image:
    """20-23s. Hard cut, silence, two lines."""
    # hard cut: plain navy, no grid
    canvas = Image.new("RGBA", (W, H), NAVY)
    d = ImageDraw.Draw(canvas)
    f1 = font_inter(72, 450)
    f2 = font_inter(72, 600)
    line1 = "You bring the hardware."
    line2 = "We build the plastic around it."
    # measure
    w1, h1 = text_size(d, line1, f1)
    w2, h2 = text_size(d, line2, f2)
    gap = 28
    total_h = h1 + gap + h2
    y0 = (H - total_h) // 2 - 10
    x1 = (W - w1) // 2
    x2 = (W - w2) // 2
    d.text((x1, y0), line1, font=f1, fill=CREAM)
    if t >= 1.15:
        a = ease_out((t - 1.15) / 0.25)
        col = (ORANGE[0], ORANGE[1], ORANGE[2], int(255 * a))
        # draw via a layer so we can fade
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.text((x2, y0 + h1 + gap), line2, font=f2, fill=col)
        canvas.alpha_composite(layer)
    return canvas


def make_network(progress: float, n=22, seed=42) -> Image.Image:
    """Procedural orange node/line graph. progress 0-1 draws it in."""
    rng = np.random.default_rng(seed)
    # even ring + interior points, centered above the caption
    cx, cy, rx, ry = W // 2, 460, 480, 260
    nodes = []
    ring = 12
    for i in range(ring):
        ang = 2 * math.pi * i / ring + 0.15
        x = cx + math.cos(ang) * rx * 0.92
        y = cy + math.sin(ang) * ry * 0.92
        nodes.append((x, y, 0.05 + 0.35 * (i / ring)))
    for i in range(n - ring):
        ang = 2 * math.pi * (i + 0.5) / (n - ring)
        rad = 0.25 + 0.55 * math.sqrt(float(rng.uniform(0.0, 1.0)))
        x = cx + math.cos(ang) * rx * rad
        y = cy + math.sin(ang) * ry * rad
        nodes.append((x, y, 0.08 + 0.40 * (i / max(1, n - ring - 1))))
    # edges: 3-nearest + a ring cycle so it stays connected
    edges = []
    for i, (x, y, ti) in enumerate(nodes):
        dist = [((x - ox) ** 2 + (y - oy) ** 2, j) for j, (ox, oy, _) in enumerate(nodes) if j != i]
        dist.sort()
        for _, j in dist[:3]:
            a, b = min(i, j), max(i, j)
            if (a, b) not in edges:
                edges.append((a, b))
    for i in range(ring):
        a, b = i, (i + 1) % ring
        a, b = min(a, b) if b else (0, ring - 1), max(a, b) if b else (0, ring - 1)
        # fix the last ring edge
    for i in range(ring):
        a, b = i, (i + 1) % ring
        if a > b:
            a, b = b, a
        if (a, b) not in edges:
            edges.append((a, b))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # lines
    for a, b in edges:
        t0 = max(nodes[a][2], nodes[b][2]) * 0.35
        p = clamp01((progress - t0) / 0.28)
        if p <= 0:
            continue
        x0, y0 = nodes[a][0], nodes[a][1]
        x1, y1 = nodes[b][0], nodes[b][1]
        x1 = x0 + (x1 - x0) * ease_out(p)
        y1 = y0 + (y1 - y0) * ease_out(p)
        alpha = int(180 * p)
        d.line([(x0, y0), (x1, y1)], fill=(ORANGE[0], ORANGE[1], ORANGE[2], alpha), width=2)
    # nodes
    for x, y, ti in nodes:
        p = clamp01((progress - ti * 0.5) / 0.25)
        if p <= 0:
            continue
        r = lerp(0, 6, ease_out(p))
        # pulse
        pulse = 0.85 + 0.15 * math.sin(progress * 8 + x * 0.01)
        alpha = int(255 * p * pulse)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(ORANGE[0], ORANGE[1], ORANGE[2], alpha))
        # soft halo
        hr = r + 5
        d.ellipse(
            [x - hr, y - hr, x + hr, y + hr],
            outline=(ORANGE[0], ORANGE[1], ORANGE[2], int(50 * p)),
            width=1,
        )
    return layer


def scene_network(t: float, keyed) -> Image.Image:
    """23-26s."""
    canvas = grid(0.45)
    canvas.alpha_composite(make_network(clamp01(t / 1.55)))
    d = ImageDraw.Draw(canvas)
    f = font_mono(24, "medium")
    text = "Designed by miners on Bittensor  ·  Subnet 42"
    a = ease_out(t / 0.5)
    col = (CREAM[0], CREAM[1], CREAM[2], int(255 * a))
    tw, th = text_size(d, text, f)
    # place below the network
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.text(((W - tw) // 2, 820), text, font=f, fill=col)
    canvas.alpha_composite(layer)
    return canvas


_LOGO_IMG = None


def scene_endcard(t: float, keyed) -> Image.Image:
    """26-30s."""
    global _LOGO_IMG
    canvas = grid(0.28)
    appear = ease_out(t / 0.6)
    if _LOGO_IMG is None:
        _LOGO_IMG = Image.open(LOGO).convert("RGBA")
    logo = _LOGO_IMG
    # mark
    mark_size = 180
    mark = logo.resize((mark_size, mark_size), Image.Resampling.LANCZOS)
    # wordmark
    dtmp = ImageDraw.Draw(canvas)
    f_cad = font_inter(96, 560)
    f_mind = font_inter(96, 560)
    f_line = font_inter(36, 400)
    f_url = font_mono(26, "medium")
    w_cad, h_cad = text_size(dtmp, "cad", f_cad)
    w_mind, _ = text_size(dtmp, "mind", f_mind)
    gap = 0
    wm_w = w_cad + w_mind + gap
    # vertical stack: mark, wordmark, tagline, url
    line = "Prompt to print."
    url = "cadmind.ai"
    w_line, h_line = text_size(dtmp, line, f_line)
    w_url, h_url = text_size(dtmp, url, f_url)
    stack_h = mark_size + 28 + h_cad + 36 + h_line + 18 + h_url
    y = (H - stack_h) // 2
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # mark
    mx = (W - mark_size) // 2
    layer.alpha_composite(mark, (mx, y))
    ld = ImageDraw.Draw(layer)
    wy = y + mark_size + 28
    wx = (W - wm_w) // 2
    ld.text((wx, wy), "cad", font=f_cad, fill=CREAM)
    ld.text((wx + w_cad + gap, wy), "mind", font=f_mind, fill=ORANGE)
    ly = wy + h_cad + 36
    ld.text(((W - w_line) // 2, ly), line, font=f_line, fill=CREAM)
    uy = ly + h_line + 18
    ld.text(((W - w_url) // 2, uy), url, font=f_url, fill=(ORANGE[0], ORANGE[1], ORANGE[2], 255))
    if appear < 0.999:
        a = layer.split()[3].point(lambda p: int(p * appear))
        layer.putalpha(a)
    canvas.alpha_composite(layer)
    # CC BY credits, independent of the centered stack
    cred_a = ease_out(max(0.0, (t - 0.45) / 0.5))
    if cred_a > 0.01:
        clayer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        cd = ImageDraw.Draw(clayer)
        f_cred = font_mono(22, "bold")
        c1 = "Footage  ·  Creative Commons Attribution"
        c2 = "YO2BTW  ·  KPrint  ·  Uncle Duncan's Electronics Shack  ·  Adafruit Industries"
        w1, h1 = text_size(cd, c1, f_cred)
        w2, h2 = text_size(cd, c2, f_cred)
        muted = (230, 228, 224, int(250 * cred_a))
        muted2 = (230, 228, 224, int(245 * cred_a))
        cy = H - 110
        cd.text(((W - w1) // 2, cy), c1, font=f_cred, fill=muted)
        cd.text(((W - w2) // 2, cy + h1 + 6), c2, font=f_cred, fill=muted2)
        canvas.alpha_composite(clayer)
    return canvas


# broll dict is attached on the keyed object via a side channel in main
_BROLL = {}


def render_frame(t: float, keyed, broll=None) -> Image.Image:
    if broll is None:
        broll = _BROLL
    if t < 5.0:
        return scene_hook(t, keyed)
    if t < 7.5:
        return scene_resolve(t - 5.0, keyed)
    if t < SHOTS_END:
        return scene_shots(t, keyed, broll)
    if t < INTERRUPT_END:
        return scene_interrupt(t - SHOTS_END, keyed)
    if t < NETWORK_END:
        return scene_network(t - INTERRUPT_END, keyed)
    return scene_endcard(t - NETWORK_END, keyed)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def env_exp(t, tau):
    return np.exp(-np.maximum(t, 0) / tau)


def synth_audio(path: Path) -> None:
    sr = 48000
    n = int(sr * DURATION)
    t = np.arange(n) / sr
    L = np.zeros(n, dtype=np.float64)
    R = np.zeros(n, dtype=np.float64)

    def place(sig, t0, dest=None):
        i0 = int(t0 * sr)
        if i0 >= n:
            return
        i1 = min(n, i0 + len(sig))
        sl = dest if dest is not None else L
        sl[i0:i1] += sig[: i1 - i0]

    def kick(t0, amp=0.85):
        tt = np.arange(int(0.18 * sr)) / sr
        freq = 110 * np.exp(-tt * 18) + 38
        phase = 2 * np.pi * np.cumsum(freq) / sr
        sig = amp * np.sin(phase) * env_exp(tt, 0.055)
        # click
        click = 0.25 * amp * np.sin(2 * np.pi * 1800 * tt) * env_exp(tt, 0.004)
        place(sig + click, t0, L)
        place(sig + click, t0, R)

    def hat(t0, amp=0.12, decay=0.025, pan=0.0):
        tt = np.arange(int(0.08 * sr)) / sr
        rng = np.random.default_rng(int(t0 * 1000) + 7)
        noise = rng.standard_normal(len(tt))
        # crude highpass
        noise = np.convolve(noise, np.array([1, -1], dtype=float), mode="same")
        sig = amp * noise * env_exp(tt, decay)
        l = sig * (1 - max(0, pan))
        r = sig * (1 + min(0, pan) * -1 if pan < 0 else 1 - 0)  # messy
        # simpler pan: pan -1 left +1 right
        l = sig * math.sqrt(0.5 * (1 - pan))
        r = sig * math.sqrt(0.5 * (1 + pan))
        place(l, t0, L)
        place(r, t0, R)

    def clap(t0, amp=0.22):
        tt = np.arange(int(0.12 * sr)) / sr
        rng = np.random.default_rng(int(t0 * 2000) + 3)
        noise = rng.standard_normal(len(tt))
        # band-ish: difference of smoothing
        kernel = np.hanning(21)
        kernel /= kernel.sum()
        sm = np.convolve(noise, kernel, mode="same")
        sig = amp * (noise - 0.5 * sm) * env_exp(tt, 0.035)
        # a couple of delayed bursts
        sig2 = np.zeros_like(sig)
        dly = int(0.012 * sr)
        if dly < len(sig):
            sig2[dly:] += 0.6 * sig[:-dly]
        place((sig + sig2) * 0.7, t0, L)
        place((sig + sig2) * 0.7, t0, R)

    def whoosh(t0, dur=0.45, amp=0.12):
        tt = np.arange(int(dur * sr)) / sr
        rng = np.random.default_rng(11)
        noise = rng.standard_normal(len(tt))
        # rising amplitude + simple high-shelf feel
        rise = (tt / dur) ** 2
        fade = env_exp(tt - dur * 0.6, 0.08)
        sig = amp * noise * rise * fade
        place(sig * 0.7, t0, L)
        place(sig * 0.9, t0, R)

    def typ(t0, amp=0.06):
        tt = np.arange(int(0.018 * sr)) / sr
        rng = np.random.default_rng(int(t0 * 8000) + 1)
        noise = rng.standard_normal(len(tt))
        tone = np.sin(2 * np.pi * rng.uniform(2400, 3400) * tt)
        sig = amp * (0.7 * noise + 0.3 * tone) * env_exp(tt, 0.006)
        pan = rng.uniform(-0.3, 0.3)
        place(sig * math.sqrt(0.5 * (1 - pan)), t0, L)
        place(sig * math.sqrt(0.5 * (1 + pan)), t0, R)

    def tone(t0, freq, dur, amp, tau, dest_both=True):
        tt = np.arange(int(dur * sr)) / sr
        sig = amp * np.sin(2 * np.pi * freq * tt) * env_exp(tt, tau)
        # mild fade in 8ms
        fi = int(0.008 * sr)
        sig[:fi] *= np.linspace(0, 1, fi)
        place(sig, t0, L)
        place(sig, t0, R)

    # ambient sub throughout hook
    drone = 0.05 * np.sin(2 * np.pi * 55 * t) * np.exp(-0.5 * np.clip(t, 0, 8) * 0)  # constant
    hook_env = np.zeros(n)
    hook_env[: int(5.2 * sr)] = 1
    # fade out into resolve
    fade = int(0.4 * sr)
    i5 = int(5.0 * sr)
    hook_env[i5 : i5 + fade] = np.linspace(1, 0.4, fade)
    hook_env[i5 + fade : int(7.5 * sr)] = 0.25  # under resolve
    L += drone * hook_env * 0.8
    R += drone * hook_env * 0.8

    # typing clicks ~15 cps from 1.05-3.85
    start_type, end_type = 1.05, 3.85
    for i, ch in enumerate(PROMPT):
        tt = start_type + (i / max(1, len(PROMPT) - 1)) * (end_type - start_type)
        if ch != " ":
            typ(tt, amp=0.045 + 0.01 * (i % 3))

    whoosh(4.95, dur=0.55, amp=0.10)

    # pickup kicks into first b-roll cut
    kick(7.00, 0.35)
    kick(7.50, 0.70)
    whoosh(7.35, dur=0.40, amp=0.08)

    # montage + b-roll 7.50–26.50: 120 BPM
    t0 = 7.50
    while t0 < SHOTS_END - 1e-6:
        kick(t0, 0.78)
        # backbeat clap on the off-beats of each second (*.00 kept as downbeat)
        if abs((t0 * 2) % 2 - 1) < 1e-6:
            clap(t0, 0.18)
        t0 += 0.5
    t0 = 7.75
    while t0 < SHOTS_END:
        hat(t0, amp=0.07, pan=0.25 if int(t0 * 4) % 2 == 0 else -0.25)
        t0 += 0.25
    # light whooshes at the start of each b-roll pair
    for wt in (13.00, 17.50, 21.50):
        whoosh(wt, dur=0.35, amp=0.07)

    # hard silence during the pattern-interrupt card
    i_cut = int(SHOTS_END * sr)
    fade_n = int(0.04 * sr)
    ramp = np.linspace(1, 0, fade_n)
    L[i_cut - fade_n : i_cut] *= ramp
    R[i_cut - fade_n : i_cut] *= ramp
    L[i_cut : int(INTERRUPT_END * sr)] = 0
    R[i_cut : int(INTERRUPT_END * sr)] = 0

    # network: soft kicks every 1s + high shimmer
    for tt in (INTERRUPT_END, INTERRUPT_END + 1.0, INTERRUPT_END + 2.0):
        if tt < NETWORK_END:
            kick(tt, 0.42)
            hat(tt + 0.5, amp=0.05)
    i_n0, i_n1 = int(INTERRUPT_END * sr), int(NETWORK_END * sr)
    ts = np.arange(i_n1 - i_n0) / sr
    shim = 0.03 * (
        np.sin(2 * np.pi * 880 * ts) * 0.4
        + np.sin(2 * np.pi * 1320 * ts) * 0.3
        + np.sin(2 * np.pi * 1760.5 * ts) * 0.2
    ) * (0.3 + 0.7 * (ts / max(ts[-1], 1e-6)))
    fo = int(0.15 * sr)
    if fo < len(shim):
        shim[-fo:] *= np.linspace(1, 0, fo)
    L[i_n0:i_n1] += shim
    R[i_n0:i_n1] += shim * 0.85

    # end card sting: low fifth, clean
    tone(NETWORK_END, 110, 2.6, 0.16, 1.4)
    tone(NETWORK_END, 165, 2.6, 0.10, 1.2)
    tone(NETWORK_END + 0.08, 220, 1.8, 0.07, 0.9)
    kick(NETWORK_END, 0.5)

    # fade last 0.5s
    fade_n = int(0.5 * sr)
    ramp = np.linspace(1, 0, fade_n)
    L[-fade_n:] *= ramp
    R[-fade_n:] *= ramp

    # peak normalize to ~-1.2 dB
    peak = max(1e-9, max(np.max(np.abs(L)), np.max(np.abs(R))))
    gain = 0.87 / peak
    L = np.clip(L * gain, -1, 1)
    R = np.clip(R * gain, -1, 1)
    # 16-bit stereo wav
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.stack([L, R], axis=1)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    print("audio", path, "peak_in", float(peak), "gain", gain)


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------
def contact_sheet(keyed, path: Path) -> None:
    times = [2.70, 23.70, 33.20]
    labels = ["0:03  Hook", "0:24  Printed, in use", "0:33  End card"]
    frames = [render_frame(t, keyed, _BROLL).convert("RGB") for t in times]
    # 1920x1080 sheet, three 584x328 panels with captions
    sheet = Image.new("RGB", (1920, 1080), (16, 20, 28))
    d = ImageDraw.Draw(sheet)
    title_f = font_inter(28, 550)
    cap_f = font_mono(18, "medium")
    d.text((80, 36), "CADmind promo v2  ·  contact sheet", font=title_f, fill=CREAM)
    pw, ph = 560, 315  # 16:9
    gap = 40
    total = 3 * pw + 2 * gap
    x0 = (1920 - total) // 2
    y = 280
    for i, (fr, lab) in enumerate(zip(frames, labels)):
        thumb = fr.resize((pw, ph), Image.Resampling.LANCZOS)
        x = x0 + i * (pw + gap)
        # orange hairline
        d.rectangle([x - 1, y - 1, x + pw, y + ph], outline=(255, 79, 18))
        sheet.paste(thumb, (x, y))
        tw, _ = text_size(d, lab, cap_f)
        d.text((x + (pw - tw) // 2, y + ph + 18), lab, font=cap_f, fill=CREAM)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)
    print("contact", path)


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------
def encode(keyed) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-i",
        str(AUDIO),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-shortest",
        "-movflags",
        "+faststart",
        str(OUT_MP4),
    ]
    print("ffmpeg", " ".join(cmd), flush=True)
    err_path = PREVIEW / "ffmpeg.log"
    with open(err_path, "w") as errf:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=errf,
        )
        assert proc.stdin is not None
        try:
            for i in range(NFRAMES):
                t = i / FPS
                # last frame is 29.966s; that's correct for 30.0s at 30fps
                fr = render_frame(t, keyed).convert("RGB")
                proc.stdin.write(fr.tobytes())
                if i % 30 == 0:
                    print(f"  frame {i}/{NFRAMES} t={t:.2f}", flush=True)
            proc.stdin.close()
            proc.wait(timeout=240)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
            raise
    stderr = err_path.read_text(errors="replace")
    if proc.returncode != 0:
        sys.stderr.write(stderr[-4000:])
        raise SystemExit(f"ffmpeg failed: {proc.returncode}")
    print(stderr[-1500:])
    print("wrote", OUT_MP4)


def previews(keyed) -> None:
    times = [
        0.2, 1.5, 2.7, 4.2, 5.4, 6.6, 8.2, 10.4, 12.5, 13.7,
        16.9, 18.2, 19.7, 21.0, 22.2, 23.8, 25.0, 25.8, 26.5, 27.2, 28.8, 30.6, 33.2,
    ]
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for t in times:
        fr = render_frame(t, keyed, _BROLL).convert("RGB")
        p = PREVIEW / f"frame_{t:04.1f}.png"
        fr.save(p)
        print("preview", p)


def main():
    preview_only = "--preview" in sys.argv
    KEYED.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    keyed = preprocess()
    # save a keyed QA image
    keyed["bench_psu_2"].save(PREVIEW / "keyed_bench_psu_2.png")
    keyed["mini_itx_2"].save(PREVIEW / "keyed_mini_itx_2.png")
    global _BROLL
    _BROLL = load_broll()
    if preview_only:
        previews(keyed)
        contact_sheet(keyed, OUT_CONTACT)
        return
    synth_audio(AUDIO)
    previews(keyed)
    contact_sheet(keyed, OUT_CONTACT)
    encode(keyed)
    # duration check
    import json

    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,bit_rate:stream=codec_name,width,height,avg_frame_rate,nb_frames,codec_type,sample_rate",
            "-of",
            "json",
            str(OUT_MP4),
        ]
    )
    print(probe.decode())


if __name__ == "__main__":
    main()
