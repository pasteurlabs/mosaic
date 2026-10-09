#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draw the Mosaic wordmark (docs/logo.webp) from the benchmark's own fields.

The letters MOSAIC are tiled with irregular stones (a relaxed Voronoi
tessellation clipped to the glyphs). Each stone shows a crop of a real
solution field from ``mosaic-results/``: vorticity and speed of the 2-D and
3-D Navier–Stokes flows and gradient magnitudes ∂L/∂IC, in a few colour maps.
The grout between stones is transparent, so the logo works on light and dark
backgrounds.

Needs benchmark results under ``mosaic-results/`` (see Getting Started) and a
heavy sans font (Arial Black by default). The output is committed; rerun only
to redesign it.

Usage:
    uv run python docs/figures/make_logo.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "mosaic-results"
OUT = ROOT / "docs" / "logo.webp"

FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf",
]
WORD = "MOSAIC"
WIDTH = 1800  # output px (shown at half size, i.e. 2× for sharp screens)
SUPER = 2  # supersampling for anti-aliased grout
STONE = 50  # typical stone spacing in output px
GROUT = 3.0  # grout width in output px
SEED = 5

JAGGED = 0.35  # stone-edge irregularity, as a fraction of the stone spacing
OUTLINE = 0.08  # letter-outline irregularity, same units

# Colour maps built from the docs palette only (indigo, paper, terracotta,
# green, saffron): diverging for signed vorticity, sequential otherwise.
_DIVERGING = LinearSegmentedColormap.from_list(
    "mosaic_div", ["#1f3263", "#2f4b8f", "#e9e4da", "#c8462b", "#7a2412"]
)
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "mosaic_seq", ["#1d2b4f", "#2f4b8f", "#2f7a57", "#e3a02f", "#f3d58a"]
)
CMAPS = {"vorticity": _DIVERGING, "speed": _SEQUENTIAL, "gradient": _SEQUENTIAL}


# ── Fields ───────────────────────────────────────────────────────────────────


def _vorticity(v: np.ndarray) -> np.ndarray:
    vx, vy = v[..., 0], v[..., 1]
    dvy = np.roll(vy, -1, 0) - np.roll(vy, 1, 0)
    dvx = np.roll(vx, -1, 1) - np.roll(vx, 1, 1)
    return (dvy - dvx) / 2


def _velocity_slices(arr: np.ndarray) -> list[np.ndarray]:
    """(N, N, 1, 2) → its 2-D velocity; (N, N, N, 3) → the mid-z slice."""
    if arr.ndim == 4 and arr.shape[2] == 1 and arr.shape[3] == 2:
        return [arr[:, :, 0, :]]
    if arr.ndim == 4 and arr.shape[3] == 3:
        return [arr[:, :, arr.shape[2] // 2, :2]]
    return []


def collect_fields() -> list[tuple[str, np.ndarray]]:
    """(kind, 2-D array) pairs from every velocity / gradient field ≥ 32²."""
    fields = []
    for path in sorted(glob.glob(str(RESULTS / "**" / "*.npz"), recursive=True)):
        if "jacobian" in path:
            continue
        try:
            npz = np.load(path, allow_pickle=False)
        except Exception:  # noqa: S112 — skip unreadable archives
            continue
        is_grad = "gradient_fields" in path
        for key in npz.files:
            for v in _velocity_slices(npz[key]):
                if min(v.shape[:2]) < 32:
                    continue
                if is_grad and key != "ic":
                    fields.append(("gradient", np.hypot(v[..., 0], v[..., 1])))
                else:
                    fields.append(("vorticity", _vorticity(v)))
                    fields.append(("speed", np.hypot(v[..., 0], v[..., 1])))
    return [(k, f) for k, f in fields if np.isfinite(f).all() and np.ptp(f) > 0]


def _colorize(field: np.ndarray, kind: str) -> np.ndarray:
    if kind == "vorticity":
        m = np.percentile(np.abs(field), 99) or 1.0
        x = np.clip(field / m, -1, 1)
        t = (
            np.sign(x) * np.sqrt(np.abs(x)) + 1
        ) / 2  # lift weak features off the centre
    else:
        lo, hi = np.percentile(field, [1, 99])
        t = (field - lo) / ((hi - lo) or 1.0)
    return CMAPS[kind](np.clip(t, 0, 1))[..., :3]


# ── Stones ───────────────────────────────────────────────────────────────────


def letter_mask(width: int) -> np.ndarray:
    """Anti-aliased glyph coverage in [0, 1], ``width`` px wide."""
    font_path = next((f for f in FONTS if Path(f).exists()), None)
    if font_path is None:
        raise SystemExit("No heavy sans font found; edit FONTS in make_logo.py")
    size = width // 4
    font = ImageFont.truetype(font_path, size)
    box = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), WORD, font=font)
    pad = size // 20
    img = Image.new("L", (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad))
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), WORD, font=font, fill=255)
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    return np.asarray(img, dtype=float) / 255


def rough_outline(
    mask: np.ndarray, spacing: float, rng: np.random.Generator
) -> np.ndarray:
    """Displace the glyph outline with smooth noise, like an edge of laid stones."""
    amp = OUTLINE * spacing
    dy, dx = (
        ndimage.gaussian_filter(rng.standard_normal(mask.shape), spacing / 4)
        for _ in range(2)
    )
    dy, dx = (d / (np.abs(d).max() or 1.0) * amp for d in (dy, dx))
    yy, xx = np.indices(mask.shape, dtype=float)
    return ndimage.map_coordinates(mask, [yy + dy, xx + dx], order=1, mode="constant")


def stones(mask: np.ndarray, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Label map of a relaxed Voronoi tessellation inside ``mask`` (−1 outside).

    Seeds are spread uniformly over the glyphs (one per ``spacing``² of area)
    and evened out with a few Lloyd steps, which gives stone-like cells of
    similar size but irregular shape.
    """
    inside = np.argwhere(mask > 0.5)
    n = max(8, round(len(inside) / spacing**2))
    pts = inside[rng.choice(len(inside), n, replace=False)].astype(float)
    for _ in range(4):
        _, lab = cKDTree(pts).query(inside)
        for k in range(n):
            sel = inside[lab == k]
            if len(sel):
                pts[k] = sel.mean(axis=0)
    # Warp the final assignment with smooth noise so stone edges wobble like
    # hand-cut stones instead of running dead straight.
    warp = [
        ndimage.gaussian_filter(rng.standard_normal(mask.shape), spacing / 5)
        for _ in range(2)
    ]
    warp = [w / (np.abs(w).max() or 1.0) * JAGGED * spacing for w in warp]
    shifted = inside + np.stack([w[inside[:, 0], inside[:, 1]] for w in warp], axis=1)
    _, lab = cKDTree(pts).query(shifted)
    labels = np.full(mask.shape, -1, dtype=int)
    labels[inside[:, 0], inside[:, 1]] = lab
    return labels


def grout(labels: np.ndarray, width: float) -> np.ndarray:
    """Boolean map of stone borders, about ``width`` px wide."""
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    return ndimage.binary_dilation(edge, iterations=max(1, round(width / 2)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    fields = collect_fields()
    if not fields:
        raise SystemExit(f"No fields found under {RESULTS}")
    mask = rough_outline(letter_mask(WIDTH * SUPER), STONE * SUPER, rng)
    labels = stones(mask, STONE * SUPER, rng)
    rgb = np.zeros((*mask.shape, 3))
    for k in range(labels.max() + 1):
        ys, xs = np.nonzero(labels == k)
        if not len(ys):
            continue
        y0, x0 = ys.min(), xs.min()
        hh, ww = ys.max() + 1 - y0, xs.max() + 1 - x0
        # A random window of a random field, stretched over the stone's
        # bounding box; nearly flat windows are redrawn so every stone shows
        # some structure.
        for _ in range(8):
            kind, field = fields[rng.integers(len(fields))]
            colors = _colorize(field, kind)
            n0, n1 = field.shape
            frac = rng.uniform(0.55, 1.0)
            c0, c1 = max(4, int(n0 * frac)), max(4, int(n1 * frac))
            oy, ox = rng.integers(0, n0 - c0 + 1), rng.integers(0, n1 - c1 + 1)
            crop = colors[oy : oy + c0, ox : ox + c1]
            if crop.std(axis=(0, 1)).mean() > 0.12:
                break
        tile = ndimage.zoom(crop, (hh / c0, ww / c1, 1), order=1)
        tile = np.clip(
            np.pad(
                tile,
                (
                    (0, max(0, hh - tile.shape[0])),
                    (0, max(0, ww - tile.shape[1])),
                    (0, 0),
                ),
                mode="edge",
            ),
            0,
            1,
        )
        rgb[ys, xs] = tile[ys - y0, xs - x0]
    alpha = mask * ~grout(labels, GROUT * SUPER)
    img = Image.fromarray(np.uint8(np.dstack([rgb, alpha]) * 255), "RGBA")
    img = img.resize((WIDTH, round(img.height / SUPER)), Image.LANCZOS)
    img.save(OUT, "WEBP", quality=88, method=6)  # ~140 KB with alpha
    print(
        f"Wrote {OUT} ({img.width}×{img.height}, {labels.max() + 1} stones, {len(fields)} fields)"
    )


if __name__ == "__main__":
    main()
