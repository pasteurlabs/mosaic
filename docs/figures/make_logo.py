#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draw the Mosaic wordmark (docs/logo.webp), a cleaner take on docs/logo.png.

Like the original stained-glass logo, the banner is a field of glass panes
(a relaxed Voronoi tessellation) with dark leading between them, and the word
MOSAIC is spelled by the colour of the panes. The original set cool letters
in a warm field; this version inverts it, with warm letters (terracotta,
orange, saffron, yellow) in a cool field (indigo, blue, teal, green), which
reads more clearly and sits well on the neutral site. A pane belongs
to a letter when the glyph covers most of it, so letter edges follow pane
edges. Panes are flat colour with a soft glass highlight instead of the
original's textures, and their edges are slightly warped like hand-cut glass.

Needs a heavy sans font (Arial Black by default). The output is committed;
rerun only to redesign it.

Usage:
    uv run python docs/figures/make_logo.py
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "logo.webp"

FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf",
]
WORD = "MOSAIC"
WIDTH, HEIGHT = 1800, 430  # output px (shown at half size: 2× for sharp screens)
SUPER = 2  # supersampling for anti-aliased leading
PANE = 34  # typical pane spacing in the field, output px
DENSER = 3.0  # pane density on the letters relative to the field
LEADING = 3.0  # width of the dark lines between panes, output px
JAGGED = 0.18  # pane-edge irregularity, as a fraction of the pane spacing
SEED = 3

LEAD_COLOR = (32, 33, 36)
FIELD = ["#2f4b8f", "#24607f", "#2f7a57", "#3c5aa0", "#1f6f6a"]  # cool glass
LETTERS = ["#e2683f", "#ec8a3c", "#e3a02f", "#f0bf45"]  # warm glass
TRACKING = 0.16  # extra space between letters, as a fraction of the font size


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]) / 255


def _pane_color(palette: list[str], rng: np.random.Generator) -> np.ndarray:
    """A palette colour with small lightness/saturation variation, like panes
    cut from different sheets of glass."""
    h, lum, s = colorsys.rgb_to_hls(*_rgb(palette[rng.integers(len(palette))]))
    lum = float(np.clip(lum * rng.uniform(0.9, 1.1), 0, 1))
    s = float(np.clip(s * rng.uniform(0.9, 1.05), 0, 1))
    return np.array(colorsys.hls_to_rgb(h, lum, s))


def letter_mask(w: int, h: int) -> np.ndarray:
    """Glyph coverage in [0, 1] for WORD (with extra tracking), centred."""
    font_path = next((f for f in FONTS if Path(f).exists()), None)
    if font_path is None:
        raise SystemExit("No heavy sans font found; edit FONTS in make_logo.py")
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))

    def layout(
        size: int,
    ) -> tuple[list[tuple[str, float]], tuple[float, float, float, float]]:
        font = ImageFont.truetype(font_path, size)
        x, placed, top, bottom = 0.0, [], 1e9, -1e9
        for ch in WORD:
            box = measure.textbbox((0, 0), ch, font=font)
            placed.append((ch, x - box[0]))
            x += box[2] - box[0] + TRACKING * size
            top, bottom = min(top, box[1]), max(bottom, box[3])
        return placed, (0.0, top, x - TRACKING * size, bottom)

    _, box = layout(100)
    # Fit the word into 88 % of the width and 76 % of the height.
    size = int(100 * min(0.88 * w / (box[2] - box[0]), 0.76 * h / (box[3] - box[1])))
    placed, box = layout(size)
    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (w, h))
    draw = ImageDraw.Draw(img)
    x0 = (w - box[2]) / 2
    y0 = (h - (box[3] - box[1])) / 2 - box[1]
    for ch, x in placed:
        draw.text((x0 + x, y0), ch, font=font, fill=255)
    return np.asarray(img, dtype=float) / 255


def panes(glyphs: np.ndarray, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Label map of a relaxed, slightly warped Voronoi tessellation.

    Panes are smaller on and around the letters (density ``DENSER`` times the
    field's), so glyph edges are resolved while the field stays calm. Lloyd
    steps use density-weighted centroids, which keeps that contrast.
    """
    density = 1 + (DENSER - 1) * np.clip(
        ndimage.gaussian_filter(glyphs, spacing / 6) * 1.3, 0, 1
    )
    yy, xx = np.indices(glyphs.shape)
    grid = np.column_stack([yy.ravel(), xx.ravel()]).astype(float)
    weight = density.ravel()
    n = round(weight.sum() / spacing**2)
    start = rng.choice(len(grid), n, replace=False, p=weight / weight.sum())
    pts = grid[start] + rng.uniform(-0.5, 0.5, (n, 2))
    step = max(1, len(grid) // 400_000)  # Lloyd steps on a subsample
    sub, sub_w = grid[::step], weight[::step]
    for _ in range(5):
        _, lab = cKDTree(pts).query(sub)
        wsum = np.bincount(lab, weights=sub_w, minlength=n)
        ok = wsum > 0
        for dim in range(2):
            num = np.bincount(lab, weights=sub_w * sub[:, dim], minlength=n)
            pts[ok, dim] = num[ok] / wsum[ok]
    warp = [
        ndimage.gaussian_filter(rng.standard_normal(glyphs.shape), spacing / 5)
        for _ in range(2)
    ]
    # Smaller panes get proportionally smaller wobble.
    scale = JAGGED * spacing / np.sqrt(density)
    warp = [d / (np.abs(d).max() or 1.0) * scale for d in warp]
    _, lab = cKDTree(pts).query(grid + np.column_stack([d.ravel() for d in warp]))
    return lab.reshape(glyphs.shape)


def leading(labels: np.ndarray, width: float) -> np.ndarray:
    """Boolean map of the lines between panes, about ``width`` px wide."""
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    return ndimage.binary_dilation(edge, iterations=max(1, round(width / 2)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    h, w = HEIGHT * SUPER, WIDTH * SUPER
    glyphs = letter_mask(w, h)
    labels = panes(glyphs, PANE * SUPER, rng)
    idx = np.arange(labels.max() + 1)
    coverage = ndimage.mean(glyphs, labels, index=idx)
    colors = np.array(
        [_pane_color(LETTERS if c > 0.5 else FIELD, rng) for c in coverage]
    )

    # Soft glass highlight: each pane brightens towards a point near its centre.
    yy, xx = np.indices((h, w))
    jitter = rng.uniform(-0.3, 0.3, (2, len(idx))) * PANE * SUPER
    hy = ndimage.mean(yy, labels, index=idx) + jitter[0]
    hx = ndimage.mean(xx, labels, index=idx) + jitter[1]
    dist = np.hypot(yy - hy[labels], xx - hx[labels]) / (PANE * SUPER)
    glow = np.clip(1 - dist, 0, 1) ** 2 * 0.22

    rgb = colors[labels]
    rgb = rgb + (1 - rgb) * glow[..., None]
    rgb[leading(labels, LEADING * SUPER)] = np.array(LEAD_COLOR) / 255
    img = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255), "RGB")
    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
    img.save(OUT, "WEBP", quality=90, method=6)
    print(f"Wrote {OUT} ({WIDTH}×{HEIGHT}, {len(idx)} panes)")


if __name__ == "__main__":
    main()
