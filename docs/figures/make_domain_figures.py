#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draw the four benchmark-task schematics as SVG (docs/figures/domain_*.svg).

Each figure shows the control variable, the physical process and the
optimization objective of one benchmark task, in the docs palette. The SVGs are
written by hand (no plotting library) so they stay small, crisp and editable.

Usage:
    python docs/figures/make_domain_figures.py
"""

from __future__ import annotations

import base64
import math
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent

W, H = 600, 340

INK = "#1d1e20"
INK2 = "#2f3134"
MUTED = "#5f6368"
BORDER = "#e1e2e4"
PAPER = "#f7f7f6"
SAND = "#f1f1ef"
# One colour per concept across all figures.
ACCENT = "#a8401f"  # forces and forcing: load, drag, heat flux
SAFFRON = "#e3a02f"
INDIGO = "#2f4b8f"  # the control the optimizer changes
GREEN = "#2f7a57"  # the observation or target
TINT = "#eef0f4"
WALL = 11  # thickness of a fixed-boundary band

SANS = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
SERIF = "Georgia, 'Times New Roman', serif"
MONO = "'JetBrains Mono', Menlo, Consolas, monospace"


# ── Building blocks ──────────────────────────────────────────────────────────


def _defs(extra: str = "") -> str:
    markers = "".join(
        f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,1 L9,5 L0,9 z" fill="{color}"/></marker>'
        for name, color in (
            ("ink", INK2),
            ("accent", ACCENT),
            ("indigo", INDIGO),
            ("muted", MUTED),
            ("saffron", SAFFRON),
        )
    )
    blobs = "".join(
        f'<radialGradient id="blob-{name}"><stop offset="0" stop-color="{c}" '
        f'stop-opacity="0.85"/><stop offset="0.55" stop-color="{c}" '
        f'stop-opacity="0.35"/><stop offset="1" stop-color="{c}" stop-opacity="0"/>'
        "</radialGradient>"
        for name, c in (
            ("accent", ACCENT),
            ("indigo", INDIGO),
            ("green", GREEN),
            ("saffron", SAFFRON),
        )
    )
    markers += "".join(
        f'<marker id="ah-{name}-sm" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse">'
        f'<path d="M0,1 L9,5 L0,9 z" fill="{color}"/></marker>'
        for name, color in (
            ("accent", ACCENT),
            ("indigo", INDIGO),
            ("white", "#ffffff"),
        )
    )
    wall = (
        '<pattern id="wall" width="5.5" height="5.5" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="5.5" height="5.5" fill="#dcdee3"/>'
        f'<line x1="0" y1="0" x2="0" y2="5.5" stroke="{MUTED}" stroke-width="1.1"/></pattern>'
    )
    return f"<defs>{markers}{blobs}{wall}{extra}</defs>"


def _svg(body: str, title: str, extra_defs: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" aria-label="{title}">'
        f"<title>{title}</title>"
        f"<style>text{{font-family:{SANS};fill:{INK}}}"
        f".eyebrow{{font-family:{MONO};font-size:13px;font-weight:500;letter-spacing:.04em;fill:{INDIGO}}}"
        f".eyebrow.target{{fill:{GREEN}}}"
        f".eyebrow.quiet{{fill:{MUTED}}}"
        f".label{{font-size:19px;fill:{INK2}}}"
        f".small{{font-size:16px;fill:{MUTED}}}"
        f".dim{{font-size:15px;fill:{MUTED}}}"
        f".halo{{paint-order:stroke;stroke:#ffffff;stroke-width:4px;stroke-linejoin:round}}"
        f".m{{font-family:{SERIF};font-style:italic}}"
        f".up{{font-family:{SERIF};font-style:normal}}"
        "</style>"
        f"{_defs(extra_defs)}{body}</svg>\n"
    )


def _label(
    x: float,
    y: float,
    eyebrow: str,
    text: str,
    anchor: str = "start",
    kind: str = "",
) -> str:
    cls = f"eyebrow {kind}".strip()
    return (
        f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{eyebrow}</text>'
        f'<text x="{x}" y="{y + 24}" class="label" text-anchor="{anchor}">{text}</text>'
    )


def _objective(formula: str, width: float = 360) -> str:
    """Bottom-centred objective box: eyebrow on the left, formula on the right."""
    x0 = (W - width) / 2
    y0 = H - 50
    return (
        f'<rect x="{x0}" y="{y0}" width="{width}" height="42" rx="6" fill="{SAND}" '
        f'stroke="{BORDER}"/>'
        f'<text x="{x0 + 16}" y="{y0 + 26}" class="eyebrow quiet">OBJECTIVE</text>'
        f'<text x="{x0 + 100 + (width - 100) / 2}" y="{y0 + 28}" text-anchor="middle" '
        f'style="font-size:20px">{formula}</text>'
    )


# Sub/superscripts use explicit dy offsets (baseline-shift is not supported
# consistently across renderers), followed by a zero-width reset.
def _sub(s: str) -> str:
    return f'<tspan dy="0.3em" style="font-size:70%">{s}</tspan><tspan dy="-0.3em">\u200b</tspan>'


def _sup(s: str) -> str:
    return f'<tspan dy="-0.45em" style="font-size:70%">{s}</tspan><tspan dy="0.45em">\u200b</tspan>'


def _m(s: str) -> str:
    return f'<tspan class="m">{s}</tspan>'


def _up(s: str) -> str:
    return f'<tspan class="up">{s}</tspan>'


def _b(s: str) -> str:
    """Bold upright symbol (vectors)."""
    return f'<tspan class="up" font-weight="bold">{s}</tspan>'


def _path(points: list[tuple[float, float]]) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _blob(cx: float, cy: float, r: float, color: str, opacity: float = 1.0) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#blob-{color})" opacity="{opacity}"/>'


def _wall(x: float, y: float, w: float, h: float, edge: tuple) -> str:
    """A fixed boundary: hatched band with a solid edge facing the domain."""
    x1, y1, x2, y2 = edge
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="url(#wall)"/>'
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="2"/>'
    )


def _hatch(x: float, y0: float, y1: float, side: int = -1) -> str:
    """Vertical wall band on the left (side=-1) or right of *x*."""
    left = x - WALL if side < 0 else x
    return _wall(left, y0, WALL, y1 - y0, (x, y0, x, y1))


def _hatch_h(y: float, x0: float, x1: float, side: int = -1) -> str:
    """Horizontal wall band above (side=-1) or below *y*."""
    top = y - WALL if side < 0 else y
    return _wall(x0, top, x1 - x0, WALL, (x0, y, x1, y))


def _inflow_arrow(x0: float, x1: float, y: float) -> str:
    """Filled arrow in one shape, with a thin outline for use on dark fields."""
    head = min(8.0, (x1 - x0) * 0.6)
    pts = [
        (x0, y - 1.3),
        (x1 - head, y - 1.3),
        (x1 - head, y - 4),
        (x1, y),
        (x1 - head, y + 4),
        (x1 - head, y + 1.3),
        (x0, y + 1.3),
    ]
    return (
        f'<path d="{_path(pts)} z" fill="{INDIGO}" stroke="#ffffff" '
        'stroke-width="0.7" stroke-linejoin="round"/>'
    )


def _frange(a: float, b: float, step: float):
    v = a
    while v <= b + 1e-9:
        yield v
        v += step


def _arrow(x1, y1, x2, y2, color="ink", width=2.0, dash="") -> str:
    c = {
        "ink": INK2,
        "accent": ACCENT,
        "indigo": INDIGO,
        "muted": MUTED,
        "saffron": SAFFRON,
    }[color]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{width}"'
        f' marker-end="url(#ah-{color})"{d}/>'
    )


# ── F2: DFG 2D-1 channel, inflow optimization for drag ───────────────────────


def _speed_field_png(nx: int, ny: int, cx: float, cy: float, r: float) -> str:
    """Illustrative speed field as a PNG data URI, in the overview figure's colormap."""
    stops = [
        (0x2E, 0x1E, 0x3B),
        (0x41, 0x3D, 0x7B),
        (0x37, 0x65, 0x9E),
        (0x34, 0x8F, 0xA7),
        (0x40, 0xB7, 0xAD),
        (0x8B, 0xDA, 0xB2),
        (0xDE, 0xF5, 0xE5),
    ]
    rows = []
    for j in range(ny):
        y = (j + 0.5) / ny
        row = bytearray([0])
        for i in range(nx):
            x = (i + 0.5) / nx * 2.2
            yy = (1 - y) * 0.41
            speed = 4 * y * (1 - y)
            d = math.hypot(x - cx, yy - cy)
            speed *= 1 + 0.22 * math.exp(-(((x - cx) / 0.09) ** 2))
            if x > cx:
                wake = math.exp(-(((yy - cy) / (0.05 + 0.03 * (x - cx))) ** 2))
                speed *= 1 - 0.97 * wake * math.exp(-(x - cx) / 0.9)
            speed *= 1 - math.exp(-((((d / r) - 1) / 0.7) ** 2)) if d > r else 0.0
            t = max(0.0, min(0.999, speed / 1.2)) * (len(stops) - 1)
            k = int(t)
            row += bytes(
                round(a + (b - a) * (t - k))
                for a, b in zip(stops[k], stops[k + 1], strict=True)
            )
        rows.append(bytes(row))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", nx, ny, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def fig_ns_grid() -> str:
    # To scale: channel 2.2 x 0.41, cylinder D = 0.1 centered at (0.2, 0.2).
    px = 250
    x0, x1, yt = 12, 12 + 2.2 * px, 112
    yb = yt + 0.41 * px
    cx, cy, r = x0 + 0.2 * px, yb - 0.2 * px, 0.05 * px
    field = _speed_field_png(330, 62, 0.2, 0.2, 0.05)
    parts = [
        (
            f'<image href="{field}" x="{x0}" y="{yt}" width="{x1 - x0}" height="{yb - yt}" '
            'preserveAspectRatio="none"/>'
        ),
        _hatch_h(yt, x0, x1, side=-1),
        _hatch_h(yb, x0, x1, side=1),
    ]

    # Inflow (the control): arrows from the inlet, scaled by the profile.
    n = 8
    for i in range(1, n):
        u = 4 * (i / n) * (1 - i / n)
        parts.append(_inflow_arrow(x0 + 2, x0 + 8 + 30 * u, yt + (yb - yt) * i / n))

    # Cylinder and drag force.
    parts += [
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{INK}"/>',
        _arrow(cx + r + 2, cy, cx + r + 40, cy, "accent", 1.9),
        (
            f'<text x="{cx + r + 46}" y="{cy + 6}" class="label halo" style="fill:{ACCENT}">'
            f"{_m('F')}{_sub(_m('D'))}</text>"
        ),
    ]

    # Outflow.
    for i in (1, 2, 3):
        y = yt + (yb - yt) * i / 4
        parts.append(_arrow(x1 + 3, y, x1 + 20, y, "muted", 1.3))

    parts += [
        f'<text x="{(x0 + x1) / 2}" y="{yt - WALL - 9}" class="small" text-anchor="middle">no-slip walls</text>',
        f'<text x="{x1}" y="{yb + WALL + 22}" class="small" text-anchor="end">outlet {_m("p")} = 0</text>',
        _label(x0, 22, "CONTROL", f"inflow {_m('u')}({_m('y')}), fixed flow rate"),
        _label(W - 14, 22, "PROCESS", "steady, Re = 20", anchor="end", kind="quiet"),
        _objective(
            f"{_up('min')}{_sub(_m('u'))} {_m('F')}{_sub(_m('D'))}",
            250,
        ),
    ]
    return _svg(
        "".join(parts),
        "Navier–Stokes 2D: optimize the inflow profile to minimize drag on a cylinder",
    )


# ── F3: 3D Navier–Stokes, initial condition recovery ─────────────────────────


def _cube(
    x0: float, y0: float, s: float, dx: float, dy: float, field: str, cid: str
) -> str:
    """Oblique cube whose three visible faces show ``field`` (unit-square art)."""
    faces = {
        "front": (f"{s},0,0,{s},{x0},{y0}", f"M{x0},{y0} h{s} v{s} h{-s} z", ""),
        "top": (
            f"{s},0,{-dx},{-dy},{x0 + dx},{y0 + dy}",
            f"M{x0},{y0} L{x0 + dx},{y0 + dy} h{s} L{x0 + s},{y0} z",
            '<rect width="1" height="1" fill="#fff" opacity="0.35"/>',
        ),
        "side": (
            f"{dx},{dy},0,{s},{x0 + s},{y0}",
            f"M{x0 + s},{y0} L{x0 + s + dx},{y0 + dy} v{s} L{x0 + s},{y0 + s} z",
            f'<rect width="1" height="1" fill="{INK}" opacity="0.07"/>',
        ),
    }
    out = []
    for name, (matrix, clip, shade) in faces.items():
        out.append(
            f'<clipPath id="{cid}-{name}"><path d="{clip}"/></clipPath>'
            f'<g clip-path="url(#{cid}-{name})"><g transform="matrix({matrix})">'
            f'<rect width="1" height="1" fill="{PAPER}"/>{field}{shade}</g></g>'
        )
    # Hidden back edges (dashed), then visible edges.
    bx, by = x0 + dx, y0 + dy
    out.append(
        f'<path d="M{bx},{by + s} L{bx},{by} M{bx},{by + s} h{s} M{bx},{by + s} L{x0},{y0 + s}" '
        f'fill="none" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>'
        f'<path d="M{x0},{y0} h{s} v{s} h{-s} z M{x0},{y0} L{bx},{by} h{s} L{x0 + s},{y0} '
        f'M{bx + s},{by} v{s} L{x0 + s},{y0 + s}" fill="none" stroke="{INK2}" '
        f'stroke-width="1.5" stroke-linejoin="round"/>'
    )
    return "".join(out)


def fig_ns_3d_grid() -> str:
    # The recovery runs start from a random divergence-free field.
    initial = "".join(
        _blob(x, y, r, "indigo", o)
        for x, y, r, o in (
            (0.2, 0.24, 0.24, 1.0),
            (0.62, 0.16, 0.2, 0.55),
            (0.86, 0.5, 0.22, 1.0),
            (0.44, 0.52, 0.2, 0.55),
            (0.16, 0.74, 0.22, 0.55),
            (0.64, 0.84, 0.24, 1.0),
        )
    )
    evolved = "".join(
        _blob(x, y, r, "green", o)
        for x, y, r, o in (
            (0.26, 0.3, 0.36, 0.85),
            (0.74, 0.28, 0.32, 0.45),
            (0.3, 0.78, 0.32, 0.45),
            (0.76, 0.76, 0.36, 0.85),
        )
    )
    s, dx, dy = 118, 44, -30
    xa, xb, y0 = 40, 396, 112
    parts = [
        _cube(xa, y0, s, dx, dy, initial, "c0"),
        _cube(xb, y0, s, dx, dy, evolved, "c1"),
        (
            f'<path d="M{xa + s + dx + 14},{y0 + 42} C262,{y0 + 18} 336,{y0 + 18} '
            f'{xb - 12},{y0 + 42}" '
            f'fill="none" stroke="{INK2}" stroke-width="2" marker-end="url(#ah-ink)"/>'
        ),
        f'<text x="299" y="{y0 + 66}" class="small" text-anchor="middle">Navier–Stokes</text>',
        f'<text x="299" y="{y0 + 90}" class="label" text-anchor="middle">{_m("t")} = 0 → {_m("T")}</text>',
        (
            f'<text x="299" y="{y0 + s + 24}" class="small" text-anchor="middle">'
            f"periodic in {_m('x')}, {_m('y')}, {_m('z')}</text>"
        ),
        _label(14, 22, "CONTROL", f"initial velocity {_b('v')}{_sub('0')}"),
        _label(
            W - 14,
            22,
            "TARGET",
            f"velocity {_b('v')}({_m('T')})",
            anchor="end",
            kind="target",
        ),
        _objective(
            f"{_up('min')}{_sub(_b('v') + '₀')} "
            f"‖{_b('v')}({_m('T')}; {_b('v')}₀) − {_b('v')}{_sub('target')}‖{_sup('2')}",
            410,
        ),
    ]
    return _svg(
        "".join(parts),
        "Navier–Stokes 3D: recover the initial velocity field from a later snapshot",
    )


# ── S: cantilever compliance minimization ────────────────────────────────────


def fig_structural_mesh() -> str:
    # To scale: 2 x 1 slab, clamped at x = 0, loaded upward at the free bottom corner.
    x0, x1, y0, y1 = 150, 450, 84, 234
    load = (x1, y1)

    def chord(x: float, top: bool) -> float:
        if not top:
            return y1 - 8
        t = (x - x0) / (x1 - x0)
        return y0 + 8 + (y1 - y0 - 16) * t**2.2

    struts = []
    for top, width in ((True, 12), (False, 12)):
        pts = [(x, chord(x, top)) for x in _frange(x0, x1, 5)]
        struts.append(f'<path d="{_path(pts)}" stroke-width="{width}"/>')
    xs = [x0 + 5, 221, 285, 343, 390, 426]
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        w = 9 - i * 1.4
        struts.append(
            f'<path d="M{a},{chord(a, True)} L{b},{chord(b, False)} M{a},{chord(a, False)} '
            f'L{b},{chord(b, True)}" stroke-width="{w}"/>'
        )
    cell = 14
    grid = "".join(
        f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}"/>'
        for x in _frange(x0 + cell, x1 - 1, cell)
    ) + "".join(
        f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}"/>'
        for y in _frange(y0 + cell, y1 - 1, cell)
    )
    blur = '<filter id="soft" x="-5%" y="-5%" width="110%" height="110%"><feGaussianBlur stdDeviation="1.7"/></filter>'
    parts = [
        f'<clipPath id="beam"><rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}"/></clipPath>',
        f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="{TINT}"/>',
        (
            f'<g clip-path="url(#beam)"><g filter="url(#soft)" fill="none" stroke="{INDIGO}" '
            f'stroke-linecap="round" stroke-linejoin="round">{"".join(struts)}</g></g>'
        ),
        f'<g stroke="{INDIGO}" stroke-opacity="0.1" stroke-width="1">{grid}</g>',
        (
            f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" '
            f'stroke="{INK2}" stroke-width="1.5"/>'
        ),
        _hatch(x0, y0 - 10, y1 + 10, side=-1),
        f'<text x="{x0 - WALL - 10}" y="{(y0 + y1) / 2 + 6}" class="small" text-anchor="end">clamped</text>',
        _arrow(load[0], load[1] + 44, load[0], load[1] + 8, "accent", 2.6),
        f'<circle cx="{load[0]}" cy="{load[1]}" r="4.5" fill="{ACCENT}"/>',
        f'<text x="{load[0] + 12}" y="{load[1] + 26}" class="label" style="fill:{ACCENT}">{_m("F")}</text>',
        f'<text x="{load[0] + 30}" y="{load[1] + 26}" class="small">corner load</text>',
        _label(14, 22, "CONTROL", f"element densities {_m('ρ')}{_sub(_m('e'))}"),
        _objective(
            f"{_up('min')}{_sub(_m('ρ'))} {_b('f')}{_sup('⊤')}"
            f"{_b('u')}({_m('ρ')})"
            f'<tspan dx="12" class="small" style="font-size:15px">s.t.</tspan>'
            f'<tspan dx="8">Σ</tspan>{_sub(_m("e"))} {_m("v")}{_sub(_m("e"))}{_m("ρ")}{_sub(_m("e"))} ≤ '
            f"{_m('V')}{_sub(_m('f'))}",
            430,
        ),
    ]
    return _svg(
        "".join(parts),
        "Structural mechanics: place material in a clamped beam to minimize compliance",
        blur,
    )


# ── H: steady heat conduction, conductivity inversion ────────────────────────


def _temperature(nx: int, ny: int) -> list[list[float]]:
    """Solve div(k grad T) = 0 on the slab by SOR: T = 0 left, unit flux in on the right."""
    h = 2.0 / nx

    def k(x: float, y: float) -> float:
        g = sum(
            math.exp(-((x - cx) ** 2 + (y - 0.5) ** 2) / (2 * 0.15**2))
            for cx in (0.6, 1.4)
        )
        return 0.1 + 0.9 * min(g, 1.0)

    kk = [[k(i * h, j * h) for i in range(nx + 1)] for j in range(ny + 1)]
    temp = [[0.0] * (nx + 1) for _ in range(ny + 1)]
    for _ in range(900):
        for j in range(ny + 1):
            jm, jp = max(j - 1, 0), min(j + 1, ny)
            for i in range(1, nx + 1):
                ip = min(i + 1, nx)
                kw = (kk[j][i] + kk[j][i - 1]) / 2
                ke = (kk[j][i] + kk[j][ip]) / 2 if i < nx else 0.0
                ks = (kk[j][i] + kk[jm][i]) / 2 if j > 0 else 0.0
                kn = (kk[j][i] + kk[jp][i]) / 2 if j < ny else 0.0
                flux = h if i == nx else 0.0
                new = (
                    kw * temp[j][i - 1]
                    + ke * temp[j][ip]
                    + ks * temp[jm][i]
                    + kn * temp[jp][i]
                    + flux
                ) / (kw + ke + ks + kn)
                temp[j][i] += 1.8 * (new - temp[j][i])
    return temp


def _isotherms(temp, x0, y0, sx, sy, levels) -> str:
    """Marching squares: one path of line segments per contour level."""
    out = []
    for level in levels:
        d = []
        for j in range(len(temp) - 1):
            for i in range(len(temp[0]) - 1):
                corners = (
                    (i, j, temp[j][i]),
                    (i + 1, j, temp[j][i + 1]),
                    (i + 1, j + 1, temp[j + 1][i + 1]),
                    (i, j + 1, temp[j + 1][i]),
                )
                hits = []
                for (ax, ay, a), (bx, by, b) in zip(
                    corners, corners[1:] + corners[:1], strict=True
                ):
                    if (a < level) != (b < level):
                        t = (level - a) / (b - a)
                        hits.append((ax + (bx - ax) * t, ay + (by - ay) * t))
                if len(hits) == 2:
                    (px, py), (qx, qy) = hits
                    d.append(
                        f"M{x0 + px * sx:.1f},{y0 + py * sy:.1f}"
                        f"L{x0 + qx * sx:.1f},{y0 + qy * sy:.1f}"
                    )
        out.append(f'<path d="{"".join(d)}"/>')
    return "".join(out)


def fig_thermal_mesh() -> str:
    # To scale: 2 x 1 slab, two Gaussian conductivity peaks as in the recovery target.
    x0, x1, y0, y1 = 150, 450, 84, 234
    w, h = x1 - x0, y1 - y0
    nx, ny = 64, 32
    temp = _temperature(nx, ny)
    tmax = max(max(row) for row in temp)
    levels = [tmax * (i + 0.5) / 9 for i in range(9)]
    cond = "".join(
        _blob(x0 + w * fx, (y0 + y1) / 2, 0.37 * h, "indigo") for fx in (0.3, 0.7)
    )
    flux = [
        _arrow(x1 + 38, y0 + h * f, x1 + 7, y0 + h * f, "accent", 1.8)
        for f in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    parts = [
        f'<clipPath id="kclip"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath>',
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="{TINT}"/>',
        f'<g clip-path="url(#kclip)">{cond}</g>',
        (
            f'<g fill="none" stroke="{GREEN}" stroke-width="1.3" stroke-linecap="round">'
            f"{_isotherms(temp, x0, y0, w / nx, h / ny, levels)}</g>"
        ),
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="{INK2}" stroke-width="1.5"/>',
        _hatch(x0, y0 - 10, y1 + 10, side=-1),
        f'<text x="{x0 - WALL - 10}" y="{(y0 + y1) / 2 + 6}" class="label" text-anchor="end">{_m("T")} = 0</text>',
        *flux,
        f'<text x="{x1 + 22}" y="{y1 + 26}" class="label" text-anchor="middle" style="fill:{ACCENT}">{_m("q")}</text>',
        f'<text x="{(x0 + x1) / 2}" y="{y1 + 26}" class="small" text-anchor="middle">insulated top and bottom</text>',
        _label(14, 22, "CONTROL", f"conductivity {_m('k')}({_m('x')}, {_m('y')})"),
        _label(
            W - 14,
            22,
            "OBSERVATION",
            f"isotherms of {_m('T')}{_sub('obs')}",
            anchor="end",
            kind="target",
        ),
        _objective(
            f"{_up('min')}{_sub(_m('k'))} ‖{_m('T')}({_m('k')}) − {_m('T')}{_sub('obs')}‖{_sup('2')}",
            320,
        ),
    ]
    return _svg(
        "".join(parts),
        "Heat transfer: invert for the conductivity field from an observed temperature",
    )


FIGURES = {
    "domain_ns_grid.svg": fig_ns_grid,
    "domain_ns_3d_grid.svg": fig_ns_3d_grid,
    "domain_structural_mesh.svg": fig_structural_mesh,
    "domain_thermal_mesh.svg": fig_thermal_mesh,
}


def main() -> None:
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print(f"Wrote {OUT / name}")


if __name__ == "__main__":
    main()
