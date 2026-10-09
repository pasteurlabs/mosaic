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

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent

W, H = 720, 380

INK = "#1d1e20"
INK2 = "#2f3134"
MUTED = "#5f6368"
BORDER = "#e1e2e4"
PAPER = "#f7f7f6"
SAND = "#f1f1ef"
ACCENT = "#a8401f"
SAFFRON = "#e3a02f"
INDIGO = "#2f4b8f"
GREEN = "#2f7a57"
PLUM = "#6e3f8f"

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
            ("plum", PLUM),
            ("saffron", SAFFRON),
        )
    )
    markers += "".join(
        f'<marker id="ah-{name}-sm" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse">'
        f'<path d="M0,1 L9,5 L0,9 z" fill="{color}"/></marker>'
        for name, color in (("accent", ACCENT), ("indigo", INDIGO))
    )
    return f"<defs>{markers}{blobs}{extra}</defs>"


def _svg(body: str, title: str, extra_defs: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" aria-label="{title}">'
        f"<title>{title}</title>"
        f"<style>text{{font-family:{SANS};fill:{INK}}}"
        f".eyebrow{{font-family:{MONO};font-size:11px;font-weight:500;letter-spacing:.04em;fill:{ACCENT}}}"
        f".eyebrow.quiet{{fill:{MUTED}}}"
        f".label{{font-size:15px;fill:{INK2}}}"
        f".small{{font-size:12px;fill:{MUTED}}}"
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
    quiet: bool = False,
) -> str:
    cls = "eyebrow quiet" if quiet else "eyebrow"
    return (
        f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{eyebrow}</text>'
        f'<text x="{x}" y="{y + 21}" class="label" text-anchor="{anchor}">{text}</text>'
    )


def _objective(formula: str, width: float = 360) -> str:
    """Bottom-centred objective box: eyebrow on the left, formula on the right."""
    x0 = (W - width) / 2
    y0 = 314
    return (
        f'<rect x="{x0}" y="{y0}" width="{width}" height="46" rx="8" fill="{SAND}" '
        f'stroke="{BORDER}"/>'
        f'<text x="{x0 + 16}" y="{y0 + 27}" class="eyebrow quiet">OBJECTIVE</text>'
        f'<text x="{x0 + 100 + (width - 100) / 2}" y="{y0 + 29}" text-anchor="middle" '
        f'style="font-size:17px">{formula}</text>'
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


def _hatch(x: float, y0: float, y1: float, side: int = -1, step: float = 12) -> str:
    """A wall: solid line with 45° hatch ticks on one side (side=-1 → left/top)."""
    ticks = "".join(
        f'<line x1="{x}" y1="{y}" x2="{x + side * 10}" y2="{y + 10}" />'
        for y in _frange(y0, y1 - 10, step)
    )
    return (
        f'<g stroke="{INK2}" stroke-width="1.4">{ticks}</g>'
        f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" stroke="{INK2}" stroke-width="3"/>'
    )


def _hatch_h(y: float, x0: float, x1: float, side: int = -1, step: float = 12) -> str:
    ticks = "".join(
        f'<line x1="{x}" y1="{y}" x2="{x + 10}" y2="{y + side * 10}" />'
        for x in _frange(x0, x1 - 10, step)
    )
    return (
        f'<g stroke="{INK2}" stroke-width="1.2" opacity="0.7">{ticks}</g>'
        f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{INK2}" stroke-width="2.5"/>'
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


# ── F2: 2D channel flow, inflow optimization for drag ────────────────────────


def fig_ns_grid() -> str:
    x0, x1, yt, yb = 40, 680, 96, 256
    yc = (yt + yb) / 2
    cx, cy, r = 236, yc, 19
    parts = [
        f'<rect x="{x0}" y="{yt}" width="{x1 - x0}" height="{yb - yt}" fill="#eef0f4"/>',
        _hatch_h(yt, x0, x1, side=-1),
        _hatch_h(yb, x0, x1, side=1),
    ]

    # Streamlines: deflect around the cylinder, then oscillate in the wake.
    for y0 in (112, 128, 144, 160, 192, 208, 224, 240):
        off = y0 - yc
        pts = []
        for i in range(141):
            x = 110 + i * 4
            push = (
                22
                * math.copysign(1, off)
                * math.exp(-(((x - cx) / 38) ** 2))
                * math.exp(-abs(off) / 45)
            )
            ramp = max(0.0, min(1.0, (x - 270) / 120))
            wig = (
                7
                * ramp
                * math.sin(2 * math.pi * (x - 270) / 115 + (0.0 if off < 0 else 0.9))
            )
            pts.append((x, y0 + push + wig * math.exp(-abs(off) / 70)))
        parts.append(
            f'<path d="{_path(pts)}" fill="none" stroke="{INDIGO}" stroke-opacity="0.35" stroke-width="1.3"/>'
        )

    # Von Kármán street: alternating vortices.
    for i, vx in enumerate((318, 388, 458, 528, 598)):
        up = i % 2 == 0
        vy = yc - 20 if up else yc + 20
        col, mk = (ACCENT, "accent") if up else (INDIGO, "indigo")
        rr = 9 - i * 0.6
        a0, a1 = (200, -70) if up else (160, 430)
        sx, sy = (
            vx + rr * math.cos(math.radians(a0)),
            vy - rr * math.sin(math.radians(a0)),
        )
        ex, ey = (
            vx + rr * math.cos(math.radians(a1)),
            vy - rr * math.sin(math.radians(a1)),
        )
        sweep = 1 if up else 0
        parts.append(
            f'<path d="M{sx:.1f},{sy:.1f} A{rr:.1f},{rr:.1f} 0 1 {sweep} {ex:.1f},{ey:.1f}" '
            f'fill="none" stroke="{col}" stroke-width="1.6" stroke-opacity="0.8" marker-end="url(#ah-{mk}-sm)"/>'
        )

    # Inflow profile (the control): parabolic arrows plus the profile curve.
    tips = []
    for y in _frange(yt + 10, yb - 10, 15):
        s = 1 - ((y - yc) / (yc - yt)) ** 2
        L = 10 + 56 * s
        parts.append(_arrow(x0 + 6, y, x0 + 6 + L, y, "indigo", 1.6))
        tips.append((x0 + 6 + L + 3, y))
    parts.append(
        f'<path d="{_path([(x0 + 9, yt + 4), *tips, (x0 + 9, yb - 4)])}" fill="none" '
        f'stroke="{ACCENT}" stroke-width="2" stroke-dasharray="4 3"/>'
    )

    # Cylinder and drag force.
    parts += [
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{INK2}"/>',
        _arrow(cx + r + 4, cy, cx + r + 44, cy, "accent", 2.4),
        f'<text x="{cx + 8}" y="{cy + r + 22}" class="label">drag {_m("F")}{_sub(_m("D"))}</text>',
    ]

    # Outflow.
    for y in _frange(yt + 22, yb - 22, 23):
        parts.append(_arrow(x1 - 26, y, x1 - 6, y, "muted", 1.4))

    parts += [
        _label(x0, 40, "CONTROL", f"inflow profile {_m('u')}({_m('y')})"),
        _label(
            x1, 40, "PROCESS", "incompressible Navier–Stokes", anchor="end", quiet=True
        ),
        _objective(
            f"{_up('min')}{_sub(_m('u') + '(' + _m('y') + ')')} {_m('F')}{_sub(_m('D'))}",
            300,
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
    tgv = "".join(
        _blob(
            0.25 + 0.5 * i,
            0.25 + 0.5 * j,
            0.3,
            "accent" if (i + j) % 2 == 0 else "indigo",
        )
        for i in range(2)
        for j in range(2)
    ) + _blob(0.5, 0.5, 0.12, "indigo", 0.6)
    evolved = "".join(
        _blob(x, y, r, c)
        for x, y, r, c in (
            (0.22, 0.3, 0.26, "green"),
            (0.62, 0.22, 0.2, "plum"),
            (0.75, 0.68, 0.3, "green"),
            (0.35, 0.75, 0.22, "plum"),
            (0.5, 0.48, 0.16, "plum"),
        )
    )
    s, dx, dy = 150, 58, -40
    parts = [
        _cube(70, 120, s, dx, dy, tgv, "c0"),
        _cube(440, 120, s, dx, dy, evolved, "c1"),
        (
            f'<path d="M296,170 C340,140 390,140 430,170" fill="none" stroke="{INK2}" '
            f'stroke-width="2" marker-end="url(#ah-ink)"/>'
        ),
        '<text x="363" y="128" class="small" text-anchor="middle">Navier–Stokes rollout</text>',
        f'<text x="363" y="195" class="label" text-anchor="middle">{_m("t")} = 0 → {_m("T")}</text>',
        _label(70, 40, "CONTROL", f"initial velocity {_b('v')}{_sub('0')}"),
        _label(
            650,
            40,
            "TARGET",
            f"velocity at time {_m('T')}, {_b('v')}({_m('T')})",
            anchor="end",
        ),
        _objective(
            f"{_up('min')}{_sub(_b('v') + '₀')} "
            f"‖{_b('v')}({_m('T')}; "
            f"{_b('v')}₀) − "
            f"{_b('v')}{_sub('target')}‖{_sup('2')}",
            400,
        ),
    ]
    return _svg(
        "".join(parts),
        "Navier–Stokes 3D: recover the initial velocity field from a later snapshot",
    )


# ── S: cantilever compliance minimization ────────────────────────────────────


def fig_structural_mesh() -> str:
    x0, x1, y0, y1 = 170, 510, 100, 270
    tip = (x1, (y0 + y1) / 2)

    def chord(x: float, top: bool) -> float:
        t = ((x - x0) / (x1 - x0)) ** 1.5
        edge = y0 + 8 if top else y1 - 8
        return edge + (tip[1] - edge) * t

    struts = []
    for top in (True, False):
        pts = [(x, chord(x, top)) for x in _frange(x0, x1, 6)]
        struts.append(f'<path d="{_path(pts)}" stroke-width="15"/>')
    xs = [x0 + 6, 250, 325, 395, 455]
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        w = 11 - i * 2
        struts.append(
            f'<path d="M{a},{chord(a, True)} L{b},{chord(b, False)} M{a},{chord(a, False)} '
            f'L{b},{chord(b, True)}" stroke-width="{w}"/>'
        )
    grid = "".join(
        f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}"/>'
        for x in _frange(x0 + 17, x1 - 1, 17)
    ) + "".join(
        f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}"/>'
        for y in _frange(y0 + 17, y1 - 1, 17)
    )
    blur = '<filter id="soft" x="-5%" y="-5%" width="110%" height="110%"><feGaussianBlur stdDeviation="2.2"/></filter>'
    parts = [
        f'<clipPath id="beam"><rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}"/></clipPath>',
        f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="#eef0f4"/>',
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
        f'<text x="{x0 - 14}" y="{y1 + 34}" class="small" text-anchor="middle">clamped</text>',
        f'<circle cx="{tip[0]}" cy="{tip[1]}" r="4" fill="{ACCENT}"/>',
        _arrow(tip[0] + 34, tip[1] - 46, tip[0] + 34, tip[1] + 30, "accent", 2.6),
        f'<text x="{tip[0] + 48}" y="{tip[1] - 2}" class="label">{_m("F")}</text>',
        f'<text x="{tip[0] + 48}" y="{tip[1] + 16}" class="small">tip load</text>',
        _label(x0, 48, "CONTROL", f"element densities {_m('ρ')}{_sub(_m('e'))}"),
        f'<text x="{x1}" y="{y1 + 34}" class="small" text-anchor="end">dark = material, light = void</text>',
        _objective(
            f"{_up('min')}{_sub(_m('ρ'))} {_b('f')}{_sup('⊤')}"
            f"{_b('u')}({_m('ρ')})"
            f'<tspan dx="14" class="small" style="font-size:13px">s.t.</tspan>'
            f'<tspan dx="8">Σ</tspan>{_sub(_m("e"))} {_m("v")}{_sub(_m("e"))}{_m("ρ")}{_sub(_m("e"))} ≤ '
            f"{_m('V')}{_sub(_m('f'))}",
            470,
        ),
    ]
    return _svg(
        "".join(parts),
        "Structural mechanics: place material in a clamped beam to minimize compliance",
        blur,
    )


# ── H: steady heat conduction, conductivity inversion ────────────────────────


def fig_thermal_mesh() -> str:
    s = 160
    lx, rx, y0 = 80, 480, 92
    temp = (
        '<radialGradient id="temp" cx="0.45" cy="0.42" r="0.75">'
        f'<stop offset="0" stop-color="{ACCENT}"/><stop offset="0.3" stop-color="#d9773f"/>'
        f'<stop offset="0.55" stop-color="{SAFFRON}" stop-opacity="0.75"/>'
        f'<stop offset="0.8" stop-color="#efe6d6"/><stop offset="1" stop-color="#c9d3ea"/>'
        "</radialGradient>"
    )
    cond = "".join(
        _blob(lx + s * x, y0 + s * y, s * r, c)
        for x, y, r, c in (
            (0.3, 0.3, 0.3, "indigo"),
            (0.48, 0.48, 0.2, "indigo"),
            (0.72, 0.68, 0.3, "green"),
        )
    )
    squiggles = []
    for y in _frange(y0 + 22, y0 + s - 20, 29):
        pts = [
            (lx + s + 5 + t, y + 2.5 * math.sin(2 * math.pi * t / 12))
            for t in range(19)
        ]
        pts.append((lx + s + 5 + 30, y))
        squiggles.append(
            f'<path d="{_path(pts)}" fill="none" stroke="{SAFFRON}" stroke-width="2" marker-end="url(#ah-saffron)"/>'
        )
    parts = [
        # Conductivity field (control) with Dirichlet edge and heat input.
        f'<clipPath id="kclip"><rect x="{lx}" y="{y0}" width="{s}" height="{s}"/></clipPath>',
        f'<rect x="{lx}" y="{y0}" width="{s}" height="{s}" fill="#eef0f4"/>',
        f'<g clip-path="url(#kclip)">{cond}</g>',
        f'<rect x="{lx}" y="{y0}" width="{s}" height="{s}" fill="none" stroke="{INK2}" stroke-width="1.5"/>',
        f'<rect x="{lx - 6}" y="{y0}" width="6" height="{s}" fill="{MUTED}"/>',
        (
            f'<text transform="translate({lx - 16},{y0 + s / 2}) rotate(-90)" class="small" '
            f'text-anchor="middle">{_m("T")} = {_m("T")}{_sub("0")}</text>'
        ),
        *squiggles,
        (
            f'<text x="{lx + s / 2}" y="{y0 + s + 26}" class="label" text-anchor="middle">'
            f"∇·({_m('k')}∇{_m('T')}) + {_m('q')} = 0</text>"
        ),
        # Observed temperature field.
        f'<rect x="{rx}" y="{y0}" width="{s}" height="{s}" fill="url(#temp)"/>',
        f'<rect x="{rx}" y="{y0}" width="{s}" height="{s}" fill="none" stroke="{INK2}" stroke-width="1.5"/>',
        (
            f'<text x="{rx + s / 2}" y="{y0 + s + 26}" class="label" text-anchor="middle">'
            f"{_m('T')}{_sub('obs')} = {_m('T')}({_m('x')}, {_m('y')}; {_m('k')}*)</text>"
        ),
        # Forward solve and inversion.
        _arrow(300, 140, 462, 140, "ink", 2.2),
        '<text x="381" y="128" class="small" text-anchor="middle">steady-state solve</text>',
        _arrow(462, 206, 300, 206, "accent", 2.2),
        (
            f'<text x="381" y="226" class="small" text-anchor="middle" style="fill:{ACCENT}">'
            f"invert: find {_m('k')}({_m('x')}, {_m('y')})</text>"
        ),
        (
            f'<text x="381" y="242" class="small" text-anchor="middle" style="fill:{ACCENT}">'
            f"that reproduces {_m('T')}{_sub('obs')}</text>"
        ),
        _label(lx, 40, "CONTROL", f"conductivity {_m('k')}({_m('x')}, {_m('y')})"),
        _label(
            rx + s, 40, "OBSERVATION", "temperature field", anchor="end", quiet=True
        ),
        _objective(
            f"{_up('min')}{_sub(_m('k'))} ‖{_m('T')}({_m('k')}) − {_m('T')}{_sub('obs')}‖{_sup('2')}",
            330,
        ),
    ]
    return _svg(
        "".join(parts),
        "Heat transfer: invert for the conductivity field from an observed temperature",
        temp,
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
