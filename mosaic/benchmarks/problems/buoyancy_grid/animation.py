# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-scale corrected fields and errors for the held-out buoyancy rollout."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mosaic.benchmarks.core.config import Problem


def _center_velocity(states: np.ndarray) -> np.ndarray:
    ux = np.pad(states[..., :-1, :, 0], ((0, 0), (1, 1), (0, 0)))
    uy = np.pad(states[..., :, :-1, 1], ((0, 0), (0, 0), (1, 1)))
    return np.stack(
        ((ux[:, :-1] + ux[:, 1:]) / 2, (uy[:, :, :-1] + uy[:, :, 1:]) / 2), axis=-1
    )


def _reduce(fields: np.ndarray, maximum: int = 32) -> np.ndarray:
    n = fields.shape[-1]
    if n <= maximum:
        return fields
    # Block means retain the local mean error magnitude, without signed-error
    # cancellation. Non-divisible sizes retain their original resolution.
    factor = (n + maximum - 1) // maximum
    if n % factor:
        return fields
    return fields.reshape(len(fields), n // factor, factor, n // factor, factor).mean(
        axis=(2, 4)
    )


_COLORS_PER_FIELD = 96
_GRAY_COLORS = 48


def _field_indices(normalized: np.ndarray, group: int) -> np.ndarray:
    """Map a fixed [0,1] scale into its field's disjoint palette block."""
    count = _GRAY_COLORS if group == 2 else _COLORS_PER_FIELD
    return (
        np.rint(np.clip(normalized, 0, 1) * (count - 1)) + group * _COLORS_PER_FIELD
    ).astype(np.uint8)


def _palette() -> list[int]:
    # Both zero-error colors are black; positive errors use separate warm/cool
    # sequential ramps. Indices 240..255 remain reserved for labels/background.
    anchors = (
        [[0, 0, 0], [55, 0, 8], [145, 15, 35], [225, 60, 55], [255, 180, 150]],
        [[0, 0, 0], [0, 18, 65], [0, 75, 155], [20, 155, 220], [165, 225, 255]],
    )
    palette = np.zeros((256, 3), dtype=np.uint8)
    for group, ramp in enumerate(anchors):
        colors = np.stack(
            [
                np.interp(
                    np.linspace(0, 4, _COLORS_PER_FIELD),
                    np.arange(5),
                    np.asarray(ramp)[:, c],
                )
                for c in range(3)
            ],
            axis=1,
        )
        start = group * _COLORS_PER_FIELD
        palette[start : start + _COLORS_PER_FIELD] = colors.astype(np.uint8)
    palette[192:240] = np.linspace(0, 255, _GRAY_COLORS).astype(np.uint8)[:, None]
    palette[253] = [100, 100, 100]
    palette[255] = [255, 255, 255]
    return palette.ravel().tolist()


def write_error_gif(
    cfg: Problem, out: Path, result: dict, snapshots: dict
) -> Path | None:
    """Save all available solvers together, with one fixed scale per physical field.

    Scalar error is absolute scalar difference. Velocity error is the norm of
    the difference of cell-centered velocity vectors, not speed differences.
    Input frame zero is retained; each subsequent frame is one solver interval.
    """
    names = [str(name) for name in snapshots.get("solver_names", [])]
    metrics = {r["solver"]: r.get("metrics", {}) for r in result.get("results", [])}
    rows = []
    for solver in cfg.solvers:
        name = solver.name
        if name not in names or metrics.get(name, {}).get("status") != "ok":
            continue
        index = names.index(name)
        keys = [
            f"{arm}_{index}" for arm in ("reference", "uncorrected", "solver_in_loop")
        ]
        if not all(key in snapshots for key in keys):
            continue
        reference, baseline, corrected = [np.asarray(snapshots[key]) for key in keys]
        if (
            reference.ndim != 4
            or reference.shape[-1] != 3
            or reference.shape[1] != reference.shape[2]
        ):
            raise ValueError("Expected rollout shape (frames,N,N,3)")
        if any(
            a.shape != reference.shape or not np.isfinite(a).all()
            for a in (reference, baseline, corrected)
        ):
            raise ValueError(f"Invalid saved rollout for {name}")
        ref_velocity = _center_velocity(reference)
        errors = [
            np.abs(baseline[..., 2] - reference[..., 2]),
            np.abs(corrected[..., 2] - reference[..., 2]),
            np.linalg.norm(_center_velocity(baseline) - ref_velocity, axis=-1),
            np.linalg.norm(_center_velocity(corrected) - ref_velocity, axis=-1),
        ]
        fields = [
            corrected[..., 2],
            np.linalg.norm(_center_velocity(corrected), axis=-1),
        ]
        rows.append((name, [_reduce(a) for a in errors], [_reduce(a) for a in fields]))
    if not rows:
        return None
    frames = len(rows[0][1][0])
    if any(len(a) != frames for _, errors, _ in rows for a in errors):
        raise ValueError("Combined error animation requires equal frame counts")
    maxima = [
        max(float(a.max()) for _, errors, _ in rows for a in errors[start : start + 2])
        for start in (0, 2)
    ]
    scales = [max(v, 1e-12) for v in maxima]
    field_ranges = [
        (
            min(float(a[0].min()) for _, _, a in rows),
            max(float(a[0].max()) for _, _, a in rows),
        ),
        (0.0, max(float(a[1].max()) for _, _, a in rows)),
    ]
    physics = result.get("params", {}).get("physics", {})
    interval = float(physics["dt"]) * int(physics["steps"])
    extent = float(physics.get("domain_extent", cfg.domain_extent))
    palette = _palette()
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    panel, gap, left, top = 144, 12, 125, 116
    width, height = left + 6 * (panel + gap), top + len(rows) * (panel + 28) + 38
    images = []
    for frame in range(frames):
        image = Image.new("P", (width, height), color=255)
        image.putpalette(palette)
        draw = ImageDraw.Draw(image)
        draw.text(
            (12, 8),
            f"Held-out corrected fields and errors | time {frame * interval:.2f} | frame {frame}/{frames - 1}",
            fill=254,
            font=font,
        )
        draw.text(
            (12, 29),
            "Own-solver restricted fine reference; fixed scales over all solvers and times",
            fill=253,
            font=small,
        )
        for column, title, group, limits, span in (
            (0, "Corrected temperature", 2, field_ranges[0], 1),
            (1, "Temperature absolute error", 0, (0, maxima[0]), 2),
            (3, "Corrected speed", 2, field_ranges[1], 1),
            (4, "Velocity vector error", 1, (0, maxima[1]), 2),
        ):
            x = left + column * (panel + gap)
            bar_width = span * panel + (span - 1) * gap
            draw.text((x, 50), title, fill=254, font=small)
            colors = _field_indices(np.linspace(0, 1, bar_width), group)
            for pixel, color in enumerate(colors):
                draw.line((x + pixel, 72, x + pixel, 80), fill=int(color))
            draw.text((x, 82), f"{limits[0]:.3g}", fill=254, font=small)
            draw.text(
                (x + bar_width - 55, 82), f"{limits[1]:.3g}", fill=254, font=small
            )
        for i, (name, errors, fields) in enumerate(rows):
            y = top + i * (panel + 28)
            draw.text((8, y + panel // 2), name, fill=254, font=font)
            panels = (
                (fields[0], 2, field_ranges[0], "Selected correction"),
                (errors[0], 0, (0, scales[0]), "Uncorrected error"),
                (errors[1], 0, (0, scales[0]), "Corrected error"),
                (fields[1], 2, field_ranges[1], "Selected correction"),
                (errors[2], 1, (0, scales[1]), "Uncorrected error"),
                (errors[3], 1, (0, scales[1]), "Corrected error"),
            )
            for column, (values, group, limits, label) in enumerate(panels):
                x = left + column * (panel + gap)
                normalized = (values[frame].T[::-1] - limits[0]) / max(
                    limits[1] - limits[0], 1e-12
                )
                indices = _field_indices(normalized, group)
                tile = Image.frombytes(
                    "P", (indices.shape[1], indices.shape[0]), indices.tobytes()
                )
                tile.putpalette(palette)
                image.paste(
                    tile.resize((panel, panel), Image.Resampling.NEAREST), (x, y)
                )
                draw.text((x, y + panel + 3), label, fill=254, font=small)
        draw.text(
            (12, height - 28),
            f"Box [0,{extent:g}]^2; block-mean fields/errors at up to 32x32 when divisible; 10 fps",
            fill=253,
            font=small,
        )
        images.append(image)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "error_evolution.gif"
    images[0].save(
        target,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        optimize=False,
        disposal=2,
    )
    images[-1].convert("RGB").save(target.with_suffix(".png"))
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                "solvers": [name for name, _, _ in rows],
                "frames": frames,
                "frame_interval": interval,
                "fps": 10,
                "scalar_max": maxima[0],
                "velocity_max": maxima[1],
                "columns": [
                    "corrected_temperature",
                    "uncorrected_temperature_error",
                    "corrected_temperature_error",
                    "corrected_speed",
                    "uncorrected_velocity_error",
                    "corrected_velocity_error",
                ],
                "actual_field_ranges": {
                    "temperature": list(field_ranges[0]),
                    "speed": list(field_ranges[1]),
                },
                "normalization": "None; physical-state error units",
                "palettes": {
                    "temperature": "sequential red",
                    "velocity": "sequential blue",
                    "zero": "black",
                    "actual_fields": "sequential grayscale",
                },
                "spatial_reduction": "Block mean of error magnitude, after velocity centering and differencing",
                "selected_updates": {
                    name: metrics[name]["solver_in_loop"]["selected_update"]
                    for name, _, _ in rows
                },
            },
            indent=2,
        )
        + "\n"
    )
    return target
