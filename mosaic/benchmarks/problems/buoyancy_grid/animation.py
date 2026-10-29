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


def _panel_indices(values: np.ndarray, limits: tuple | list, group: int) -> np.ndarray:
    """Map one field independently, with no blending or dependence on another field."""
    normalized = (values - limits[0]) / max(limits[1] - limits[0], 1e-12)
    return _field_indices(normalized, group)


def write_error_gif(
    cfg: Problem, out: Path, result: dict, snapshots: dict
) -> Path | None:
    """Save separate actual/error heatmaps; paired scales stay fixed per row.

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
            baseline[..., 2],
            corrected[..., 2],
            np.linalg.norm(_center_velocity(baseline), axis=-1),
            np.linalg.norm(_center_velocity(corrected), axis=-1),
        ]
        rows.append((name, [_reduce(a) for a in errors], [_reduce(a) for a in fields]))
    if not rows:
        return None
    frames = len(rows[0][1][0])
    if any(len(a) != frames for _, errors, _ in rows for a in errors):
        raise ValueError("Combined error animation requires equal frame counts")
    row_scales = {}
    for name, errors, fields in rows:
        row_scales[name] = {
            "temperature_error_max": max(float(a.max()) for a in errors[:2]),
            "velocity_error_max": max(float(a.max()) for a in errors[2:]),
            "temperature_range": [
                min(0.0, *(float(a.min()) for a in fields[:2])),
                max(0.0, *(float(a.max()) for a in fields[:2])),
            ],
            "speed_range": [0.0, max(float(a.max()) for a in fields[2:])],
        }
    physics = result.get("params", {}).get("physics", {})
    interval = float(physics["dt"]) * int(physics["steps"])
    extent = float(physics.get("domain_extent", cfg.domain_extent))
    palette = _palette()
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    panel, left, top = 112, 125, 119
    offsets = [0, 115, 237, 352, 484, 599, 721, 836]
    row_height = panel + 62
    width, height = left + offsets[-1] + panel + 12, top + len(rows) * row_height + 38
    images = []
    for frame in range(frames):
        image = Image.new("P", (width, height), color=255)
        image.putpalette(palette)
        draw = ImageDraw.Draw(image)
        draw.text(
            (12, 8),
            f"Held-out fields and errors | time {frame * interval:.2f} | frame {frame}/{frames - 1}",
            fill=254,
            font=font,
        )
        draw.text(
            (12, 29),
            "Gray: actual fields. Red: absolute temperature error. Blue: velocity vector-error magnitude.",
            fill=253,
            font=small,
        )
        draw.text(
            (12, 46),
            "Scales differ by row; paired arms share scales fixed over time. Compare colors within each row.",
            fill=253,
            font=small,
        )
        for group, title in enumerate(("Temperature", "Speed and velocity error")):
            x = left + offsets[4 * group]
            draw.text((x, 66), title, fill=254, font=font)
            for arm, label in enumerate(("Uncorrected", "Selected correction")):
                ax = x + offsets[2 * arm]
                draw.text((ax, 84), label, fill=254, font=small)
                draw.text((ax, 101), "Field", fill=253, font=small)
                draw.text((ax + 115, 101), "Error", fill=253, font=small)
        for i, (name, errors, fields) in enumerate(rows):
            y = top + i * row_height
            draw.text((8, y + panel // 2), name, fill=254, font=font)
            scale = row_scales[name]
            for column, (values, error) in enumerate(zip(fields, errors, strict=True)):
                group = column // 2
                limits = scale["temperature_range" if group == 0 else "speed_range"]
                maximum = scale[
                    "temperature_error_max" if group == 0 else "velocity_error_max"
                ]
                for subcolumn, array, bounds, palette_group in (
                    (0, values, limits, 2),
                    (1, error, (0, maximum), group),
                ):
                    indices = _panel_indices(
                        array[frame].T[::-1], bounds, palette_group
                    )
                    tile = Image.frombytes(
                        "P", (indices.shape[1], indices.shape[0]), indices.tobytes()
                    )
                    tile.putpalette(palette)
                    image.paste(
                        tile.resize((panel, panel), Image.Resampling.NEAREST),
                        (left + offsets[2 * column + subcolumn], y),
                    )
            for group, field in enumerate(("temperature", "velocity")):
                x = left + offsets[4 * group]
                maximum = scale[f"{field}_error_max"]
                limits = scale["temperature_range" if group == 0 else "speed_range"]
                for bar, title, bounds, palette_group in (
                    (0, "Field", limits, 2),
                    (1, "Error", (0, maximum), group),
                ):
                    bx = x + bar * 237
                    draw.text(
                        (bx, y + panel + 4),
                        f"{title}: {bounds[0]:.3g} to {bounds[1]:.3g}",
                        fill=254,
                        font=small,
                    )
                    colors = _field_indices(np.linspace(0, 1, 210), palette_group)
                    for pixel, color in enumerate(colors):
                        draw.line(
                            (bx + pixel, y + panel + 23, bx + pixel, y + panel + 30),
                            fill=int(color),
                        )
        draw.text(
            (12, height - 28),
            f"Box [0,{extent:g}]^2; block means up to 32x32; own-solver fine reference; 10 fps",
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
                "columns": [
                    "uncorrected_temperature",
                    "uncorrected_temperature_error",
                    "corrected_temperature",
                    "corrected_temperature_error",
                    "uncorrected_speed",
                    "uncorrected_velocity_error",
                    "corrected_speed",
                    "corrected_velocity_error",
                ],
                "row_scales": row_scales,
                "layout": {
                    "panel_size": panel,
                    "column_offsets": offsets,
                    "left": left,
                    "top": top,
                    "row_height": row_height,
                },
                "normalization": (
                    "Per solver and physical field, shared across both arms and all frames; "
                    "not comparable across rows"
                ),
                "palettes": {
                    "temperature": "sequential red",
                    "velocity": "sequential blue",
                    "zero_error": "black",
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
