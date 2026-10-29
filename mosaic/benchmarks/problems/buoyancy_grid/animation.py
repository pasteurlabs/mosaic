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


def _field_indices(normalized: np.ndarray, group: int) -> np.ndarray:
    """Map a fixed [0,1] scale into its field's disjoint palette block."""
    count = _COLORS_PER_FIELD
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
    for name, errors, _ in rows:
        row_scales[name] = {
            "temperature_error_max": max(float(a.max()) for a in errors[:2]),
            "velocity_error_max": max(float(a.max()) for a in errors[2:]),
        }
    orientation = next((row for row in rows if row[0] == "JAX-CFD"), rows[0])
    orientation_fields = [orientation[2][1], orientation[2][3]]
    orientation_ranges = [
        [
            min(0.0, float(orientation_fields[0].min())),
            max(0.0, float(orientation_fields[0].max())),
        ],
        [0.0, float(orientation_fields[1].max())],
    ]
    physics = result.get("params", {}).get("physics", {})
    interval = float(physics["dt"]) * int(physics["steps"])
    extent = float(physics.get("domain_extent", cfg.domain_extent))
    palette = _palette()
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    panel, left, top = 144, 125, 316
    offsets = [0, 150, 312, 462]
    row_height = panel + 48
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
            f"Orientation: selected {orientation[0]} rollout (actual fields, not errors).",
            fill=253,
            font=small,
        )
        draw.text(
            (12, 46),
            "Errors below use each solver's own fine reference; compare colors within each row.",
            fill=253,
            font=small,
        )
        for group, title in enumerate(("Actual temperature", "Actual speed")):
            x = left + offsets[group * 2] + 75
            draw.text((x, 67), title, fill=254, font=small)
            indices = _panel_indices(
                orientation_fields[group][frame].T[::-1],
                orientation_ranges[group],
                group,
            )
            tile = Image.frombytes(
                "P", (indices.shape[1], indices.shape[0]), indices.tobytes()
            )
            tile.putpalette(palette)
            image.paste(tile.resize((panel, panel), Image.Resampling.NEAREST), (x, 85))
            bounds = orientation_ranges[group]
            draw.text(
                (x, 233),
                f"Field: {bounds[0]:.3g} to {bounds[1]:.3g}",
                fill=254,
                font=small,
            )
            for pixel, color in enumerate(
                _field_indices(np.linspace(0, 1, panel), group)
            ):
                draw.line((x + pixel, 251, x + pixel, 257), fill=int(color))
        draw.text(
            (12, 269),
            "Error scales: paired arms share each row's maximum, fixed over all frames.",
            fill=253,
            font=small,
        )
        for column, title in enumerate(
            (
                "Uncorrected temp error",
                "Corrected temp error",
                "Uncorrected vel error",
                "Corrected vel error",
            )
        ):
            draw.text((left + offsets[column], 294), title, fill=254, font=small)
        for i, (name, errors, _) in enumerate(rows):
            y = top + i * row_height
            draw.text((8, y + panel // 2), name, fill=254, font=font)
            scale = row_scales[name]
            for column, error in enumerate(errors):
                group = column // 2
                maximum = scale[
                    "temperature_error_max" if group == 0 else "velocity_error_max"
                ]
                indices = _panel_indices(error[frame].T[::-1], (0, maximum), group)
                tile = Image.frombytes(
                    "P", (indices.shape[1], indices.shape[0]), indices.tobytes()
                )
                tile.putpalette(palette)
                image.paste(
                    tile.resize((panel, panel), Image.Resampling.NEAREST),
                    (left + offsets[column], y),
                )
            for group, field in enumerate(("temperature", "velocity")):
                x = left + offsets[2 * group]
                maximum = scale[f"{field}_error_max"]
                draw.text(
                    (x, y + panel + 4),
                    f"Absolute error: 0 to {maximum:.3g}",
                    fill=254,
                    font=small,
                )
                for pixel, color in enumerate(
                    _field_indices(np.linspace(0, 1, 294), group)
                ):
                    draw.line(
                        (x + pixel, y + panel + 23, x + pixel, y + panel + 29),
                        fill=int(color),
                    )
        draw.text(
            (12, height - 28),
            f"Box [0,{extent:g}]^2; block means up to 32x32; 10 fps",
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
                    "uncorrected_temperature_error",
                    "corrected_temperature_error",
                    "uncorrected_velocity_error",
                    "corrected_velocity_error",
                ],
                "row_scales": row_scales,
                "orientation": {
                    "solver": orientation[0],
                    "method": "selected solver_in_loop correction",
                    "temperature_range": orientation_ranges[0],
                    "speed_range": orientation_ranges[1],
                    "selection": "JAX-CFD when available, otherwise first available solver",
                },
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
                    "actual_fields": "temperature red, speed blue; independent physical scales",
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
