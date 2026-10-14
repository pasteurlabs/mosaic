# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Held-out fields, rollout errors, checkpoint selection and measured cost."""

from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_experiment_result, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.charts.common import (
    save_chart,
    solver_encodings,
)

_METHODS = {
    "uncorrected": ("Uncorrected", "#777777"),
    "solver_in_loop": ("Solver in the loop", "#177c71"),
}


def _speed(state: np.ndarray) -> np.ndarray:
    ux = np.pad(state[:-1, :, 0], ((1, 1), (0, 0)))
    uy = np.pad(state[:, :-1, 1], ((0, 0), (1, 1)))
    return np.hypot((ux[:-1] + ux[1:]) / 2, (uy[:, :-1] + uy[:, 1:]) / 2)


def _fields(states: dict, labels: dict, extent: float, title: str) -> Any:
    final = {key: value[-1] for key, value in states.items()}
    panels = []
    for field in ("Buoyancy scalar", "Speed", "Scalar error"):
        values = {
            key: (
                _speed(state)
                if field == "Speed"
                else state[..., 2] - final["reference"][..., 2]
                if field == "Scalar error"
                else state[..., 2]
            )
            for key, state in final.items()
        }
        low = min(0.0, *(float(value.min()) for value in values.values()))
        high = max(float(value.max()) for value in values.values())
        if field == "Scalar error":
            high = max(abs(low), abs(high), 1e-8)
            low = -high
        elif high == low:
            high = low + 1e-8
        rows = []
        for key, value in values.items():
            n = value.shape[0]
            i, j = np.indices(value.shape)
            rows.append(
                pd.DataFrame(
                    {
                        "method": labels[key],
                        "x": i.ravel() * extent / n,
                        "x2": (i.ravel() + 1) * extent / n,
                        "y": j.ravel() * extent / n,
                        "y2": (j.ravel() + 1) * extent / n,
                        "value": value.ravel(),
                    }
                )
            )
        panel = (
            alt.Chart(pd.concat(rows, ignore_index=True))
            .mark_rect()
            .encode(
                x=alt.X("x:Q", title="x", scale=alt.Scale(domain=[0, extent])),
                x2="x2:Q",
                y=alt.Y("y:Q", title="y", scale=alt.Scale(domain=[0, extent])),
                y2="y2:Q",
                color=alt.Color(
                    "value:Q",
                    title=field,
                    scale=alt.Scale(
                        domain=[low, high],
                        scheme="redblue" if field == "Scalar error" else "inferno",
                        reverse=field == "Scalar error",
                    ),
                ),
                tooltip=[
                    "method:N",
                    alt.Tooltip("value:Q", format=".5g"),
                    "x:Q",
                    "y:Q",
                ],
            )
            .properties(width=180, height=180)
            .facet(
                column=alt.Column("method:N", sort=list(labels.values()), title=None)
            )
        )
        panels.append(panel)
    return (
        alt.vconcat(*panels).resolve_scale(color="independent").properties(title=title)
    )


def _rollout(metrics: dict, times: np.ndarray, selected_label: str, name: str) -> Any:
    rows = []
    labels = ["Uncorrected", selected_label]
    for method in _METHODS:
        label = selected_label if method == "solver_in_loop" else "Uncorrected"
        for field, key in (
            ("Velocity", "velocity_mse"),
            ("Buoyancy scalar", "scalar_mse"),
        ):
            rows.extend(
                {
                    "time": float(time),
                    "error": float(np.sqrt(value)),
                    "method": label,
                    "field": field,
                }
                for time, value in zip(times[1:], metrics[method][key][1:], strict=True)
            )
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_line()
        .encode(
            x=alt.X("time:Q", title="Physical time"),
            y=alt.Y(
                "error:Q",
                title="Normalized RMS error",
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(domain=labels, range=[v[1] for v in _METHODS.values()]),
                title=None,
            ),
            tooltip=[
                "method:N",
                "field:N",
                "time:Q",
                alt.Tooltip("error:Q", format=".5g"),
            ],
        )
        .properties(width=300, height=220)
        .facet(column=alt.Column("field:N", title=None))
        .properties(title=f"{name}: held-out rollout")
    )


def _validation(trained: dict, name: str) -> Any:
    rows = [
        {
            "update": point["update"],
            "loss": point["loss"],
            "lr": f"{candidate['learning_rate']:g}",
        }
        for candidate in trained["candidates"]
        for point in candidate["validation_checkpoints"]
    ]
    selected = trained["candidates"][trained["selected_candidate"]]
    base = (
        alt.Chart(pd.DataFrame(rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("update:Q", title="Completed optimizer updates"),
            y=alt.Y("loss:Q", title="Full-rollout validation loss"),
            color=alt.Color("lr:N", title="Learning rate"),
            tooltip=["update:Q", "loss:Q", "lr:N"],
        )
    )
    marker = (
        alt.Chart(
            pd.DataFrame(
                [
                    {
                        "update": trained["selected_update"],
                        "loss": selected["validation_loss"],
                    }
                ]
            )
        )
        .mark_point(shape="diamond", filled=True, color="black", size=130)
        .encode(x="update:Q", y="loss:Q", tooltip=["update:Q", "loss:Q"])
    )
    return (base + marker).properties(
        width=620,
        height=220,
        title=f"{name}: selected checkpoint (diamond); update 0 is no correction",
    )


def _cost(trained: dict, name: str, updates: int) -> Any:
    training = trained["total_training_seconds"]
    rows = [
        {"phase": "Training", "start": 0, "end": training, "seconds": training},
        {
            "phase": "Validation checkpoints",
            "start": training,
            "end": training + trained["total_validation_seconds"],
            "seconds": trained["total_validation_seconds"],
        },
    ]
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar()
        .encode(
            x=alt.X(
                "start:Q",
                title="Measured training and validation time across candidates (s)",
            ),
            x2="end:Q",
            color=alt.Color(
                "phase:N",
                scale=alt.Scale(
                    domain=["Training", "Validation checkpoints"],
                    range=["#177c71", "#aaaaaa"],
                ),
                title=None,
            ),
            tooltip=["phase:N", alt.Tooltip("seconds:Q", format=".3f")],
        )
        .properties(
            width=620,
            height=70,
            title=f"{name}: {updates} completed updates; selected update {trained['selected_update']}",
        )
    )


def plot_solver_in_loop(
    cfg: Problem,
    *,
    suffix: str = "",
    exp_key: str = "solver_in_loop",
    save: bool = True,
    **_: Any,
) -> list:
    """Write paired PNG/Vega-Lite artifacts for the standard interactive docs."""
    out = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    result = load_experiment_result(out)
    if result is None:
        return []
    snapshots = try_load_npz(out / "corrector_fields.npz")
    names = [str(name) for name in snapshots.get("solver_names", [])]
    charts = []
    for record in result.get("results", []):
        name, metrics = record["solver"], record.get("metrics", {})
        if metrics.get("status") != "ok" or name not in names:
            continue
        index = names.index(name)
        keys = ["reference", *_METHODS]
        if any(f"{key}_{index}" not in snapshots for key in keys):
            continue
        states = {key: snapshots[f"{key}_{index}"] for key in keys}
        physics = result.get("params", {}).get("physics", {})
        times = (
            np.arange(len(states["reference"]))
            * physics.get("dt", 0.04)
            * physics.get("steps", 2)
        )
        trained = metrics["solver_in_loop"]
        label = (
            "Selected: no correction"
            if trained["selected_update"] == 0
            else f"Solver in the loop (update {trained['selected_update']})"
        )
        built = {
            f"fields_{name}": _fields(
                states,
                {
                    "reference": "Restricted reference",
                    "uncorrected": "Uncorrected",
                    "solver_in_loop": label,
                },
                physics.get("domain_extent", 1.0),
                f"{name}: held-out plume at t={times[-1]:.2f}; shared scales per row",
            ),
            f"rollout_error_{name}": _rollout(metrics, times, label, name),
            f"validation_checkpoints_{name}": _validation(trained, name),
            f"training_cost_{name}": _cost(trained, name, metrics["updates"]),
        }
        for stem, chart in built.items():
            if save:
                save_chart(chart, stem, out)
            charts.append(chart)
    return charts


def plot_forward_metrics(
    cfg: Problem,
    *,
    suffix: str = "",
    exp_key: str = "physical_laws",
    save: bool = True,
    **_: Any,
) -> list:
    """Plot coupled-flow diagnostics without treating the scalar as velocity."""
    out = results_dir() / cfg.name / "forward" / f"{exp_key}{suffix}"
    result = load_experiment_result(out)
    if result is None:
        return []
    labels = {
        "velocity_rel_error": "Velocity relative L2 difference",
        "scalar_rel_error": "Scalar relative L2 difference",
        "divergence_rms": "MAC divergence RMS",
        "scalar_mass_rel_drift": "Absolute relative scalar mass drift",
        "scalar_min": "Scalar minimum",
        "scalar_max": "Scalar maximum",
        "kinetic_energy": "Kinetic energy per unit area",
        "wall_normal_padding_max": "Wall-normal storage padding magnitude",
    }
    rows = []
    for entry in result.get("results", []):
        metrics = entry.get("metrics") or {}
        for name, value in metrics.items():
            if name not in labels or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                rows.append(
                    {
                        "solver": entry["solver"],
                        "sweep": float(entry["sweep_value"]),
                        "diagnostic": labels[name],
                        "value": value,
                    }
                )
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    enc = solver_encodings(frame["solver"])
    chart = (
        alt.Chart(enc.relabel(frame))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "sweep:Q", title=(result.get("sweep") or {}).get("key", "Sweep value")
            ),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
            color=alt.Color("solver:N", scale=enc.color),
            strokeDash=alt.StrokeDash("solver:N", scale=enc.dash),
            tooltip=[
                "solver:N",
                "diagnostic:N",
                "sweep:Q",
                alt.Tooltip("value:Q", format=".5g"),
            ],
        )
        .properties(width=280, height=180)
        .facet(facet=alt.Facet("diagnostic:N", title=None), columns=3)
        .resolve_scale(y="independent")
        .properties(title=f"{cfg.name}: {exp_key.replace('_', ' ')}")
    )
    if save:
        save_chart(chart, "diagnostics", out)
    return [chart]
