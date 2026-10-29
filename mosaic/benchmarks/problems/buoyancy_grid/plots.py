# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined buoyancy rollout, validation, and physical-diagnostic plots."""

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


def plot_solver_in_loop(
    cfg: Problem,
    *,
    suffix: str = "",
    exp_key: str = "solver_in_loop",
    save: bool = True,
    **_: Any,
) -> list:
    """One error animation, one rollout comparison, and one validation plot."""
    out = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    result = load_experiment_result(out)
    if result is None:
        return []
    rows, checkpoints = [], []
    physics = result.get("params", {}).get("physics", {})
    interval = physics.get("dt", 0.005) * physics.get("steps", 8)
    for entry in result.get("results", []):
        metrics = entry.get("metrics") or {}
        if metrics.get("status") != "ok":
            continue
        name = entry["solver"]
        for method, label in [
            ("uncorrected", "Uncorrected"),
            ("solver_in_loop", "Corrected"),
        ]:
            for field, key in [("Velocity", "velocity_mse"), ("Scalar", "scalar_mse")]:
                rows.extend(
                    {
                        "solver": name,
                        "method": label,
                        "field": field,
                        "time": frame * interval,
                        "error": float(np.sqrt(value)),
                    }
                    for frame, value in enumerate(metrics[method][key])
                    if frame > 0
                )
        trained = metrics["solver_in_loop"]
        candidate = trained["candidates"][trained["selected_candidate"]]
        checkpoints.extend(
            {
                "solver": name,
                "update": point["update"],
                "loss": point["loss"],
                "selected": point["update"] == trained["selected_update"],
            }
            for point in candidate["validation_checkpoints"]
        )
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    enc = solver_encodings(frame["solver"])
    rollout = (
        alt.Chart(enc.relabel(frame))
        .mark_line()
        .encode(
            x=alt.X("time:Q", title="Physical time"),
            y=alt.Y(
                "error:Q", title="Normalized RMS error", scale=alt.Scale(zero=True)
            ),
            color=alt.Color("solver:N", scale=enc.color),
            strokeDash=alt.StrokeDash(
                "method:N",
                scale=alt.Scale(
                    domain=["Uncorrected", "Corrected"], range=[[6, 3], [1, 0]]
                ),
            ),
            tooltip=[
                "solver:N",
                "method:N",
                "field:N",
                "time:Q",
                alt.Tooltip("error:Q", format=".5g"),
            ],
        )
        .properties(width=380, height=240)
        .facet(column=alt.Column("field:N", title=None))
        .resolve_scale(y="independent")
        .properties(title="Held-out rollout: corrected and uncorrected errors")
    )
    base = alt.Chart(enc.relabel(pd.DataFrame(checkpoints))).encode(
        x=alt.X("update:Q", title="Completed optimizer updates"),
        y=alt.Y(
            "loss:Q", title="Full-rollout validation loss", scale=alt.Scale(zero=True)
        ),
        color=alt.Color("solver:N", scale=enc.color),
        tooltip=[
            "solver:N",
            "update:Q",
            alt.Tooltip("loss:Q", format=".5g"),
            "selected:N",
        ],
    )
    validation = (
        base.mark_line(point=True)
        + base.transform_filter(alt.datum.selected).mark_point(
            shape="diamond", filled=True, size=120
        )
    ).properties(
        width=800,
        height=240,
        title="Validation loss: selected checkpoints marked by diamonds",
    )
    charts = {"rollout_errors": rollout, "validation_loss": validation}
    if save:
        # Remove only retired generated plot artifacts from earlier renders.
        for pattern in (
            "fields_*",
            "rollout_error_*",
            "validation_checkpoints_*",
            "training_cost_*",
        ):
            for path in out.glob(pattern):
                if path.name.endswith((".png", ".vl.json")):
                    path.unlink()
        for stem, chart in charts.items():
            save_chart(chart, stem, out)
        from .animation import write_error_gif

        write_error_gif(cfg, out, result, try_load_npz(out / "corrector_fields.npz"))
    return list(charts.values())


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
            y=alt.Y(
                "value:Q",
                title=None,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=".3~g"),
            ),
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
