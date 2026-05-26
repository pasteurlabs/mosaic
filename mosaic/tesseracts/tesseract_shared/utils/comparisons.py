# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic factory functions for multi-solver comparison plots.

All factories return callables compatible with ComparisonPlot.fn:
    fn(inputs: BaseModel, results: dict[str, BaseModel], axes: list[Axes]) -> None

The `get_field` parameter is a callable that extracts a numpy array from a
schema instance, decoupling the plot logic from specific field names.

Example usage in a problem definition::

    from tesseract_shared.utils.comparisons import (
        make_ensemble_deviation_plot,
        make_gradient_cosine_plot,
    )

    _compare_deviation = make_ensemble_deviation_plot(
        get_field=lambda out: np.asarray(out.result),
        title="Velocity deviation from ensemble mean",
    )
    # n_axes=lambda n: n  (one axis per solver)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel


def make_ensemble_deviation_plot(
    get_field: Callable[[BaseModel], np.ndarray],
    title: str = "|field − ensemble mean|",
) -> Callable:
    """Side-by-side deviation maps showing each solver's distance from the ensemble mean.

    Args:
        get_field: Extracts the comparison array from an OutputSchema instance.
                   Must return a real-valued ndarray; final two axes are (H, W).
        title:     Figure suptitle.

    Returns:
        fn(inputs, results, axes) -> None   [use n_axes=lambda n: n]
    """

    def _plot(inputs: BaseModel, results: dict[str, BaseModel], axes: list) -> None:
        fig = axes[0].get_figure()
        names = list(results)
        fields = [get_field(results[s]) for s in names]
        mean_field = np.mean(fields, axis=0)

        dev_maps = {
            s: np.linalg.norm((f - mean_field).reshape(*f.shape[:-1], -1), axis=-1)
            if f.ndim > 2
            else np.abs(f - mean_field)
            for s, f in zip(names, fields, strict=False)
        }
        rms = {s: float(np.sqrt(np.mean(dev_maps[s] ** 2))) for s in names}
        dev_vmax = max(m.max() for m in dev_maps.values()) or 1.0

        for ax, name in zip(axes, names, strict=False):
            data = dev_maps[name]
            if data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            im = ax.imshow(data.T, origin="lower", cmap="hot", vmin=0, vmax=dev_vmax)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"{name}\nRMS={rms[name]:.4f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(title)

    return _plot


def make_gradient_cosine_plot(
    get_field: Callable[[BaseModel], np.ndarray],
    field_label: str = "∇L",
    title: str = "Gradient comparison",
) -> Callable:
    """Per-solver gradient magnitude maps + cross-solver cosine similarity matrix.

    Args:
        get_field:   Extracts the gradient array from a cotangent InputSchema instance.
        field_label: Label used in per-solver subplot titles.
        title:       Figure suptitle.

    Returns:
        fn(inputs, grads, axes) -> None   [use n_axes=lambda n: n + 1]
    """

    def _plot(inputs: BaseModel, grads: dict[str, BaseModel], axes: list) -> None:
        fig = axes[0].get_figure()
        names = list(grads)
        n = len(names)

        raw = {s: get_field(grads[s]).ravel() for s in names}
        mags = {
            s: np.linalg.norm(
                get_field(grads[s]).reshape(
                    get_field(grads[s]).shape[0], get_field(grads[s]).shape[1], -1
                ),
                axis=-1,
            )
            for s in names
        }
        vmax = (
            np.percentile(np.concatenate([m.ravel() for m in mags.values()]), 99) or 1.0
        )

        cos_sim = np.zeros((n, n))
        for i, ni in enumerate(names):
            gi = raw[ni]
            for j, nj in enumerate(names):
                gj = raw[nj]
                denom = np.linalg.norm(gi) * np.linalg.norm(gj)
                cos_sim[i, j] = np.dot(gi, gj) / (denom + 1e-30)

        for ax, name in zip(axes[:n], names, strict=False):
            im = ax.imshow(
                mags[name].T, origin="lower", cmap="viridis", vmin=0, vmax=vmax
            )
            fig.colorbar(im, ax=ax)
            ax.set_title(f"{name}\n‖{field_label}‖", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        ax_sim = axes[n]
        im = ax_sim.imshow(cos_sim, vmin=-1, vmax=1, cmap="RdBu_r")
        fig.colorbar(im, ax=ax_sim)
        ax_sim.set_xticks(range(n))
        ax_sim.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax_sim.set_yticks(range(n))
        ax_sim.set_yticklabels(names, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax_sim.text(
                    j, i, f"{cos_sim[i, j]:.2f}", ha="center", va="center", fontsize=7
                )
        ax_sim.set_title("Cosine similarity\n(gradient direction)")

        fig.suptitle(title)

    return _plot
