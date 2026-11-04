#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the forced Kolmogorov flow loop on the paper page (docs/figures/kolmogorov.mp4).

Runs Exponax, one of the benchmarked solvers, with the physical parameters of
the Kolmogorov setup (mosaic/benchmarks/problems/navier_stokes_grid/kolmogorov.py)
through its public API, and renders the vorticity in the docs palette. It is an
illustration of the setup, not a benchmark result.

Needs Exponax and ffmpeg, which the docs environment does not include:

    uv run --no-project --with exponax==0.2.0 --with jax==0.10.0 \
        --with matplotlib python docs/scripts/make_kolmogorov_animation.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import exponax as ex
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parents[1] / "figures"

# The Kolmogorov setup: f = (sin 4y, 0) on [0, 2π]², ν = 2e-3, drag 0.1.
DOMAIN_EXTENT = 2 * jnp.pi
VISCOSITY = 2e-3
DRAG = 0.1
WAVENUMBER = 4
N = 256
DT = 0.001  # stable through the spin-up at this resolution

SPIN_UP = 40.0  # time to reach the statistically steady state
DURATION = 8.0  # time shown in the loop
FRAME_EVERY = 0.05
FPS = 24

INDIGO, PAPER, ACCENT = "#2f4b8f", "#f7f7f6", "#a8401f"


def simulate() -> jax.Array:
    stepper = ex.stepper.KolmogorovFlowVorticity(
        2,
        DOMAIN_EXTENT,
        N,
        DT,
        diffusivity=VISCOSITY,
        drag=-DRAG,
        injection_mode=WAVENUMBER,
        injection_scale=1.0,
    )
    ic = ex.ic.RandomTruncatedFourierSeries(2, cutoff=5, max_one=True)
    q = ic(N, key=jax.random.PRNGKey(0))

    steps_per_frame = round(FRAME_EVERY / DT)
    advance = jax.jit(ex.repeat(stepper, steps_per_frame))
    for _ in range(round(SPIN_UP / FRAME_EVERY)):
        q = advance(q)
    frames = []
    for _ in range(round(DURATION / FRAME_EVERY)):
        q = advance(q)
        frames.append(q[0])
    return jnp.stack(frames)


def render(frames: jax.Array) -> None:
    cmap = LinearSegmentedColormap.from_list("mosaic", [INDIGO, PAPER, ACCENT])
    lim = float(jnp.percentile(jnp.abs(frames), 99.5))
    with tempfile.TemporaryDirectory() as tmp:
        for i, q in enumerate(frames):
            fig = plt.figure(figsize=(4.8, 4.8), dpi=100)
            ax = fig.add_axes((0, 0, 1, 1))
            # Exponax's q is minus the vorticity; plot the vorticity.
            ax.imshow(-q.T, origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
            ax.set_axis_off()
            fig.savefig(f"{tmp}/{i:04d}.png")
            if i == 0:
                fig.savefig(OUT / "kolmogorov.jpg", pil_kwargs={"quality": 85})
            plt.close(fig)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(FPS), "-i", f"{tmp}/%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                "-movflags", "+faststart",
                str(OUT / "kolmogorov.mp4"),
            ],
            check=True,
        )  # fmt: skip


def main() -> None:
    render(simulate())
    print(f"Wrote {OUT / 'kolmogorov.mp4'} and {OUT / 'kolmogorov.jpg'}")


if __name__ == "__main__":
    main()
