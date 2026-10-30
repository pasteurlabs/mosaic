# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closure of chained calls against one call, on the Kolmogorov coarse configuration.

From each of the ground truth's test states, a chain of 40 one-interval calls
is compared with one call over 2, 16 and 40 intervals (relative L2 distance).
With ``--state``, the chain hands each call the native state the previous one
returned; without it, every call restarts from the canonical velocity.

Usage: python closure.py TAG SOLVER... [--state]

SOLVER is the image prefix (``jax_cfd``, ``ins``, ``xlb``, ``pict``,
``exponax``, ``warp_ns``, ``phiflow``); the image is
``{SOLVER}_navier_stokes_grid:{TAG}``. Prints one JSON line per solver.
"""

import json
import sys
from typing import Any

import numpy as np
from tesseract_core import Tesseract

from mosaic.benchmarks.problems.navier_stokes_grid import kolmogorov as ko

N = ko.COARSE_N
DT = ko.COARSE_DT
STEPS = round(ko.INTERVAL / DT)
CHAIN = 40
COMPARED = (2, 16, 40)
# Inputs a solver needs beyond the canonical ones.
EXTRA = {"jax_cfd": {"density": np.array([1.0], np.float32), "inner_steps": 1}}


def inputs(solver: str, v: np.ndarray, steps: int, state: Any, use_state: bool) -> dict:
    """Inputs of one call of ``solver`` from ``v``, optionally with native state."""
    d = {
        "v0": v.astype(np.float32),
        "viscosity": np.array([ko.VISCOSITY], np.float32),
        "dt": np.array([DT], np.float32),
        "steps": steps,
        "domain_extent": float(ko.DOMAIN_EXTENT),
        "forcing": ko.forcing_field(N),
        "linear_drag": np.array([ko.DRAG], np.float32),
        **EXTRA.get(solver, {}),
    }
    if use_state:
        d["return_state"] = True
        if state is not None:
            d["state"] = state
    return d


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 distance of ``a`` from ``b``."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def closure(solver: str, tag: str, use_state: bool) -> dict:
    """Median and max distance between the chain and one call, per length."""
    truth = ko.load_ground_truth()
    starts = [ko.initial_state(truth, s, N) for s in ko.GROUND_TRUTH_SEEDS["test"]]
    errors = {k: [] for k in COMPARED}
    with Tesseract.from_image(f"{solver}_navier_stokes_grid:{tag}", gpus=["all"]) as t:
        for v0 in starts:
            v, state, chain = v0, None, []
            for _ in range(CHAIN):
                out = t.apply(inputs(solver, v, STEPS, state, use_state))
                v, state = np.asarray(out["result"]), out.get("state")
                chain.append(v)
            for k in COMPARED:
                # Same flags as the chain, so a solver that switches its
                # integrator when a state is involved uses one integrator.
                one = t.apply(inputs(solver, v0, k * STEPS, None, use_state))
                errors[k].append(rel(chain[k - 1], np.asarray(one["result"])))
    return {
        f"intervals_{k}": {"median": float(np.median(e)), "max": float(np.max(e))}
        for k, e in errors.items()
    }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_state = "--state" in sys.argv
    tag, solvers = args[0], args[1:]
    for solver in solvers:
        result = closure(solver, tag, use_state)
        print(json.dumps({"solver": solver, "state": use_state, **result}), flush=True)
