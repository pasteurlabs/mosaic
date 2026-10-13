# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-box buoyancy flow and neural velocity-corrector training."""

from functools import partial

from mosaic.benchmarks.core.config import Problem, discover_solvers
from mosaic.benchmarks.core.status_checks import (
    max_error,
    max_peer_k,
    max_rel_err,
    min_cosine,
)
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.problems.shared.cost import spatial_cost
from mosaic.benchmarks.problems.shared.forward import agreement
from mosaic.benchmarks.problems.shared.plots.cost import plot_cost
from mosaic.benchmarks.problems.shared.plots.forward import plot_agreement
from mosaic.benchmarks.problems.shared.plots.gradient import plot_fd_check
from mosaic.benchmarks.problems.shared.plots.ics import plot_ic
from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles

from .gradient import fd_check
from .physics import (
    diffusion_ic,
    diffusion_reference,
    interacting_plumes,
    make_inputs,
    plume,
    scalar_field,
)
from .plots import plot_solver_in_loop
from .status import corrector_execution
from .training import solver_in_loop

_solvers = discover_solvers("buoyancy-grid")
apply_styles(_solvers)

problem = Problem(
    name="buoyancy-grid",
    category_label="Buoyancy Flow (Grid)",
    description=(
        "Two unequal warm plumes interact in a closed box. An advected scalar drives upward "
        "buoyancy, and the resulting incompressible velocity transports that scalar. "
        "We solve $u_t+(u\\cdot\\nabla)u=-\\nabla p+\\nu\\Delta u+b\\theta e_y$, "
        "$\\nabla\\cdot u=0$, and "
        "$\\theta_t+u\\cdot\\nabla\\theta=\\kappa\\Delta\\theta$. "
        "A neural velocity corrector is trained through recurrent solver steps. "
        "Held-out rollouts compare corrected and uncorrected velocity "
        "and scalar error against the same solver at finer resolution. Inspired by "
        "[Um et al. (2020)](https://arxiv.org/abs/2007.00016); this CI-runnable benchmark "
        "is not a reproduction of their full training campaign."
    ),
    bc_description=(
        "Closed square [0,1]²: no-slip velocity walls "
        "and zero normal scalar derivative. No inflow or externally randomized forcing. "
        "Velocity is face-centered; the scalar is cell-centered."
    ),
    tesseract_dir="buoyancy-grid",
    solvers=list(_solvers.values()),
    make_inputs=make_inputs,
    ic_key="state0",
    output_key="result",
    domain_extent=1.0,
    error_fn=l2_error_rel,
    reference=diffusion_reference,
)

_ic_plot = partial(
    plot_ic, ic_to_2d=scalar_field, field_cmap="inferno", field_symmetric=False
)
problem.add_ic(
    "plume",
    plume,
    description="Warm Gaussian plume at rest; seed sets its position and width.",
    plot_params={"N": 48},
    plot=_ic_plot,
)
problem.add_ic(
    "interacting_plumes",
    interacting_plumes,
    description="Two unequal warm plumes at rest, with gentle seeded position and width changes.",
    plot_params={"N": 64},
    plot=_ic_plot,
)
problem.add_ic(
    "diffusion",
    diffusion_ic,
    description="Cosine scalar mode with homogeneous Neumann walls.",
    plot_params={"N": 48},
    plot=_ic_plot,
)

problem.add_experiment(
    "forward/diffusion",
    agreement,
    ic={"name": "diffusion", "seed": 0},
    physics={
        "N": [16, 24, 32],
        "nu": 0.001,
        "diffusivity": 0.01,
        "buoyancy": 0.0,
        "dt": 0.02,
        "steps": 20,
    },
    status_check=[max_error(0.01)],
    plot=plot_agreement,
    plot_description="Scalar diffusion error against the analytic Neumann cosine solution, with buoyancy disabled.",
)
problem.add_experiment(
    "gradient/fd_check",
    fd_check,
    ic={"name": "plume", "seed": 0},
    physics={
        "N": 16,
        "nu": 0.001,
        "diffusivity": 0.001,
        "buoyancy": 1.0,
        "dt": 0.04,
        "steps": 4,
    },
    fd={"eps_values": [0.1, 0.01, 0.001], "n_dirs": 3},
    status_check=[min_cosine(0.99), max_rel_err(0.05)],
    plot=plot_fd_check,
    plot_description=(
        "Directional finite differences of the coupled velocity–scalar transition "
        "after four warmup steps, away from transport derivative kinks at zero velocity."
    ),
)
problem.add_experiment(
    "cost/forward",
    spatial_cost,
    ic={"name": "plume", "seed": 0},
    physics={
        "N": [16, 24, 32],
        "nu": 0.001,
        "diffusivity": 0.001,
        "buoyancy": 1.0,
        "dt": 0.04,
        "steps": 4,
    },
    cost={"n_trials": 3},
    status_check=[max_peer_k(20)],
    plot=plot_cost,
    plot_description="Coupled forward-step cost after compilation warmup.",
)
problem.add_experiment(
    "optimization/solver_in_loop",
    solver_in_loop,
    runs=[
        {
            "ic": {"name": "interacting_plumes", "seed": 0},
            "physics": {
                "N": 64,
                "nu": 0.0005,
                "diffusivity": 0.0005,
                "buoyancy": 1.0,
                "dt": 0.005,
                "steps": 8,
            },
            "training": {
                "updates": 256,
                "unroll": 8,
                "train_frames": 100,
                "eval_frames": 100,
                "reference_factor": 2,
                "learning_rates": [0.003],
                "hidden_channels": 8,
                "train_seed": 0,
                "validation_seed": 1,
                "test_seed": 100,
                "model_seed": 0,
                "reference_tolerance": 0.05,
                "fd_epsilon": 0.003,
                "fd_tolerance": 0.05,
            },
        }
    ],
    status_check=[corrector_execution],
    plot=plot_solver_in_loop,
    plot_description=(
        "Compare solver-in-the-loop correction with the uncorrected solver. Measure held-out rollout "
        "accuracy and training cost, with reference refinement and gradient correctness "
        "checks; superiority is measured, not required."
    ),
)
