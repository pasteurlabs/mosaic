# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run DAFoam for tesseract_api.py, under DAFoam's own Python.

    python dafoam_driver.py <case_dir> <request.json> <result.npz>

Saves the force on the cylinder per unit depth, ``body_force`` (steps, 2).
With a ``window`` in the request, it also saves the gradient of the x force
averaged over the last ``window`` steps with respect to the inlet face
velocities.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import openmdao.api as om
from dafoam.mphys.mphys_dafoam import DAFoamBuilderUnsteady
from mpi4py import MPI  # noqa: F401  (initialises MPI before DAFoam)

case, request_path, result_path = sys.argv[1:4]
request = json.loads(Path(request_path).read_text())
inlet_u = np.asarray(request["inlet_u"])
depth = request["depth"]
window = request.get("window")
os.chdir(case)


def force(direction: list[float], time_op: dict) -> dict:
    """A DAFoam force function on the cylinder, per unit depth."""
    return {
        "type": "force",
        "source": "patchToFace",
        "patches": ["cylinder"],
        "directionMode": "fixedDirection",
        "direction": direction,
        "scale": 1.0 / depth,
        **time_op,
    }


# DAFoam averages over the trailing max(2, round(nStepsFrac * steps)) steps,
# so a one-step window is the final value instead.
steps = request["steps"]
if window is None or window >= 2:
    fx_time_op = {"timeOp": "average", "nStepsFrac": (window or steps) / steps}
else:
    fx_time_op = {"timeOp": "final"}

options = {
    "designSurfaces": ["cylinder"],
    "solverName": "DAPimpleFoam",
    "useAD": {"mode": "reverse"},
    "primalBC": {"useWallFunction": False},
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 20,
        "PCMatUpdateInterval": 1,
        "readZeroFields": True,
        "additionalOutput": ["U", "p", "phi"],
    },
    "function": {
        "fx": force([1.0, 0.0, 0.0], fx_time_op),
        "fy": force([0.0, 1.0, 0.0], {"timeOp": "final"}),
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-10,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "gmresMaxIters": 2000,
        "gmresRestart": 2000,
    },
    "normalizeStates": {
        "U": request["u_ref"],
        "p": request["u_ref"] ** 2 / 2.0,
        "phi": 1.0,
    },
    "inputInfo": {
        "inlet_u": {
            "type": "patchField",
            "patches": ["inlet"],
            "fieldName": "U",
            "fieldType": "vector",
            "indices": [0],
            "distributed": False,
            "components": ["solver", "function"],
        },
    },
    "unsteadyCompOutput": {"fx": ["fx"], "fy": ["fy"]},
}


class Model(om.Group):
    """The inlet velocities feeding DAFoam's unsteady solver."""

    def setup(self) -> None:
        """Add the inputs and the solver."""
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        self.add_subsystem(
            "cfd", DAFoamBuilderUnsteady(solver_options=options), promotes=["*"]
        )

    def configure(self) -> None:
        """Set the inlet velocities and the x force as the objective."""
        self.dvs.add_output("inlet_u", val=inlet_u)
        self.add_design_var("inlet_u")
        self.add_objective("fx")


prob = om.Problem(model=Model(), reports=False)
prob.setup(mode="rev")
prob.run_model()

# DAFoam keeps only the time-operated function values, so the force history is
# re-evaluated from the time steps the primal wrote.
solver = prob.model.cfd.solver.DASolver
dt = solver.solver.getDeltaT()
body_force = np.zeros((steps, 2))
for k in range(1, steps + 1):
    solver.solver.setTime(k * dt, k)
    solver.readStateVars(k * dt, dt)
    body_force[k - 1] = (
        solver.solver.calcFunction("fx"),
        solver.solver.calcFunction("fy"),
    )
result = {"body_force": body_force}

if window is not None:
    totals = prob.compute_totals(of=["fx"], wrt=["inlet_u"])
    result["d_fx_d_inlet_u"] = totals[("fx", "inlet_u")].ravel()

np.savez(result_path, **result)
