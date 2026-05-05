"""Registry of available problem configs, keyed by CLI name."""

from __future__ import annotations

from benchmarks.core.config import ProblemConfig


def _registry() -> dict[str, ProblemConfig]:
    from benchmarks.problems.navier_stokes_3d_grid import CONFIG as ns_3d_grid
    from benchmarks.problems.navier_stokes_grid import CONFIG as ns_grid
    from benchmarks.problems.structural_mesh import CONFIG as structural_mesh
    from benchmarks.problems.thermal_mesh import CONFIG as thermal_mesh

    return {
        "ns-grid": ns_grid,
        "ns-3d-grid": ns_3d_grid,
        "structural-mesh": structural_mesh,
        "thermal-mesh": thermal_mesh,
    }


def get_config(name: str) -> ProblemConfig:
    reg = _registry()
    if name not in reg:
        raise ValueError(f"Unknown problem {name!r}. Choose from: {list(reg)}")
    return reg[name]


PROBLEMS: list[str] = list(_registry().keys())
