"""Registry of available problem configs, keyed by CLI name."""

from __future__ import annotations

from mosaic.benchmarks.core.config import ProblemConfig


def _registry() -> dict[str, ProblemConfig]:
    from mosaic.benchmarks.problems.navier_stokes_3d_grid import CONFIG as ns_3d_grid
    from mosaic.benchmarks.problems.navier_stokes_grid import CONFIG as ns_grid
    from mosaic.benchmarks.problems.structural_mesh import CONFIG as structural_mesh
    from mosaic.benchmarks.problems.thermal_mesh import CONFIG as thermal_mesh

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
