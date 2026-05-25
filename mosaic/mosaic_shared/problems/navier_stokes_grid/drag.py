"""Canonical x-direction drag for ns-grid solvers.

Surface-integral form on a uniform Cartesian grid:

    drag_x = Σ_face_cells [ ±p · dx ]  −  Σ_face_cells [ ±ν · u_x ]

where the sums run over fluid cells whose +x or −x neighbour is solid (the
faces immediately bordering the obstacle).  Pressure contributes with the
outward face normal sign (+x face of solid → +p, −x face → −p) and the viscous
shear contributes with the opposite sign so that a uniform +x flow past a
cylinder yields a negative drag value (force-on-fluid convention, matching
jax-cfd / PhiFlow).

Two implementations live here so each solver can use the one matching its
autodiff framework.  Both consume the same canonical inputs:

    ux           — (nx, ny) collocated x-velocity, after time-averaging.
    pressure     — (nx, ny) collocated pressure (force-per-density units).
    solid_mask   — (nx, ny) bool, True inside the obstacle.
    viscosity    — scalar kinematic viscosity ν.
    dx           — scalar physical cell spacing (= domain_extent / nx).

Returns shape (1,) float32 — drag in physical units (force per unit depth,
divided by density, since solvers run with ρ=1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax.numpy as jnp
    import torch

__all__ = ["drag_jax", "drag_torch"]


def drag_jax(
    ux: "jnp.ndarray",
    pressure: "jnp.ndarray",
    solid_mask: "jnp.ndarray",
    viscosity: Any,
    dx: Any,
) -> "jnp.ndarray":
    """Surface-integral drag for jax-based solvers (PhiFlow, XLB, jax-cfd)."""
    import jax.numpy as jnp

    fluid_mask = ~solid_mask
    solid_right = jnp.roll(solid_mask, -1, axis=0)
    solid_left = jnp.roll(solid_mask, 1, axis=0)
    surf_right = fluid_mask & solid_right  # fluid cell whose +x nbr is solid
    surf_left = fluid_mask & solid_left  # fluid cell whose −x nbr is solid

    p_drag = jnp.sum(
        jnp.where(surf_right, pressure * dx, 0.0)
        + jnp.where(surf_left, -pressure * dx, 0.0)
    )
    visc_drag = jnp.sum(
        jnp.where(surf_right, -viscosity * ux, 0.0)
        + jnp.where(surf_left, viscosity * ux, 0.0)
    )
    return jnp.reshape((p_drag + visc_drag).astype(jnp.float32), (1,))


def drag_torch(
    ux: "torch.Tensor",
    pressure: "torch.Tensor",
    solid_mask: "torch.Tensor",
    viscosity: Any,
    dx: Any,
) -> "torch.Tensor":
    """Surface-integral drag for PyTorch-based solvers (PICT)."""
    import torch

    fluid_mask = ~solid_mask
    solid_right = torch.roll(solid_mask, shifts=-1, dims=0)
    solid_left = torch.roll(solid_mask, shifts=1, dims=0)
    surf_right = fluid_mask & solid_right
    surf_left = fluid_mask & solid_left

    z = torch.zeros((), dtype=pressure.dtype, device=pressure.device)
    p_drag = (
        torch.where(surf_right, pressure * dx, z)
        + torch.where(surf_left, -pressure * dx, z)
    ).sum()
    visc_drag = (
        torch.where(surf_right, -viscosity * ux, z)
        + torch.where(surf_left, viscosity * ux, z)
    ).sum()
    return (p_drag + visc_drag).reshape(1).to(torch.float32)
