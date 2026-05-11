import os
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import meshio
from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from mosaic_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.types import make_differentiable
from tesseract_core.runtime import ShapeDType
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

crt_file_path = os.path.dirname(__file__)


class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho", "source"])):
    pass


class OutputSchema(
    make_differentiable(
        _CanonicalOutputSchema, ["thermal_compliance", "identification_error"]
    )
):
    pass


# ---------------------------------------------------------------------------
# Identification error helper
# ---------------------------------------------------------------------------


def _compute_identification_error(  # mosaic:physics
    T_nodes: jnp.ndarray, target_temperature: jnp.ndarray
) -> jnp.ndarray:
    """Compute ||T - target_temperature||²_2 (scalar, float32).

    Args:
        T_nodes: Nodal temperature field, shape (n_nodes,).
        target_temperature: Per-node target, shape (n_target,). Trimmed to
            min(n_nodes, n_target) for safety.

    Returns:
        Scalar identification error.
    """
    n = min(T_nodes.shape[0], target_temperature.shape[0])
    diff = T_nodes[:n] - target_temperature[:n]
    return jnp.sum(diff**2).astype(jnp.float32)


#
# Helper functions
#


class HeatConduction(Problem):  # mosaic:physics
    """Steady-state heat conduction problem with SIMP conductivity interpolation."""

    def custom_init(self, von_neumann_value_fns: list[Callable]) -> None:
        """Initialize custom problem parameters.

        Args:
            von_neumann_value_fns: List of functions for Neumann (heat flux) boundary conditions.
        """
        self.fe = self.fes[0]
        self.fe.flex_inds = jnp.arange(len(self.fe.cells))
        self.von_neumann_value_fns = von_neumann_value_fns

    def get_tensor_map(self) -> Callable:
        """Get the Fourier heat conduction constitutive map.

        Returns:
            Callable that computes heat flux from temperature gradient and density.
        """

        def thermal_flux(
            u_grad: jnp.ndarray, theta: jnp.ndarray, src: jnp.ndarray
        ) -> jnp.ndarray:
            k_max = 1.0
            k_min = 1e-3 * k_max
            penal = 3.0

            k = k_min + (k_max - k_min) * theta[0] ** penal

            # Fourier's law: q = k * grad(T)
            # (sign handled by JAX-FEM weak form convention)
            # src is passed through but ignored here (handled by get_mass_map)
            return k * u_grad

        return thermal_flux

    def get_surface_maps(self) -> list[Callable]:
        """Get Neumann boundary condition functions (prescribed heat flux).

        Returns:
            List of Neumann boundary condition value functions.
        """
        return self.von_neumann_value_fns

    def set_params(self, params) -> None:
        """Set density and optional source parameters.

        Args:
            params: Either a density array of shape (n_flex_cells, 1) for
                topology-optimisation mode, or a tuple (rho, source) where
                rho has shape (n_flex_cells, 1) and source has shape
                (n_cells,) for source-identification mode.
        """
        if isinstance(params, (tuple, list)):
            rho, source_flat = params
        else:
            rho = params
            source_flat = jnp.zeros(self.fe.num_cells)

        full_params = jnp.ones((self.fe.num_cells, rho.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(rho)
        thetas = jnp.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params

        # Always include source quads in internal_vars so get_mass_map kernel
        # signature is consistent regardless of whether source is active.
        src = source_flat[: self.fe.num_cells, None, None]  # (num_cells, 1, 1)
        src_quads = jnp.repeat(
            src, self.fe.num_quads, axis=1
        )  # (num_cells, num_quads, 1)
        self.internal_vars = [thetas, src_quads]

    def get_mass_map(self):
        """Body-force (volumetric heat source) contribution.

        Returns a mass map f(u, x, theta, src) = -src so that the mass
        kernel adds  -∫ f·v dΩ  to the residual, i.e. moves the source
        term to the RHS:  R = K·T - ∫ f·v dΩ = 0.
        """

        def _source_fn(
            u: jnp.ndarray, x: jnp.ndarray, theta: jnp.ndarray, src: jnp.ndarray
        ) -> jnp.ndarray:
            return -src  # shape (vec,) = (1,); contributes -f·v to residual

        return _source_fn

    def compute_thermal_compliance(self, sol: jnp.ndarray) -> jnp.ndarray:
        """Compute thermal compliance via surface integral.

        Thermal compliance C = ∮ q_n * T dΓ (work done by heat flux on temperature field).

        Args:
            sol: Solution temperature field.

        Returns:
            Thermal compliance value (scalar).
        """
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        T_face = (
            sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :]
            * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        )
        T_face = jnp.sum(T_face, axis=2)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        flux = -jax.vmap(jax.vmap(neumann_fn))(T_face, subset_quad_points)
        val = jnp.sum(flux * T_face * nanson_scale[:, :, None])
        return val


def setup(  # mosaic:init
    pts: jnp.ndarray = None,
    cells: jnp.ndarray = None,
    boundary_conditions: dict | None = None,
) -> tuple[HeatConduction, Callable]:
    """Setup the heat conduction problem and its differentiable solver.

    Args:
        pts: Array of mesh vertex positions.
        cells: Array of hexahedral cell definitions.
        boundary_conditions: Dict from MeshBC.model_dump() with 'dirichlet' and
            'neumann' keys. dirichlet.values shape (n_groups, 1) or None for zero.
            neumann.values shape (n_groups, 1).

    Returns:
        Tuple of (problem, fwd_pred) where problem is the configured HeatConduction
        problem instance and fwd_pred is the differentiable forward solver.
    """
    ele_type = "HEX8"
    meshio_mesh = meshio.Mesh(points=pts, cells={"hexahedron": cells})
    mesh = Mesh(pts, meshio_mesh.cells_dict["hexahedron"])

    bc = boundary_conditions or {}
    d_bc = bc.get("dirichlet") or {}
    n_bc = bc.get("neumann") or {}

    dirichlet_mask = jnp.array(d_bc.get("mask", []), dtype=jnp.int32)
    n_d_groups = (
        int(jnp.max(dirichlet_mask).item())
        if dirichlet_mask.size > 0 and int(jnp.max(dirichlet_mask)) > 0
        else 0
    )
    raw_dv = d_bc.get("values")
    dirichlet_values = (
        jnp.array(raw_dv, dtype=jnp.float32)
        if raw_dv is not None
        else jnp.zeros((n_d_groups, 1), dtype=jnp.float32)
    )

    von_neumann_mask = jnp.array(n_bc.get("mask", []), dtype=jnp.int32)
    von_neumann_values = jnp.array(
        n_bc.get("values", jnp.zeros((0, 1), dtype=jnp.float32)), dtype=jnp.float32
    )
    n_vn_groups = von_neumann_values.shape[0]

    def make_location_fn(masks: jnp.ndarray, idx: int) -> Callable:
        def _location_fn(point: jnp.ndarray, index: int):
            return (
                jnp.sum(jax.lax.dynamic_index_in_dim(masks, index, 0, keepdims=False))
                == idx + 1
            ).astype(jnp.bool_)

        return _location_fn

    def make_dirichlet_value_fn(i: int) -> Callable:
        def _value_fn(point: jnp.ndarray):
            return dirichlet_values[i, 0]

        return _value_fn

    def make_neumann_value_fn(i: int) -> Callable:
        def _value_fn_vn(u: jnp.ndarray, x: jnp.ndarray):
            return von_neumann_values[i]

        return _value_fn_vn

    # Build Dirichlet BC: one entry per group (scalar DOF)
    d_location_fns = [make_location_fn(dirichlet_mask, i) for i in range(n_d_groups)]
    d_value_fns = [make_dirichlet_value_fn(i) for i in range(n_d_groups)]

    # For scalar heat conduction (vec=1): single DOF per node (temperature)
    dirichlet_bc_info = [d_location_fns, [0] * n_d_groups, d_value_fns]

    # Build Neumann BC: one entry per group
    vn_location_fns = [
        make_location_fn(von_neumann_mask, i) for i in range(n_vn_groups)
    ]
    vn_value_fns = [make_neumann_value_fn(i) for i in range(n_vn_groups)]

    problem = HeatConduction(
        mesh,
        vec=1,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=vn_location_fns,
        additional_info=(vn_value_fns,),
    )

    fwd_pred = ad_wrapper(
        problem,
        solver_options={"umfpack_solver": {}},
        adjoint_solver_options={"umfpack_solver": {}},
    )
    return problem, fwd_pred


def apply_fn(inputs: dict) -> dict:  # mosaic:physics
    """Compute the thermal compliance given a density field.

    Args:
        inputs: Dictionary containing input parameters and density field.

    Returns:
        Dictionary containing the thermal compliance and identification error.
    """
    problem, fwd_pred = setup(
        pts=inputs["hex_mesh"]["points"][: inputs["hex_mesh"]["n_points"]],
        cells=inputs["hex_mesh"]["faces"][: inputs["hex_mesh"]["n_faces"]],
        boundary_conditions=inputs["boundary_conditions"],
    )

    n_cells = inputs["hex_mesh"]["n_faces"]
    rho = inputs["rho"][:n_cells]
    rho = rho[:, None]  # (n_cells,) → (n_cells, 1) as required by JAX-FEM

    source = inputs["source"][:n_cells]  # (n_cells,) volumetric heat source

    # Pass (rho, source) as a tuple so that JAX AD can differentiate w.r.t.
    # both fields via the ad_wrapper adjoint.  set_params unpacks the tuple
    # and stores source as internal_vars alongside the SIMP density thetas.
    sol_list = fwd_pred((rho, source))
    sol = sol_list[0]
    thermal_compliance = problem.compute_thermal_compliance(sol)

    target_temperature = jnp.array(inputs.get("target_temperature", jnp.zeros(1)))
    # JAX-FEM's weak form uses the sign convention K·T = -f, so the solved
    # temperature field is the negative of the physical temperature returned by
    # FEniCS/Firedrake/deal.II (which use K·T = +f).  Negate sol[:,0] so that
    # identification_error = sum((-T_jaxfem - T_target)^2) is consistent with
    # the other three solvers' sum((T_nodal - T_target)^2) formulation.
    T_nodal = -sol[:, 0]
    id_error = _compute_identification_error(T_nodal, target_temperature)

    return {
        "thermal_compliance": thermal_compliance.astype(jnp.float32),
        "identification_error": id_error,
    }


#
# Tesseract endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Compute the thermal compliance given a density field."""
    return apply_fn(inputs.model_dump())


def jacobian_vector_product(  # mosaic:grad:rho,source:autodiff
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    assert jvp_inputs <= {"rho", "source"}
    assert jvp_outputs <= {"thermal_compliance", "identification_error"}

    inputs_dict = inputs.model_dump()
    filtered_apply = filter_func(apply_fn, inputs_dict, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs_dict, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


def vector_jacobian_product(  # mosaic:grad:rho,source:autodiff
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute vector-Jacobian product for specified inputs and outputs.

    Args:
        inputs: InputSchema instance containing input parameters and density field.
        vjp_inputs: Set of input variable names for which to compute gradients.
        vjp_outputs: Set of output variable names with respect to which to compute gradients.
        cotangent_vector: Dictionary containing cotangent vectors for the specified outputs.

    Returns:
        Dictionary containing the vector-Jacobian product for the specified inputs.
    """
    assert vjp_inputs <= {"rho", "source"}
    assert vjp_outputs <= {"thermal_compliance", "identification_error"}

    inputs = inputs.model_dump()

    filtered_apply = filter_func(apply_fn, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    out = vjp_func(cotangent_vector)[0]
    return out


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Calculate output shape of apply from the shape of its inputs."""
    return {
        "thermal_compliance": ShapeDType(shape=(), dtype="float32"),
        "identification_error": ShapeDType(shape=(), dtype="float32"),
    }
