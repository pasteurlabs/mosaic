import os
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import meshio
from jax_fem.generate_mesh import Mesh

# Import JAX-FEM specific modules
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from mosaic_shared.problems.structural_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.structural_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.types import make_differentiable
from tesseract_core.runtime import ShapeDType
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, "data")


class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho"])):
    pass


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["compliance"])):
    pass


#
# Helper functions
#


# Define constitutive relationship
# Adapted from JAX-FEM
# https://github.com/deepmodeling/jax-fem/blob/1bdbf060bb32951d04ed9848c238c9a470fee1b4/demos/topology_optimization/example.py
class Elasticity(Problem):  # mosaic:physics
    """Linear elasticity problem with custom constitutive law."""

    def custom_init(self, von_neumann_value_fns: list[Callable]) -> None:
        """Initialize custom problem parameters.

        Args:
            von_neumann_value_fns: List of functions for van Neumann boundary conditions.
        """
        self.fe = self.fes[0]
        self.fe.flex_inds = jnp.arange(len(self.fe.cells))

        self.von_neumann_value_fns = von_neumann_value_fns

    def get_tensor_map(self) -> Callable:
        """Get the stress-strain constitutive relationship tensor map.

        Returns:
            Callable that computes stress from strain gradient and density.
        """

        def stress(u_grad: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
            Emax = 70.0e3
            Emin = 1e-3 * Emax
            penal = 3.0

            E = Emin + (Emax - Emin) * theta[0] ** penal

            nu = 0.3
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

            epsilon = 0.5 * (u_grad + u_grad.T)

            sigma = lmbda * jnp.trace(epsilon) * jnp.eye(self.dim) + 2.0 * mu * epsilon
            return sigma

        return stress

    def get_surface_maps(self) -> list[Callable]:
        """Get surface traction boundary condition functions.

        Returns:
            List of van Neumann boundary condition value functions.
        """
        return self.von_neumann_value_fns

    def set_params(self, params: jnp.ndarray) -> None:
        """Set density parameters for topology optimization.

        Args:
            params: Density field array for the flexible elements.
        """
        # Override base class method.
        full_params = jnp.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = jnp.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def compute_compliance(self, sol: jnp.ndarray) -> jnp.ndarray:
        """Compute structural compliance via surface integral.

        Args:
            sol: Solution displacement field.

        Returns:
            Compliance value (scalar).
        """
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        u_face = (
            sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :]
            * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        )
        u_face = jnp.sum(u_face, axis=2)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points)
        val = jnp.sum(traction * u_face * nanson_scale[:, :, None])
        return val


# Module-level cache for the compiled elasticity solver.
# Keyed by (n_nodes, n_cells, n_dirichlet_nodes, n_neumann_nodes) to detect
# mesh/BC changes while avoiding re-compilation on every forward/VJP call.
_setup_cache: dict = {}


def setup(  # mosaic:init
    pts: jnp.ndarray = None,
    cells: jnp.ndarray = None,
    boundary_conditions: dict | None = None,
) -> tuple[Elasticity, Callable]:
    """Setup the elasticity problem and its differentiable solver.

    Args:
        pts: Array of mesh vertex positions.
        cells: Array of hexahedral cell definitions.
        boundary_conditions: Dict from MeshBC.model_dump() with 'dirichlet' and
            'neumann' keys. dirichlet.values shape (n_groups, 3) or None for zero.
            neumann.values shape (n_groups, 3).

    Returns:
        Tuple of (problem, fwd_pred) where problem is the configured Elasticity
        problem instance and fwd_pred is the differentiable forward solver.
    """
    # Cache key: mesh shape + BC structure + BC values.
    # Must include Neumann values so that F_total sweeps compile a fresh graph
    # for each force magnitude rather than reusing the first cached value.
    # The Dirichlet values are always zero for this cantilever problem, but we
    # include them for correctness.
    bc = boundary_conditions or {}
    d_bc = bc.get("dirichlet") or {}
    n_bc = bc.get("neumann") or {}
    d_mask_raw = d_bc.get("mask", [])
    n_mask_raw = n_bc.get("mask", [])
    _d_values_raw = d_bc.get("values")
    _n_values_raw = n_bc.get("values")
    import numpy as _np

    d_values_raw = (
        _np.asarray(_d_values_raw).ravel().tolist() if _d_values_raw is not None else []
    )
    n_values_raw = (
        _np.asarray(_n_values_raw).ravel().tolist() if _n_values_raw is not None else []
    )
    cache_key = (
        pts.shape,
        cells.shape,
        tuple(d_mask_raw) if len(d_mask_raw) < 1000 else hash(bytes(d_mask_raw)),
        tuple(n_mask_raw) if len(n_mask_raw) < 1000 else hash(bytes(n_mask_raw)),
        # Include BC values so F_total changes invalidate the cache.
        tuple(float(v) for v in d_values_raw),
        tuple(float(v) for v in n_values_raw),
    )
    if cache_key in _setup_cache:
        return _setup_cache[cache_key]

    ele_type = "HEX8"
    meshio_mesh = meshio.Mesh(points=pts, cells={"hexahedron": cells})
    mesh = Mesh(pts, meshio_mesh.cells_dict["hexahedron"])

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
        else jnp.zeros((n_d_groups, 3), dtype=jnp.float32)
    )

    von_neumann_mask = jnp.array(n_bc.get("mask", []), dtype=jnp.int32)
    von_neumann_values = jnp.array(
        n_bc.get("values", jnp.zeros((0, 3), dtype=jnp.float32)), dtype=jnp.float32
    )
    n_vn_groups = von_neumann_values.shape[0]

    def make_location_fn(masks: jnp.ndarray, idx: int) -> Callable:
        def _location_fn(point: jnp.ndarray, index: int):
            return (
                jnp.sum(jax.lax.dynamic_index_in_dim(masks, index, 0, keepdims=False))
                == idx + 1
            ).astype(jnp.bool_)

        return _location_fn

    def make_dirichlet_value_fn(i: int, d: int) -> Callable:
        def _value_fn(point: jnp.ndarray):
            return dirichlet_values[i, d]

        return _value_fn

    def make_neumann_value_fn(i: int) -> Callable:
        def _value_fn_vn(u: jnp.ndarray, x: jnp.ndarray):
            return von_neumann_values[i]

        return _value_fn_vn

    # Build Dirichlet BC: one entry per (group, DOF)
    d_location_fns = []
    d_value_fns = []
    d_dofs = []
    for i in range(n_d_groups):
        for dof in range(3):
            d_location_fns.append(make_location_fn(dirichlet_mask, i))
            d_value_fns.append(make_dirichlet_value_fn(i, dof))
            d_dofs.append(dof)

    dirichlet_bc_info = [d_location_fns, d_dofs, d_value_fns]

    # Build Neumann BC: one entry per group
    vn_location_fns = [
        make_location_fn(von_neumann_mask, i) for i in range(n_vn_groups)
    ]
    vn_value_fns = [make_neumann_value_fn(i) for i in range(n_vn_groups)]

    # Define forward problem
    problem = Elasticity(
        mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=vn_location_fns,
        additional_info=(vn_value_fns,),
    )

    # Apply the automatic differentiation wrapper
    # This is a critical step that makes the problem solver differentiable
    fwd_pred = ad_wrapper(
        problem,
        solver_options={"umfpack_solver": {}},
        adjoint_solver_options={"umfpack_solver": {}},
    )
    result = (problem, fwd_pred)
    _setup_cache[cache_key] = result
    return result


def apply_fn(inputs: dict) -> dict:  # mosaic:physics
    """Compute the compliance of the structure given a density field.

    Args:
        inputs: Dictionary containing input parameters and density field.

    Returns:
        Dictionary containing the compliance of the structure.
    """
    # no stop grads
    problem, fwd_pred = setup(
        pts=inputs["hex_mesh"]["points"][: inputs["hex_mesh"]["n_points"]],
        cells=inputs["hex_mesh"]["faces"][: inputs["hex_mesh"]["n_faces"]],
        boundary_conditions=inputs["boundary_conditions"],
    )

    rho = inputs["rho"][: inputs["hex_mesh"]["n_faces"]]
    rho = rho[:, None]  # (n_faces,) → (n_faces, 1) as required by JAX-FEM

    sol_list = fwd_pred(rho)
    sol = sol_list[0]
    compliance = problem.compute_compliance(sol)

    return {"compliance": compliance.astype(jnp.float32)}


#
# Tesseract endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Compute the compliance of the structure given a density field."""
    return apply_fn(inputs.model_dump())


def jacobian_vector_product(  # mosaic:grad:rho:autodiff
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    assert jvp_inputs <= {"rho"}
    assert jvp_outputs <= {"compliance"}

    inputs_dict = inputs.model_dump()
    filtered_apply = filter_func(apply_fn, inputs_dict, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs_dict, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


def vector_jacobian_product(  # mosaic:grad:rho:autodiff
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
    assert vjp_inputs <= {"rho"}
    assert vjp_outputs <= {"compliance"}

    inputs = inputs.model_dump()

    filtered_apply = filter_func(apply_fn, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    out = vjp_func(cotangent_vector)[0]
    return out


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Calculate output shape of apply from the shape of its inputs."""
    return {"compliance": ShapeDType(shape=(), dtype="float32")}
