using IncompressibleNavierStokes
using Zygote

# ---------------------------------------------------------------------------
# Setup cache (indexed by (n, L, ndim))
# ---------------------------------------------------------------------------
# mosaic:init

const _SETUP_CACHE = Dict{Tuple{Int,Float64,Int}, Tuple}()

# ---------------------------------------------------------------------------
# Channel setup cache 2-D (indexed by (n, L)) — DCT pressure, psolver_transform
# ---------------------------------------------------------------------------

const _CHANNEL_2D_SETUP_CACHE = Dict{Tuple{Int,Float64}, Tuple}()

function get_setup_and_psolver(n::Int, L::Float64, ndim::Int)
    key = (n, L, ndim)
    if !haskey(_SETUP_CACHE, key)
        ax = LinRange(0.0, L, n + 1)
        if ndim == 2
            setup = Setup(;
                x = (ax, ax),
                boundary_conditions = (;
                    u = (
                        (PeriodicBC(), PeriodicBC()),
                        (PeriodicBC(), PeriodicBC()),
                    ),
                ),
            )
        else
            setup = Setup(;
                x = (ax, ax, ax),
                boundary_conditions = (;
                    u = (
                        (PeriodicBC(), PeriodicBC()),
                        (PeriodicBC(), PeriodicBC()),
                        (PeriodicBC(), PeriodicBC()),
                    ),
                ),
            )
        end
        psolver = psolver_spectral(setup)
        _SETUP_CACHE[key] = (setup, psolver)
    end
    return _SETUP_CACHE[key]
end


# ---------------------------------------------------------------------------
# Ghost cell utilities — 2-D  (n,n,2) <-> (n+2,n+2,2)
# ---------------------------------------------------------------------------

# mosaic:util
"""Add periodic ghost cells: (n,n,2) -> (n+2,n+2,2)."""
function add_ghosts_2d(u_inner::AbstractArray, n::Int)
    u_x = cat(u_inner[n:n, :, :], u_inner, u_inner[1:1, :, :]; dims=1)
    return cat(u_x[:, n:n, :], u_x, u_x[:, 1:1, :]; dims=2)
end

"""Strip ghost cells: (n+2,n+2,2) -> (n,n,2)."""
strip_ghosts_2d(u::AbstractArray, n::Int) = u[2:n+1, 2:n+1, :]


# ---------------------------------------------------------------------------
# Ghost cell utilities — 3-D  (n,n,n,3) <-> (n+2,n+2,n+2,3)
# ---------------------------------------------------------------------------

"""Add periodic ghost cells: (n,n,n,3) -> (n+2,n+2,n+2,3)."""
function add_ghosts_3d(u_inner::AbstractArray, n::Int)
    u_x  = cat(u_inner[n:n, :, :, :], u_inner, u_inner[1:1, :, :, :]; dims=1)
    u_xy = cat(u_x[:, n:n, :, :], u_x, u_x[:, 1:1, :, :]; dims=2)
    return cat(u_xy[:, :, n:n, :], u_xy, u_xy[:, :, 1:1, :]; dims=3)
end

"""Strip ghost cells: (n+2,n+2,n+2,3) -> (n,n,n,3)."""
strip_ghosts_3d(u::AbstractArray, n::Int) = u[2:n+1, 2:n+1, 2:n+1, :]


# ---------------------------------------------------------------------------
# Grid interpolation helpers — 2-D
# ---------------------------------------------------------------------------

"""Collocated (n,n,2) → staggered (n,n,2) via periodic linear interpolation."""
function coloc_to_stag_2d(u::AbstractArray, n::Int)
    ux = u[:, :, 1]
    uy = u[:, :, 2]
    ux_s = 0.5 .* (ux .+ cat(ux[2:end, :], ux[1:1, :]; dims=1))
    uy_s = 0.5 .* (uy .+ cat(uy[:, 2:end], uy[:, 1:1]; dims=2))
    return cat(reshape(ux_s, n, n, 1), reshape(uy_s, n, n, 1); dims=3)
end

"""Staggered (n,n,2) → collocated (n,n,2)."""
function stag_to_coloc_2d(u::AbstractArray, n::Int)
    ux_s = u[:, :, 1]
    uy_s = u[:, :, 2]
    ux = 0.5 .* (cat(ux_s[end:end, :], ux_s[1:end-1, :]; dims=1) .+ ux_s)
    uy = 0.5 .* (cat(uy_s[:, end:end], uy_s[:, 1:end-1]; dims=2) .+ uy_s)
    return cat(reshape(ux, n, n, 1), reshape(uy, n, n, 1); dims=3)
end


# ---------------------------------------------------------------------------
# Grid interpolation helpers — 3-D
# ---------------------------------------------------------------------------

"""Collocated (n,n,n,3) → staggered (n,n,n,3) via periodic linear interpolation."""
function coloc_to_stag_3d(u::AbstractArray, n::Int)
    ux = u[:, :, :, 1]
    uy = u[:, :, :, 2]
    uz = u[:, :, :, 3]
    ux_s = 0.5 .* (ux .+ cat(ux[2:end, :, :], ux[1:1, :, :]; dims=1))
    uy_s = 0.5 .* (uy .+ cat(uy[:, 2:end, :], uy[:, 1:1, :]; dims=2))
    uz_s = 0.5 .* (uz .+ cat(uz[:, :, 2:end], uz[:, :, 1:1]; dims=3))
    return cat(
        reshape(ux_s, n, n, n, 1),
        reshape(uy_s, n, n, n, 1),
        reshape(uz_s, n, n, n, 1);
        dims=4,
    )
end

"""Staggered (n,n,n,3) → collocated (n,n,n,3)."""
function stag_to_coloc_3d(u::AbstractArray, n::Int)
    ux_s = u[:, :, :, 1]
    uy_s = u[:, :, :, 2]
    uz_s = u[:, :, :, 3]
    ux = 0.5 .* (cat(ux_s[end:end, :, :], ux_s[1:end-1, :, :]; dims=1) .+ ux_s)
    uy = 0.5 .* (cat(uy_s[:, end:end, :], uy_s[:, 1:end-1, :]; dims=2) .+ uy_s)
    uz = 0.5 .* (cat(uz_s[:, :, end:end], uz_s[:, :, 1:end-1]; dims=3) .+ uz_s)
    return cat(
        reshape(ux, n, n, n, 1),
        reshape(uy, n, n, n, 1),
        reshape(uz, n, n, n, 1);
        dims=4,
    )
end


# ---------------------------------------------------------------------------
# Forward passes (RK4, non-mutating, Zygote-differentiable)
# ---------------------------------------------------------------------------

# mosaic:physics
"""2-D forward: v0 (n,n,2) → v_out (n,n,2)."""
function ns_forward_2d(v0::AbstractArray, rhs, setup, psolver,
                       nu::Real, dt::Real, steps::Int, n::Int)
    p = (; viscosity = nu)
    v0_stag = coloc_to_stag_2d(v0, n)
    u = add_ghosts_2d(v0_stag, n)
    u = project(u, setup; psolver)
    u = add_ghosts_2d(strip_ghosts_2d(u, n), n)

    for _ in 1:steps
        k1 = rhs(u, p, 0.0)
        k2 = rhs(add_ghosts_2d(strip_ghosts_2d(u .+ (dt/2) .* k1, n), n), p, 0.0)
        k3 = rhs(add_ghosts_2d(strip_ghosts_2d(u .+ (dt/2) .* k2, n), n), p, 0.0)
        k4 = rhs(add_ghosts_2d(strip_ghosts_2d(u .+ dt .* k3, n), n), p, 0.0)
        u = add_ghosts_2d(strip_ghosts_2d(
            u .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4), n), n)
    end

    return stag_to_coloc_2d(strip_ghosts_2d(u, n), n)
end

"""3-D forward: v0 (n,n,n,3) → v_out (n,n,n,3)."""
function ns_forward_3d(v0::AbstractArray, rhs, setup, psolver,
                       nu::Real, dt::Real, steps::Int, n::Int)
    p = (; viscosity = nu)
    v0_stag = coloc_to_stag_3d(v0, n)
    u = add_ghosts_3d(v0_stag, n)
    u = project(u, setup; psolver)
    u = add_ghosts_3d(strip_ghosts_3d(u, n), n)

    for _ in 1:steps
        k1 = rhs(u, p, 0.0)
        k2 = rhs(add_ghosts_3d(strip_ghosts_3d(u .+ (dt/2) .* k1, n), n), p, 0.0)
        k3 = rhs(add_ghosts_3d(strip_ghosts_3d(u .+ (dt/2) .* k2, n), n), p, 0.0)
        k4 = rhs(add_ghosts_3d(strip_ghosts_3d(u .+ dt .* k3, n), n), p, 0.0)
        u = add_ghosts_3d(strip_ghosts_3d(
            u .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4), n), n)
    end

    return stag_to_coloc_3d(strip_ghosts_3d(u, n), n)
end


# ---------------------------------------------------------------------------
# Public API (called from Python via juliacall)
# ---------------------------------------------------------------------------

# mosaic:io
"""
    ns_apply(v0_np, nu, dt, steps, n, L) -> v_out_np

Forward pass.
  2-D: v0_np (n,n,2) Float32 → returns (n,n,2) Float32
  3-D: v0_np (n,n,n,3) Float32 → returns (n,n,n,3) Float32
"""
function ns_apply(v0_np, nu::Float64, dt::Float64, steps::Int, n::Int, L::Float64)
    v0   = Float32.(v0_np)
    ndim = size(v0, ndims(v0))  # last dim: 2 or 3
    setup, psolver = get_setup_and_psolver(n, L, ndim)
    rhs  = create_right_hand_side(setup, psolver)

    if ndim == 2
        v_out = ns_forward_2d(v0, rhs, setup, psolver, nu, dt, steps, n)
    else
        v_out = ns_forward_3d(v0, rhs, setup, psolver, nu, dt, steps, n)
    end
    return Float32.(v_out)
end

# mosaic:grad:v0,viscosity,dt:adjoint
"""
    ns_vjp(v0_np, cotangent_np, nu, dt, steps, n, L)
        -> (grad_v0, grad_nu, grad_dt, grad_L)

VJP. Shapes match v0_np. grad_L is always 0.0 (L is structural).

grad_v0 and grad_dt are computed via Zygote reverse-mode AD.
grad_nu is computed via central finite differences because INS.jl's
`diffusion` rrule returns NoTangent() for the viscosity argument (the
library registers its own ChainRulesCore rrule that only differentiates
through the velocity field, not through nu). This is a scalar FD and
therefore cheap: two extra forward passes.
"""
function ns_vjp(v0_np, cotangent_np, nu::Float64, dt::Float64,
                steps::Int, n::Int, L::Float64)
    v0   = Float32.(v0_np)
    cot  = Float32.(cotangent_np)
    ndim = size(v0, ndims(v0))

    setup, psolver = get_setup_and_psolver(n, L, ndim)
    rhs  = create_right_hand_side(setup, psolver)

    if ndim == 2
        fwd = (v, dt_) -> ns_forward_2d(v, rhs, setup, psolver, nu, dt_, steps, n)
    else
        fwd = (v, dt_) -> ns_forward_3d(v, rhs, setup, psolver, nu, dt_, steps, n)
    end

    # Zygote pullback for grad_v0 and grad_dt.
    # nu is captured as a constant here so Zygote does not attempt to
    # differentiate through INS.jl's diffusion rrule (which returns
    # NoTangent() for viscosity). grad_nu is handled separately below.
    _, back = Zygote.pullback(fwd, v0, Float32(dt))
    grads = back(cot)
    grad_v0  = Float32.(grads[1])
    grad_dt  = Float64(something(grads[2], 0.0))

    # grad_nu via central finite differences (scalar nu, two extra forward passes).
    # ε is chosen relative to nu so the FD stencil is accurate regardless of scale.
    eps_nu = max(1f-4, Float32(abs(nu)) * 1f-3)
    if ndim == 2
        f_plus  = ns_forward_2d(v0, rhs, setup, psolver, nu + eps_nu, Float32(dt), steps, n)
        f_minus = ns_forward_2d(v0, rhs, setup, psolver, nu - eps_nu, Float32(dt), steps, n)
    else
        f_plus  = ns_forward_3d(v0, rhs, setup, psolver, nu + eps_nu, Float32(dt), steps, n)
        f_minus = ns_forward_3d(v0, rhs, setup, psolver, nu - eps_nu, Float32(dt), steps, n)
    end
    jvp_nu  = (f_plus .- f_minus) ./ (2 * eps_nu)  # ∂f/∂nu (same shape as cot)
    grad_nu = Float64(sum(cot .* jvp_nu))

    return (
        grad_v0,
        grad_nu,
        grad_dt,
        Float64(0.0),
    )
end



# ---------------------------------------------------------------------------
# Channel flow with cylinder obstacle — 2-D
#
# Domain: [0,L]^2 periodic in y, inflow at x=0, outflow at x=L.
# Brinkman penalization zeros velocity inside the solid obstacle mask.
#
# The pressure Poisson problem uses psolver_transform (DCT-based) with all-
# DirichletBC boundaries.  psolver_direct (LU) was previously used but proved
# ill-conditioned for Brinkman penalization: the post-projection zeroing of
# velocity inside the obstacle creates divergence that the LU solver amplifies
# until the field diverges.  psolver_transform avoids this
# by working in frequency space (diagonal system).
#
# Gradient path:
#   inflow_np is injected into the left ghost column after every velocity
#   update using a pure functional cat+masking operation, so Zygote can
#   differentiate through it with respect to inflow_np.  The obstacle mask
#   is treated as a non-differentiable constant (Float32 array not traced by
#   Zygote).  psolver_transform is self-adjoint so the ChainRulesCore rrule
#   on poisson() applies correctly in reverse.
# ---------------------------------------------------------------------------

# mosaic:init
"""Build (or return cached) 2-D channel setup + psolver_transform for n×n grid.

We configure all four boundaries as DirichletBC so that INS.jl builds the
DCT-based pressure Laplacian.  psolver_transform (DCT in x, DCT in y) is used
instead of psolver_direct (LU) because the Brinkman volume penalization causes
a near-singular system for the LU solver: the penalization zeroes velocity
inside the obstacle after each pressure projection, creating divergence that the
LU solve amplifies until the field diverges.  The DCT solver is spectrally
diagonal and avoids this amplification.

psolver_transform requirements (verified here):
  - all BCs are PeriodicBC or DirichletBC — satisfied (all DirichletBC).
  - uniform grid — satisfied (LinRange).
"""
function get_channel_2d_setup_and_psolver(n::Int, L::Float64)
    key = (n, L)
    if !haskey(_CHANNEL_2D_SETUP_CACHE, key)
        ax = LinRange(0.0, L, n + 1)
        setup = Setup(;
            x = (ax, ax),
            boundary_conditions = (;
                u = (
                    (DirichletBC(), DirichletBC()),
                    (DirichletBC(), DirichletBC()),
                ),
            ),
        )
        psolver = psolver_transform(setup)
        _CHANNEL_2D_SETUP_CACHE[key] = (setup, psolver)
    end
    return _CHANNEL_2D_SETUP_CACHE[key]
end

# mosaic:util
"""
    _apply_inflow_ghost_2d(u, inflow_field, mask_inflow, n1, n2)

Return a new velocity array where the left ghost column (x = 1) is set to
inflow_field (ux component) with uy = 0.

inflow_field: (n, 2) — inflow velocity along the left edge, components [ux, uy].
u:            (n1, n2, 2) — full staggered field including ghost cells.
mask_inflow:  (n1, n2, 2) — binary mask with 1 at left ghost column for both components.

Pure functional — Zygote can differentiate through it w.r.t. inflow_field.
"""
function _apply_inflow_ghost_2d(
    u::AbstractArray,
    inflow_field::AbstractArray,
    mask_inflow::AbstractArray,
    n1::Int, n2::Int,
)
    T = eltype(inflow_field)
    nl = size(inflow_field, 1)  # = n
    pad = (n2 - nl) ÷ 2        # = 1 ghost cell on each y-side
    # Pad inflow to full n2 height (one zero ghost cell on each y-side)
    inflow_padded = cat(
        zeros(T, pad, 2),
        inflow_field,
        zeros(T, pad, 2);
        dims = 1,
    )  # (n2, 2)
    # Build (1, n2, 2) slice for the left ghost x-column
    inflow_slice = cat(
        reshape(inflow_padded[:, 1], 1, n2, 1),
        reshape(inflow_padded[:, 2], 1, n2, 1);
        dims = 3,
    )  # (1, n2, 2)
    # Extend to full (n1, n2, 2) with zeros to the right
    inflow_full = cat(inflow_slice, zeros(T, n1 - 1, n2, 2); dims = 1)
    # Mask-based replacement: u_out = u*(1-mask) + inflow_full*mask
    return u .* (1 .- mask_inflow) .+ inflow_full .* mask_inflow
end

"""
    _apply_outflow_ghost_2d(u, n1, n2)

Zero-gradient (Neumann) outflow at x=L: copy the rightmost interior column
into the right ghost column (x = n1).

Pure functional — Zygote can differentiate through it.
"""
function _apply_outflow_ghost_2d(u::AbstractArray, n1::Int, n2::Int)
    # Interior columns: indices 2..n1-1 (0-indexed ghost: col 1 = left ghost, col n1 = right ghost)
    # Rightmost interior column index (1-based) = n1 - 1
    right_interior = u[n1-1:n1-1, :, :]  # (1, n2, 2)
    # Replace right ghost column (index n1) with interior copy
    return cat(u[1:n1-1, :, :], right_interior; dims = 1)
end

"""
    _apply_brinkman_2d(u, obstacle_mask, n1, n2)

Zero velocity inside the Brinkman obstacle mask.

obstacle_mask: (n, n, 2) collocated interior mask (1=solid, 0=fluid).
u:             (n1, n2, 2) staggered field including ghost cells.

We pad the obstacle mask with zeros for ghost cells before applying.
Pure functional.
"""
function _apply_brinkman_2d(u::AbstractArray, obstacle_mask::AbstractArray, n1::Int, n2::Int)
    T = eltype(u)
    n = size(obstacle_mask, 1)  # interior grid size
    # Pad obstacle mask with zero ghost cells on all sides → (n1, n2, 2)
    mask_padded = cat(
        zeros(T, 1, n2, 2),
        cat(
            zeros(T, n, 1, 2),
            obstacle_mask,
            zeros(T, n, 1, 2);
            dims = 2,
        ),
        zeros(T, 1, n2, 2);
        dims = 1,
    )  # (n1, n2, 2)
    return u .* (1 .- mask_padded)
end

# mosaic:physics
"""
    _channel_2d_rhs(u, inflow_field, obstacle_mask, nu, t, setup, psolver, mask_inflow)

One NS RHS evaluation for 2-D channel flow with Brinkman obstacle:
  1. Apply no-slip BC to all walls (zeros all ghost cells via DirichletBC setup).
  2. Override left ghost column with inflow_field (differentiable injection).
  3. Override right ghost column with zero-gradient outflow (differentiable).
  4. Apply periodic ghost cells in y-direction (top/bottom wrapping).
  5. Zero velocity inside Brinkman obstacle (non-differentiable mask).
  6. Compute NS momentum RHS via INS.jl navierstokes().
  7. Apply time-derivative BC.
  8. Project to divergence-free space via psolver_direct.
  9. Apply Brinkman mask again after projection.
"""
function _channel_2d_rhs(
    u::AbstractArray,
    inflow_field::AbstractArray,
    obstacle_mask::AbstractArray,
    nu::Real,
    t::Real,
    setup,
    psolver,
    mask_inflow::AbstractArray,
)
    n1, n2 = setup.N
    n = n1 - 2  # interior grid size
    T = eltype(inflow_field)
    # Step 1: zero all ghost cells (DirichletBC walls)
    u_bc = apply_bc_u(u, t, setup)
    # Step 2: inject inflow at left ghost column (differentiable)
    u_in = _apply_inflow_ghost_2d(u_bc, inflow_field, mask_inflow, n1, n2)
    # Step 3: zero-gradient outflow at right ghost column (differentiable)
    u_out = _apply_outflow_ghost_2d(u_in, n1, n2)
    # Step 4: periodic ghost cells in y-direction (top/bottom).
    # Extract interior block and the left/right x-ghost columns.
    u_inner   = u_out[2:n1-1, 2:n2-1, :]  # (n, n, 2) interior
    left_col  = u_out[1:1,    2:n2-1, :]  # (1, n, 2) inflow x-ghost, interior y
    right_col = u_out[n1:n1,  2:n2-1, :]  # (1, n, 2) outflow x-ghost, interior y
    # Periodic y: bottom ghost = top interior row, top ghost = bottom interior row
    bot_y_ghost = u_inner[:, n:n, :]   # (n, 1, 2)
    top_y_ghost = u_inner[:, 1:1, :]   # (n, 1, 2)
    # Interior block with y ghost cells added
    inner_ypad = cat(bot_y_ghost, u_inner, top_y_ghost; dims = 2)  # (n, n2, 2)
    # Rebuild x-ghost columns with matching y ghost corners
    left_ypad  = cat(u_out[1:1, n2-1:n2-1, :], left_col,  u_out[1:1, 2:2, :]; dims = 2)  # (1, n2, 2)
    right_ypad = cat(u_out[n1:n1, n2-1:n2-1, :], right_col, u_out[n1:n1, 2:2, :]; dims = 2)  # (1, n2, 2)
    u_yperiodic = cat(left_ypad, inner_ypad, right_ypad; dims = 1)  # (n1, n2, 2)
    # Step 5: Brinkman obstacle mask (non-differentiable)
    u_brink = _apply_brinkman_2d(u_yperiodic, obstacle_mask, n1, n2)
    # Step 6: NS momentum RHS
    f = navierstokes((; u = u_brink), t; setup, viscosity = T(nu))
    # Step 7: BC on time derivative
    du = apply_bc_u(f.u, t, setup; dudt = true)
    # Step 8: pressure projection
    u_proj = project(du, setup; psolver)
    # Step 9: Brinkman after projection
    return _apply_brinkman_2d(u_proj, obstacle_mask, n1, n2)
end

"""
    ns_channel_2d_forward(v0, inflow_field, obstacle_mask, nu, dt, steps, setup, psolver, mask_inflow)

RK4 time integration for 2-D channel flow with Brinkman cylinder.

v0:            (n, n, 2) Float32 initial velocity (collocated).
inflow_field:  (n, 2)    Float32 differentiable inflow velocity at x=0.
obstacle_mask: (n, n, 2) Float32 Brinkman mask (1=solid, 0=fluid).

Returns: (n1, n2, 2) Float32 full staggered velocity field (includes ghost cells).
"""
function ns_channel_2d_forward(
    v0::AbstractArray,
    inflow_field::AbstractArray,
    obstacle_mask::AbstractArray,
    nu::Real, dt::Real, steps::Int,
    setup, psolver,
    mask_inflow::AbstractArray,
)
    n1, n2 = setup.N
    n = n1 - 2
    T = eltype(inflow_field)
    # Convert collocated IC to staggered
    v0_stag = coloc_to_stag_2d(v0, n)  # (n, n, 2)
    # Build initial ghost array consistent with channel BCs:
    #   x: left ghost = inflow, right ghost = outflow (copy rightmost interior col)
    #   y: periodic (bottom ghost = top interior row, top ghost = bottom interior row)
    # Interior with periodic y ghost cells
    bot_y_ghost = v0_stag[:, n:n, :]   # (n, 1, 2) — periodic: ghost from top interior
    top_y_ghost = v0_stag[:, 1:1, :]   # (n, 1, 2) — periodic: ghost from bottom interior
    inner_ypad  = cat(bot_y_ghost, v0_stag, top_y_ghost; dims = 2)  # (n, n2, 2)
    # Left x-ghost column: inflow values, with periodic y corners
    inflow_bot_corner = reshape(inflow_field[n:n, :], 1, 1, 2)   # (1, 1, 2)
    inflow_top_corner = reshape(inflow_field[1:1, :], 1, 1, 2)   # (1, 1, 2)
    inflow_interior   = reshape(inflow_field, 1, n, 2)             # (1, n, 2)
    left_col_ypad = cat(inflow_bot_corner, inflow_interior, inflow_top_corner; dims = 2)  # (1, n2, 2)
    # Right x-ghost column: zero-gradient outflow (copy rightmost interior column)
    right_interior_col = v0_stag[n:n, :, :]   # (1, n, 2)
    right_bot_corner   = v0_stag[n:n, n:n, :]  # (1, 1, 2)
    right_top_corner   = v0_stag[n:n, 1:1, :]  # (1, 1, 2)
    right_col_ypad = cat(right_bot_corner, right_interior_col, right_top_corner; dims = 2)  # (1, n2, 2)
    u = cat(left_col_ypad, inner_ypad, right_col_ypad; dims = 1)  # (n1, n2, 2)
    # Apply Brinkman to initial field
    u = _apply_brinkman_2d(u, obstacle_mask, n1, n2)
    for _ in 1:steps
        k1 = _channel_2d_rhs(u, inflow_field, obstacle_mask, nu, 0.0, setup, psolver, mask_inflow)
        k2 = _channel_2d_rhs(u .+ (dt / 2) .* k1, inflow_field, obstacle_mask, nu, 0.0, setup, psolver, mask_inflow)
        k3 = _channel_2d_rhs(u .+ (dt / 2) .* k2, inflow_field, obstacle_mask, nu, 0.0, setup, psolver, mask_inflow)
        k4 = _channel_2d_rhs(u .+ dt .* k3, inflow_field, obstacle_mask, nu, 0.0, setup, psolver, mask_inflow)
        u = u .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        # Enforce Brinkman constraint on state after each RK4 step so the
        # state always satisfies the obstacle mask. Without this, the state
        # can accumulate nonzero velocity inside the obstacle between steps,
        # and the hard-zero in _channel_2d_rhs step 5 creates a discontinuity
        # that grows until the field diverges.
        u = _apply_brinkman_2d(u, obstacle_mask, n1, n2)
    end
    return u
end

# mosaic:io
"""
    ns_apply_channel_2d(v0_np, inflow_np, obstacle_np, nu, dt, steps, n, L) -> v_out_np

Forward pass for 2-D channel flow with Brinkman cylinder obstacle.

v0_np:       (n, n, 2)   Float32 — initial velocity (collocated).
inflow_np:   (n, 2)      Float32 — x- and y-velocity of inflow at x=0 face.
obstacle_np: (n, n, 2)   Float32 — Brinkman mask (1=solid, 0=fluid), broadcast over velocity components.

Returns: (n, n, 2) Float32 collocated interior velocity field.
"""
function ns_apply_channel_2d(
    v0_np,
    inflow_np,
    obstacle_np,
    nu::Float64, dt::Float64, steps::Int, n::Int, L::Float64,
)
    v0      = Float32.(v0_np)       # (n, n, 2)
    inflow  = Float32.(inflow_np)   # (n, 2)
    obs     = Float32.(obstacle_np) # (n, n, 2)

    setup, psolver = get_channel_2d_setup_and_psolver(n, L)
    n1, n2 = setup.N  # = (n+2, n+2)

    # Inflow mask: 1 at left ghost column (x = 1) for both velocity components
    mask_inflow = zeros(Float32, n1, n2, 2)
    mask_inflow[1, :, 1] .= 1f0
    mask_inflow[1, :, 2] .= 1f0

    u_full = ns_channel_2d_forward(
        v0, inflow, obs, Float32(nu), Float32(dt), steps,
        setup, psolver, mask_inflow,
    )

    return Float32.(stag_to_coloc_2d(strip_ghosts_2d(u_full, n), n))
end

# mosaic:grad:inflow_profile,viscosity,dt:adjoint
"""
    ns_vjp_channel_2d(v0_np, inflow_np, obstacle_np, nu, dt, steps, n, L, cotangent_np)
        -> (grad_inflow, grad_nu, grad_dt, grad_L)

VJP for 2-D channel flow w.r.t. inflow_np.

grad_inflow: (n, 2) Float32 — gradient of loss w.r.t. inflow velocity field.
obstacle_np is treated as non-differentiable (held constant in pullback).
grad_nu, grad_dt, grad_L: zeros.
"""
function ns_vjp_channel_2d(
    v0_np,
    inflow_np,
    obstacle_np,
    nu::Float64, dt::Float64, steps::Int, n::Int, L::Float64,
    cotangent_np,
)
    v0          = Float32.(v0_np)
    inflow      = Float32.(inflow_np)
    obs         = Float32.(obstacle_np)
    cot_coloc   = Float32.(cotangent_np)  # (n, n, 2) collocated cotangent

    setup, psolver = get_channel_2d_setup_and_psolver(n, L)
    n1, n2 = setup.N

    mask_inflow = zeros(Float32, n1, n2, 2)
    mask_inflow[1, :, 1] .= 1f0
    mask_inflow[1, :, 2] .= 1f0

    # Forward function for pullback: differentiate w.r.t. inflow only.
    # v0 and obstacle_mask are held constant (non-differentiable).
    function fwd(inflow_f)
        u_full = ns_channel_2d_forward(
            v0, inflow_f, obs, Float32(nu), Float32(dt), steps,
            setup, psolver, mask_inflow,
        )
        stag_to_coloc_2d(strip_ghosts_2d(u_full, n), n)
    end

    _, back = Zygote.pullback(fwd, inflow)
    grads = back(cot_coloc)

    grad_inflow = Float32.(something(grads[1], zeros(Float32, n, 2)))
    return (
        grad_inflow,
        Float64(0.0),
        Float64(0.0),
        Float64(0.0),
    )
end


# ---------------------------------------------------------------------------
# Channel flow — tail-window drag averaging
#
# Drag is computed inside Julia so that Zygote can differentiate through
# the accumulated mean drag w.r.t. inflow_field.
#
# Drag formula (viscous term only, pressure assumed zero — same approximation
# as _compute_drag_numpy in the Python layer):
#   drag = ν * Σ_{surface cells} (surf_left - surf_right) * ux
# where surf_right / surf_left are the fluid cells immediately right/left
# of the solid obstacle boundary.
#
# The obstacle surface mask is pre-computed from obstacle_mask and is held
# constant (non-differentiable); only ux (which depends on inflow_field)
# is differentiated through.
# ---------------------------------------------------------------------------

# mosaic:physics
"""
    _compute_drag_julia_2d(u_coloc, solid_mask, nu)

Compute x-direction viscous drag on the obstacle from a collocated velocity
field u_coloc of shape (n, n, 2).

solid_mask: (n, n) Bool or Float32 — 1 inside the obstacle.
nu:         kinematic viscosity.

Returns a scalar (Float32) drag value (Zygote-differentiable w.r.t. u_coloc).
"""
function _compute_drag_julia_2d(
    u_coloc::AbstractArray,
    solid_mask::AbstractArray,
    nu::Real,
)
    n = size(u_coloc, 1)
    ux = u_coloc[:, :, 1]  # (n, n)

    fluid_mask = 1f0 .- solid_mask  # (n, n) — 1 in fluid, 0 in solid

    # Neighbour solid cells in the x-direction (periodic wrap is fine here;
    # solid cells are interior so wrap does not introduce artefacts at the
    # boundary for a centred cylinder).
    solid_right = cat(solid_mask[2:n, :], solid_mask[1:1, :]; dims = 1)   # shift right
    solid_left  = cat(solid_mask[n:n, :], solid_mask[1:n-1, :]; dims = 1) # shift left

    # Fluid cells adjacent to solid on right/left in x
    surf_right = fluid_mask .* solid_right   # (n, n) — fluid cell with solid to its right (+x)
    surf_left  = fluid_mask .* solid_left    # (n, n) — fluid cell with solid to its left (-x)

    # Viscous drag: ν * Σ (surf_left - surf_right) * ux
    drag = nu * sum((surf_left .- surf_right) .* ux)
    return drag
end

# mosaic:physics
"""
    ns_channel_2d_forward_with_drag(v0, inflow_field, obstacle_mask, solid_mask,
                                     nu, dt, steps, setup, psolver, mask_inflow)

RK4 time integration for 2-D channel flow with Brinkman cylinder.
Accumulates drag over the last `steps ÷ 2` steps (tail-window mean).

Returns: (u_full, mean_drag, mean_velocity)
  u_full:        (n1, n2, 2) staggered velocity field including ghost cells.
  mean_drag:     scalar Float32 — mean x-drag over tail window.
  mean_velocity: (n, n, 2)   Float32 — tail-window mean collocated velocity (RANS).
"""
function ns_channel_2d_forward_with_drag(
    v0::AbstractArray,
    inflow_field::AbstractArray,
    obstacle_mask::AbstractArray,
    solid_mask::AbstractArray,
    nu::Real, dt::Real, steps::Int,
    setup, psolver,
    mask_inflow::AbstractArray,
)
    n1, n2 = setup.N
    n = n1 - 2
    T = eltype(inflow_field)
    nu_T = T(nu)

    # Initial staggered state (same as ns_channel_2d_forward)
    v0_stag = coloc_to_stag_2d(v0, n)
    bot_y_ghost = v0_stag[:, n:n, :]
    top_y_ghost = v0_stag[:, 1:1, :]
    inner_ypad  = cat(bot_y_ghost, v0_stag, top_y_ghost; dims = 2)
    inflow_bot_corner = reshape(inflow_field[n:n, :], 1, 1, 2)
    inflow_top_corner = reshape(inflow_field[1:1, :], 1, 1, 2)
    inflow_interior   = reshape(inflow_field, 1, n, 2)
    left_col_ypad = cat(inflow_bot_corner, inflow_interior, inflow_top_corner; dims = 2)
    right_interior_col = v0_stag[n:n, :, :]
    right_bot_corner   = v0_stag[n:n, n:n, :]
    right_top_corner   = v0_stag[n:n, 1:1, :]
    right_col_ypad = cat(right_bot_corner, right_interior_col, right_top_corner; dims = 2)
    u = cat(left_col_ypad, inner_ypad, right_col_ypad; dims = 1)
    u = _apply_brinkman_2d(u, obstacle_mask, n1, n2)

    n_tail = max(1, steps ÷ 2)
    tail_start = steps - n_tail + 1  # 1-based step index when tail window begins

    # Accumulate drag and velocity over the tail window as running sums (differentiable).
    drag_sum = zero(T)
    velocity_sum = zeros(T, n, n, 2)

    for step_i in 1:steps
        k1 = _channel_2d_rhs(u, inflow_field, obstacle_mask, nu, 0.0, setup, psolver, mask_inflow)
        k2 = _channel_2d_rhs(u .+ (dt / 2) .* k1, inflow_field, obstacle_mask, nu, 0.0, setup, psolver, mask_inflow)
        k3 = _channel_2d_rhs(u .+ (dt / 2) .* k2, inflow_field, obstacle_mask, nu, 0.0, setup, psolver, mask_inflow)
        k4 = _channel_2d_rhs(u .+ dt .* k3, inflow_field, obstacle_mask, nu, 0.0, setup, psolver, mask_inflow)
        u = u .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        # Enforce Brinkman constraint on state after each RK4 step so the
        # state always satisfies the obstacle mask. Without this, the state
        # can accumulate nonzero velocity inside the obstacle between steps,
        # and the hard-zero in _channel_2d_rhs step 5 creates a discontinuity
        # that grows until the field diverges.
        u = _apply_brinkman_2d(u, obstacle_mask, n1, n2)

        if step_i >= tail_start
            u_coloc = stag_to_coloc_2d(strip_ghosts_2d(u, n), n)
            drag_sum = drag_sum + _compute_drag_julia_2d(u_coloc, solid_mask, nu_T)
            velocity_sum = velocity_sum .+ u_coloc
        end
    end

    mean_drag = drag_sum / T(n_tail)
    mean_velocity = velocity_sum ./ T(n_tail)
    return u, mean_drag, mean_velocity
end

# mosaic:io
"""
    ns_apply_channel_2d_drag_window(v0_np, inflow_np, obstacle_np, nu, dt, steps, n, L)
        -> (v_out_np, mean_drag, mean_velocity)

Forward pass for 2-D channel flow with Brinkman cylinder obstacle.
Returns the final collocated velocity field, the tail-window mean drag,
and the tail-window mean velocity (RANS) field (averaged over the last steps÷2 steps).

v0_np:       (n, n, 2)   Float32 — initial velocity (collocated).
inflow_np:   (n, 2)      Float32 — x- and y-velocity of inflow at x=0 face.
obstacle_np: (n, n, 2)   Float32 — Brinkman mask (1=solid, 0=fluid).

Returns:
  v_out_np:      (n, n, 2) Float32 collocated interior velocity field.
  mean_drag:     Float32 scalar — mean viscous drag over last steps÷2 steps.
  mean_velocity: (n, n, 2) Float32 — tail-window mean collocated velocity (RANS).
"""
function ns_apply_channel_2d_drag_window(
    v0_np,
    inflow_np,
    obstacle_np,
    nu::Float64, dt::Float64, steps::Int, n::Int, L::Float64,
)
    v0      = Float32.(v0_np)
    inflow  = Float32.(inflow_np)
    obs     = Float32.(obstacle_np)

    setup, psolver = get_channel_2d_setup_and_psolver(n, L)
    n1, n2 = setup.N

    mask_inflow = zeros(Float32, n1, n2, 2)
    mask_inflow[1, :, 1] .= 1f0
    mask_inflow[1, :, 2] .= 1f0

    # Build solid_mask (n, n) from the first component of obstacle_mask.
    # obstacle_mask shape is (n, n, 2); component [:, :, 1] is the solid indicator.
    solid_mask = obs[:, :, 1]  # (n, n) Float32

    u_full, mean_drag, mean_velocity = ns_channel_2d_forward_with_drag(
        v0, inflow, obs, solid_mask, Float32(nu), Float32(dt), steps,
        setup, psolver, mask_inflow,
    )

    v_out = Float32.(stag_to_coloc_2d(strip_ghosts_2d(u_full, n), n))
    return v_out, Float32(mean_drag), Float32.(mean_velocity)
end

# mosaic:grad:inflow_profile,viscosity,dt:adjoint
"""
    ns_vjp_channel_2d_drag_window(v0_np, inflow_np, obstacle_np, nu, dt, steps, n, L,
                                   cotangent_result_np, cotangent_drag)
        -> (grad_inflow, grad_nu, grad_dt, grad_L)

VJP for 2-D channel flow w.r.t. inflow_np, differentiating through both the
final velocity field AND the tail-window mean drag.

cotangent_result_np: (n, n, 2) Float32 — cotangent on result (final velocity).
cotangent_drag:      Float32 scalar   — cotangent on mean_drag.

grad_inflow: (n, 2) Float32 — gradient of loss w.r.t. inflow velocity field.
"""
function ns_vjp_channel_2d_drag_window(
    v0_np,
    inflow_np,
    obstacle_np,
    nu::Float64, dt::Float64, steps::Int, n::Int, L::Float64,
    cotangent_result_np,
    cotangent_drag::Float64,
)
    v0          = Float32.(v0_np)
    inflow      = Float32.(inflow_np)
    obs         = Float32.(obstacle_np)
    cot_result  = Float32.(cotangent_result_np)  # (n, n, 2)
    cot_drag    = Float32(cotangent_drag)         # scalar

    setup, psolver = get_channel_2d_setup_and_psolver(n, L)
    n1, n2 = setup.N

    mask_inflow = zeros(Float32, n1, n2, 2)
    mask_inflow[1, :, 1] .= 1f0
    mask_inflow[1, :, 2] .= 1f0

    solid_mask = obs[:, :, 1]  # (n, n)

    # Forward function: inflow → (collocated_result, mean_drag).
    # Zygote traces through both outputs so the cotangents on both can flow back.
    # mean_velocity is also computed but not differentiated (cotangent = 0).
    function fwd(inflow_f)
        u_full, mean_drag, _mean_vel = ns_channel_2d_forward_with_drag(
            v0, inflow_f, obs, solid_mask, Float32(nu), Float32(dt), steps,
            setup, psolver, mask_inflow,
        )
        v_out = stag_to_coloc_2d(strip_ghosts_2d(u_full, n), n)
        return v_out, mean_drag
    end

    _, back = Zygote.pullback(fwd, inflow)
    # Cotangent tuple: (cotangent_on_result, cotangent_on_drag)
    grads = back((cot_result, cot_drag))

    grad_inflow = Float32.(something(grads[1], zeros(Float32, n, 2)))
    return (
        grad_inflow,
        Float64(0.0),
        Float64(0.0),
        Float64(0.0),
    )
end
