using IncompressibleNavierStokes
using Zygote

# ---------------------------------------------------------------------------
# Setup cache (indexed by (n, L, ndim))
# ---------------------------------------------------------------------------

const _SETUP_CACHE = Dict{Tuple{Int,Float64,Int}, Tuple}()

function get_setup_and_psolver(n::Int, L::Float64, ndim::Int)
    key = (n, L, ndim)
    if !haskey(_SETUP_CACHE, key)
        # Float32 grid coordinates make the whole solve run in Float32: the
        # non-mutating INS.jl operators allocate their outputs from the setup
        # grid eltype, so a Float64 grid would silently promote the state (and
        # the Zygote pullback) to Float64 after the first projection.
        ax = LinRange(0f0, Float32(L), n + 1)
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
    half = eltype(u)(1) / 2  # eltype-preserving (a 0.5 literal promotes Float32 → Float64)
    ux = u[:, :, 1]
    uy = u[:, :, 2]
    ux_s = half .* (ux .+ cat(ux[2:end, :], ux[1:1, :]; dims=1))
    uy_s = half .* (uy .+ cat(uy[:, 2:end], uy[:, 1:1]; dims=2))
    return cat(reshape(ux_s, n, n, 1), reshape(uy_s, n, n, 1); dims=3)
end

"""Staggered (n,n,2) → collocated (n,n,2)."""
function stag_to_coloc_2d(u::AbstractArray, n::Int)
    half = eltype(u)(1) / 2
    ux_s = u[:, :, 1]
    uy_s = u[:, :, 2]
    ux = half .* (cat(ux_s[end:end, :], ux_s[1:end-1, :]; dims=1) .+ ux_s)
    uy = half .* (cat(uy_s[:, end:end], uy_s[:, 1:end-1]; dims=2) .+ uy_s)
    return cat(reshape(ux, n, n, 1), reshape(uy, n, n, 1); dims=3)
end


# ---------------------------------------------------------------------------
# Grid interpolation helpers — 3-D
# ---------------------------------------------------------------------------

"""Collocated (n,n,n,3) → staggered (n,n,n,3) via periodic linear interpolation."""
function coloc_to_stag_3d(u::AbstractArray, n::Int)
    half = eltype(u)(1) / 2
    ux = u[:, :, :, 1]
    uy = u[:, :, :, 2]
    uz = u[:, :, :, 3]
    ux_s = half .* (ux .+ cat(ux[2:end, :, :], ux[1:1, :, :]; dims=1))
    uy_s = half .* (uy .+ cat(uy[:, 2:end, :], uy[:, 1:1, :]; dims=2))
    uz_s = half .* (uz .+ cat(uz[:, :, 2:end], uz[:, :, 1:1]; dims=3))
    return cat(
        reshape(ux_s, n, n, n, 1),
        reshape(uy_s, n, n, n, 1),
        reshape(uz_s, n, n, n, 1);
        dims=4,
    )
end

"""Staggered (n,n,n,3) → collocated (n,n,n,3)."""
function stag_to_coloc_3d(u::AbstractArray, n::Int)
    half = eltype(u)(1) / 2
    ux_s = u[:, :, :, 1]
    uy_s = u[:, :, :, 2]
    uz_s = u[:, :, :, 3]
    ux = half .* (cat(ux_s[end:end, :, :], ux_s[1:end-1, :, :]; dims=1) .+ ux_s)
    uy = half .* (cat(uy_s[:, end:end, :], uy_s[:, 1:end-1, :]; dims=2) .+ uy_s)
    uz = half .* (cat(uz_s[:, :, end:end], uz_s[:, :, 1:end-1]; dims=3) .+ uz_s)
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

"""
Collocated v0 → staggered, ghosted, divergence-free initial state.

Shared by `ns_apply` and the VJP rollout so both start from the same state.
`project` leaves the ghost layer stale, and the in-place RK stepper evaluates
its first stage on the start state without refreshing the BCs, so the
ghosts are refreshed here.
"""
function initial_state(v0::AbstractArray, setup, psolver, n::Int)
    u = if ndims(v0) == 3
        add_ghosts_2d(coloc_to_stag_2d(v0, n), n)
    else
        add_ghosts_3d(coloc_to_stag_3d(v0, n), n)
    end
    u = project(u, setup; psolver)
    return apply_bc_u(u, zero(eltype(u)), setup)
end

"""2-D forward: v0 (n,n,2) → v_out (n,n,2)."""
function ns_forward_2d(v0::AbstractArray, rhs, setup, psolver,
                       nu::Real, dt::Real, steps::Int, n::Int)
    p = (; viscosity = nu)
    u = initial_state(v0, setup, psolver, n)

    # rhs (create_right_hand_side) applies the periodic BCs to its input and
    # projects its output, so no ghost refresh is needed between stages; the
    # state's own ghost entries go stale but are never read and are stripped
    # at the end.
    for _ in 1:steps
        k1 = rhs(u, p, 0.0)
        k2 = rhs(u .+ (dt/2) .* k1, p, 0.0)
        k3 = rhs(u .+ (dt/2) .* k2, p, 0.0)
        k4 = rhs(u .+ dt .* k3, p, 0.0)
        u = u .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end

    return stag_to_coloc_2d(strip_ghosts_2d(u, n), n)
end

"""3-D forward: v0 (n,n,n,3) → v_out (n,n,n,3)."""
function ns_forward_3d(v0::AbstractArray, rhs, setup, psolver,
                       nu::Real, dt::Real, steps::Int, n::Int)
    p = (; viscosity = nu)
    u = initial_state(v0, setup, psolver, n)

    # See ns_forward_2d for why no per-stage ghost refresh is needed.
    for _ in 1:steps
        k1 = rhs(u, p, 0.0)
        k2 = rhs(u .+ (dt/2) .* k1, p, 0.0)
        k3 = rhs(u .+ (dt/2) .* k2, p, 0.0)
        k4 = rhs(u .+ dt .* k3, p, 0.0)
        u = u .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end

    return stag_to_coloc_3d(strip_ghosts_3d(u, n), n)
end


# ---------------------------------------------------------------------------
# Body force and linear drag (2-D)
# ---------------------------------------------------------------------------

"""
Collocated body force (n,n,2) → projected, ghosted staggered force, or `nothing`.

The force is averaged onto the faces like v0 and projected once, so that
adding it to the projected right-hand side keeps the stages divergence-free.
"""
function staggered_force(force, setup, psolver, n::Int)
    isnothing(force) && return nothing
    f = add_ghosts_2d(coloc_to_stag_2d(Float32.(force), n), n)
    return project(f, setup; psolver)
end

"""
`rhs` plus a constant body force `f` and a linear drag `-drag * u`.

The drag acts on the stage velocity, which is divergence-free, so the sum
stays projected. Returns `rhs` itself when there is neither.
"""
function forced_rhs(rhs, f, drag::Real)
    isnothing(f) && drag == 0 && return rhs
    return function (u, p, t)
        du = rhs(u, p, t)
        if !isnothing(f)
            du = du .+ f
        end
        if drag != 0
            du = du .- eltype(u)(drag) .* u
        end
        return du
    end
end


# ---------------------------------------------------------------------------
# Public API (called from Python via juliacall)
# ---------------------------------------------------------------------------

"""
    ns_apply(v0_np, nu, dt, steps, n, L) -> v_out_np

Forward pass.
  2-D: v0_np (n,n,2) Float32 → returns (n,n,2) Float32
  3-D: v0_np (n,n,n,3) Float32 → returns (n,n,n,3) Float32

Forward-only rollouts use the in-place `solve_unsteady` time stepper (no AD
tape needed), which is several times faster than the non-mutating rollout
used for the VJP primal. `RKMethods.RK44` applies the same per-stage BC +
projection as the projected RHS in `ns_forward_*`, so the two paths agree to
round-off.
"""
function ns_apply(v0_np, nu::Float64, dt::Float64, steps::Int, n::Int, L::Float64;
                  force = nothing, drag::Float64 = 0.0)
    T = Float32
    v0   = T.(v0_np)
    ndim = size(v0, ndims(v0))  # last dim: 2 or 3
    setup, psolver = get_setup_and_psolver(n, L, ndim)

    if !isnothing(force) || drag != 0
        # A body force or drag runs the same RK4 loop as the VJP, so that both
        # integrate the same right-hand side. 2-D only.
        rhs = forced_rhs(
            create_right_hand_side(setup, psolver),
            staggered_force(force, setup, psolver, n),
            drag,
        )
        return T.(ns_forward_2d(v0, rhs, setup, psolver, T(nu), T(dt), steps, n))
    end

    u = initial_state(v0, setup, psolver, n)

    state, _ = solve_unsteady(;
        setup,
        tlims = (T(0), T(steps) * T(dt)),
        start = (; u),
        method = RKMethods.RK44(; T),
        psolver,
        Δt = T(dt),
        params = (; viscosity = T(nu)),
    )

    v_out = if ndim == 2
        stag_to_coloc_2d(strip_ghosts_2d(state.u, n), n)
    else
        stag_to_coloc_3d(strip_ghosts_3d(state.u, n), n)
    end
    return T.(v_out)
end

"""
    ns_vjp(v0_np, cotangent_np, nu, dt, steps, n, L)
        -> (grad_v0, grad_nu, grad_dt, grad_L)

VJP. Shapes match v0_np. grad_L is always 0.0 (L is structural).

grad_v0, grad_dt, and grad_nu all come from a single Zygote reverse-mode
pullback. IncompressibleNavierStokes >= 5 provides the viscosity cotangent
in its `diffusion` rrule (the diffusive term is linear in nu, so this is
exact), which removed the need for the finite-difference fallback (and its
two extra forward rollouts per VJP call) that earlier versions required.
"""
function ns_vjp(v0_np, cotangent_np, nu::Float64, dt::Float64,
                steps::Int, n::Int, L::Float64;
                force = nothing, drag::Float64 = 0.0)
    v0   = Float32.(v0_np)
    cot  = Float32.(cotangent_np)
    ndim = size(v0, ndims(v0))

    setup, psolver = get_setup_and_psolver(n, L, ndim)
    rhs  = forced_rhs(
        create_right_hand_side(setup, psolver),
        staggered_force(force, setup, psolver, n),
        drag,
    )

    if ndim == 2
        fwd = (v, dt_, nu_) -> ns_forward_2d(v, rhs, setup, psolver, nu_, dt_, steps, n)
    else
        fwd = (v, dt_, nu_) -> ns_forward_3d(v, rhs, setup, psolver, nu_, dt_, steps, n)
    end

    _, back = Zygote.pullback(fwd, v0, Float32(dt), Float32(nu))
    grads = back(cot)
    grad_v0  = Float32.(grads[1])
    grad_dt  = Float64(something(grads[2], 0.0))
    grad_nu  = Float64(something(grads[3], 0.0))

    return (
        grad_v0,
        grad_nu,
        grad_dt,
        Float64(0.0),
    )
end
