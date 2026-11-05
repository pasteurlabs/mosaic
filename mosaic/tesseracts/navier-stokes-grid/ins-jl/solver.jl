# mosaic:util
using IncompressibleNavierStokes
using Zygote
using CUDA
using LinearAlgebra
using SparseArrays

const FFTW = IncompressibleNavierStokes.FFTW

# Run on the GPU when the container has one, otherwise on INS.jl's default CPU
# backend. Inputs are copied to the device once per call and results back.
const _ON_GPU = CUDA.functional()
const _BACKEND_KW = _ON_GPU ? (; backend = CUDABackend()) : (;)
to_device(a) = _ON_GPU ? CuArray(a) : a

# mosaic:init
# ---------------------------------------------------------------------------
# Setup cache (indexed by (n, L, ndim), or by the channel's grid); holds the
# current setup only
# ---------------------------------------------------------------------------

const _SETUP_CACHE = Dict{Tuple, Tuple}()

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
                _BACKEND_KW...,
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
                _BACKEND_KW...,
            )
        end
        psolver = psolver_spectral(setup)
        empty!(_SETUP_CACHE)
        _SETUP_CACHE[key] = (setup, psolver)
    end
    return _SETUP_CACHE[key]
end


# mosaic:util
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


# mosaic:physics
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

# mosaic-intervention: hand-written-rk4-rollout
"""
One classic RK4 step of the projected right-hand side `rhs`.

rhs (create_right_hand_side) applies the periodic BCs to its input and
projects its output, so no ghost refresh is needed between stages; the
state's own ghost entries go stale but are never read and are stripped at
the end.
"""
function rk4_step(u, rhs, p, dt::Real)
    k1 = rhs(u, p, 0.0)
    k2 = rhs(u .+ (dt/2) .* k1, p, 0.0)
    k3 = rhs(u .+ (dt/2) .* k2, p, 0.0)
    k4 = rhs(u .+ dt .* k3, p, 0.0)
    return u .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

"""2-D forward: v0 (n,n,2) → v_out (n,n,2)."""
function ns_forward_2d(v0::AbstractArray, rhs, setup, psolver,
                       nu::Real, dt::Real, steps::Int, n::Int)
    p = (; viscosity = nu)
    u = initial_state(v0, setup, psolver, n)
    for _ in 1:steps
        u = rk4_step(u, rhs, p, dt)
    end
    return stag_to_coloc_2d(strip_ghosts_2d(u, n), n)
end

"""3-D forward: v0 (n,n,n,3) → v_out (n,n,n,3)."""
function ns_forward_3d(v0::AbstractArray, rhs, setup, psolver,
                       nu::Real, dt::Real, steps::Int, n::Int)
    p = (; viscosity = nu)
    u = initial_state(v0, setup, psolver, n)
    for _ in 1:steps
        u = rk4_step(u, rhs, p, dt)
    end
    return stag_to_coloc_3d(strip_ghosts_3d(u, n), n)
end

"""
2-D forward that also keeps the velocity after every `every` steps:
v0 (n,n,2) → host array (steps ÷ every, n, n, 2). Not differentiated.
"""
function ns_snapshots_2d(v0::AbstractArray, rhs, setup, psolver,
                         nu::Real, dt::Real, steps::Int, every::Int, n::Int)
    p = (; viscosity = nu)
    u = initial_state(v0, setup, psolver, n)
    frames = Array{eltype(u)}(undef, steps ÷ every, n, n, 2)
    for step in 1:steps
        u = rk4_step(u, rhs, p, dt)
        if step % every == 0
            frames[step ÷ every, :, :, :] = Array(stag_to_coloc_2d(strip_ghosts_2d(u, n), n))
        end
    end
    return frames
end
# mosaic-intervention-end: hand-written-rk4-rollout


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
    f = add_ghosts_2d(coloc_to_stag_2d(to_device(Float32.(force)), n), n)
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


# mosaic:util
# ---------------------------------------------------------------------------
# Native state (2-D): the high-face velocities and the returned velocity
# ---------------------------------------------------------------------------

# Largest amplification the lift applies to a Fourier mode; see
# mosaic_shared/problems/navier_stokes_grid/grid_layout.py.
const MAX_LIFT_GAIN = 32

"""Per-mode inverse of the face-to-centre average along one axis, gain-capped."""
function lift_filter(n::Int)
    transfer = (1 .+ cispi.(-2 .* FFTW.fftfreq(n))) ./ 2
    return to_device([abs(t) * MAX_LIFT_GAIN >= 1 ? 1 / t : zero(t) for t in transfer])
end

"""High-face field (n,n,2) whose average to the cell centres is `d` (n,n,2)."""
function lift_coloc_to_stag_2d(d::AbstractArray, n::Int)
    inv = Zygote.@ignore lift_filter(n)
    ux = real.(FFTW.ifft(FFTW.fft(d[:, :, 1], 1) .* inv, 1))
    uy = real.(FFTW.ifft(FFTW.fft(d[:, :, 2], 2) .* transpose(inv), 2))
    T = eltype(d)
    return cat(reshape(T.(ux), n, n, 1), reshape(T.(uy), n, n, 1); dims=3)
end

# mosaic:physics
"""
Ghosted staggered state continuing from the native `state` (n,n,4) with `v0`.

Only the change from the velocity returned with `state` to `v0` is lifted onto
the faces and projected, so an unchanged `v0` continues exactly.
"""
function continued_state(v0::AbstractArray, state::AbstractArray, setup, psolver, n::Int)
    correction = lift_coloc_to_stag_2d(v0 .- state[:, :, 3:4], n)
    correction = project(add_ghosts_2d(correction, n), setup; psolver)
    u = add_ghosts_2d(state[:, :, 1:2], n) .+ correction
    return apply_bc_u(u, zero(eltype(u)), setup)
end

# mosaic-intervention: hand-written-rk4-rollout
"""2-D forward from `v0`, or continuing from `state`; returns (v_out, native)."""
function ns_forward_native_2d(v0::AbstractArray, state, rhs, setup, psolver,
                              nu::Real, dt::Real, steps::Int, n::Int)
    p = (; viscosity = nu)
    u = if isnothing(state)
        initial_state(v0, setup, psolver, n)
    else
        continued_state(v0, state, setup, psolver, n)
    end
    for _ in 1:steps
        u = rk4_step(u, rhs, p, dt)
    end
    faces = strip_ghosts_2d(u, n)
    v_out = stag_to_coloc_2d(faces, n)
    return v_out, cat(faces, v_out; dims=3)
end
# mosaic-intervention-end: hand-written-rk4-rollout


# ---------------------------------------------------------------------------
# Public API (called from Python via juliacall)
# ---------------------------------------------------------------------------

# mosaic:io
"""
    ns_apply(v0_np, nu, dt, steps, n, L; force, drag, snapshot_every, state,
             return_state) -> v_out_np

Forward pass.
  2-D: v0_np (n,n,2) Float32 → returns (n,n,2) Float32
  3-D: v0_np (n,n,n,3) Float32 → returns (n,n,n,3) Float32

With `snapshot_every > 0` (2-D only), returns the velocity after every
`snapshot_every` steps instead, (steps ÷ snapshot_every, n, n, 2), from the
same RK4 loop as the forced run; the last frame is the final velocity.
With a native `state` or `return_state` (2-D only), returns (v_out, native)
from that loop, continuing from `state` if it is given.

Unforced rollouts use the in-place `solve_unsteady` time stepper (no AD
tape needed), which is several times faster than the non-mutating rollout
used for the VJP primal. `RKMethods.RK44` applies the same per-stage BC +
projection as the projected RHS in `ns_forward_*`, so the two paths agree to
round-off.
"""
function ns_apply(v0_np, nu::Float64, dt::Float64, steps::Int, n::Int, L::Float64;
                  force = nothing, drag::Float64 = 0.0, snapshot_every::Int = 0,
                  state = nothing, return_state::Bool = false)
    T = Float32
    v0   = to_device(T.(v0_np))
    ndim = size(v0, ndims(v0))  # last dim: 2 or 3
    setup, psolver = get_setup_and_psolver(n, L, ndim)

    if !isnothing(state) || return_state
        rhs = forced_rhs(
            create_right_hand_side(setup, psolver),
            staggered_force(force, setup, psolver, n),
            drag,
        )
        st = isnothing(state) ? nothing : to_device(T.(state))
        v_out, native = ns_forward_native_2d(
            v0, st, rhs, setup, psolver, T(nu), T(dt), steps, n,
        )
        return Array(T.(v_out)), Array(T.(native))
    end

    if snapshot_every > 0
        rhs = forced_rhs(
            create_right_hand_side(setup, psolver),
            staggered_force(force, setup, psolver, n),
            drag,
        )
        return ns_snapshots_2d(
            v0, rhs, setup, psolver, T(nu), T(dt), steps, snapshot_every, n
        )
    end

    if !isnothing(force) || drag != 0
        # A body force or drag runs the same RK4 loop as the VJP, so that both
        # integrate the same right-hand side. 2-D only.
        rhs = forced_rhs(
            create_right_hand_side(setup, psolver),
            staggered_force(force, setup, psolver, n),
            drag,
        )
        return Array(T.(ns_forward_2d(v0, rhs, setup, psolver, T(nu), T(dt), steps, n)))
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
    return Array(T.(v_out))
end

# mosaic:grad:v0,state,viscosity,dt:adjoint
"""
    ns_vjp(v0_np, cotangent_np, nu, dt, steps, n, L; force, drag, state,
           return_state, state_cotangent)
        -> (grad_v0, grad_nu, grad_dt, grad_L[, grad_state])

VJP. Shapes match v0_np. grad_L is always 0.0 (L is structural).

grad_v0, grad_dt, and grad_nu all come from a single Zygote reverse-mode
pullback. IncompressibleNavierStokes >= 5 provides the viscosity cotangent
in its `diffusion` rrule (the diffusive term is linear in nu, so this is
exact), which removed the need for the finite-difference fallback (and its
two extra forward rollouts per VJP call) that earlier versions required.
"""
function ns_vjp(v0_np, cotangent_np, nu::Float64, dt::Float64,
                steps::Int, n::Int, L::Float64;
                force = nothing, drag::Float64 = 0.0,
                state = nothing, return_state::Bool = false,
                state_cotangent = nothing)
    v0   = to_device(Float32.(v0_np))
    cot  = to_device(Float32.(cotangent_np))
    ndim = size(v0, ndims(v0))

    setup, psolver = get_setup_and_psolver(n, L, ndim)
    rhs  = forced_rhs(
        create_right_hand_side(setup, psolver),
        staggered_force(force, setup, psolver, n),
        drag,
    )

    if !isnothing(state) || return_state
        # With a native state, also the cotangent of the returned state and
        # the gradient with respect to the incoming one (2-D only).
        st = isnothing(state) ? nothing : to_device(Float32.(state))
        fwd = (v, dt_, nu_, s) -> ns_forward_native_2d(
            v, s, rhs, setup, psolver, nu_, dt_, steps, n,
        )
        _, back = Zygote.pullback(fwd, v0, Float32(dt), Float32(nu), st)
        cot_state = if isnothing(state_cotangent)
            nothing
        else
            to_device(Float32.(state_cotangent))
        end
        grads = back((cot, cot_state))
        grad_state = if isnothing(st)
            nothing
        else
            Array(Float32.(something(grads[4], zero(st))))
        end
        return (
            Array(Float32.(something(grads[1], zero(v0)))),
            Float64(something(grads[3], 0.0)),
            Float64(something(grads[2], 0.0)),
            Float64(0.0),
            grad_state,
        )
    end

    if ndim == 2
        fwd = (v, dt_, nu_) -> ns_forward_2d(v, rhs, setup, psolver, nu_, dt_, steps, n)
    else
        fwd = (v, dt_, nu_) -> ns_forward_3d(v, rhs, setup, psolver, nu_, dt_, steps, n)
    end

    _, back = Zygote.pullback(fwd, v0, Float32(dt), Float32(nu))
    grads = back(cot)
    grad_v0  = Array(Float32.(grads[1]))
    grad_dt  = Float64(something(grads[2], 0.0))
    grad_nu  = Float64(something(grads[3], 0.0))

    return (
        grad_v0,
        grad_nu,
        grad_dt,
        Float64(0.0),
    )
end


# mosaic:init
# ---------------------------------------------------------------------------
# 2-D channel with a cylinder (Schäfer–Turek DFG 2D-1)
# ---------------------------------------------------------------------------
#
# No-slip DirichletBC on the inlet and the walls, PressureBC (p = 0, zero
# normal velocity gradient) on the outlet. Every right-hand side adds the
# inflow profile to the inlet faces (`inlet_field`) and the cylinder's
# Brinkman penalization force (`channel_momentum`).
#
# Fields carry one ghost layer. Cell (i, j) of the shared 0-based staircase is
# pressure index (i + 2, j + 2). u[I, J, 1] is the x face to the right of
# pressure cell I, so u[1, :, 1] is the inlet and u[nx + 1, :, 1] the outlet;
# u[I, J, 2] is the y face above pressure cell J.

function get_channel_setup_and_psolver(nx::Int, ny::Int, Lx::Float64, Ly::Float64)
    key = (:channel, nx, ny, Lx, Ly)
    if !haskey(_SETUP_CACHE, key)
        # Float32 grid coordinates, as in get_setup_and_psolver.
        noslip = DirichletBC()
        setup = Setup(;
            x = (
                LinRange(0f0, Float32(Lx), nx + 1),
                LinRange(0f0, Float32(Ly), ny + 1),
            ),
            boundary_conditions = (; u = ((noslip, PressureBC()), (noslip, noslip))),
            _BACKEND_KW...,
        )
        psolver = psolver_channel(setup)
        empty!(_SETUP_CACHE)
        _SETUP_CACHE[key] = (setup, psolver)
    end
    return _SETUP_CACHE[key]
end

# mosaic:physics
# mosaic-intervention: separable-pressure-solver
"""
Exact pressure solver for the channel, by diagonalizing the Laplacian.

On a uniform grid INS.jl's pressure Laplacian for the channel (zero normal
gradient at the inlet and the walls, p = 0 at the outlet) is the Kronecker sum
of an x and a y operator. Both are read off `laplacian_mat(setup)` and
diagonalized once, so a solve is four dense matrix products. They run in
Float64, like INS.jl's CPU `psolver_direct`.
"""
function psolver_channel(setup)
    (; Np, Ip) = setup
    nx, ny = Np
    L = Float64.(sparse(laplacian_mat(setup)))
    cell = LinearIndices((nx, ny))
    # The y operator has zero-gradient ends, so its diagonal is minus the sum
    # of its off-diagonal row. The rest of L's diagonal belongs to x.
    Ay = [j == k ? 0.0 : L[cell[1, j], cell[1, k]] for j in 1:ny, k in 1:ny]
    Ay[diagind(Ay)] .= -vec(sum(Ay; dims = 2))
    Ax = [i == k ? 0.0 : L[cell[i, 1], cell[k, 1]] for i in 1:nx, k in 1:nx]
    Ax[diagind(Ax)] .= [L[cell[i, 1], cell[i, 1]] for i in 1:nx] .- Ay[1, 1]
    # The Float32 grid spacings differ by round-off, which leaves a mismatch
    # of a few 1e-6 of L's largest entry.
    mismatch = maximum(abs, kron(sparse(1.0I, ny, ny), sparse(Ax)) +
                            kron(sparse(Ay), sparse(1.0I, nx, nx)) - L)
    mismatch < 1e-4 * maximum(abs, L) ||
        error("the channel's pressure Laplacian is not separable (mismatch $mismatch)")
    λx, Vx = eigen(Symmetric(Ax))
    λy, Vy = eigen(Symmetric(Ay))
    Vx, Vy, inv_λ = to_device(Vx), to_device(Vy), to_device(1 ./ (λx .+ λy'))
    function psolve!(p)
        f = Float64.(view(p, Ip))
        p[Ip] .= eltype(p).(Vx * ((Vx' * f * Vy) .* inv_λ) * Vy')
        p
    end
end
# mosaic-intervention-end: separable-pressure-solver

# mosaic-intervention: brinkman-cylinder
"""
Penalization time η of the Brinkman force -χ u / η.

The force is integrated explicitly. RK4 is stable for dt / η up to about 2.8
on the penalty alone, and at η = dt / 2.5 a run at 20 cells per diameter
diverged. η = dt leaves room for the diffusion on the same faces.
"""
penalization_time(dt) = dt
# mosaic-intervention-end: brinkman-cylinder

# mosaic:init
"""
Device arrays for one channel run: the coefficient χ / η of the penalization
force, the weights that sum it into the force on the cylinder, the inlet
selector and the fluid cells.

χ is 1 on every face touching a cell of `solid` (nx, ny), the staircase the
other grid solvers share, and 0 elsewhere.
"""
function channel_fields(solid::AbstractMatrix{Bool}, setup, eta::Float64, dx::Float64)
    T = Float32
    S = falses(setup.N)
    S[2:end-1, 2:end-1] .= solid
    chi = zeros(T, setup.N..., 2)
    chi[1:end-1, :, 1] .= S[1:end-1, :] .| S[2:end, :]
    chi[:, 1:end-1, 2] .= S[:, 1:end-1] .| S[:, 2:end]
    weight = T(dx^2 / eta) .* chi
    inlet = zeros(T, setup.N..., 2)
    inlet[1, 2:end-1, 1] .= 1
    return (;
        chi_eta = to_device(chi ./ T(eta)),
        weight_x = to_device(cat(weight[:, :, 1], zero(weight[:, :, 2]); dims = 3)),
        weight_y = to_device(cat(zero(weight[:, :, 1]), weight[:, :, 2]; dims = 3)),
        inlet = to_device(inlet),
        fluid = to_device(T.(.!solid)),
    )
end

# mosaic:util
# mosaic-intervention: differentiable-inlet
"""
The inflow profile on the inlet faces, zero elsewhere.

`channel_momentum` adds it to the state after INS.jl's no-slip inlet BC, where
Zygote sees it. INS.jl's rrule for `apply_bc_u` gives a DirichletBC value no
tangent.
"""
function inlet_field(inlet, profile)
    pad = zero(profile[1:1])
    return inlet .* reshape(vcat(pad, profile, pad), 1, :, 1)
end
# mosaic-intervention-end: differentiable-inlet

# mosaic:physics
"""
Momentum right-hand side before the projection: that of
`create_right_hand_side`, with the inflow profile `uin` and the cylinder's
penalization force.
"""
function channel_momentum(u, uin, chi_eta, setup, nu)
    T = eltype(u)
    # mosaic-intervention: differentiable-inlet
    u = apply_bc_u(u, zero(T), setup) .+ uin
    # mosaic-intervention-end: differentiable-inlet
    # mosaic-intervention: brinkman-cylinder
    f = navierstokes((; u), zero(T); setup, viscosity = nu).u .- chi_eta .* u
    # mosaic-intervention-end: brinkman-cylinder
    return apply_bc_u(f, zero(T), setup; dudt = true)
end

# mosaic:util
"""
Cell-centered (nx,ny,2) → ghosted staggered channel field. The outlet face
takes the last cell's value; the inlet, wall and ghost faces are zero.
"""
function channel_coloc_to_stag(v::AbstractArray)
    half = eltype(v)(1) / 2
    vx, vy = v[:, :, 1], v[:, :, 2]
    ux = half .* (vx .+ vcat(vx[2:end, :], vx[end:end, :]))  # (nx, ny)
    uy = half .* (vy[:, 1:end-1] .+ vy[:, 2:end])            # (nx, ny - 1)
    row(a) = zero(a[1:1, :])
    col(a) = zero(a[:, 1:1])
    ux = vcat(row(ux), ux, row(ux))
    ux = hcat(col(ux), ux, col(ux))
    uy = vcat(row(uy), uy, row(uy))
    uy = hcat(col(uy), uy, col(uy), col(uy))
    return cat(ux, uy; dims = 3)
end

"""Ghosted staggered channel field with its boundary values set → cell centers (nx,ny,2)."""
function channel_stag_to_coloc(u::AbstractArray, nx::Int, ny::Int)
    half = eltype(u)(1) / 2
    ux = half .* (u[1:nx, 2:ny+1, 1] .+ u[2:nx+1, 2:ny+1, 1])
    uy = half .* (u[2:nx+1, 1:ny, 2] .+ u[2:nx+1, 2:ny+1, 2])
    return cat(ux, uy; dims = 3)
end

# mosaic:physics
# mosaic-intervention: hand-written-rk4-rollout
"""
Channel forward: v0 (nx,ny,2) at the cell centers → the velocity (nx,ny,2) at
the cell centers, zero in solid cells; the force on the cylinder after every
step (steps, 2); and the final staggered state.

The force is the penalization's integral ∑ χ u / η Δx², the momentum the
cylinder takes out of the fluid per unit time. Shared by `channel_apply` and
`channel_vjp`, so the VJP differentiates exactly what apply returns.
"""
function channel_forward(v0::AbstractArray, profile, nu::Real, dt::Real,
                         steps::Int, setup, psolver, fields)
    T = eltype(v0)
    nx, ny = setup.Np
    uin = inlet_field(fields.inlet, profile)
    rhs(u) = project(channel_momentum(u, uin, fields.chi_eta, setup, nu), setup; psolver)
    u = channel_coloc_to_stag(v0)
    u = project(apply_bc_u(u, zero(T), setup) .+ uin, setup; psolver)
    force = Zygote.Buffer(zeros(T, steps, 2))
    for k in 1:steps
        k1 = rhs(u)
        k2 = rhs(u .+ (dt/2) .* k1)
        k3 = rhs(u .+ (dt/2) .* k2)
        k4 = rhs(u .+ dt .* k3)
        u = u .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        force[k, 1] = dot(fields.weight_x, u)
        force[k, 2] = dot(fields.weight_y, u)
    end
    u_bc = apply_bc_u(u, zero(T), setup) .+ uin
    velocity = channel_stag_to_coloc(u_bc, nx, ny) .* fields.fluid
    return velocity, copy(force), u
end
# mosaic-intervention-end: hand-written-rk4-rollout

"""
Kinematic pressure (nx, ny) of the channel state `u`: the pressure that the
projection of its right-hand side solves for, zero at the outlet.
"""
function channel_pressure(u, profile, nu::Real, setup, psolver, fields)
    T = eltype(u)
    uin = inlet_field(fields.inlet, profile)
    du = channel_momentum(u, uin, fields.chi_eta, setup, nu)
    p = poisson(psolver, scalewithvolume(divergence(du, setup), setup))
    return apply_bc_p(p, zero(T), setup)[setup.Ip]
end

# mosaic:io
"""
    channel_apply(v0_np, solid_np, profile_np, nu, dt, steps, Lx, Ly)
        -> (velocity, pressure, body_force)

Forward pass on the DFG channel of size Lx × Ly. `v0_np` (nx,ny,2) and the
returned velocity are at the cell centers, `solid_np` (nx,ny) is the
cylinder's staircase and `profile_np` (ny,) the inlet velocity. Returns the
pressure (nx,ny) and the force on the cylinder after every step (steps,2).
"""
function channel_apply(v0_np, solid_np, profile_np, nu::Float64, dt::Float64,
                       steps::Int, Lx::Float64, Ly::Float64)
    T = Float32
    nx, ny = size(solid_np)
    setup, psolver = get_channel_setup_and_psolver(nx, ny, Lx, Ly)
    fields = channel_fields(Bool.(solid_np), setup, penalization_time(dt), Lx / nx)
    profile = to_device(T.(profile_np))
    velocity, force, u = channel_forward(
        to_device(T.(v0_np)), profile, T(nu), T(dt), steps, setup, psolver, fields,
    )
    pressure = channel_pressure(u, profile, T(nu), setup, psolver, fields)
    return Array(velocity), Array(pressure), force
end

# mosaic:grad:v0,inflow_profile,viscosity,dt:adjoint
"""
    channel_vjp(v0_np, solid_np, profile_np, nu, dt, steps, Lx, Ly,
                ct_result_np, ct_force_np)
        -> (grad_v0, grad_profile, grad_nu, grad_dt)

VJP of `channel_apply`'s velocity and force; the pressure carries no
gradient. η stays at the given dt, so grad_dt holds the penalization fixed.
"""
function channel_vjp(v0_np, solid_np, profile_np, nu::Float64, dt::Float64,
                     steps::Int, Lx::Float64, Ly::Float64, ct_result_np, ct_force_np)
    T = Float32
    nx, ny = size(solid_np)
    setup, psolver = get_channel_setup_and_psolver(nx, ny, Lx, Ly)
    fields = channel_fields(Bool.(solid_np), setup, penalization_time(dt), Lx / nx)
    v0 = to_device(T.(v0_np))
    profile = to_device(T.(profile_np))
    fwd = (v, prof, dt_, nu_) ->
        channel_forward(v, prof, nu_, dt_, steps, setup, psolver, fields)[1:2]
    _, back = Zygote.pullback(fwd, v0, profile, T(dt), T(nu))
    grads = back((to_device(T.(ct_result_np)), T.(ct_force_np)))
    return (
        Array(something(grads[1], zero(v0))),
        Array(something(grads[2], zero(profile))),
        Float64(something(grads[4], 0.0)),
        Float64(something(grads[3], 0.0)),
    )
end
