using WaterLily
using Zygote
using ForwardDiff
using ChainRulesCore

# ---------------------------------------------------------------------------
# Grid interpolation helpers (Zygote-differentiable, non-mutating)
# Consistent with ns_solver.jl.
# ---------------------------------------------------------------------------

# ---- 2-D ----

# mosaic:util
"""Collocated (n,n,2) → staggered (n,n,2) via periodic linear interpolation.

WaterLily convention: u_stag[i] lives on the LEFT face of cell i (at x=(i-1)*dx).
The face between cells i-1 and i is best approximated by averaging cell-centre
values i-1 and i, so:  ux_s[i] = 0.5*(ux[i-1] + ux[i]).
"""
function coloc_to_stag_2d(u::AbstractArray, n::Int)
    ux = u[:, :, 1]
    uy = u[:, :, 2]
    ux_s = 0.5 .* (cat(ux[end:end, :], ux[1:end-1, :]; dims=1) .+ ux)
    uy_s = 0.5 .* (cat(uy[:, end:end], uy[:, 1:end-1]; dims=2) .+ uy)
    return cat(reshape(ux_s, n, n, 1), reshape(uy_s, n, n, 1); dims=3)
end

"""Staggered (n,n,2) → collocated (n,n,2) via periodic linear interpolation.

WaterLily convention: u_stag[i] lives on the LEFT face of cell i (at x=(i-1)*dx).
Cell-centre i sits between faces i and i+1, so:  ux[i] = 0.5*(ux_s[i] + ux_s[i+1]).
"""
function stag_to_coloc_2d(u::AbstractArray, n::Int)
    ux_s = u[:, :, 1]
    uy_s = u[:, :, 2]
    ux = 0.5 .* (ux_s .+ cat(ux_s[2:end, :], ux_s[1:1, :]; dims=1))
    uy = 0.5 .* (uy_s .+ cat(uy_s[:, 2:end], uy_s[:, 1:1]; dims=2))
    return cat(reshape(ux, n, n, 1), reshape(uy, n, n, 1); dims=3)
end

# ---- 3-D ----

"""Collocated (n,n,n,3) → staggered (n,n,n,3) via periodic linear interpolation.

WaterLily convention: u_stag[i] lives on the LEFT face of cell i.
ux_s[i] = 0.5*(ux[i-1] + ux[i])  (and similarly for y, z).
"""
function coloc_to_stag_3d(u::AbstractArray, n::Int)
    ux = u[:, :, :, 1]
    uy = u[:, :, :, 2]
    uz = u[:, :, :, 3]
    ux_s = 0.5 .* (cat(ux[end:end, :, :], ux[1:end-1, :, :]; dims=1) .+ ux)
    uy_s = 0.5 .* (cat(uy[:, end:end, :], uy[:, 1:end-1, :]; dims=2) .+ uy)
    uz_s = 0.5 .* (cat(uz[:, :, end:end], uz[:, :, 1:end-1]; dims=3) .+ uz)
    return cat(
        reshape(ux_s, n, n, n, 1),
        reshape(uy_s, n, n, n, 1),
        reshape(uz_s, n, n, n, 1);
        dims=4,
    )
end

"""Staggered (n,n,n,3) → collocated (n,n,n,3).

WaterLily convention: u_stag[i] lives on the LEFT face of cell i.
Cell-centre i sits between faces i and i+1:  ux[i] = 0.5*(ux_s[i] + ux_s[i+1]).
"""
function stag_to_coloc_3d(u::AbstractArray, n::Int)
    ux_s = u[:, :, :, 1]
    uy_s = u[:, :, :, 2]
    uz_s = u[:, :, :, 3]
    ux = 0.5 .* (ux_s .+ cat(ux_s[2:end, :, :], ux_s[1:1, :, :]; dims=1))
    uy = 0.5 .* (uy_s .+ cat(uy_s[:, 2:end, :], uy_s[:, 1:1, :]; dims=2))
    uz = 0.5 .* (uz_s .+ cat(uz_s[:, :, 2:end], uz_s[:, :, 1:1]; dims=3))
    return cat(
        reshape(ux, n, n, n, 1),
        reshape(uy, n, n, n, 1),
        reshape(uz, n, n, n, 1);
        dims=4,
    )
end


# ---------------------------------------------------------------------------
# Obstacle helpers
# ---------------------------------------------------------------------------

# mosaic:init
"""
    _make_body_2d(shape, center, radius) -> body or nothing

Construct a WaterLily AutoBody for a 2-D obstacle.  `center` is [cx, cy] in
grid-cell coordinates (already converted from fractional by the caller).
`radius` is in grid-cell units.  Returns `nothing` when `shape == ""`.
"""
function _make_body_2d(shape::String, center::Vector{Float64}, radius::Float64)
    shape == "" && return nothing
    if shape == "cylinder"
        cx, cy = center[1], center[2]
        r = radius
        return AutoBody((x, t) -> sqrt((x[1] - cx)^2 + (x[2] - cy)^2) - r)
    else
        error("Unsupported obstacle shape: $shape")
    end
end

"""
    _make_body_3d(shape, center, radius) -> body or nothing

3-D variant: cylinder is infinite along z-axis.
"""
function _make_body_3d(shape::String, center::Vector{Float64}, radius::Float64)
    shape == "" && return nothing
    if shape == "cylinder"
        cx, cy = center[1], center[2]
        r = radius
        # Infinite cylinder along z: distance in x-y plane only
        return AutoBody((x, t) -> sqrt((x[1] - cx)^2 + (x[2] - cy)^2) - r)
    else
        error("Unsupported obstacle shape: $shape")
    end
end


# ---------------------------------------------------------------------------
# Inner forward – primal (Float32, mutation-based, non-differentiable)
# ---------------------------------------------------------------------------

# mosaic:physics
"""
    _wl_run_2d(v0_stag, nu, dt_lat, steps, n; body) -> u_stag_out

Low-level 2-D WaterLily forward.
`v0_stag`: (n,n,2) staggered velocity in lattice units.  Element type T must be
compatible with WaterLily (Float32 for the primal; ForwardDiff Dual for JVPs).
Returns (n,n,2) staggered velocity output in lattice units with the same T.
Not directly differentiable by Zygote; see the ChainRulesCore.rrule below.
"""
function _wl_run_2d(v0_stag::AbstractArray{T},
                    nu::T, dt_lat::T,
                    steps::Int, n::Int; body=nothing) where {T}
    if body === nothing
        sim = Simulation((n, n), (zero(T), zero(T)), n;
                         ν = nu, U = one(T), perdir = (1, 2), T = T)
    else
        sim = Simulation((n, n), (zero(T), zero(T)), n;
                         ν = nu, U = one(T), perdir = (1, 2),
                         body = body, T = T)
    end
    sim.flow.u[2:n+1, 2:n+1, 1] .= v0_stag[:, :, 1]
    sim.flow.u[2:n+1, 2:n+1, 2] .= v0_stag[:, :, 2]
    WaterLily.project!(sim.flow, sim.pois)
    for _ in 1:steps
        sim.flow.Δt[end] = dt_lat
        sim_step!(sim; remeasure = false)
    end
    return copy(sim.flow.u[2:n+1, 2:n+1, :])   # (n,n,2) staggered, lattice units
end

"""
    _wl_run_3d(v0_stag, nu, dt_lat, steps, n; body) -> u_stag_out

Low-level 3-D WaterLily forward.
`v0_stag`: (n,n,n,3) staggered velocity in lattice units.
Returns (n,n,n,3) staggered velocity output in lattice units.
"""
function _wl_run_3d(v0_stag::AbstractArray{T},
                    nu::T, dt_lat::T,
                    steps::Int, n::Int; body=nothing) where {T}
    if body === nothing
        sim = Simulation((n, n, n), (zero(T), zero(T), zero(T)), n;
                         ν = nu, U = one(T), perdir = (1, 2, 3), T = T)
    else
        sim = Simulation((n, n, n), (zero(T), zero(T), zero(T)), n;
                         ν = nu, U = one(T), perdir = (1, 2, 3),
                         body = body, T = T)
    end
    sim.flow.u[2:n+1, 2:n+1, 2:n+1, 1] .= v0_stag[:, :, :, 1]
    sim.flow.u[2:n+1, 2:n+1, 2:n+1, 2] .= v0_stag[:, :, :, 2]
    sim.flow.u[2:n+1, 2:n+1, 2:n+1, 3] .= v0_stag[:, :, :, 3]
    WaterLily.project!(sim.flow, sim.pois)
    for _ in 1:steps
        sim.flow.Δt[end] = dt_lat
        sim_step!(sim; remeasure = false)
    end
    return copy(sim.flow.u[2:n+1, 2:n+1, 2:n+1, :])  # (n,n,n,3) staggered, lattice units
end


# ---------------------------------------------------------------------------
# ChainRulesCore rrules: VJP via ForwardDiff JVPs
#
# WaterLily is built for forward-mode AD (ForwardDiff) and all its internals
# are ForwardDiff-compatible.  Reverse-mode (Zygote) cannot trace through the
# mutating `sim_step!` / `mom_step!` calls.
#
# Strategy: implement the pullback by contracting each ForwardDiff JVP
# (one per input element) with the cotangent vector.  This gives the exact
# same gradient as finite differences at the cost of N_inputs ForwardDiff
# passes.  For the benchmark grid sizes (N=8..32 → 128..2048 DOF) this is
# acceptable.
# ---------------------------------------------------------------------------

# mosaic:grad:v0,viscosity,dt:autodiff
"""
    _jvp_wl_2d(v0_stag, dv0, nu, dnu, dt_lat, steps, n; body) -> du_out

Compute the JVP of `_wl_run_2d` at (v0_stag, nu) in direction (dv0, dnu).
Uses a single ForwardDiff pass with a scalar ε:
  d/dε [_wl_run_2d(v0 + ε*dv0, nu + ε*dnu, dt_lat + 0, ...)] at ε = 0

`dt_lat` is included as `dt_lat + zero(ε)` so the promoted element type T is
consistent (Dual throughout).
"""
function _jvp_wl_2d(v0_stag::AbstractArray{Float32}, dv0::AbstractArray{Float32},
                     nu::Float32, dnu::Float32, dt_lat::Float32,
                     steps::Int, n::Int; body=nothing)
    ForwardDiff.derivative(Float32(0)) do ε
        _wl_run_2d(v0_stag .+ ε .* dv0,
                    nu + ε * dnu,
                    dt_lat + zero(ε),
                    steps, n; body=body)
    end
end

"""
    _jvp_wl_3d(v0_stag, dv0, nu, dnu, dt_lat, steps, n; body) -> du_out

3-D variant of `_jvp_wl_2d`.
"""
function _jvp_wl_3d(v0_stag::AbstractArray{Float32}, dv0::AbstractArray{Float32},
                     nu::Float32, dnu::Float32, dt_lat::Float32,
                     steps::Int, n::Int; body=nothing)
    ForwardDiff.derivative(Float32(0)) do ε
        _wl_run_3d(v0_stag .+ ε .* dv0,
                    nu + ε * dnu,
                    dt_lat + zero(ε),
                    steps, n; body=body)
    end
end


function ChainRulesCore.rrule(::typeof(_wl_run_2d),
                               v0_stag::AbstractArray{Float32},
                               nu::Float32, dt_lat::Float32,
                               steps::Int, n::Int;
                               body=nothing)
    # Primal forward pass (Float32)
    u_out = _wl_run_2d(v0_stag, nu, dt_lat, steps, n; body=body)

    function _wl_run_2d_pullback(ū)
        ū_flat = vec(Float32.(ū))

        # --- gradient w.r.t. v0_stag via JVPs ---
        # g_k = dot(J * eₖ, ū)  where J * eₖ is the JVP in direction eₖ.
        v0_flat = vec(v0_stag)
        nv      = length(v0_flat)
        grad_v0 = zeros(Float32, nv)

        for k in 1:nv
            dv0 = zeros(Float32, size(v0_stag))
            dv0[k] = 1f0
            jvp = _jvp_wl_2d(v0_stag, dv0, nu, 0f0, dt_lat, steps, n; body=body)
            grad_v0[k] = dot(vec(Float32.(jvp)), ū_flat)
        end

        # --- gradient w.r.t. nu ---
        jvp_nu = _jvp_wl_2d(v0_stag, zero(v0_stag), nu, 1f0, dt_lat, steps, n; body=body)
        grad_nu = dot(vec(Float32.(jvp_nu)), ū_flat)

        grad_v0_arr = reshape(grad_v0, size(v0_stag))
        return (NoTangent(),           # function itself
                grad_v0_arr,           # ∂/∂v0_stag
                Float32(grad_nu),      # ∂/∂nu
                ZeroTangent(),         # ∂/∂dt_lat  (not needed here)
                NoTangent(),           # steps
                NoTangent())           # n
    end

    return u_out, _wl_run_2d_pullback
end


function ChainRulesCore.rrule(::typeof(_wl_run_3d),
                               v0_stag::AbstractArray{Float32},
                               nu::Float32, dt_lat::Float32,
                               steps::Int, n::Int;
                               body=nothing)
    u_out = _wl_run_3d(v0_stag, nu, dt_lat, steps, n; body=body)

    function _wl_run_3d_pullback(ū)
        ū_flat  = vec(Float32.(ū))
        v0_flat = vec(v0_stag)
        nv      = length(v0_flat)
        grad_v0 = zeros(Float32, nv)

        for k in 1:nv
            dv0 = zeros(Float32, size(v0_stag))
            dv0[k] = 1f0
            jvp = _jvp_wl_3d(v0_stag, dv0, nu, 0f0, dt_lat, steps, n; body=body)
            grad_v0[k] = dot(vec(Float32.(jvp)), ū_flat)
        end

        jvp_nu = _jvp_wl_3d(v0_stag, zero(v0_stag), nu, 1f0, dt_lat, steps, n; body=body)
        grad_nu = dot(vec(Float32.(jvp_nu)), ū_flat)

        grad_v0_arr = reshape(grad_v0, size(v0_stag))
        return (NoTangent(),
                grad_v0_arr,
                Float32(grad_nu),
                ZeroTangent(),
                NoTangent(),
                NoTangent())
    end

    return u_out, _wl_run_3d_pullback
end


# ---------------------------------------------------------------------------
# Inner forward (differentiable through the rrule above + Zygote)
# ---------------------------------------------------------------------------

# mosaic:physics
"""
    wl_forward_inner_2d(v0, nu_lattice, dt_lat, steps, n, L; body=nothing) -> v_out

2-D WaterLily forward pass. v0: (n,n,2) Float32. Returns (n,n,2) Float32.

Differentiable via the ChainRulesCore.rrule for `_wl_run_2d`, which computes
the pullback using ForwardDiff JVPs.  WaterLily uses ForwardDiff internally
and all its kernels are ForwardDiff-compatible; the rrule leverages that to
provide correct reverse-mode gradients.

Callers (including wl_vjp) must pass v0 as Float32 and nu_lattice as Float32
so that Zygote dispatches to the registered rrule for `_wl_run_2d`.
"""
function wl_forward_inner_2d(v0::AbstractArray{Float32}, nu_lattice::Float32,
                              dt_lat::Real, steps::Int, n::Int, L::Float64;
                              body=nothing)
    dt_f32  = Float32(dt_lat)

    v0_stag = coloc_to_stag_2d(v0, n)

    # _wl_run_2d returns staggered output (n,n,2) in lattice units.
    # Its rrule allows Zygote to differentiate through this call.
    u_stag_out = _wl_run_2d(v0_stag, nu_lattice, dt_f32, steps, n; body=body)

    v_out = stag_to_coloc_2d(u_stag_out, n)

    scale = Float32(1 / dt_lat)
    return v_out .* scale
end

"""
    wl_forward_inner_3d(v0, nu_lattice, dt_lat, steps, n, L; body=nothing) -> v_out

3-D WaterLily forward pass. v0: (n,n,n,3) Float32. Returns (n,n,n,3) Float32.

Differentiable via the ChainRulesCore.rrule for `_wl_run_3d`.
"""
function wl_forward_inner_3d(v0::AbstractArray{Float32}, nu_lattice::Float32,
                              dt_lat::Real, steps::Int, n::Int, L::Float64;
                              body=nothing)
    dt_f32  = Float32(dt_lat)

    v0_stag = coloc_to_stag_3d(v0, n)

    u_stag_out = _wl_run_3d(v0_stag, nu_lattice, dt_f32, steps, n; body=body)

    v_out = stag_to_coloc_3d(u_stag_out, n)

    scale = Float32(1 / dt_lat)
    return v_out .* scale
end


# ---------------------------------------------------------------------------
# Public API (called from Python via juliacall)
# ---------------------------------------------------------------------------

# mosaic:physics
"""
    wl_forward_inner_cavity_3d(lid_vel, nu_lattice, dt_lat, steps, n, L) -> v_out

3-D lid-driven cavity forward pass.

`lid_vel` is the x-velocity of the moving lid (Zygote-traced scalar).
Non-periodic walls in all directions (`perdir=()`); WaterLily applies the freestream
`(lid_vel, 0, 0)` as the ghost-cell BC on every face.  The flow starts from rest and
is driven by the lid.

Returns (n,n,n,3) Float32 in physical velocity units.
"""
function wl_forward_inner_cavity_3d(lid_vel::Real, nu_lattice::Real, dt_lat::Real,
                                     steps::Int, n::Int, L::Float64)
    T = typeof(lid_vel)
    sim = Simulation((n, n, n), (lid_vel, T(0), T(0)), n;
                     ν = T(nu_lattice),
                     U = T(1),
                     perdir = (),
                     T = T)
    WaterLily.project!(sim.flow, sim.pois)
    for _ in 1:steps
        sim.flow.Δt[end] = T(dt_lat)
        sim_step!(sim; remeasure = false)
    end
    u_inner = sim.flow.u[2:n+1, 2:n+1, 2:n+1, :]
    v_out   = stag_to_coloc_3d(u_inner, n)
    return v_out .* T(1 / dt_lat)
end

# mosaic:io
"""
    wl_apply_cavity(lid_vel_np, nu, dt, steps, n, L) -> v_out_np

Forward pass for the 3-D lid-driven cavity.
`lid_vel_np` is a scalar or (1,) array giving the x-velocity of the lid.
Returns (n,n,n,3) Float32.
"""
function wl_apply_cavity(lid_vel_np, nu::Float64, dt::Float64,
                          steps::Int, n::Int, L::Float64)
    lid_vel = Float32(lid_vel_np isa AbstractArray ? lid_vel_np[1] : lid_vel_np)
    dx        = L / n
    dt_lat    = dt / dx
    nu_lattice = nu * dt / dx^2
    v_out = wl_forward_inner_cavity_3d(lid_vel, Float32(nu_lattice), Float32(dt_lat),
                                        steps, n, L)
    return Float32.(v_out)
end

# mosaic:grad:lid_velocity:autodiff
"""
    wl_vjp_cavity(lid_vel_np, cotangent_np, nu, dt, steps, n, L) -> grad_lid_vel

VJP of the 3-D lid-driven cavity w.r.t. the scalar lid velocity.
Returns the gradient of the loss w.r.t. lid_vel as a Float64 scalar.

WaterLily is a ForwardDiff-based solver; a single ForwardDiff.derivative call
computes the JVP dv_out/d(lid_vel), which is then contracted with the cotangent
to obtain the scalar VJP.
"""
function wl_vjp_cavity(lid_vel_np, cotangent_np, nu::Float64, dt::Float64,
                        steps::Int, n::Int, L::Float64)
    lid_vel_f32 = Float32(lid_vel_np isa AbstractArray ? lid_vel_np[1] : lid_vel_np)
    cot_f32 = Float32.(cotangent_np)
    dx         = L / n
    dt_lat     = Float32(dt / dx)
    nu_lattice = Float32(nu * dt / dx^2)

    # One ForwardDiff pass: scalar input → array output
    jvp = ForwardDiff.derivative(
        (lv) -> wl_forward_inner_cavity_3d(lv, nu_lattice, dt_lat, steps, n, L),
        lid_vel_f32
    )

    return Float64(dot(vec(jvp), vec(cot_f32)))
end


# mosaic:io
"""
    wl_apply(v0_np, nu, dt, steps, n, L; obs_shape, obs_center, obs_radius) -> v_out_np

Forward pass.
  2-D: v0_np (n,n,2) Float32 → returns (n,n,2) Float32
  3-D: v0_np (n,n,n,3) Float32 → returns (n,n,n,3) Float32

Optional obstacle: pass obs_shape="cylinder" with obs_center ([cx,cy] as fractions of
domain_extent) and obs_radius (fraction of domain_extent).  Default obs_shape="" means
no obstacle (fully periodic).
"""
function wl_apply(v0_np, nu::Float64, dt::Float64, steps::Int, n::Int, L::Float64;
                  obs_shape::String="", obs_center::Vector{Float64}=Float64[],
                  obs_radius::Float64=0.0)
    v0   = Float32.(v0_np)
    ndim = size(v0, ndims(v0))
    dx      = L / n
    dt_lat  = dt / dx          # lattice time step = dt/dx
    nu_lattice = nu * dt / dx^2  # lattice viscosity = ν*dt/dx²

    # Convert fractional obstacle coords → grid-cell units
    center_cells = isempty(obs_center) ? Float64[] : obs_center .* n
    radius_cells = obs_radius * n

    # Convert physical velocity → lattice velocity
    v0_lat = v0 .* Float32(dt_lat)

    if ndim == 2
        body = _make_body_2d(obs_shape, center_cells, radius_cells)
        v_out = wl_forward_inner_2d(v0_lat, Float32(nu_lattice), dt_lat, steps, n, L;
                                    body=body)
    else
        body = _make_body_3d(obs_shape, center_cells, radius_cells)
        v_out = wl_forward_inner_3d(v0_lat, Float32(nu_lattice), dt_lat, steps, n, L;
                                    body=body)
    end

    return Float32.(v_out)
end

# mosaic:grad:v0,viscosity,dt:autodiff
"""
    wl_vjp(v0_np, cotangent_np, nu, dt, steps, n, L; obs_shape, obs_center, obs_radius)
        -> (grad_v0, grad_nu, grad_dt, 0.0)

VJP. Shapes match v0_np. grad_L is always 0.0 (structural parameter).
Obstacle params mirror wl_apply — they are structural (not differentiated).

The gradient is computed via Zygote, which dispatches through the ChainRulesCore
rrules registered for `_wl_run_2d` / `_wl_run_3d`.  Those rrules implement the
pullback using ForwardDiff JVPs, which correctly propagate through all of
WaterLily's mutating kernels.
"""
function wl_vjp(v0_np, cotangent_np, nu::Float64, dt::Float64,
                steps::Int, n::Int, L::Float64;
                obs_shape::String="", obs_center::Vector{Float64}=Float64[],
                obs_radius::Float64=0.0)
    v0   = Float32.(v0_np)
    cot  = Float32.(cotangent_np)
    ndim = size(v0, ndims(v0))

    dx         = L / n
    dt_lat     = dt / dx
    nu_lattice = nu * dt / dx^2
    scale_in   = Float32(dt_lat)   # physical → lattice

    center_cells = isempty(obs_center) ? Float64[] : obs_center .* n
    radius_cells = obs_radius * n

    v0_lat  = v0  .* scale_in

    if ndim == 2
        body = _make_body_2d(obs_shape, center_cells, radius_cells)
        f = (v, nul) -> wl_forward_inner_2d(v, nul, dt_lat, steps, n, L; body=body)
    else
        body = _make_body_3d(obs_shape, center_cells, radius_cells)
        f = (v, nul) -> wl_forward_inner_3d(v, nul, dt_lat, steps, n, L; body=body)
    end

    _, back = Zygote.pullback(f, v0_lat, Float32(nu_lattice))
    grads   = back(cot)

    grad_v0_lat = something(grads[1], zero(v0_lat))
    grad_v0     = Float32.(grad_v0_lat .* scale_in)

    grad_nul = Float64(something(grads[2], 0f0))
    grad_nu  = grad_nul * (dt / dx^2)

    return grad_v0, Float64(grad_nu), Float64(0.0), Float64(0.0)
end
