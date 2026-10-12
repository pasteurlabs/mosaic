# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

module MosaicWaterLily
using WaterLily

# Native WaterLily grid has unit spacing: velocity is converted to cells/s.
# Tangential ghost reflection implements no-slip (upstream defaults to free-slip).
function walls!(u)
    u[1:2, :, 1] .= 0
    u[end, :, 1] .= 0
    u[:, 1:2, 2] .= 0
    u[:, end, 2] .= 0
    u[:, 1, 1] .= -u[:, 2, 1]
    u[:, end, 1] .= -u[:, end-1, 1]
    u[1, :, 2] .= -u[2, :, 2]
    u[end, :, 2] .= -u[end-1, :, 2]
    return u
end

function scalarwalls!(c)
    c[1, :] .= c[2, :]
    c[end, :] .= c[end-1, :]
    c[:, 1] .= c[:, 2]
    c[:, end] .= c[:, end-1]
end

# Adapter-owned conservative upwind transport, with zero flux at closed walls.
function scalar_rhs!(r, c, u, kappa)
    fill!(r, 0)
    nx, ny = size(c)
    for j = 2:ny-1, i = 3:nx-1
        v = u[i, j, 1]
        flux = max(v, zero(v)) * c[i-1, j] + min(v, zero(v)) * c[i, j] -
               kappa * (c[i, j] - c[i-1, j])
        r[i-1, j] -= flux
        r[i, j] += flux
    end
    for j = 3:ny-1, i = 2:nx-1
        v = u[i, j, 2]
        flux = max(v, zero(v)) * c[i, j-1] + min(v, zero(v)) * c[i, j] -
               kappa * (c[i, j] - c[i, j-1])
        r[i, j-1] -= flux
        r[i, j] += flux
    end
    return r
end

function momentum_rhs!(a, c, b)
    WaterLily.conv_diff!(a.f, a.u, a.σ, a.λ; ν=a.ν, perdir=())
    for j = 3:size(c, 2)-1, i = 2:size(c, 1)-1
        a.f[i, j, 2] += b * (c[i, j] + c[i, j-1]) / 2
    end
end

function project!(a, p)
    walls!(a.u)
    # Reset the pressure guess each projection: no hidden warmstart is needed
    # to restart from the public velocity/scalar state. Native PCG resets its
    # residual/search direction when solving the new right-hand side.
    fill!(p.x, 0)
    fill!(p.z, 0)
    for I in WaterLily.inside(p.x)
        p.z[I] = WaterLily.div(I, a.u)
    end
    WaterLily.solver!(p; tol=1e-7, itmx=1000)
    for d = 1:2, I in WaterLily.inside(p.x)
        a.u[I, d] -= p.L[I, d] * WaterLily.∂(d, I, p.x)
    end
    walls!(a.u)
end

function simulate(state, nu, kappa, b, dt, steps, extent)
    T = promote_type(eltype(state), typeof(nu), typeof(kappa), typeof(b), typeof(dt))
    n = size(state, 1)
    h = extent / n
    a = WaterLily.Flow((n, n), (0, 0); ν=nu / h^2, T=T, Δt=dt)
    a.u[3:n+1, 2:n+1, 1] .= state[1:n-1, :, 1] ./ h
    a.u[2:n+1, 3:n+1, 2] .= state[:, 1:n-1, 2] ./ h
    walls!(a.u)
    c = zeros(T, n+2, n+2)
    c[2:n+1, 2:n+1] .= state[:, :, 3]
    scalarwalls!(c)
    p = WaterLily.Poisson(a.p, a.μ₀, a.σ)
    r = similar(c)
    c0 = similar(c)

    # Coupled Heun predictor/corrector with native momentum fluxes and a
    # native pressure solve at both stages; no adaptive timestep/history.
    for _ = 1:steps
        copyto!(a.u⁰, a.u)
        copyto!(c0, c)
        scalar_rhs!(r, c, a.u, kappa / h^2)
        momentum_rhs!(a, c, b / h)
        a.u .= a.u⁰ .+ dt .* a.f
        c .= c0 .+ dt .* r
        scalarwalls!(c)
        project!(a, p)

        scalar_rhs!(r, c, a.u, kappa / h^2)
        momentum_rhs!(a, c, b / h)
        a.u .= (a.u⁰ .+ a.u .+ dt .* a.f) ./ 2
        c .= (c0 .+ c .+ dt .* r) ./ 2
        scalarwalls!(c)
        project!(a, p)
    end

    out = zeros(T, n, n, 3)
    out[1:n-1, :, 1] .= h .* a.u[3:n+1, 2:n+1, 1]
    out[:, 1:n-1, 2] .= h .* a.u[2:n+1, 3:n+1, 2]
    out[:, :, 3] .= c[2:n+1, 2:n+1]
    return out
end
end
