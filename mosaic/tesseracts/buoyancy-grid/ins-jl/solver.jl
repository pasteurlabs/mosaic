# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

using IncompressibleNavierStokes
using Zygote

const setups = Dict{Tuple{Int,Float64},Tuple}()

function get_setup(n, extent)
    get!(setups, (n, Float64(extent))) do
        axis = range(0.0, Float64(extent); length=n+1)
        setup = Setup(; x=(axis, axis), boundary_conditions=(;
            u=((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
            temp=((SymmetricBC(), SymmetricBC()), (SymmetricBC(), SymmetricBC()))))
        # Native closed-wall Neumann inverse with an upstream Poisson adjoint.
        # Float64 avoids the transform solver's documented Float32 accuracy loss.
        (setup, psolver_transform(setup))
    end
end

function pad(field)
    n = size(field, 1)
    rows = vcat(zeros(eltype(field), 1, n), field, zeros(eltype(field), 1, n))
    hcat(zeros(eltype(field), n+2, 1), rows, zeros(eltype(field), n+2, 1))
end

function rollout(state, setup, psolver, viscosity, diffusivity, buoyancy, dt, steps)
    n = size(state, 1)
    # Native u[i,j,1] is the right face of cell[i,j]. Preserve the canonical
    # staggered DOFs directly; padded outer faces never enter the computation.
    ux = vcat(state[1:n-1, :, 1], zeros(eltype(state), 1, n))
    uy = hcat(state[:, 1:n-1, 2], zeros(eltype(state), n, 1))
    u = apply_bc_u(cat(pad(ux), pad(uy); dims=3), 0.0, setup)
    temp = apply_bc_temp(pad(state[:, :, 3]), 0.0, setup)
    method = RKMethods.RK44(; T=Float64)
    stepper = create_stepper(method; setup, psolver, state=(; u, temp), t=0.0)
    params = (; viscosity, conductivity=diffusivity, gdir=2,
              gravity=buoyancy, dodissipation=false)
    # RK4 carries no history beyond u,temp; reconstructing halos is exact.
    for _ in 1:steps
        stepper = timestep(method, boussinesq, stepper, dt; params)
    end
    final = stepper.state
    ux = vcat(final.u[2:n, 2:n+1, 1], zeros(eltype(state), 1, n))
    uy = hcat(final.u[2:n+1, 2:n, 2], zeros(eltype(state), n, 1))
    cat(ux, uy, final.temp[2:n+1, 2:n+1]; dims=3)
end

function solve(state, viscosity, diffusivity, buoyancy, dt, steps, extent)
    setup, psolver = get_setup(size(state, 1), extent)
    rollout(state, setup, psolver, viscosity, diffusivity, buoyancy, dt, steps)
end

function state_vjp(state, viscosity, diffusivity, buoyancy, dt, steps, extent, cotangent)
    # Geometry/cache construction stays outside the differentiated state map.
    setup, psolver = get_setup(size(state, 1), extent)
    _, back = Zygote.pullback(state) do initial
        rollout(initial, setup, psolver, viscosity, diffusivity, buoyancy, dt, steps)
    end
    only(back(cotangent))
end
