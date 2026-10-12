# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

using Oceananigans
using Oceananigans.TimeSteppers: time_step!

function make_model(n; extent=1.0, viscosity=0.001, diffusivity=0.001, buoyancy=1.0)
    # Mosaic's vertical y coordinate is Oceananigans' z; the native y axis is flat.
    grid = RectilinearGrid(CPU(), Float64;
        size=(n, n), x=(0, extent), z=(0, extent),
        topology=(Bounded, Flat, Bounded), halo=(3, 3))
    # Normal velocity is impermeable by default; explicitly impose no-slip tangents.
    u_bcs = FieldBoundaryConditions(
        bottom=ValueBoundaryCondition(0), top=ValueBoundaryCondition(0))
    w_bcs = FieldBoundaryConditions(
        west=ValueBoundaryCondition(0), east=ValueBoundaryCondition(0))
    scalar_bcs = FieldBoundaryConditions(
        west=GradientBoundaryCondition(0), east=GradientBoundaryCondition(0),
        bottom=GradientBoundaryCondition(0), top=GradientBoundaryCondition(0))
    # Native buoyancy is g * alpha * T. T remains the unscaled recurrent scalar,
    # including when the requested buoyancy coefficient is zero or negative.
    eos = LinearEquationOfState(thermal_expansion=buoyancy, haline_contraction=0.0)
    buoyancy_model = SeawaterBuoyancy(gravitational_acceleration=1.0,
        equation_of_state=eos, constant_salinity=0.0)
    return NonhydrostaticModel(grid;
        buoyancy=buoyancy_model, tracers=:T, advection=UpwindBiased(order=3),
        timestepper=:RungeKutta3,
        closure=ScalarDiffusivity(ν=viscosity, κ=diffusivity),
        boundary_conditions=(u=u_bcs, w=w_bcs, T=scalar_bcs))
end

function unpack!(model, state)
    n = size(state, 1)
    for k in 1:n, i in 1:n-1
        model.velocities.u[i+1, 1, k] = state[i, k, 1]
    end
    for k in 1:n-1, i in 1:n
        model.velocities.w[i, 1, k+1] = state[i, k, 2]
    end
    for k in 1:n, i in 1:n
        model.tracers.T[i, 1, k] = state[i, k, 3]
    end
    return nothing
end

function pack!(out, model)
    n = size(out, 1)
    for k in 1:n, i in 1:n
        out[i, k, 1] = i < n ? model.velocities.u[i+1, 1, k] : 0.0
        out[i, k, 2] = k < n ? model.velocities.w[i, 1, k+1] : 0.0
        out[i, k, 3] = model.tracers.T[i, 1, k]
    end
    return nothing
end

function advance!(out, model, state, dt, steps)
    unpack!(model, state)
    # RK3 overwrites its stage tendencies each step; no AB history is discarded.
    for _ in 1:steps
        time_step!(model, dt)
    end
    pack!(out, model)
    return nothing
end

function solve(state, viscosity, diffusivity, buoyancy, dt, steps, extent)
    model = make_model(size(state, 1); extent, viscosity, diffusivity, buoyancy)
    out = zero(state)
    advance!(out, model, state, dt, steps)
    return out
end
