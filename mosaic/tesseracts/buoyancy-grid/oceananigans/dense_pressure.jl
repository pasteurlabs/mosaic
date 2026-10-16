# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

# Same uniform-grid MAC Neumann inverse as the native FFTW pressure solver.
# Explicit separable cosine matrices avoid FFTW calls without Enzyme rules.
# The rest of the Boussinesq discretization and RK3 remain native Oceananigans.
struct DenseNeumannSolver{G,T}
    grid::G
    basis::Matrix{T}
    inverse_eigenvalues::Matrix{T}
end

# Only geometry and constant transform coefficients are stored here, never state.
Enzyme.EnzymeRules.inactive_type(::Type{<:DenseNeumannSolver}) = true
const pressure_bases = Dict{Tuple{Int,Float64},Tuple{Matrix{Float64},Matrix{Float64}}}()

function DenseNeumannSolver(grid)
    n = grid.Nx
    grid.Nz == n || error("The buoyancy adapter requires a square grid.")
    basis, inverse_eigenvalues = get!(pressure_bases, (n, Float64(grid.Lx))) do
        basis = [sqrt(k == 0 ? 1/n : 2/n) * cos(pi*k*(i-0.5)/n)
                 for k in 0:n-1, i in 1:n]
        h = grid.Lx / n
        eigenvalues = [-4 * (sin(pi*i/(2n))^2 + sin(pi*j/(2n))^2) / h^2
                       for i in 0:n-1, j in 0:n-1]
        inverse_eigenvalues = map(x -> iszero(x) ? 0.0 : inv(x), eigenvalues)
        (basis, inverse_eigenvalues)
    end
    return DenseNeumannSolver(grid, basis, inverse_eigenvalues)
end

function solve_for_pressure!(pressure, solver::DenseNeumannSolver, ::Nothing, velocity, dt)
    n = solver.grid.Nx
    h = solver.grid.Lx / n
    rhs = Matrix{eltype(pressure)}(undef, n, n)
    for k in 1:n, i in 1:n
        rhs[i, k] = (velocity.u[i+1, 1, k] - velocity.u[i, 1, k] +
                     velocity.w[i, 1, k+1] - velocity.w[i, 1, k]) / h
    end
    coefficients = solver.basis * rhs * transpose(solver.basis)
    coefficients .*= solver.inverse_eigenvalues
    result = transpose(solver.basis) * coefficients * solver.basis
    for k in 1:n, i in 1:n
        pressure[i, 1, k] = result[i, k]
    end
    return pressure
end
