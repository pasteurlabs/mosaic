# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

using Enzyme

function coupled_vjp(state, nu, kappa, b, dt, steps, extent, cotangent)
    n = size(state, 1)
    x = vcat(vec(state), [nu, kappa, b, dt])
    # Grid topology is constant: specialize the differentiated function on the
    # pressure solver type rather than introducing a union of mutable solvers.
    pressure_mode = Val(n >= 8 && n % 4 == 0)
    function objective(z)
        initial = reshape(view(z, 1:3n*n), n, n, 3)
        result = MosaicWaterLily.simulate(
            initial, z[end-3], z[end-2], z[end-1], z[end], steps, extent, pressure_mode
        )
        return sum(result .* cotangent)
    end
    gradient = zeros(eltype(x), length(x))
    # Only the captured cotangent/grid are constant. All state and physical
    # parameters are active entries of x, differentiated through every step.
    Enzyme.autodiff(
        Enzyme.Reverse, Enzyme.Const(objective), Enzyme.Active,
        Enzyme.Duplicated(x, gradient)
    )
    return reshape(gradient[1:3n*n], n, n, 3), gradient[end-3:end]
end
