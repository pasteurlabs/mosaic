# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

using FFTW
using Zygote

"""Collocated (n,n,2) → periodic high-face staggered velocity."""
function collocated_to_staggered_periodic_2d(u::AbstractArray, n::Int)
    half = eltype(u)(0.5)
    ux = u[:, :, 1]
    uy = u[:, :, 2]
    ux_s = half .* (ux .+ cat(ux[2:end, :], ux[1:1, :]; dims=1))
    uy_s = half .* (uy .+ cat(uy[:, 2:end], uy[:, 1:1]; dims=2))
    return cat(reshape(ux_s, n, n, 1), reshape(uy_s, n, n, 1); dims=3)
end

"""Periodic high-face staggered (n,n,2) → collocated velocity."""
function staggered_to_collocated_periodic_2d(u::AbstractArray, n::Int)
    half = eltype(u)(0.5)
    ux_s = u[:, :, 1]
    uy_s = u[:, :, 2]
    ux = half .* (cat(ux_s[end:end, :], ux_s[1:end-1, :]; dims=1) .+ ux_s)
    uy = half .* (cat(uy_s[:, end:end], uy_s[:, 1:end-1]; dims=2) .+ uy_s)
    return cat(reshape(ux, n, n, 1), reshape(uy, n, n, 1); dims=3)
end

"""Collocated (n,n,n,3) → periodic high-face staggered velocity."""
function collocated_to_staggered_periodic_3d(u::AbstractArray, n::Int)
    half = eltype(u)(0.5)
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

"""Periodic high-face staggered (n,n,n,3) → collocated velocity."""
function staggered_to_collocated_periodic_3d(u::AbstractArray, n::Int)
    half = eltype(u)(0.5)
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

"""Fourier multiplier for the supported-mode inverse of face averaging."""
function periodic_staggered_lift_multiplier(
    n::Int,
    ::Type{T};
    max_gain::Real=32.0,
) where {T<:AbstractFloat}
    max_gain >= 1 || throw(ArgumentError("max_gain must be at least 1"))
    return map(0:n-1) do k
        signed_k = k <= n ÷ 2 ? k : k - n
        theta = T(2π * signed_k / n)
        transfer = T(0.5) * (one(Complex{T}) + cis(-theta))
        abs(transfer) * T(max_gain) >= one(T) ? inv(transfer) : zero(Complex{T})
    end
end
Zygote.@nograd periodic_staggered_lift_multiplier

"""Right-invert one periodic face-to-centre averaging axis."""
function lift_collocated_to_staggered_periodic(
    u::AbstractArray,
    axis::Int;
    max_gain::Real=32.0,
)
    n = size(u, axis)
    inverse_transfer = periodic_staggered_lift_multiplier(
        n,
        eltype(u);
        max_gain=max_gain,
    )
    transfer_shape = ntuple(dim -> dim == axis ? n : 1, ndims(u))
    multiplier = reshape(inverse_transfer, transfer_shape)
    return real.(ifft(fft(u, (axis,)) .* multiplier, (axis,)))
end

"""Lift a canonical 2-D correction onto periodic high faces."""
function lift_collocated_to_staggered_periodic_2d(u::AbstractArray, n::Int)
    ux = lift_collocated_to_staggered_periodic(u[:, :, 1], 1)
    uy = lift_collocated_to_staggered_periodic(u[:, :, 2], 2)
    return cat(reshape(ux, n, n, 1), reshape(uy, n, n, 1); dims=3)
end

"""Lift a canonical 3-D correction onto periodic high faces."""
function lift_collocated_to_staggered_periodic_3d(u::AbstractArray, n::Int)
    ux = lift_collocated_to_staggered_periodic(u[:, :, :, 1], 1)
    uy = lift_collocated_to_staggered_periodic(u[:, :, :, 2], 2)
    uz = lift_collocated_to_staggered_periodic(u[:, :, :, 3], 3)
    return cat(
        reshape(ux, n, n, n, 1),
        reshape(uy, n, n, n, 1),
        reshape(uz, n, n, n, 1);
        dims=4,
    )
end
