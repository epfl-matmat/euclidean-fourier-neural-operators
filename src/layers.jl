"""
Implementation of a Fourier Neural Operator (FNO) in Julia using Lux.jl, for the purpose of mapping electron densities to exchange-correlation potentials.

On shapes:
Following Lux.jl's conventions we use the following shape: (N1, N2, N3, Channels, Batch).
After `rfft` this gives: (N1÷2+1, N2, N3, Channels, Batch) complex.
Since a single datapoint has no channels and no batch dimension, we will always start and end the FNO model with a `lift` and `project` layer to reshape the input and output to the correct shape and add/remove channels.
"""

"""
    ChannelMix(in_chs => out_chs)

Identical to `Conv((1, 1, 1), in_chs => out_chs)` in the forward pass, but a plain
matmul instead of `NNlib.conv` because the second-order (nested) derivative of
`conv` under Reactant is silently wrong (EnzymeAD/Reactant.jl#3064). Fine as a
permanent choice. NB old checkpoints don't load - the Conv weight was
`(1,1,1,Ci,Co)`, ChannelMix is `(Co,Ci)`.
"""
struct ChannelMix <: LuxCore.AbstractLuxLayer
    in_chs::Int
    out_chs::Int
end
ChannelMix(ch::Pair{<:Integer, <:Integer}) = ChannelMix(first(ch), last(ch))

LuxCore.initialparameters(rng::AbstractRNG, l::ChannelMix) =
    (; weight = glorot_uniform(rng, Float32, l.out_chs, l.in_chs),
       bias = zeros32(rng, l.out_chs))

function (l::ChannelMix)(x::AbstractArray{T, 5}, ps, st::NamedTuple) where {T}
    N1, N2, N3, Ci, B = size(x)
    @assert Ci == l.in_chs "input channel dim $Ci does not match layer in_chs $(l.in_chs)"
    xc = permutedims(x, (4, 1, 2, 3, 5))              # (Ci, N1, N2, N3, B)
    y = ps.weight * reshape(xc, Ci, :) .+ ps.bias     # (Co, N1·N2·N3·B)
    y = reshape(y, l.out_chs, N1, N2, N3, B)
    return permutedims(y, (2, 3, 4, 1, 5)), st        # (N1, N2, N3, Co, B)
end

"""
Shape the input density correctly and add channels
"""
lift(C) = Chain(
    WrappedFunction(x -> reshape(x, size(x)..., 1, 1)),
    ChannelMix(1 => C),
)

"""
Project the output of the FNO back to a single channel
"""
project(C) = Chain(
    ChannelMix(C => 1),
    WrappedFunction(x -> reshape(x, size(x)[1:(end - 2)]...)),
)


##########################################################################################
# First, FNO with discrete kernel R
# This is the established FNO as one can find it in literature and code. We will later
# extend it to a continuous kernel R.
struct DiscreteSpectralConv3D <: LuxCore.AbstractLuxLayer
    modes::NTuple{3, Int}
    in_chs::Int
    out_chs::Int
end
function DiscreteSpectralConv3D(modes::NTuple{3, Int}, ch::Pair{<:Integer, <:Integer})
    return DiscreteSpectralConv3D(modes, first(ch), last(ch))
end

# Weight initialization following https://neuraloperator.github.io/dev/theory_guide/fno.html
# Stores real and imaginary parts separately (Float32) so that Reactant/XLA can trace
# and the optimizer state stays real-valued.
function LuxCore.initialparameters(rng::AbstractRNG, l::DiscreteSpectralConv3D)
    M1, M2, M3 = l.modes
    Ci, Co = l.in_chs, l.out_chs
    scale = Float32(1 / (Ci * Co))
    mk_re() = scale .* rand(rng, Float32, M1, M2, M3, Co, Ci)
    mk_im() = scale .* rand(rng, Float32, M1, M2, M3, Co, Ci)
    return (
        R1_re = mk_re(), R1_im = mk_im(),
        R2_re = mk_re(), R2_im = mk_im(),
        R3_re = mk_re(), R3_im = mk_im(),
        R4_re = mk_re(), R4_im = mk_im(),
    )
end

_to_complex(re, im) = Complex.(re, im)

function _modewise_matvec(W::AbstractArray{T, 5}, x::AbstractArray{S, 5}) where {T, S}
    M1, M2, M3, Co, Ci = size(W)
    @assert size(x, 1) == M1 && size(x, 2) == M2 && size(x, 3) == M3 && size(x, 4) == Ci
    B = size(x, 5)
    Wr = reshape(W, M1, M2, M3, Co, Ci, 1)
    xr = reshape(x, M1, M2, M3, 1, Ci, B)
    return dropdims(sum(Wr .* xr; dims = 5); dims = 5)
end

function _modewise_matvec(W::AbstractArray{T, 6}, x::AbstractArray{S, 5}) where {T, S}
    M1, M2, M3, Co, Ci, B = size(W)
    @assert size(x, 1) == M1 && size(x, 2) == M2 && size(x, 3) == M3 && size(x, 4) == Ci && size(x, 5) == B
    xr = reshape(x, M1, M2, M3, 1, Ci, B)
    return dropdims(sum(W .* xr; dims = 5); dims = 5)
end

function (l::DiscreteSpectralConv3D)(x::AbstractArray{T, 5}, ps, st::NamedTuple) where {T}
    M1, M2, M3 = l.modes
    N1, N2, N3, Ci, B = size(x)

    @assert Ci == l.in_chs   "input channel dim $Ci does not match layer in_chs $(l.in_chs)"
    @assert M1 <= N1 ÷ 2 + 1 "M1=$M1 too large for spatial dim of size $N1"
    @assert 2 * M2 <= N2     "M2=$M2 too large; need 2*M2 <= $N2"
    @assert 2 * M3 <= N3     "M3=$M3 too large; need 2*M3 <= $N3"

    x_hat = rfft(x, 1:3)                          # (N1÷2+1, N2, N3, Ci, B) complex

    # similar (not zeros) so out lives on the same device as x_hat (GPU, Reactant traced, ...)
    out = fill!(similar(x_hat, N1 ÷ 2 + 1, N2, N3, l.out_chs, B), 0)

    R1 = _to_complex(ps.R1_re, ps.R1_im)
    R2 = _to_complex(ps.R2_re, ps.R2_im)
    R3 = _to_complex(ps.R3_re, ps.R3_im)
    R4 = _to_complex(ps.R4_re, ps.R4_im)

    corners = [
        (1:M1, 1:M2, 1:M3),
        (1:M1, 1:M2, (N3 - M3 + 1):N3),
        (1:M1, (N2 - M2 + 1):N2, 1:M3),
        (1:M1, (N2 - M2 + 1):N2, (N3 - M3 + 1):N3),
    ]
    out[corners[1]..., :, :] .= _modewise_matvec(R1, x_hat[corners[1]..., :, :])
    out[corners[2]..., :, :] .= _modewise_matvec(R2, x_hat[corners[2]..., :, :])
    out[corners[3]..., :, :] .= _modewise_matvec(R3, x_hat[corners[3]..., :, :])
    out[corners[4]..., :, :] .= _modewise_matvec(R4, x_hat[corners[4]..., :, :])

    return irfft(out, N1, 1:3), st                # (N1, N2, N3, Co, B) real
end

fno_block(m_max, C) = Chain(
    Parallel(
        +,
        DiscreteSpectralConv3D((m_max, m_max, m_max), C => C),
        ChannelMix(C => C)
    ),
    WrappedFunction(gelu),
)







##########################################################################################
# Euclidean FNO (EFNO): the spectral kernel is a continuous function of the physical
# wavevector k, evaluated at k = B·m for whatever lattice is at hand.
abstract type AbstractSpectralKernel <: LuxCore.AbstractLuxLayer end

struct GaussianSpectralKernel{C, S<:AbstractFloat} <: AbstractSpectralKernel
    in_chs::Int
    out_chs::Int
    centers::C           # μ_j, e.g. a range of length `n_basis`
    sigma::S             # shared width σ of all basis functions
end
function GaussianSpectralKernel(ch::Pair{<:Integer, <:Integer}; t_max, n_basis=16, T::Type{<:AbstractFloat}=Float64)
    ch_in, ch_out = ch
    centers = range(T(0), T(t_max); length=n_basis)
    sigma = T(t_max / n_basis)
    return GaussianSpectralKernel(ch_in, ch_out, centers, sigma)
end
function LuxCore.initialparameters(rng::AbstractRNG, l::GaussianSpectralKernel)
    return (; M=zeros(Float32, length(l.centers), l.out_chs, l.in_chs))
end
# k_sq_grid holds |k|²; the kernel is κ̂(k) = Σ_j exp(-(|k|²-μ_j)²/(2σ²)) M_j.
function (kern::GaussianSpectralKernel)(k_sq_grid::AbstractArray{<:Any, 3}, ps, st)
    k_sq = reshape(k_sq_grid, :, 1)
    c = reshape(kern.centers, 1, :)
    Phi = exp.(-(k_sq .- c) .^ 2 ./ (2 * kern.sigma^2))
    R = Phi * reshape(ps.M, length(kern.centers), :)
    return reshape(R, size(k_sq_grid)..., kern.out_chs, kern.in_chs), st
end

# Batched dispatch: k_sq_grid is (M1, N2, N3, B) with per-batch lattices.
# Returns R of shape (M1, N2, N3, Co, Ci, B) for _modewise_matvec 6D dispatch.
function (kern::GaussianSpectralKernel)(k_sq_grid::AbstractArray{<:Any, 4}, ps, st)
    M1, N2, N3, B = size(k_sq_grid)
    k_sq = reshape(k_sq_grid, :, 1)
    c = reshape(kern.centers, 1, :)
    Phi = exp.(-(k_sq .- c) .^ 2 ./ (2 * kern.sigma^2))
    R = Phi * reshape(ps.M, length(kern.centers), :)
    R = reshape(R, M1, N2, N3, B, kern.out_chs, kern.in_chs)
    return permutedims(R, (1, 2, 3, 5, 6, 4)), st
end

# |k|² on the rfft grid (first dimension truncated to its positive frequencies).
function k_sq_grid(lattice, N1, N2, N3)
    kx, ky, kz = _k_cartesian(lattice, N1, N2, N3)
    k_sq = kx .^ 2 .+ ky .^ 2 .+ kz .^ 2
    return k_sq[1:N1÷2+1, :, :]
end


struct EuclideanSpectralConv3D{K} <: LuxCore.AbstractLuxContainerLayer{(:spectral_kernel,)}
    spectral_kernel::K
end
LuxCore.initialparameters(rng::AbstractRNG, l::EuclideanSpectralConv3D) =
    (; spectral_kernel=LuxCore.initialparameters(rng, l.spectral_kernel))
LuxCore.initialstates(rng::AbstractRNG, l::EuclideanSpectralConv3D) =
    (; spectral_kernel=LuxCore.initialstates(rng, l.spectral_kernel), k_sq_grid=nothing)

"""Return a copy of the (possibly nested) state `st` with every `k_sq_grid` entry replaced."""
set_k_sq_grid(st::NamedTuple, k_sq_grid) =
    map(v -> v isa NamedTuple ? set_k_sq_grid(v, k_sq_grid) : v,
        merge(st, haskey(st, :k_sq_grid) ? (; k_sq_grid) : (;)))

function (l::EuclideanSpectralConv3D)(x::AbstractArray{T,5}, ps, st::NamedTuple) where {T}
    @assert st.k_sq_grid !== nothing "k_sq_grid must be set in the layer state before calling the layer; use `st = set_k_sq_grid(st, k_sq_grid(lattice, grid_size...))`"
    N1, N2, N3, Ci, B = size(x)
    @assert Ci == l.spectral_kernel.in_chs "input channel dim $Ci does not match layer in_chs $(l.spectral_kernel.in_chs)"
    x_hat = rfft(x, 1:3)                          # (N1÷2+1, N2, N3, Ci, B) complex
    R, _ = l.spectral_kernel(st.k_sq_grid, ps.spectral_kernel, st.spectral_kernel)  # (N1÷2+1, N2, N3, Co, Ci) real
    out = _modewise_matvec(R, x_hat)              # (N1÷2+1, N2, N3, Co, B) complex
    return irfft(out, N1, 1:3), st                # (N1, N2, N3, Co, B) real
end

# t_max=40 + Gaussian tails ≈ dataset band limit 2·Ecut = 44.1 bohr⁻² (Ecut = 600 eV everywhere)
efno_block(C; t_max=40, n_basis=16, T::Type{<:AbstractFloat}=Float64) = Chain(
    Parallel(
        +,
        EuclideanSpectralConv3D(
            GaussianSpectralKernel(C => C; t_max, n_basis, T)
        ),
        ChannelMix(C => C)
    ),
    WrappedFunction(gelu),
)

"""
Batched variants of lift/project for inputs with an explicit batch dimension.
Input: (N1, N2, N3, B) → output: (N1, N2, N3, B).
Parameter structure is identical to lift/project, so ps can be shared.
"""
batched_lift(C) = Chain(
    WrappedFunction(x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), 1, size(x, 4))),
    ChannelMix(1 => C),
)
batched_project(C) = Chain(
    ChannelMix(C => 1),
    WrappedFunction(x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), size(x, 5))),
)
