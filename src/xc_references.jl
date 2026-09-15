# Everything in this file works in atomic units: densities in e⁻/bohr³,
# lattices in bohr (rows = lattice vectors), potentials in hartree.
# Use `to_atomic_units` on raw dataset structures first.

# Cartesian wave-vector components G_cart[α][i,j,k] for an FFT grid on `lattice`
# (rows = lattice vectors). Returns three 3D arrays indexed by (i,j,k).
function _k_cartesian(lattice, Nx, Ny, Nz)
    B = 2π .* transpose(inv(lattice))             # rows = reciprocal vectors
    hx = reshape(fftfreq(Nx, Nx), Nx, 1, 1)
    hy = reshape(fftfreq(Ny, Ny), 1, Ny, 1)
    hz = reshape(fftfreq(Nz, Nz), 1, 1, Nz)
    return ntuple(α -> hx .* B[1, α] .+ hy .* B[2, α] .+ hz .* B[3, α], 3)
end

# Spectral gradient: returns (∂f/∂x, ∂f/∂y, ∂f/∂z) Cartesian (units match f / lattice).
function _grad_fft(f::AbstractArray{<:Real, 3}, lattice)
    fhat = fft(f)
    k = _k_cartesian(lattice, size(f)...)
    return ntuple(α -> real(ifft(im .* k[α] .* fhat)), 3)
end

# Spectral divergence of a Cartesian vector field F = (Fx, Fy, Fz).
function _div_fft(F::NTuple{3, <:AbstractArray{<:Real, 3}}, lattice)
    k = _k_cartesian(lattice, size(F[1])...)
    div_hat = im .* (k[1] .* fft(F[1]) .+ k[2] .* fft(F[2]) .+ k[3] .* fft(F[3]))
    return real(ifft(div_hat))
end


# ─── Functionals ───────────────────────────────────────────────────────────────

"""
    vxc_gga(ρ, lattice; x_func, c_func) -> Array{Float64,3}

Generic GGA exchange-correlation potential in hartree, using libxc functional symbols
`x_func` (exchange) and `c_func` (correlation). `ρ` is on a periodic FFT grid in
e⁻/bohr³; `lattice` is the 3×3 bohr cell with lattice vectors as ROWS (VASP convention).
Gradient and divergence are computed spectrally (FFT) - exact for periodic
band-limited data, handles non-orthogonal lattices naturally.

The GGA potential is  vrho − 2 ∇·(vsigma · ∇ρ).
"""
function vxc_gga(
        ρ::AbstractArray{<:Real, 3}, lattice::AbstractMatrix;
        x_func::Symbol, c_func::Symbol
    )
    n_raw = ρ                                    # unclipped, may have tiny negatives
    ∇n = _grad_fft(n_raw, lattice)               # gradient of raw ρ - no clipping kinks
    σ = ∇n[1] .^ 2 .+ ∇n[2] .^ 2 .+ ∇n[3] .^ 2

    # Clip only when handing to libxc; the FFT gradient stays smooth.
    n_safe = max.(n_raw, 0.0)
    fx = Functional(x_func; n_spin = 1)
    fc = Functional(c_func; n_spin = 1)
    rx = evaluate(fx; rho = vec(n_safe), sigma = vec(σ))
    rc = evaluate(fc; rho = vec(n_safe), sigma = vec(σ))

    vrho = reshape(rx.vrho .+ rc.vrho, size(ρ))
    vsigma = reshape(rx.vsigma .+ rc.vsigma, size(ρ))

    F = ntuple(α -> 2 .* vsigma .* ∇n[α], 3)
    return vrho .- _div_fft(F, lattice)
end

"PBE potential (hartree) - see [`vxc_gga`](@ref)."
vxc_pbe(ρ, lattice) = vxc_gga(ρ, lattice; x_func = :gga_x_pbe, c_func = :gga_c_pbe)

