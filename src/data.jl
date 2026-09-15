"""
    to_atomic_units(s::NamedTuple) -> NamedTuple

Convert one raw ML-RPA structure (as stored in `data/mlrpa_dataset.jld2`, VASP
conventions: lattice in Å, energies/potentials in eV, `rho` as e⁻ × V_cell) to
atomic units: lattice in bohr, `rho` in e⁻/bohr³, `E_xc`/`vxc`/`vxc_shift`/`Ecut`
in hartree. Fractional `positions` and `grid_size` are unchanged.

Apply directly after loading:

    structures = to_atomic_units.(load("data/mlrpa_dataset.jld2", "structures"))
"""
function to_atomic_units(s::NamedTuple)
    Å = austrip(1u"Å")
    eV = austrip(1u"eV")
    lattice = s.lattice .* Å                  # bohr
    V_cell = abs(det(lattice))                # bohr³
    return (;
        s...,
        lattice,
        rho = s.rho ./ V_cell,                # e⁻ × V_cell → e⁻/bohr³
        vxc = s.vxc .* eV,
        vxc_shift = s.vxc_shift * eV,
        E_xc = s.E_xc * eV,
        Ecut = s.Ecut * eV,
    )
end

function fourier_resample(f::AbstractArray{T,3}, target::NTuple{3,Int}) where T
    src = size(f)
    src == target && return f
    f_hat = fftshift(fft(f))
    out_hat = zeros(Complex{float(T)}, target)
    for d in 1:3
        @assert iseven(src[d]) && iseven(target[d]) "fourier_resample requires even grid dims"
    end
    # For each dim: copy the central n = min(src, target) modes (centered after fftshift)
    copy_ranges = ntuple(d -> let n = min(src[d], target[d])
        s_start = (src[d] - n) ÷ 2 + 1
        t_start = (target[d] - n) ÷ 2 + 1
        (s_start:s_start+n-1, t_start:t_start+n-1)
    end, 3)
    out_hat[copy_ranges[1][2], copy_ranges[2][2], copy_ranges[3][2]] .=
        f_hat[copy_ranges[1][1], copy_ranges[2][1], copy_ranges[3][1]]
    return real.(ifft(ifftshift(out_hat))) .* (prod(target) / prod(src))
end

function resample_to_grid(s::NamedTuple, target::NTuple{3,Int})
    s.grid_size == target && return s
    return (; s...,
        grid_size = target,
        rho = fourier_resample(s.rho, target),
        vxc = fourier_resample(s.vxc, target),
    )
end
