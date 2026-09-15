#!/usr/bin/env julia
#
# Parse the ML-RPA dataset (Riemelmoser, Verdi, Kaltak, Kresse, JCTC 2023,
# DOI: 10.1021/acs.jctc.3c00848) from the raw MLPOT text file into the
# JLD2 file used by all scripts in this repository.
#
# Download the raw MLPOT file (approx. 7 GB of text) from Phaidra:
#   https://phaidra.univie.ac.at/detail/o:1675367
#
# Usage (from the repository root):
#   julia --project=. experiments/exp2_xc_potentials/parse_mlpot.jl [input_file] [output_file]
# Defaults: [input_file]=MLPOT, [output_file]=data/mlrpa_dataset.jld2
#

using JLD2
using ProgressMeter

"""
Parse a single structure block from MLPOT file.
Returns a NamedTuple with all extracted data.

MLPOT block format (text, concatenated for 189 structures):

    Training EXC: -20.0404072118580
       1.00000000000000
         9.000000    0.000000    0.000000
         ...
     charge density:
      108  120  140
      [density values, 5 per line]
     xc-potential:
      108  120  140
      [potential values, 5 per line]
"""
function parse_structure(lines)
    # Line 1: E_xc (in eV)
    E_xc = parse(Float64, split(lines[1], ":")[2])

    # Line 2: scaling factor
    scale = parse(Float64, lines[2])

    # Lines 3-5: lattice vectors (in Å, VASP convention)
    lattice = zeros(3, 3)
    for i in 1:3
        lattice[i, :] = parse.(Float64, split(lines[2+i]))
    end
    lattice .*= scale

    # Line 6: atom type(s)
    atom_types_str = strip(lines[6])

    # Line 7: number of atoms (per type)
    n_atoms_per_type = parse.(Int, split(lines[7]))
    n_atoms = sum(n_atoms_per_type)

    # Line 8: coordinate type (Direct or Cartesian)
    coord_type = strip(lines[8])

    # Lines 9 to 9+n_atoms-1: atomic positions
    positions = zeros(3, n_atoms)
    for i in 1:n_atoms
        positions[:, i] = parse.(Float64, split(lines[8+i]))
    end

    # Find metadata lines
    i_ecut = findfirst(l -> contains(l, "energy cutoff"), lines)
    i_vxc_shift = findfirst(l -> contains(l, "xc-potential shift"), lines)
    i_rho = findfirst(startswith(" charge density:"), lines)
    i_vxc = findfirst(startswith(" xc-potential:"), lines)

    Ecut = parse(Float64, match(r"[\d.]+", lines[i_ecut]).match)
    vxc_shift = parse(Float64, split(lines[i_vxc_shift], ":")[2])

    # Grid size
    grid_size = Tuple(parse.(Int, split(lines[i_rho+1])))
    n_grid = prod(grid_size)

    # Parse charge density
    rho_lines = lines[i_rho+2:i_vxc-1]
    rho = Vector{Float64}(undef, n_grid)
    idx = 1
    for line in rho_lines
        for val in split(line)
            rho[idx] = parse(Float64, val)
            idx += 1
        end
    end
    rho = reshape(rho, grid_size)

    # Parse xc-potential
    vxc_lines = lines[i_vxc+2:end]
    vxc = Vector{Float64}(undef, n_grid)
    idx = 1
    for line in vxc_lines
        for val in split(line)
            vxc[idx] = parse(Float64, val)
            idx += 1
        end
    end
    vxc = reshape(vxc, grid_size)

    (; E_xc, lattice, atom_types_str, n_atoms_per_type, coord_type,
       positions, Ecut, vxc_shift, grid_size, rho, vxc)
end

function main(input_file="MLPOT", output_file=joinpath("data", "mlrpa_dataset.jld2"))
    println("Loading $input_file...")
    @time lines = readlines(input_file)
    println("  $(length(lines)) lines")

    # Find structure boundaries
    indices = findall(startswith("Training EXC"), lines)
    n_structures = length(indices)
    println("  $n_structures structures found")

    # Parse all structures
    println("\nParsing structures...")
    structures = Vector{Any}(undef, n_structures)
    @showprogress for i in 1:n_structures
        start_idx = indices[i]
        end_idx = i < n_structures ? indices[i+1] - 1 : length(lines)
        block = @view lines[start_idx:end_idx]
        structures[i] = parse_structure(block)
    end

    # Summary
    println("\nDataset summary:")
    println("================")
    by_type = Dict{String, Vector{Int}}()
    for (i, s) in enumerate(structures)
        push!(get!(by_type, s.atom_types_str, Int[]), i)
    end
    for (k, v) in sort(collect(by_type), by=x->length(x[2]), rev=true)
        grids = unique([structures[i].grid_size for i in v])
        println("  $k: $(length(v)) structures, grids: $grids")
    end

    # Save
    println("\nSaving to $output_file...")
    @time jldsave(output_file; structures, compress=true)

    filesize_mb = filesize(output_file) / 1024^2
    println("Done! Output size: $(round(filesize_mb, digits=1)) MB")
end

# Run if called as script
if abspath(PROGRAM_FILE) == @__FILE__
    input = get(ARGS, 1, "MLPOT")
    output = get(ARGS, 2, joinpath("data", "mlrpa_dataset.jld2"))
    main(input, output)
end
