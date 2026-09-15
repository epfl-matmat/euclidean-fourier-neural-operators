# Data

The ML-RPA dataset does not ship with this repository (approx. 3.4 GB).

It is derived from the raw `MLPOT` text file published by Riemelmoser, Verdi,
Kaltak and Kresse (*J. Chem. Theory Comput.* 2023, 19, 7287-7299,
DOI: 10.1021/acs.jctc.3c00848) on Phaidra:

  https://phaidra.univie.ac.at/detail/o:1675367

To regenerate `mlrpa_dataset.jld2` (~3.4 GB, 189 structures), download the raw
file and parse it (from the repository root):

```bash
julia --project=. experiments/exp2_xc_potentials/parse_mlpot.jl /path/to/MLPOT data/mlrpa_dataset.jld2
```

File format of `data/mlrpa_dataset.jld2`: `JLD2.load(path, "structures")`
returns a `Vector` of `NamedTuple`s with fields
`E_xc, lattice, atom_types_str, n_atoms_per_type, coord_type, positions, Ecut,
vxc_shift, grid_size, rho, vxc`, in VASP conventions (lattice vectors as ROWS,
in Å; `rho` stored as e⁻ × V_cell; energies/potentials in eV). Convert each
structure with `to_atomic_units(s)` before use (see the scripts).
