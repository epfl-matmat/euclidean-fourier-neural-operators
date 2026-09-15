# Shared data splits and training grids for the XC-potential experiment (Section 4.2).
# Included by train.jl and eval.jl.

using Random

"""
    get_splits(; experiment, val_fraction=0.2, seed=42)

Return `(train, val, test)` index vectors into the ML-RPA dataset for `experiment`
(`:diamond` or `:water`), following the paper protocol: train/validate on the small
structures, test on the larger ones (80/20 train/validation split within the pool).
"""
function get_splits(; experiment::Symbol, val_fraction=0.2, seed=42)
    if experiment == :diamond
        pool = collect(42:61)      # 8-atom diamond bulk
        test = collect(62:81)      # 16-atom diamond bulk
    elseif experiment == :water
        pool = collect(158:189)    # 8-molecule water
        test = collect(114:152)    # 31+32-molecule water (excluding 153:157 single-mol)
    else
        error("Unknown experiment: $experiment. Use :diamond or :water.")
    end
    shuffled = shuffle(MersenneTwister(seed), pool)
    n_val = max(1, round(Int, length(shuffled) * val_fraction))
    return (; train=shuffled[n_val+1:end], val=shuffled[1:n_val], test=collect(test))
end

# Training grids: train/val structures are Fourier-resampled to a common grid for
# efficient batched training; test structures are evaluated on their original grids.
TARGET_GRIDS = Dict(:diamond => (48, 48, 48), :water => (80, 80, 80))

"""
    vxc_errors(v, vxc_ref, w) -> (; wrms, wmae)

Density-weighted RMS and MAE of the gauge-aligned potential error.
"""
function vxc_errors(v, vxc_ref, w)
    Δ = v .- vxc_ref
    Δw = Δ .- sum(w .* Δ)
    wrms = sqrt(sum(w .* Δw .^ 2))
    wmae = sum(w .* abs.(Δw))
    return (; wrms, wmae)
end

Ha_meV = ustrip(u"meV", 1u"hartree")

const METHOD_ORDER = ["PBE", "FNO", "EFNO"]
sort_methods(names) = sort(names; by=n -> something(findfirst(==(n), METHOD_ORDER), length(METHOD_ORDER) + 1))

function load_metric_csv(path)
    lines = readlines(path)
    while !isempty(lines) && startswith(lines[1], '#')
        popfirst!(lines)
    end
    header = split(lines[1], ",")
    structure_idxs = parse.(Int, header[2:end])
    model_names = String[]
    values = Dict{String, Vector{Float64}}()
    for line in lines[2:end]
        parts = split(line, ",")
        push!(model_names, String(parts[1]))
        values[model_names[end]] = parse.(Float64, parts[2:end])
    end
    return (; model_names, structure_idxs, values)
end

function seed_groups(model_names)
    groups = Dict{String, Vector{String}}()
    for name in model_names
        base = replace(name, r"_s\d+$" => "")
        push!(get!(groups, base, String[]), name)
    end
    return groups
end

"""
    aggregate_per_structure(data) -> (; model_names, structure_idxs, values, spread)

Average per-structure metrics over seeds. Input `data` must have fields
`model_names`, `structure_idxs`, `values` (as returned by `load_metric_csv`).
"""
function aggregate_per_structure(data)
    groups = seed_groups(data.model_names)
    agg_names = sort_methods(collect(keys(groups)))
    n = length(data.structure_idxs)
    vals = Dict{String, Vector{Float64}}()
    spread = Dict{String, Vector{Float64}}()
    for base in agg_names
        seeds = groups[base]
        if length(seeds) == 1
            vals[base] = data.values[seeds[1]]
            spread[base] = zeros(n)
        else
            mat = hcat([data.values[s] for s in seeds]...)
            vals[base] = vec(mean(mat; dims=2))
            spread[base] = vec(std(mat; dims=2))
        end
    end
    return (; model_names=agg_names, structure_idxs=data.structure_idxs, values=vals, spread)
end
