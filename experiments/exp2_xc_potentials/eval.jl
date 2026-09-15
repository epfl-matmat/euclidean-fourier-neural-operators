# Experiment 2 (Section 4.2): evaluation of the trained checkpoints.
#
# Evaluates the shipped checkpoints (checkpoints/{diamond,water}_{efno,fno}.jld2,
# one per model and task; see train.jl to retrain them) together with the PBE
# baseline on the train/val pools and the larger test cells.
#
# Produces:
#   - results/{diamond,water}_wrms.csv / _wmae.csv  (per-structure, per-seed metrics, meV)
#   - results/{diamond,water}_summary.csv           (split means ± std over seeds)
#   - the medians of Table 1 (printed to stdout)
#   - figures/fig{3,4}_xc_slices_{diamond,water}.{pdf,svg} (slice visualizations)
#
# Run from the repository root:
#   julia -t 16 --project=. experiments/exp2_xc_potentials/eval.jl [--recompute]

using JLD2
using CairoMakie
using Statistics
using Unitful, UnitfulAtomic
using EuclideanFNO
using Printf

RECOMPUTE = "--recompute" in ARGS

##########################################################################################
# Checkpoints: the 4 checkpoints trained for the paper (5 seeds × 2 models × 2 tasks),
# shipped with this repository. Retrain with train.jl to reproduce.

CHECKPOINTS = Dict(
    (exp, mt) => joinpath(@__DIR__, "..", "..", "checkpoints", "$(exp)_$(mt).jld2")
    for exp in (:diamond, :water), mt in (:efno, :fno)
)

##########################################################################################
# Data

structures = load("data/mlrpa_dataset.jld2")["structures"]
println("Loaded $(length(structures)) structures")

# Splits & target grids (shared with train.jl)
include("common.jl")

# Processed structures, cached by (idx, resampled). resampled=true applies the training
# grid (for train/val structures); resampled=false keeps the raw grid (for test).
const eval_struct_cache = Dict{Tuple{Int, Bool}, Any}()

function get_eval_structure(idx, target_grid; resample::Bool)
    get!(eval_struct_cache, (idx, resample)) do
        s = to_atomic_units(structures[idx])
        if resample && target_grid !== nothing && s.grid_size != target_grid
            s = resample_to_grid(s, target_grid)
        end
        return s
    end
end

##########################################################################################
# Metrics & plot styling

##########################################################################################
# PBE cache - avoids re-running Libxc on large grids every time

RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results")
FIGURES_DIR = joinpath(@__DIR__, "..", "..", "figures")
mkpath(RESULTS_DIR)
mkpath(FIGURES_DIR)

const PBE_CACHE_PATH = joinpath(RESULTS_DIR, "pbe_cache.jld2")

function load_pbe_cache()
    isfile(PBE_CACHE_PATH) || return Dict{Tuple{Int, Any}, NamedTuple}()
    return Dict{Tuple{Int, Any}, NamedTuple}(load(PBE_CACHE_PATH, "cache"))
end

const pbe_cache = load_pbe_cache()
pbe_cache_dirty = false

function evaluate_structure_pbe(idx, s)
    get!(pbe_cache, (idx, s.grid_size)) do
        global pbe_cache_dirty = true
        w = s.rho ./ sum(s.rho)
        vxc_errors(vxc_pbe(s.rho, s.lattice), s.vxc, w)
    end
end

function flush_pbe_cache()
    pbe_cache_dirty || return
    jldsave(PBE_CACHE_PATH; cache=pbe_cache)
    global pbe_cache_dirty = false
    println("  PBE cache saved ($(length(pbe_cache)) structures)")
end

##########################################################################################
# CSV I/O

function save_metric_csv(path, model_names, structure_idxs, values)
    open(path, "w") do io
        println(io, join(["model"; string.(structure_idxs)], ","))
        for name in model_names
            println(io, join([name; [@sprintf("%.4f", v) for v in values[name]]], ","))
        end
    end
    println("  Saved $path")
end

function save_summary_csv(path, per_seed_data, splits)
    split_list = [("Train", splits.train), ("Val", splits.val),
                  ("Train+Val", vcat(splits.train, splits.val)), ("Test", splits.test)]
    groups = seed_groups(per_seed_data.model_names)
    base_names = sort_methods(collect(keys(groups)))
    idx_to_pos = Dict(idx => i for (i, idx) in enumerate(per_seed_data.structure_idxs))

    cols = ["model"]
    for (s, _) in split_list
        push!(cols, "$(s)_mean", "$(s)_std")
    end
    open(path, "w") do io
        println(io, join(cols, ","))
        for base in base_names
            seeds = groups[base]
            row = [base]
            for (_, idxs) in split_list
                positions = [idx_to_pos[idx] for idx in idxs]
                per_seed_means = [mean(per_seed_data.values[s][positions]) for s in seeds]
                push!(row, @sprintf("%.2f", mean(per_seed_means)))
                push!(row, length(seeds) > 1 ? @sprintf("%.2f", std(per_seed_means)) : "")
            end
            println(io, join(row, ","))
        end
    end
    println("  Saved $path")
end

##########################################################################################
# Compute all per-structure metrics for an experiment

function evaluate_all(loaded_models, eval_structs)
    model_names = String[]
    wrms = Dict{String, Vector{Float64}}()
    wmae = Dict{String, Vector{Float64}}()
    all_idxs = sort!(collect(keys(eval_structs)))

    for model_type in (:efno, :fno)
        haskey(loaded_models, model_type) || continue
        models = loaded_models[model_type]
        label = uppercase(string(model_type))
        for (si, loaded) in enumerate(models)
            seed_label = length(models) > 1 ? "$(label)_s$(si-1)" : label
            println("    $seed_label: L=$(loaded.config.n_blocks) C=$(loaded.config.channels) step=$(loaded.step)")
            w_vals, m_vals = Float64[], Float64[]
            for idx in all_idxs
                s = eval_structs[idx]
                w = s.rho ./ sum(s.rho)
                errs = vxc_errors(predict_vxc(loaded, s), s.vxc, w)
                push!(w_vals, errs.wrms * Ha_meV)
                push!(m_vals, errs.wmae * Ha_meV)
            end
            push!(model_names, seed_label)
            wrms[seed_label] = w_vals
            wmae[seed_label] = m_vals
        end
    end

    w_vals, m_vals = Float64[], Float64[]
    for idx in all_idxs
        errs = evaluate_structure_pbe(idx, eval_structs[idx])
        push!(w_vals, errs.wrms * Ha_meV)
        push!(m_vals, errs.wmae * Ha_meV)
    end
    flush_pbe_cache()
    push!(model_names, "PBE")
    wrms["PBE"] = w_vals
    wmae["PBE"] = m_vals

    return (; model_names, structure_idxs=all_idxs, wrms, wmae)
end

##########################################################################################
# Slice plots

# PBE for *display only*: evaluate on a spectrally upsampled grid and lowpass back.
# The GGA potential is a nonlinear functional of ρ and ∇ρ, so it has spectral content
# above the grid Nyquist that aliases onto the grid when evaluated pointwise - visible
# as voxel-scale noise in near-vacuum patches (up to ~0.1 Ha locally). Super-resolving
# the evaluation (2×) and lowpassing removes most of that aliasing. The PBE metrics
# keep using the direct on-grid evaluation, which is the honest "what a calculation
# on this grid gives" baseline.
function smooth_vxc_pbe(s; factor=2)
    fine = EuclideanFNO.resample_to_grid(s, factor .* s.grid_size)
    v = vxc_pbe(fine.rho, fine.lattice)
    return EuclideanFNO.fourier_resample(v, s.grid_size)
end

function plot_vxc_slices(experiment, loaded_models, eval_structs, representative_idxs)
    Ha_eV = ustrip(u"eV", 1u"hartree")
    bohr_to_A = ustrip(u"Å", 1u"bohr")
    n_structs = length(representative_idxs)
    entries = [uppercase(string(mt)) => loaded_models[mt]
               for mt in (:efno, :fno) if haskey(loaded_models, mt)]
    pred_names = ["PBE"; [uppercase(string(mt)) for mt in (:fno, :efno) if haskey(loaded_models, mt)]]
    potentials = ["RPA"; pred_names]

    P = length(potentials)
    E = length(pred_names)
    n_data_cols = max(P, E)
    cb_col = n_data_cols + 2  # +1 for 3D cell col, +1 for cb
    with_theme(Theme(fontsize=20, Axis=(titlefont=:regular,), Axis3=(titlefont=:regular,))) do
    fig = Figure(size=(250 * (n_data_cols + 2), 200 * 2 * n_structs); figure_padding=5)

    row_data = map(representative_idxs) do (split_label, idx)
        s = eval_structs[idx]
        w = s.rho ./ sum(s.rho)
        ref = s.vxc
        align(p) = p .- sum(w .* (p .- ref))
        preds = Dict(name => align(predict_vxc(loaded, s)) for (name, loaded) in entries)
        preds["PBE"] = align(smooth_vxc_pbe(s))
        planar_avg = vec(mean(s.rho; dims=(1, 2)))
        if experiment == :water
            nz = length(planar_avg)
            central = (nz ÷ 3):(2 * nz ÷ 3)
            iz = central[last(findmax(planar_avg[central]))]
        else
            iz = last(findmax(planar_avg))
        end
        (; split_label, idx, s, w, ref, preds, iz)
    end
    err_clim = maximum(rd -> maximum(abs, (rd.preds["PBE"] .- rd.ref)[:, :, rd.iz] .* Ha_eV),
                       row_data)

    Tbold = "TeX Gyre Termes Bold"
    Tfont = "TeX Gyre Termes"
    panel_idx = 0
    next_prefix() = (panel_idx += 1; string(Char('a' + panel_idx - 1)) * ".")
    labeled_title(text) = rich(rich(next_prefix(), font=Tbold), " $text")

    for (si, rd) in enumerate(row_data)
        (; split_label, idx, s, ref, preds, iz) = rd
        iz_frac = (iz - 1) / size(ref, 3)
        pot_row = 2 * (si - 1) + 1
        err_row = 2 * (si - 1) + 2

        atom_labels = vcat((fill(t, n) for (t, n) in
                            zip(split(s.atom_types_str), s.n_atoms_per_type))...)
        lat_A = s.lattice .* bohr_to_A

        ax3 = Axis3(fig[pot_row:err_row, 1];
            title = "", titlealign = :left,
            xlabel = "x (Å)", ylabel = "y (Å)", zlabel = "z (Å)",
            aspect = :data, azimuth = 0.35pi, elevation = 0.15pi,
            perspectiveness = 0.3)
        Label(fig[pot_row, 1, Top()],
              rich(rich(next_prefix(), font=Tbold), " $split_label system");
              font=Tfont, fontsize=20, halign=:left, valign=:bottom,
              padding=(0, 0, 0, 0))
        plot_crystal_cell!(ax3, lat_A, s.positions, atom_labels; slice_frac = iz_frac)

        slices = Dict(n => (n == "RPA" ? ref : preds[n])[:, :, iz] .* Ha_eV for n in potentials)
        vmin = minimum(slices["RPA"])
        vmax = maximum(slices["RPA"])

        function make_axis(row, col; title="")
            return Axis(fig[row, col]; title=labeled_title(title), titlealign=:left,
                aspect = DataAspect())
        end

        gl_rpa = fig[pot_row, 2] = GridLayout()
        ax_rpa = Axis(gl_rpa[1, 1]; title=labeled_title("RPA potential"), titlealign=:left, aspect=DataAspect())
        hm_rpa = plot_slice_cartesian!(ax_rpa, ref .* Ha_eV, lat_A, iz;
            colorrange = (vmin, vmax), colormap = :viridis, rasterize = 4)
        Colorbar(gl_rpa[1, 2], hm_rpa; width=8)

        local hm_pot
        for (j, name) in enumerate(pred_names)
            ax = make_axis(pot_row, j + 2;
                title = name)
            hm_pot = plot_slice_cartesian!(ax, preds[name] .* Ha_eV, lat_A, iz;
                colorrange = (vmin, vmax), colormap = :viridis, rasterize = 4)
        end
        Colorbar(fig[pot_row, cb_col], hm_pot; label="Potential (eV)", width=12)

        gl_rho = fig[err_row, 2] = GridLayout()
        ax_rho = Axis(gl_rho[1, 1]; title=labeled_title("Density"), titlealign=:left, aspect=DataAspect())
        hm_rho = plot_slice_cartesian!(ax_rho, s.rho, lat_A, iz;
            colormap = :inferno, rasterize = 4)
        Colorbar(gl_rho[1, 2], hm_rho; width=8)

        local hm_err
        for (j, name) in enumerate(pred_names)
            ax = make_axis(err_row, j + 2;
                title = "$name − RPA")
            field3d = (preds[name] .- ref) .* Ha_eV
            hm_err = plot_slice_cartesian!(ax, field3d, lat_A, iz;
                colorrange = (-err_clim, err_clim), colormap = :RdBu, rasterize = 4)
        end
        Colorbar(fig[err_row, cb_col], hm_err; label="Error (eV)", width=12)
    end

    colsize!(fig.layout, 1, Auto(2))
    colsize!(fig.layout, cb_col, Fixed(30))
    colgap!(fig.layout, 5)
    rowgap!(fig.layout, 5)
    for si in 1:(n_structs - 1)
        rowgap!(fig.layout, 2 * si, 20)
    end

    fignum = Dict(:diamond => 3, :water => 4)[experiment]
    base = joinpath(FIGURES_DIR, "fig$(fignum)_xc_slices_$(experiment)")
    save("$base.pdf", fig); save("$base.svg", fig)
    println("  Saved $base.pdf and $base.svg")
    end # with_theme
end

##########################################################################################
# Main

table1 = Dict{Tuple{Symbol, String}, Tuple{Float64, Float64}}()  # (exp, model) → (train, test) medians

for experiment in (:diamond, :water)
    println("\n", "="^60)
    println("  $(uppercase(string(experiment)))")
    println("="^60)

    # Load the checkpoints for this experiment (one per model type)
    loaded_models = Dict{Symbol, Vector{Any}}()
    for mt in (:efno, :fno)
        haskey(CHECKPOINTS, (experiment, mt)) || continue
        path = CHECKPOINTS[(experiment, mt)]
        if !isfile(path)
            println("    Checkpoint not found: $path - skipping")
            continue
        end
        loaded_models[mt] = Any[load_model(path)]
    end

    splits = get_splits(; experiment)

    wrms_csv = joinpath(RESULTS_DIR, "$(experiment)_wrms.csv")
    wmae_csv = joinpath(RESULTS_DIR, "$(experiment)_wmae.csv")

    # Step 1: Per-structure metrics - compute or load from CSV
    all_idxs = sort(unique(vcat(splits.train, splits.val, splits.test)))

    # Train/val structures on the resampled training grids; test on raw grids
    target_grid = get(TARGET_GRIDS, experiment, nothing)
    pool = Set(vcat(splits.train, splits.val))
    eval_structs = Dict(idx => get_eval_structure(idx, target_grid; resample=(idx in pool))
                        for idx in all_idxs)

    if RECOMPUTE || !isfile(wrms_csv)
        println("  Computing metrics...")
        result = evaluate_all(loaded_models, eval_structs)
        save_metric_csv(wrms_csv, result.model_names, result.structure_idxs, result.wrms)
        save_metric_csv(wmae_csv, result.model_names, result.structure_idxs, result.wmae)
        wrms_data = (; result.model_names, result.structure_idxs, values=result.wrms)
        wmae_data = (; result.model_names, result.structure_idxs, values=result.wmae)
    else
        wrms_data = load_metric_csv(wrms_csv)
        wmae_data = load_metric_csv(wmae_csv)
        println("  Loaded $(basename(wrms_csv))")
        if !issubset(Set(all_idxs), Set(wrms_data.structure_idxs))
            error("Cached CSV does not cover all structures - rerun with --recompute")
        end
    end

    # Step 2: Aggregate across seeds & save summary tables
    wrms_agg = aggregate_per_structure(wrms_data)
    save_summary_csv(joinpath(RESULTS_DIR, "$(experiment)_summary.csv"), wrms_data, splits)
    save_summary_csv(joinpath(RESULTS_DIR, "$(experiment)_summary_wmae.csv"), wmae_data, splits)

    idx_to_pos = Dict(idx => i for (i, idx) in enumerate(wrms_agg.structure_idxs))
    for name in wrms_agg.model_names
        train_vals = wrms_agg.values[name][[idx_to_pos[idx] for idx in vcat(splits.train, splits.val)]]
        test_vals = wrms_agg.values[name][[idx_to_pos[idx] for idx in splits.test]]
        table1[(experiment, name)] = (median(train_vals), median(test_vals))
    end

    # Step 3: Slice plots (uses the first available model of each type)
    if !isempty(loaded_models)
        slice_models = Dict(mt => models[1] for (mt, models) in loaded_models)
        ref_name = "EFNO" in wrms_agg.model_names ? "EFNO" : wrms_agg.model_names[1]
        reps = Pair{String, Int}[]
        split_idx_map = Dict("Train" => splits.train, "Val" => splits.val, "Test" => splits.test)
        for sname in ["Train", "Test"]
            vals = wrms_agg.values[ref_name][[idx_to_pos[idx] for idx in split_idx_map[sname]]]
            idxs = split_idx_map[sname]
            _, i = findmin(abs.(vals .- median(vals)))
            push!(reps, sname => idxs[i])
        end
        println("  Generating vxc slice plots (structures $(last.(reps)))...")
        plot_vxc_slices(experiment, slice_models, eval_structs, reps)
    end
end

##########################################################################################
# Table 1: median WRMSE (meV) within each split.
# "Train+Val" = the full small-cell train+validation pool, "Test" = the larger cells.

println("\n", "="^64)
println("  Table 1: density-weighted RMS error (WRMSE, meV), median per split")
println("="^64)
@printf("\n  %-6s  %27s    %27s\n", "", "Diamond", "Water")
@printf("  %-6s  %12s %12s    %12s %12s\n", "Model", "Train+Val", "Test", "Train+Val", "Test")
for name in ["PBE", "FNO", "EFNO"]
    dtrain, dtest = table1[(:diamond, name)]
    wtrain, wtest = table1[(:water, name)]
    @printf("  %-6s  %12.2f %12.2f    %12.2f %12.2f\n", name, dtrain, dtest, wtrain, wtest)
end

println("\nDone.")
