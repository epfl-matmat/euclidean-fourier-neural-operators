# Experiment 2 (Section 4.2): training (E)FNO models on the ML-RPA dataset.
#
# Trains on small-cell structures (Fourier-resampled to a common grid for batched
# training), validates on held-out small-cell structures, and saves the checkpoint
# with the best validation WRMSE. Structures with the same grid shape are stacked
# into batches; gradients are compiled with Reactant/XLA and Enzyme.
#
# Defaults reproduce the paper configuration; seed s=0..4 reproduces the shipped
# checkpoints (up to hardware/XLA nondeterminism).
#
# Run (from the repository root, GPU recommended):
#   julia -t 16 --heap-size-hint=32G --project=. experiments/exp2_xc_potentials/train.jl diamond
#   julia -t 16 --heap-size-hint=32G --project=. experiments/exp2_xc_potentials/train.jl diamond model_type=fno seed=1
#   julia --project=. experiments/exp2_xc_potentials/train.jl water backend=cpu n_steps=100   # CPU smoke test
#
# Hyperparameters can be overridden as key=value pairs, e.g. lr=0.003 n_steps=50000.
# Weights & Biases logging is off by default; enable with log_wandb=1 (requires the
# WANDB_API_KEY environment variable; the project can be set via WANDB_PROJECT).

get!(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "0.75")
if any(startswith("backend=cpu"), ARGS)
    ENV["CUDA_VISIBLE_DEVICES"] = ""
end

using JLD2
using CairoMakie
using Statistics, Random, Dates
using Lux, Enzyme, Reactant
using Optimisers
using ProgressMeter
using Unitful, UnitfulAtomic
using EuclideanFNO

##########################################################################################
# Config

experiment_arg = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :diamond

# Parse key=value overrides from ARGS[2:end], e.g. `diamond n_blocks=2 channels=16`
cli_overrides = Dict{Symbol, Any}()
for arg in ARGS[2:end]
    k, v = split(arg, '=')
    cli_overrides[Symbol(k)] = if contains(v, '.')
        parse(Float64, v)
    elseif all(isdigit, v)
        parse(Int, v)
    else
        String(v)
    end
end

# Defaults = the paper configuration (production v4).
config = (;
    experiment = experiment_arg,
    val_fraction = 0.2,
    split_seed = get(cli_overrides, :seed, 0),
    channels = get(cli_overrides, :channels, 8),
    n_blocks = get(cli_overrides, :n_blocks, 2),
    t_max = 40,
    n_basis = get(cli_overrides, :n_basis, 16),
    model_type = Symbol(get(cli_overrides, :model_type, :efno)),  # :efno or :fno
    fno_m_max = get(cli_overrides, :fno_m_max, 4),
    n_steps = get(cli_overrides, :n_steps, 100000),
    lr = get(cli_overrides, :lr, 1.0e-2),
    lr_schedule = Symbol(get(cli_overrides, :lr_schedule, :constant)),  # :constant or :cosine
    seed = get(cli_overrides, :seed, 0),
    precision = let p = get(cli_overrides, :precision, "Float32")
        p == "Float64" || p == "f64" ? Float64 : Float32
    end,
    backend = get(cli_overrides, :backend, "auto"),
    eval_every = get(cli_overrides, :eval_every, 25),
    plot_every = 2000,
    checkpoint_every = 2000,
    max_batch_size = get(cli_overrides, :max_batch_size, 128),
    patience = get(cli_overrides, :patience, 20000),
    log_wandb = get(cli_overrides, :log_wandb, 0) == 1,
    wandb_group = get(cli_overrides, :group, nothing),
    wandb_tags = let raw = get(cli_overrides, :tags, nothing)
        raw === nothing ? String[] : String.(split(string(raw), ","))
    end,
)

config.backend == "auto" || Reactant.set_default_backend(config.backend)

rng = Random.default_rng()
Random.seed!(rng, config.seed)
T = config.precision

##########################################################################################
# Splits & target grids (shared with eval.jl)

include("common.jl")

splits = get_splits(; experiment=config.experiment, val_fraction=config.val_fraction,
                      seed=config.split_seed)
println("Experiment: $(config.experiment)")
println("  Train: $(length(splits.train)) structures  $(splits.train)")
println("  Val:   $(length(splits.val)) structures  $(splits.val)")
println("  Test:  $(length(splits.test)) structures (not evaluated during training)")

##########################################################################################
# Run setup (WandB + directories)

runname_keys = (:experiment => "exp", :model_type => "model", :lr => "lr",
    :channels => "C", :n_blocks => "L",
    (config.model_type === :fno ? (:fno_m_max => "M",) : (:n_basis => "B",))...)
tag = join(("$abbr=$(getproperty(config, k))" for (k, abbr) in runname_keys), "_")
runname = "$(Dates.format(now(), "yyyy-mm-dd_HHMMSS"))__$(tag)"
run_dir = "runs/$(runname)"
mkpath(run_dir)

gitcommit = try readchomp(`git rev-parse HEAD`) catch; "unknown" end
hparams = Dict{String, Any}(
    string(k) => v isa Union{Real, String} ? v : string(v) for (k, v) in pairs(config)
)
hparams["gitcommit"] = gitcommit

lg = if config.log_wandb
    haskey(ENV, "WANDB_API_KEY") ||
        error("log_wandb=1 requires the WANDB_API_KEY environment variable")
    @eval using Wandb
    wandb_kwargs = Dict{Symbol, Any}()
    config.wandb_group !== nothing && (wandb_kwargs[:group] = config.wandb_group)
    !isempty(config.wandb_tags) && (wandb_kwargs[:tags] = config.wandb_tags)
    _lg = WandbLogger(; project=get(ENV, "WANDB_PROJECT", "efno-mlrpa"), name=runname,
                        config=hparams, dir=run_dir, wandb_kwargs...)
    println(_lg)
    _lg
else
    nothing
end

##########################################################################################
# Data

structures = load("data/mlrpa_dataset.jld2")["structures"]

target_grid = get(TARGET_GRIDS, config.experiment, nothing)

##########################################################################################
# Model

make_block(C, config) = config.model_type === :fno ?
    EuclideanFNO.fno_block(config.fno_m_max, C) :
    EuclideanFNO.efno_block(C; t_max=config.t_max, n_basis=config.n_basis, T=config.precision)

_blocks = [make_block(config.channels, config) for _ in 1:config.n_blocks]
model = Chain(
    EuclideanFNO.lift(config.channels),
    _blocks...,
    EuclideanFNO.project(config.channels),
)
batched_model = Chain(
    EuclideanFNO.batched_lift(config.channels),
    _blocks...,
    EuclideanFNO.batched_project(config.channels),
)

ps, st = Lux.setup(rng, model)
T == Float64 && ((ps, st) = (Lux.f64(ps), Lux.f64(st)))

dev = reactant_device()
println("Reactant device: ", dev)
println("XLA devices: ", Reactant.devices())
psd = ps |> dev

##########################################################################################
# Eval setup: compile forward pass, compute PBE baseline

model_forward(model, rho, ps, st) = first(model(rho, ps, st))

function make_eval_setup(idx)
    s = to_atomic_units(structures[idx])
    if target_grid !== nothing && s.grid_size != target_grid
        println("  Resampling structure $idx: $(s.grid_size) → $target_grid")
        s = resample_to_grid(s, target_grid)
    end
    rhod = T.(s.rho) |> dev
    st_s = EuclideanFNO.set_k_sq_grid(st, T.(EuclideanFNO.k_sq_grid(s.lattice, s.grid_size...))) |> dev
    t = @elapsed fwd = @compile model_forward(model, rhod, psd, st_s)
    println("Structure $idx: compiled forward pass in $(round(t; digits=1)) s"); flush(stdout)
    w = s.rho ./ sum(s.rho)
    v_pbe = vxc_pbe(s.rho, s.lattice)
    pbe_err = vxc_errors(v_pbe, s.vxc, w)
    v_pbe_aligned = v_pbe .+ (sum(w .* s.vxc) - sum(w .* v_pbe))
    return (; idx, s, rhod, st_s, fwd, w, v_pbe=v_pbe_aligned, pbe_err)
end

evaluate_errors(ev, ps) =
    vxc_errors(Array(ev.fwd(model, ev.rhod, ps, ev.st_s)), ev.s.vxc, ev.w)

println("\n--- Compiling evaluation forward passes ---")
ev_trains = [make_eval_setup(idx) for idx in splits.train]
ev_vals = [make_eval_setup(idx) for idx in splits.val]

println("\n--- PBE baselines (wrms / wmae, meV) ---")
for (label, evs) in [("Train", ev_trains), ("Val", ev_vals)]
    wrms_vals = [ev.pbe_err.wrms * Ha_meV for ev in evs]
    wmae_vals = [ev.pbe_err.wmae * Ha_meV for ev in evs]
    println("  $label: wrms=$(round(mean(wrms_vals); digits=1))  wmae=$(round(mean(wmae_vals); digits=1))")
end

##########################################################################################
# Comparison figure (for val structures, saved into the run directory)

plot_val_idxs = splits.val[1:min(2, length(splits.val))]

function vxc_comparison(ev, vxc_pred; filename)
    s, v_pbe = ev.s, ev.v_pbe
    Δefno = vxc_pred .- s.vxc
    Δpbe = v_pbe .- s.vxc
    com_frac = s.positions[:, 1]
    iz = clamp(round(Int, com_frac[3] * s.grid_size[3]) + 1, 1, s.grid_size[3])
    Δmax = max(maximum(abs, Δefno), maximum(abs, Δpbe))

    fig = Figure(size=(2000, 450))
    panels = [
        ("v_xc^RPA", s.vxc, :plasma, Makie.automatic),
        ("v_xc^model (aligned)", vxc_pred, :plasma, extrema(s.vxc[:, :, iz])),
        ("v_xc^PBE (aligned)", v_pbe, :plasma, extrema(s.vxc[:, :, iz])),
        ("model − RPA", Δefno, :balance, (-Δmax, Δmax)),
        ("PBE − RPA", Δpbe, :balance, (-Δmax, Δmax)),
    ]
    for (col, (label, v, cmap, crange)) in enumerate(panels)
        ax = Axis(fig[1, col], aspect=DataAspect(),
                  xlabel="e₁ ∥ a₁ (bohr)", ylabel="e₂ ⊥ a₁ (bohr)", title="$label (Ha)")
        hm = plot_slice_cartesian!(ax, v, s.lattice, iz; atom_positions=s.positions,
                                   colormap=cmap, colorrange=crange)
        Colorbar(fig[2, col], hm, vertical=false, flipaxis=false)
    end
    save(filename, fig)
    return fig
end

function plot_val_comparisons(ps_cur)
    for idx in plot_val_idxs
        ev = ev_vals[findfirst(e -> e.idx == idx, ev_vals)]
        pred = Array(ev.fwd(model, ev.rhod, ps_cur, ev.st_s))
        pred_aligned = pred .+ (sum(ev.w .* ev.s.vxc) - sum(ev.w .* pred))
        vxc_comparison(ev, pred_aligned;
                       filename=joinpath(run_dir, "vxc_val_$(idx).png"))
    end
end

##########################################################################################
# Training setup: batched gradient compilation (group structures by grid shape)

function batched_loss(model, ps, rho, vxc, w, st_s)
    pred = first(model(rho, ps, st_s))
    Δ = pred .- vxc
    gauge = sum(w .* Δ; dims=(1, 2, 3))
    Δw = Δ .- gauge
    return sum(w .* Δw .^ 2)
end
function batched_grad_fn(model, ps, s)
    derivs, loss_val = Enzyme.gradient(
        ReverseWithPrimal, batched_loss, Const(model), ps,
        Const(s.rho), Const(s.vxc), Const(s.w), Const(s.st_s),
    )
    return derivs[2], loss_val
end

# Group training structures by grid shape, then split into sub-batches
println("\n--- Grouping training structures (max_batch_size=$(config.max_batch_size)) ---")
grid_groups = Dict{NTuple{3,Int}, Vector{Int}}()
for (i, ev) in enumerate(ev_trains)
    gs = ev.s.grid_size
    push!(get!(grid_groups, gs, Int[]), i)
end

N_total = length(ev_trains)

# Split large groups into sub-batches of at most max_batch_size
all_batches = Vector{@NamedTuple{gs::NTuple{3,Int}, member_idxs::Vector{Int}}}()
for gs in sort(collect(keys(grid_groups)))
    idxs = grid_groups[gs]
    for chunk_start in 1:config.max_batch_size:length(idxs)
        chunk = idxs[chunk_start:min(chunk_start + config.max_batch_size - 1, end)]
        push!(all_batches, (; gs, member_idxs=chunk))
    end
end
for b in all_batches
    println("  Grid $(b.gs): $(length(b.member_idxs)) structures (indices: $(map(i -> ev_trains[i].idx, b.member_idxs)))")
end
println("  $(length(all_batches)) batches total")
flush(stdout)

# Build batched setups: stack rho, vxc, w, k_sq_grid along dim 4
group_setups = Tuple(begin
    evs = [ev_trains[i] for i in b.member_idxs]
    rho = cat([T.(ev.s.rho) for ev in evs]...; dims=4) |> dev
    vxc = cat([T.(ev.s.vxc) for ev in evs]...; dims=4) |> dev
    w = cat([T.(ev.w) for ev in evs]...; dims=4) |> dev
    k_sq = cat([T.(EuclideanFNO.k_sq_grid(ev.s.lattice, ev.s.grid_size...)) for ev in evs]...; dims=4)
    st_s = EuclideanFNO.set_k_sq_grid(st, k_sq) |> dev
    (; rho, vxc, w, st_s, n=length(b.member_idxs))
end for b in all_batches)

# Fused mini-batch step: ONE gradient call + optimizer update per batch.
function minibatch_step(model, opt_state, ps, s)
    grad, loss = batched_grad_fn(model, ps, s)
    g = Lux.fmap(x -> x ./ N_total, grad)
    opt_state, ps = Optimisers.update(opt_state, ps, g)
    return opt_state, ps, loss
end

# Compile one step function per unique batch shape
println("\n--- Compiling mini-batch step functions ---")
opt_dev = reactant_device(; track_numbers=AbstractFloat)
opt_state = opt_dev(Optimisers.setup(Adam(T(config.lr)), ps))

batch_shapes = unique(size(s.rho) for s in group_setups)
shape_to_stepfn = Dict{Any, Any}()
for shape in batch_shapes
    representative = findfirst(s -> size(s.rho) == shape, group_setups)
    t = @elapsed fn = @compile minibatch_step(batched_model, opt_state, psd, group_setups[representative])
    shape_to_stepfn[shape] = fn
    println("  Shape $shape: compiled in $(round(t; digits=1)) s")
    flush(stdout)
end

step_fns = Tuple(shape_to_stepfn[size(s.rho)] for s in group_setups)
n_batches = length(group_setups)
println("$(length(batch_shapes)) unique shapes, $n_batches batches"); flush(stdout)

##########################################################################################
# Checkpointing

cpu = cpu_device()

# To restore:
#   ck = load("runs/<run>/checkpoint_latest.jld2")
#   ps_cur = ck["ps"] |> dev
#   step = ck["step"]
function save_checkpoint(filename, ps_cur, step)
    mkpath(run_dir)
    jldsave(joinpath(run_dir, filename); ps=cpu(ps_cur), step, config, gitcommit)
end

function _set_eta!(tree::NamedTuple, eta)
    for v in values(tree)
        _set_eta!(v, eta)
    end
end
function _set_eta!(leaf::Optimisers.Leaf, eta)
    r = leaf.rule
    leaf.rule = Adam(eta, r.beta, r.epsilon)
end
_set_eta!(::Any, ::Any) = nothing

##########################################################################################
# Training loop

ps_cur = psd
t0 = time()
step = 0
best_val_wrms = Inf
best_val_step = 0

println("\n--- Starting training ($(config.n_steps) steps) ---\n"); flush(stdout)

@showprogress for _ in 1:(config.n_steps - step)
    global ps_cur, opt_state, t0, step, best_val_wrms, best_val_step
    step += 1
    if config.lr_schedule === :cosine
        new_eta = Reactant.ConcreteRNumber(T(config.lr * (1 + cospi(step / config.n_steps)) / 2))
        _set_eta!(opt_state, new_eta)
    end
    t_step = @elapsed begin
        total_loss = 0.0
        for b in 1:n_batches
            opt_state, ps_cur, loss = step_fns[b](batched_model, opt_state, ps_cur, group_setups[b])
            total_loss += Float64(loss)
        end
    end
    train_loss = total_loss / N_total
    train_wrms_meV = sqrt(train_loss) * Ha_meV

    step == 1 && (t0 = time())

    logs = Dict{String, Any}(
        "train/loss" => Float64(train_loss),
        "train/mean_wrms_meV" => train_wrms_meV,
    )

    if step == 1 || step % config.eval_every == 0
        t_eval = @elapsed begin
            val_wrms = Float64[]
            val_wmae = Float64[]
            for ev in ev_vals
                m = evaluate_errors(ev, ps_cur)
                push!(val_wrms, m.wrms * Ha_meV)
                push!(val_wmae, m.wmae * Ha_meV)
                logs["val_structures/wrms_meV_$(lpad(ev.idx, 3, '0'))"] = m.wrms * Ha_meV
                logs["val_structures/wmae_meV_$(lpad(ev.idx, 3, '0'))"] = m.wmae * Ha_meV
            end
            logs["val/mean_wrms_meV"] = mean(val_wrms)
            logs["val/mean_wmae_meV"] = mean(val_wmae)
        end
        logs["perf/t_eval"] = t_eval
        if mean(val_wrms) < best_val_wrms
            best_val_wrms = mean(val_wrms)
            best_val_step = step
            save_checkpoint("checkpoint_best.jld2", ps_cur, step)
        end
        if config.patience > 0 && step - best_val_step >= config.patience
            println("\nEarly stopping: no val improvement in $(config.patience) steps (best=$(round(best_val_wrms; digits=1)) meV @ step $best_val_step)")
            break
        end
    end

    logs["perf/t_step"] = t_step
    logs["perf/steps_per_s"] = step <= 1 ? 0.0 : (step - 1) / (time() - t0)
    lg !== nothing && Wandb.log(lg, logs; step)

    if step == 1 || step % config.plot_every == 0
        try
            plot_val_comparisons(ps_cur)
        catch e
            @warn "Figure saving failed at step $step" exception=(e, catch_backtrace())
        end
    end

    step % config.checkpoint_every == 0 && save_checkpoint("checkpoint_latest.jld2", ps_cur, step)
    step % 100 == 0 && GC.gc()
end

save_checkpoint("checkpoint_latest.jld2", ps_cur, step)

println("\n--- Final validation metrics (meV) ---")
for ev in ev_vals
    m = evaluate_errors(ev, ps_cur)
    println("  [$(ev.idx)] $(uppercase(string(config.model_type))): wrms=$(round(m.wrms * Ha_meV; digits=1)) wmae=$(round(m.wmae * Ha_meV; digits=1))  " *
            "PBE: wrms=$(round(ev.pbe_err.wrms * Ha_meV; digits=1)) wmae=$(round(ev.pbe_err.wmae * Ha_meV; digits=1))")
end

lg !== nothing && close(lg)
println("\nDone. Run dir: $run_dir")
