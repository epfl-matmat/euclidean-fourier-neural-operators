using JLD2

function rebuild_model(config)
    model_type = Symbol(get(config, :model_type, :efno))
    C = config.channels
    make_block(C, config) = if model_type === :fno
        fno_block(config.fno_m_max, C)
    else
        efno_block(C; t_max=config.t_max, n_basis=config.n_basis,
                     T=get(config, :precision, Float64))
    end
    Chain(
        lift(C),
        (make_block(C, config) for _ in 1:config.n_blocks)...,
        project(C),
    )
end

function load_model(checkpoint_path)
    ck = JLD2.load(checkpoint_path)
    config = ck["config"]
    ps = Lux.f64(ck["ps"])
    model = rebuild_model(config)
    rng = Random.default_rng()
    Random.seed!(rng, get(config, :seed, 0))
    _, st = Lux.setup(rng, model)
    st = Lux.f64(st)
    model_type = Symbol(get(config, :model_type, :efno))
    return (; model, ps, st, config, step=ck["step"], model_type)
end

function predict_vxc(loaded, s)
    st_s = if loaded.model_type === :efno
        set_k_sq_grid(loaded.st, Float64.(k_sq_grid(s.lattice, s.grid_size...)))
    else
        loaded.st
    end
    return first(loaded.model(s.rho, loaded.ps, st_s))
end
