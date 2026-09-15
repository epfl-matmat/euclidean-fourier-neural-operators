# Experiment 1 (Section 4.1): transferring a learned convolution across equivalent
# periodic cells.
#
# Fits a single FNO mode table and a single EFNO Gaussian-basis symbol (both in closed
# form) to the heat-kernel symbol on a rhombic cell, then evaluates them on an
# equivalent rectangular cell and on n×n supercells.
#
# Produces:
#   - the relative L2 errors of Table 3 (printed to stdout)
#   - figures/fig1_cell_transfer.{pdf,svg} (Figure 1)
#
# Run from the repository root:
#   julia --project=. experiments/exp1_cell_transfer/exp1_cell_transfer.jl

using FFTW
using CairoMakie
using LinearAlgebra
using Printf
using EuclideanFNO

## Configuration
N0      = 48            # grid points per unit of physical length
M       = 12            # FNO modes kept per dimension
n_basis = 25            # EFNO radial basis functions
t_heat  = 0.006         # heat time
t_max   = -log(1e-6) / t_heat   # EFNO basis range in |k|²: covers the symbol down to 1e-6
src_w   = 0.03          # width of the sources
n_max   = 6             # largest supercell in the sweep

L_prim = [sqrt(3)/2 sqrt(3)/2; 0.5 -0.5]
L_conv = L_prim * [1.0 1.0; 1.0 -1.0]

## Cells, wavevectors, the target operator

# Lattice vectors are COLUMNS, so k = 2π L⁻ᵀ m.
function k_grid(L, N1, N2)
    B = 2π .* transpose(inv(L))
    hx, hy = reshape(fftfreq(N1, N1), N1, 1), reshape(fftfreq(N2, N2), 1, N2)
    return (hx .* B[1, 1] .+ hy .* B[1, 2])[1:N1÷2+1, :],
           (hx .* B[2, 1] .+ hy .* B[2, 2])[1:N1÷2+1, :]
end

symbol(kx, ky) = exp.(-t_heat .* (kx .^ 2 .+ ky .^ 2))
apply_true(f, L) = irfft(symbol(k_grid(L, size(f)...)...) .* rfft(f), size(f, 1))
relerr(pred, gt) = 100 * norm(pred .- gt) / norm(gt)

function cell(L)
    N1, N2 = (2 * round(Int, N0 * norm(L[:, i]) / 2) for i in 1:2)
    kx, ky = k_grid(L, N1, N2)
    hx, hy = reshape(fftfreq(N1, N1), N1, 1), reshape(fftfreq(N2, N2), 1, N2)
    mx, my = repeat(hx, 1, N2)[1:N1÷2+1, :], repeat(hy, N1, 1)[1:N1÷2+1, :]
    box = (abs.(mx) .<= M) .& (abs.(my) .<= M)  # FNO: index box
    k2 = kx .^ 2 .+ ky .^ 2                     # EFNO: every resolved mode
    κ = symbol(kx, ky)
    return (; L, N1, N2, box, k2, κ, κ_box=κ[box])
end

# One source per lattice site, summed over the primitive lattice: the same
# physical field in any cell.
function source(L, N1, N2)
    P = inv(L_prim)
    f = zeros(N1, N2)
    for j in 1:N2, i in 1:N1
        x = ((i - 1) / N1) .* L[:, 1] .+ ((j - 1) / N2) .* L[:, 2]
        g1 = P[1, 1] * x[1] + P[1, 2] * x[2]           # fractional coords in L_prim
        g2 = P[2, 1] * x[1] + P[2, 2] * x[2]
        d1, d2 = g1 - round(g1), g2 - round(g2)
        s = 0.0
        for n1 in -1:1, n2 in -1:1
            r = (d1 + n1) .* L_prim[:, 1] .+ (d2 + n2) .* L_prim[:, 2]
            s += exp(-sum(abs2, r) / (2 * src_w^2))
        end
        f[i, j] = s
    end
    return f
end

probe(d) = (f = source(d.L, d.N1, d.N2); (f, apply_true(f, d.L)))

## The two models, both fitted in closed form

fit_fno(c) = c.κ_box

# Gaussians in |k|², as in GaussianSpectralKernel (src/layers.jl): centers uniform
# on [0, t_max], shared width σ = t_max/n_basis. Radial, hence even and isotropic;
# modes beyond t_max get ≈0, matching the decayed symbol there.
function radial_basis(k2)
    kk = vec(k2)
    Φ = Matrix{Float64}(undef, length(kk), n_basis)
    μ = range(0.0, t_max; length=n_basis)
    σ = t_max / n_basis
    for j in 1:n_basis
        @. Φ[:, j] = exp(-(kk - μ[j])^2 / (2σ^2))
    end
    return Φ
end

function fit_efno(c)
    Φ = radial_basis(c.k2)
    G = Φ' * Φ
    return (G + 1e-10 * tr(G) / n_basis * I) \ (Φ' * vec(c.κ))
end

function apply_model(tag, p, d, f)
    R = zeros(d.N1 ÷ 2 + 1, d.N2)
    if tag === :fno
        (R[d.box] .= p)
    elseif tag === :efno
        (R .= reshape(radial_basis(d.k2) * p, size(d.k2)))
    else
        error("Unknown tag $tag")
    end
    return irfft(R .* rfft(f), d.N1)
end

## Fit on the primitive cell

d_prim = cell(L_prim)
d_conv = cell(L_conv)
tags   = [:fno, :efno]
labels = Dict(:fno => "FNO", :efno => "EFNO")
params = Dict(:fno => fit_fno(d_prim), :efno => fit_efno(d_prim))

@printf("modes / parameters:  FNO %d / %d,  EFNO %d / %d\n\n", count(d_prim.box),
    length(params[:fno]), length(d_prim.k2), length(params[:efno]))

## Evaluate: the same physical field, written in two cells

truths = Dict()
preds  = Dict()
errs   = Dict()
for (name, d) in [(:prim, d_prim), (:conv, d_conv)]
    f, truths[name] = probe(d)
    for tag in tags
        preds[(name, tag)] = apply_model(tag, params[tag], d, f)
        errs[(name, tag)] = relerr(preds[(name, tag)], truths[name])
    end
end

println("Relative L2 error on the probe [%] (Table 3, 'Single cell' columns):")
@printf("  %-6s %12s %12s\n", "", "Fitting", "Rectangular")
for tag in tags
    @printf("  %-6s %12.4g %12.4g\n", labels[tag],
        errs[(:prim, tag)], errs[(:conv, tag)])
end

## Supercell sweep

sweep = map(1:n_max) do n
    d = cell(n .* L_prim)
    f, gt = probe(d)
    (; n, err=Dict(tag => relerr(apply_model(tag, params[tag], d, f), gt) for tag in tags))
end
println("\nn×n supercells of the primitive cell [%] (Table 3, remaining columns):")
for s in sweep
    @printf("  n=%d  %s %9.4g  %s %9.4g\n", s.n,
        labels[:fno], s.err[:fno], labels[:efno], s.err[:efno])
end

## Figure (Figure 1)

Tbold = "TeX Gyre Termes Bold"
col = Dict(:fno => Makie.wong_colors()[3], :efno => Makie.wong_colors()[2])
set_theme!(paper_theme(; nrows=1, ncols=2.7))

corners(L; o=[0.0 0.0]) = [0 0; 1 0; 1 1; 0 1; 0 0] * transpose(L) .+ o

function draw_cell!(ax, field, L, win; colorrange, outlines=[corners(L)], outlinecolor=:black,
                    colormap=cgrad([:white, Makie.wong_colors()[1]]), res=400)
    N1, N2 = size(field)
    Li = inv(L)
    px, py = range(win[1]...; length=res), range(win[2]...; length=res)
    img = Matrix{Float64}(undef, res, res)
    for (jj, y) in enumerate(py), (ii, x) in enumerate(px)
        f1 = mod(Li[1, 1] * x + Li[1, 2] * y, 1.0)
        f2 = mod(Li[2, 1] * x + Li[2, 2] * y, 1.0)
        img[ii, jj] = field[floor(Int, f1 * N1)+1, floor(Int, f2 * N2)+1]
    end
    heatmap!(ax, px, py, img; colormap, colorrange, rasterize=4)
    for (n, v) in enumerate(outlines)
        lines!(ax, v[:, 1], v[:, 2]; color=(outlinecolor, 0.8), linewidth=0.9, linestyle=n == 1 ? :solid : :dash)
    end
end

crange = (0.0, maximum(truths[:prim]))

win = let c = corners(L_conv); pad = 0.15
    ((minimum(c[:, 1]) - pad, maximum(c[:, 1]) + pad),
     (minimum(c[:, 2]) - pad, maximum(c[:, 2]) + pad))
end

panel_idx = Ref(0)
next_label(text) = (panel_idx[] += 1; rich(rich(string(Char('a' + panel_idx[] - 1)) * ".", font=Tbold), " $text"))

fig = Figure(; figure_padding=(-0, 2, 2, 2))
fields_conv = [
    (next_label("Train & test data"), truths[:conv], [corners(L_conv), corners(L_prim; o=[0.0 0.5])]),
    (next_label("FNO (test)"), preds[(:conv, :fno)], [corners(L_conv)]),
    (next_label("EFNO (test)"), preds[(:conv, :efno)], [corners(L_conv)]),
]
for (c, (title, field, outs)) in enumerate(fields_conv)
    ax = Axis(fig[1, c]; title, aspect=DataAspect(), limits=win)
    hidedecorations!(ax); hidespines!(ax)
    draw_cell!(ax, field, L_conv, win; colorrange=crange, outlines=outs)
end

ax = Axis(fig[1, 4]; title=next_label("Scaling to larger cells"),
    xlabel="Supercell size n", ylabel="Error [%]",
    xticks=1:n_max, limits=((0.6, n_max + 0.4), (-5, 100)))
for tag in tags
    scatterlines!(ax, [s.n for s in sweep], [s.err[tag] for s in sweep];
        color=col[tag], markersize=4, linewidth=1.2, label=labels[tag])
end
axislegend(ax; position=:rb, framevisible=false, patchsize=(8, 4), padding=(2, 2, 0, 0))
colgap!(fig.layout, 4)
colsize!(fig.layout, 4, Relative(0.3))
resize_to_layout!(fig)

out = joinpath(@__DIR__, "..", "..", "figures", "fig1_cell_transfer")
mkpath(dirname(out))
save("$out.pdf", fig); save("$out.svg", fig)
println("\nSaved: $out.pdf and $out.svg")
