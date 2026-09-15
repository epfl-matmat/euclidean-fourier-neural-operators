# Experiment 2 (Section 4.2): overview figure (Figure 2).
# Requires the per-structure metric CSVs from eval.jl in results/.
# Produces figures/fig2_xc_overview.{pdf,svg}.
#
# Run from the repository root:
#   julia --project=. experiments/exp2_xc_potentials/plot_overview.jl

using JLD2, CairoMakie
using LinearAlgebra: det
using Statistics: mean, std
using Unitful, UnitfulAtomic
using EuclideanFNO

include("common.jl")

set_theme!(paper_theme(; nrows=1, ncols=1.5))

structures = load("data/mlrpa_dataset.jld2")["structures"]
bohr_to_A = ustrip(u"Å", 1u"bohr")
Ha_eV = ustrip(u"eV", 1u"hartree")

s = to_atomic_units(structures[62])  # first 16-atom diamond test structure
atom_labels = vcat((fill(t, n) for (t, n) in
                    zip(split(s.atom_types_str), s.n_atoms_per_type))...)
lat_A = s.lattice .* bohr_to_A
iz = last(findmax(vec(mean(s.rho; dims=(1, 2)))))
iz_frac = (iz - 1) / size(s.rho, 3)

fig = Figure(; figure_padding=(0, 5, -5, -10))

ax1 = Axis3(fig[1, 1];
    title="",
    xlabel="x (Å)", ylabel="y (Å)", zlabel="z (Å)",
    aspect=:data, azimuth=0.27pi, elevation=0.15pi, perspectiveness=0.3)
plot_crystal_cell!(ax1, lat_A, s.positions, atom_labels;
    markerscale=0.4, linewidth=0.8, strokewidth=0.5,
    slice_frac=iz_frac, slice_linewidth=1.5)

gl2 = fig[1, 2] = GridLayout()
ax2a = Axis(gl2[1, 1]; title=rich(rich("b.", font="TeX Gyre Termes Bold"), " Input: PBE density"), aspect=DataAspect(),
    ylabel="e₂ (Å)", xticklabelsvisible=false)
hm_rho = plot_slice_cartesian!(ax2a, s.rho, lat_A, iz;
    colormap=:inferno, colorrange=(0, 0.3), rasterize=4)
Colorbar(gl2[1, 2], hm_rho; width=5, ticks=0:0.1:0.3, height=Relative(0.9))

ax2b = Axis(gl2[2, 1]; title=rich(rich("c.", font="TeX Gyre Termes Bold"), " Output: RPA potential"), aspect=DataAspect(),
    xlabel="e₁ (Å)", ylabel="e₂ (Å)")
hm_vxc = plot_slice_cartesian!(ax2b, s.vxc .* Ha_eV, lat_A, iz;
    colormap=:viridis, rasterize=4)
Colorbar(gl2[2, 2], hm_vxc; width=5, height=Relative(0.9))
colgap!(gl2, 1, 5)
rowgap!(gl2, 1, 5)

RESULTS_DIR = joinpath(@__DIR__, "..", "..", "results")
_wong = Makie.wong_colors()
METHOD_COLORS = Dict("PBE" => :gray60, "EFNO" => _wong[2], "FNO" => _wong[3])
method_color(name) = get(METHOD_COLORS, name, :gray)

xfn_volume(s) = abs(det(to_atomic_units(s).lattice)) * bohr_to_A^3

SPLIT_REGIONS = Dict(
    :diamond => (train=(0, 10^1.75), test=(10^1.9, Inf)),
    :water   => (train=(0, 10^2.6), test=(10^2.85, Inf)),
)

function plot_error_volume!(ax, experiment; show_ylabel=true)
    raw = load_metric_csv(joinpath(RESULTS_DIR, "$(experiment)_wrms.csv"))
    agg = aggregate_per_structure(raw)
    xs = [xfn_volume(structures[idx]) for idx in agg.structure_idxs]
    for name in agg.model_names
        scatter!(ax, xs, agg.values[name]; color=method_color(name), marker=:circle, markersize=2)
        any(>(0), agg.spread[name]) && errorbars!(ax, xs, agg.values[name], agg.spread[name];
            color=(method_color(name), 0.8), linewidth=0.5)
    end
    regions = SPLIT_REGIONS[experiment]
    xlo, xhi = extrema(xs)
    x_min, x_max = xlo * 10^(-0.05), xhi * 10^(0.05)
    xlims!(ax, x_min, x_max)
    tr_lo, tr_hi = max(regions.train[1], x_min), min(regions.train[2], x_max)
    te_lo, te_hi = max(regions.test[1], x_min), min(regions.test[2], x_max)
    vspan!(ax, [tr_lo], [tr_hi]; color=(:gray, 0.07))
    vspan!(ax, [te_lo], [te_hi]; color=(:gray, 0.15))
    for x in (tr_lo, tr_hi, te_lo, te_hi)
        vlines!(ax, x; color=(:gray60, 0.5), linewidth=0.5)
    end
    ylims!(ax, 10, nothing)
    text!(ax, "Train"; position=(sqrt(tr_lo * tr_hi), 12), align=(:center, :bottom), fontsize=6)
    text!(ax, "Test"; position=(sqrt(te_lo * te_hi), 12), align=(:center, :bottom), fontsize=6)
end

gl3 = fig[1, 3] = GridLayout()

ax3a = Axis(gl3[1, 1];
    title=rich(rich("d.", font="TeX Gyre Termes Bold"), " Results: Diamond"), titlealign=:left,
    ylabel="Error",
    xscale=log10, yscale=log10,
    yticks=[10, 100, 1000],
    ytickformat=vs -> [string(round(Int, v)) for v in vs])
plot_error_volume!(ax3a, :diamond)

ax3b = Axis(gl3[2, 1];
    title=rich(rich("e.", font="TeX Gyre Termes Bold"), " Results: Water"), titlealign=:left,
    xlabel="Cell volume (Å³)",
    ylabel="Error",
    xscale=log10, yscale=log10,
    yticks=[10, 100, 1000],
    ytickformat=vs -> [string(round(Int, v)) for v in vs])
plot_error_volume!(ax3b, :water)

rowgap!(gl3, 1, 5)

model_elems = [MarkerElement(color=method_color(n), marker=:circle, markersize=4) for n in METHOD_ORDER]
axislegend(ax3a, model_elems, METHOD_ORDER; position=(0.6, 0.0),
    patchsize=(5,5), patchlabelgap=2, padding=(2,2,1,1), framevisible=true, rowgap=0, labelsize=7)
axislegend(ax3b, model_elems, METHOD_ORDER; position=(0.45, 0.0),
    patchsize=(5,5), patchlabelgap=2, padding=(2,2,1,1), framevisible=true, rowgap=0, labelsize=7)

Label(
      fig[1, 1, Top()], 
      rich(rich("a.", font="TeX Gyre Termes Bold"), " Diamond unit cell");
      font="TeX Gyre Termes", fontsize=9,
      padding=(0, 0, 0, 0),
      halign=:left,
      valign=:bottom,
)

colgap!(fig.layout, 1, -15)
colgap!(fig.layout, 2, 10)

resize_to_layout!(fig)
mkpath(joinpath(@__DIR__, "..", "..", "figures"))
base = joinpath(@__DIR__, "..", "..", "figures", "fig2_xc_overview")
save("$base.pdf", fig); save("$base.svg", fig); save("$base.png", fig; px_per_unit=3)
@info "Saved fig2_xc_overview.{pdf,svg,png}"  # png is used as the README hero image
