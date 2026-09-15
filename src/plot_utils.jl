"""
    plot_slice_cartesian!(ax, field, lattice, iz; atom_positions=nothing,
                          slice_thickness=0.05, colormap=:viridis, kwargs...) -> scatter

Plot a 2D slice of `field` at grid index `iz` along the third lattice direction,
rendered in Cartesian space within the (a₁, a₂) plane. Each grid point is drawn
as a filled rectangle sized to tile the cell - no interpolation, so this is
honest for non-orthogonal lattices.

`lattice` is the 3×3 matrix with lattice vectors as ROWS (VASP convention).
The in-plane orthonormal basis is e₁ ∥ a₁ and e₂ ⊥ a₁ within the (a₁, a₂) plane,
so axis labels read as `e₁ (Å)` / `e₂ (Å)` in the caller.
`atom_positions` (3×N, fractional) overlays atoms within `slice_thickness` of
the slice along the third fractional axis.
"""
function plot_slice_cartesian!(
        ax, field, lattice, iz;
        atom_positions = nothing, slice_thickness = 0.05,
        colormap = :viridis, kwargs...
    )
    a1 = lattice[1, :]
    a2 = lattice[2, :]
    Nx, Ny, Nz = size(field)

    # Orthonormal in-plane basis
    e1 = a1 / norm(a1)
    a2_perp = a2 .- dot(a2, e1) .* e1
    e2 = a2_perp / norm(a2_perp)

    # Cartesian grid positions projected onto (e₁, e₂)
    fx = range(0, 1 - 1 / Nx; length = Nx)
    fy = range(0, 1 - 1 / Ny; length = Ny)
    rx = vec([dot(fi * a1 + fj * a2, e1) for fi in fx, fj in fy])
    ry = vec([dot(fi * a1 + fj * a2, e2) for fi in fx, fj in fy])

    # Marker size: side length of one grid cell along each direction (exact tiling)
    ms = (norm(a1) / Nx, norm(a2_perp) / Ny) .* 2
    hm = scatter!(
        ax, rx, ry;
        color = vec(field[:, :, iz]), colormap = colormap,
        marker = :rect, markerspace = :data,
        markersize = ms,
        strokewidth = 0, kwargs...
    )

    if atom_positions !== nothing
        iz_frac = (iz - 1) / Nz
        near = findall(p -> abs(p - iz_frac) < slice_thickness, atom_positions[3, :])
        atom_rx = [dot(atom_positions[1, i] * a1 + atom_positions[2, i] * a2, e1) for i in near]
        atom_ry = [dot(atom_positions[1, i] * a1 + atom_positions[2, i] * a2, e2) for i in near]
        scatter!(
            ax, atom_rx, atom_ry;
            color = :white, strokecolor = :black, strokewidth = 1, markersize = 10
        )
    end

    return hm
end

"""
    plot_crystal_cell!(ax::Axis3, lattice, positions, atom_labels;
                       slice_frac=nothing, slice_thickness=0.05)

Draw a wireframe parallelepiped unit cell (`lattice`: 3×3, vectors as ROWS) with
atom spheres (`positions`: 3×N fractional, `atom_labels`: per-atom element string,
colored by element) into an `Axis3` panel. If `slice_frac` (fractional coordinate
along the third lattice direction) is given, draw the corresponding (a₁, a₂)-plane
as a transparent quad and outline atoms within `slice_thickness` of it.
"""
const ELEMENT_STYLE = Dict(
    "H"  => (color = RGBf(0.85, 0.85, 0.85), size = 6),
    "C"  => (color = RGBf(0.25, 0.25, 0.25), size = 14),
    "N"  => (color = RGBf(0.16, 0.33, 0.73), size = 13),
    "O"  => (color = RGBf(0.82, 0.16, 0.16), size = 16),
)

function plot_crystal_cell!(
        ax, lattice, positions, atom_labels;
        slice_frac = nothing, slice_thickness = 0.05, markerscale = 1.0, linewidth = 1.5,
        strokewidth = 1, slice_linewidth = 3,
    )
    a1, a2, a3 = lattice[1, :], lattice[2, :], lattice[3, :]
    corners = [zeros(3), a1, a2, a3,
               a1 .+ a2, a1 .+ a3, a2 .+ a3, a1 .+ a2 .+ a3]
    edges = [(i, j) for (i, j) in ((1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 5),
             (3, 7), (4, 6), (4, 7), (5, 8), (6, 8), (7, 8))]

    az = hasproperty(ax, :azimuth) ? ax.azimuth[] : 0.35pi
    el = hasproperty(ax, :elevation) ? ax.elevation[] : 0.15pi
    view_dir = [cos(el)*cos(az), cos(el)*sin(az), sin(el)]
    edge_depth(i, j) = dot(0.5 .* (corners[i] .+ corners[j]), view_dir)
    sort!(edges; by=((i, j),) -> edge_depth(i, j))

    function draw_edges!(idxs)
        for (i, j) in edges[idxs]
            p, q = corners[i], corners[j]
            lines!(ax, [p[1], q[1]], [p[2], q[2]], [p[3], q[3]];
                   color = :black, linewidth = linewidth)
        end
    end

    draw_edges!(1:length(edges)-3)

    if slice_frac !== nothing
        off = slice_frac .* a3
        verts = [Point3f(off), Point3f(off .+ a1), Point3f(off .+ a1 .+ a2), Point3f(off .+ a2)]
        mesh!(ax, verts, [1 2 3; 1 3 4];
              color = RGBAf(0.2, 0.5, 0.9, 0.25), transparency = true)
        for (i, j) in ((1, 2), (2, 3), (3, 4), (4, 1))
            p, q = verts[i], verts[j]
            lines!(ax, [p[1], q[1]], [p[2], q[2]], [p[3], q[3]];
                   color = RGBf(0.2, 0.5, 0.9), linewidth = slice_linewidth)
        end
    end

    cart = lattice' * positions  # 3×N cartesian
    for lab in unique(atom_labels)
        idxs = findall(==(lab), atom_labels)
        style = get(ELEMENT_STYLE, lab, (color = :purple, size = 10))
        pts = [Point3f(cart[:, i]) for i in idxs]
        ms = style.size * markerscale
        scatter!(ax, pts; color = style.color, markersize = ms,
                 strokecolor = (:black, 0.7), strokewidth = strokewidth)
    end

    draw_edges!(length(edges)-2:length(edges))

    # Tight limits: zoom the 3D view to the cell bounding box with a small margin
    bb = reduce(hcat, corners)
    pad = 0.05 * norm(a1)
    ax.limits = ntuple(d -> (extrema(bb[d, :])[1] - pad, extrema(bb[d, :])[2] + pad), 3)
    return ax
end

function paper_theme(; nrows=1, ncols=1, height_ratio=nothing)
    Tfont = "TeX Gyre Termes"
    setting = TuePlots.SETTINGS[:NEURIPS]
    main, sm = TuePlots.base_fontsize_to_sizes(setting.base_fontsize)
    hr_kw = isnothing(height_ratio) ? (;) : (; subplot_height_to_width_ratio=height_ratio)
    base = Theme(setting; nrows, ncols, font=false, fontsize=false, figsize=true, hr_kw...)
    return merge(base, Theme(
        font     = Tfont,
        fontsize = main,
        Axis = (
            titlefont      = Tfont,
            titlealign      = :left,
            xlabelfont     = Tfont,
            ylabelfont     = Tfont,
            xlabelsize     = main,
            ylabelsize     = main,
            xticklabelsize = sm,
            yticklabelsize = sm,
            xlabelpadding  = 1,
            ylabelpadding  = 2,
            xticklabelpad  = 1,
            yticklabelpad  = 2,
        ),
        Axis3 = (
            titlefont        = Tfont,
            titlealign       = :left,
            xlabelfont       = Tfont,
            ylabelfont       = Tfont,
            zlabelfont       = Tfont,
            xlabelsize       = main,
            ylabelsize       = main,
            zlabelsize       = main,
            xticklabelsize   = sm,
            yticklabelsize   = sm,
            zticklabelsize   = sm,
            xlabeloffset     = 15,
            ylabeloffset     = 15,
            zlabeloffset     = 15,
            xticklabelpad    = 1,
            yticklabelpad    = 1,
            zticklabelpad    = 1,
            xspinewidth      = 0.5,
            yspinewidth      = 0.5,
            zspinewidth      = 0.5,
            xticksize        = 2,
            yticksize        = 2,
            zticksize        = 2,
            xtickwidth       = 0.5,
            ytickwidth       = 0.5,
            ztickwidth       = 0.5,
            titlegap         = 0,
        ),
        Colorbar = (
            ticklabelsize = sm,
            labelsize     = main,
            labelfont     = Tfont,
            ticklabelfont = Tfont,
            ticksize      = 2,
            tickwidth     = 0.5,
        ),
        Legend = (
            labelsize = sm,
            labelfont = Tfont,
        ),
    ))
end
