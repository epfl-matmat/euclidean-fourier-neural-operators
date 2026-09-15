module EuclideanFNO

using Random
using CairoMakie
import TuePlots
using FFTW
using Libxc
using LinearAlgebra: norm, dot, det
using Unitful, UnitfulAtomic

using Lux, WeightInitializers

include("data.jl")
export to_atomic_units, fourier_resample, resample_to_grid
include("plot_utils.jl")
export plot_slice_cartesian!, plot_crystal_cell!, paper_theme
include("xc_references.jl")
export vxc_pbe
include("layers.jl")
include("model_io.jl")
export rebuild_model, load_model, predict_vxc

end
