
"""
 Optimizers

 A module collecting a set of deterministic inversion algorithms. Main targets are geophysical inverse problems.

 # Exports

 $(EXPORTS)
 """
module Optimizers

using LinearAlgebra
using HDF5
using DocStringExtensions
using REPL
using Logging


export lmbfgs,gaussnewton

# Line-search algos
include("Line_search/line_search_algos.jl")
# L-BFGS algo
include("BFGS/lim-mem_bfgs.jl")
# Gauss-Newton algo
include("GaussNewton/gauss_newton.jl")
# Save output
include("saveoutput.jl")

end # module
