
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

## L-BFGS algo
export lmbfgs
include("BFGS/lim-mem_bfgs.jl")



end # module
