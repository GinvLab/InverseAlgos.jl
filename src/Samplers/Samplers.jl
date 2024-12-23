

"""
Samplers

A module to solve inverse problems using Monte Carlo methods.
Main targets are geophysical inverse problems.
It contains a set of samplers such as the Hamiltonian Monte Carlo
algorithm and its variant NUTS and the Extended Metropolis algorithm.

# Exports

$(EXPORTS)
"""
module Samplers

using Distributed
using LinearAlgebra
using HDF5,JLD2
using Printf
using Dates
using UUIDs
using DocStringExtensions

export runMC, contrunMC
export HMCParams, NUTSParams
export ExtMetropParams
export clear_stflags_hdf5
export readMCoutput
export AvEpsFromHMCIter, AvEpsFromTree

# println("** Setting: LinearAlgebra.BLAS.set_num_threads(1) **")
# LinearAlgebra.BLAS.set_num_threads(1)

## main code

############################################
## Maximum number of iterations to fulfill constraints, to
##  avoid infinite loops. If reached, throws an error.
const maxitconstr = 10_000
############################################

# generic Monte Carlo functions and structures
include("genericMCfuncstructs.jl")
# Hamiltonian Monte Carlo functions and structures
include("commonHMCfuncstructs.jl")
# Metropolis-Hasings functions and structures
include("commonMeHafuncstructs.jl")
# parallel stuff for kinetic energy
include("parallelfunc.jl")

# "plain" HMC sampler
include("hammcsampler.jl")
# NUTS sampler
include("nuts.jl")
# NUTS sampler for polygonal bodies 
include("nuts_polyg.jl")
# Extended Metropolis algo
include("extmetrop.jl")

# save and print stuff
include("saveprintstuff.jl")
# utilities
include("utils.jl")

## clear status flags in HDF5 file (alternative to h5clear -s <file>)
include("clearflags_superblock_hdf5.jl")


##--------------------------
end # module
