
"""
A module to perform Kronecker-product-based linear inversion of geophysical (or other kinds of) data under Gaussian and separability assumptions. The code computes the posterior mean model and the posterior covariance matrix (or subsets of it) in an efficient manner (parallel algorithm) taking into account 3-D correlations both in the model parameters and in the observed data.

"""
module KronLinInv

export CovMats,FwdOps,KLIFactors
export kli_calcfactors,kli_posteriormean,kli_blockposteriorcov


using Distributed
using LinearAlgebra

include("common_structfunc.jl")

include("kronlininv_parallel.jl")

include("kronlininv_serial.jl")


end # module

