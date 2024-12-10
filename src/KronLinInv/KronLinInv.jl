

module KronLinInv

export CovMats,FwdOps,KLIFactors
export calcfactors,kli_posteriormean,kli_blockpostcov


using Distributed
using LinearAlgebra

include("common_structfunc.jl")

include("kronlininv_parallel.jl")

include("kronlininv_serial.jl")


end # module

