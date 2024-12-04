
############################################################
"""
$(TYPEDSIGNATURES)

Read the full output of a Monte Carlo simulation and return it in an 
   appropriate structure (depending on the algorithm used).
"""
function readMCoutput(datadir::String,simname::String ;
                      istart::Integer=1,iend::Integer=0)
  
    flname_jld2 = joinpath(datadir,simname*"_inp.jld2")
    fjd = JLD2.jldopen(flname_jld2, "r") 
    mcparams = fjd["mcparams"]
    close(fjd)
    
    if typeof(mcparams)<:AbstractHMCParams
        mcout = readHMCoutputH5(datadir,simname; istart=istart,iend=iend)

    elseif typeof(mcparams)<:AbstractMeHaParams

        if mcparams.algo==:ExtMetrop
            mcout = readExtMetropoutputH5(datadir,simname; istart=istart,iend=iend)
        end

    end
    
    return mcout
end
############################################################

"""
$(TYPEDEF)

Structure containing all the output saved from an HMC simulation.

# Fields 

$(TYPEDFIELDS)

"""
Base.@kwdef struct HMCOutput
    "Directory containing the output files"
    datadir::String
    "Name of the simulation (i.e., pars[\"simname\"])"
    simname::String
    "Starting model"
    startingmodel::Vector{Float64}
    "Unique identifier of the simulation"
    simulid::String
    "Struct containing all the parameters of the problem"
    userprob::MCSamplers.NLogPostPDF
    "Struct containing all the parameters for the MC simulation"
    hmcparams::AbstractHMCParams
    "If running a synthetic test, this can hold the 'target' model."
    refmod::Union{Vector{Float64},Nothing}
    "Array of iteration numbers"
    iter::Vector{Int64}
    "Array of iteration numbers corresponding to each saved model"
    iter_m::Vector{Int64}
    "Saved models"
    mods::Array{Float64,2}
    "Number of accepted models at each iteration"
    nacc::Vector{Int64}
    "Potential energy (= misfit value)"
    ucur::Vector{Float64}
    "Likelihood value"
    likelihoodval::Union{Vector{Float64},Vector{Union{Missing,Float64}},Nothing}
    "Prior value"
    priorval::Union{Vector{Float64},Vector{Union{Missing,Float64}},Nothing}
    "Potential energy (= misfit value) corresponding to each saved model"
    ucur_m::Vector{Float64}
    "Statistics about the simulation. They depend on the algorithm used for the simulation."
    stats::Union{Dict,Nothing}
end

############################################################

"""
$(TYPEDSIGNATURES)

Read the full output of an HMC simulation and return it in a
  HMCFullOutput structure.
"""
function readHMCoutputH5(datadir::String,simname::String ;
                         istart::Integer=1,iend::Integer=0)

    @assert istart>=1
    @assert iend>=0

    flname_jld2 = joinpath(datadir,simname*"_inp.jld2")
    flname_h5 = joinpath(datadir,simname*"_outp.h5")
    
    # get input data
    fjd = JLD2.jldopen(flname_jld2, "r") 
    mstart = fjd["startingmodel"]
    simulid_jld2 = fjd["simulid"]
    userprob = fjd["problemparams"]
    hmcparams = fjd["mcparams"]
    if haskey(fjd,"refmodel")
        refmod = fjd["refmodel"]
    else
        refmod = nothing
    end
    close(fjd)
    
    # read the file
    fid =  h5open(flname_h5,"r",swmr=true) 
    iter   = read(fid["iter"])
    # to avoid problems with fast saving runs
    nread = length(iter) 
    maxiterm = length(fid["iter_m"])
    if iend==0
        iend = maxiterm
    else
        if iend>maxiterm
            error("readhmcoutputfull(): iend higher than number of saved models.")
        end
    end
    
    iter_m = read(fid["iter_m"])[istart:iend]
    mods   = read(fid["mods"])[:,istart:iend]
    nacc   = read(fid["nacc"])[1:nread]
    ucur   = read(fid["ucur"])[1:nread]
    if haskey(fid,"likelihoodval")
        likelihoodval = read(fid["likelihoodval"])[1:nread]
    else
      likelihoodval = nothing  
    end
    if haskey(fid,"priorval")
        priorval = read(fid["priorval"])[1:nread]
    else
        priorval = nothing
    end
    ucur_m = read(fid["ucur_m"])[istart:iend]
    simulid_h5 = read(fid["simulid"])

    if hmcparams.algo==:NUTS
        stats = Dict()
        stats["epsilon"] = read(fid["epsilon"])
        stats["epsilonbar"] = read(fid["epsilonbar"])
        stats["treeheight"] = read(fid["treeheight"])
        stats["accepted_at_lfstep"] = read(fid["accepted_at_lfstep"])
        stats["lfsteps"] = read(fid["lfsteps"])

    else
        stats = nothing
    end
    
    close(fid)

    # check that the two files belong to the same simulation
    @assert simulid_jld2==simulid_h5


    out = HMCOutput(datadir=datadir,simname=simname,
                    startingmodel=mstart,
                    simulid = simulid_h5,
                    userprob = userprob,
                    hmcparams = hmcparams,
                    refmod = refmod,
                    iter = iter,
                    iter_m = iter_m,
                    mods = mods,
                    nacc = nacc,
                    ucur = ucur,
                    likelihoodval = likelihoodval,
                    priorval = priorval,
                    ucur_m = ucur_m,
                    stats = stats )

    return out
end

############################################################
############################################################
"""
$(TYPEDEF)

Structure containing all the output saved from an MeHa simulation.

# Fields 

$(TYPEDFIELDS)

"""
Base.@kwdef struct ExtMetropOutput
    "Directory containing the output files"
    datadir::String
    "Name of the simulation (i.e., pars[\"simname\"])"
    simname::String
    "Starting model"
    startingmodel::Vector{Float64}
    "Unique identifier of the simulation"
    simulid::String
    "Struct containing all the parameters of the problem"
    userprob::MCSamplers.NLogPostPDF
    "Struct containing all the parameters for the MC simulation"
    mcparams::ExtMetropParams
    "If running a synthetic test, this can hold the 'target' model."
    refmod::Union{Vector{Float64},Nothing}
    "Array of iteration numbers"
    iter::Vector{Int64}
    "Array of iteration numbers corresponding to each saved model"
    iter_m::Vector{Int64}
    "Saved models"
    mods::Array{Float64,2}
    "Number of accepted models at each iteration"
    nacc::Vector{Int64}
    "Negative log-likelihood"
    nllk::Vector{Float64}
    "Negative log-likelihood corresponding to each saved model"
    nllk_m::Vector{Float64}
end

############################################################


"""
$(TYPEDSIGNATURES)

Read the full output of an HMC simulation and return it in a
  HMCFullOutput structure.
"""
function readExtMetropoutputH5(datadir::String,simname::String ;
                               istart::Integer=1,iend::Integer=0)

    @assert istart>=1
    @assert iend>=0

    flname_jld2 = joinpath(datadir,simname*"_inp.jld2")
    flname_h5 = joinpath(datadir,simname*"_outp.h5")

    # get input data
    fjd = JLD2.jldopen(flname_jld2, "r") 
    mstart = fjd["startingmodel"]
    simulid_jld2 = fjd["simulid"]
    userprob = fjd["problemparams"]
    mcparams = fjd["mcparams"]
    if haskey(fjd,"refmodel")
        refmod = fjd["refmodel"]
    else
        refmod = nothing
    end
    close(fjd)

    # which algo was used?
    algo = mcparams.algo
    @assert (algo in [:ExtMetrop])

    # read the file
    fid =  h5open(flname_h5,"r",swmr=true) 
    iter   = read(fid["iter"])
    # to avoid problems with fast saving runs
    nread = length(iter) 
    maxiterm = length(fid["iter_m"])
    if iend==0
        iend = maxiterm
    else
        if iend>maxiterm
            error("readextmetropoutputfull(): iend higher than number of saved models.")
        end
    end
 
    iter_m = read(fid["iter_m"])[istart:iend]
    mods   = read(fid["mods"])[:,istart:iend]
    nacc   = read(fid["nacc"])[1:nread]
    nllk   = read(fid["nllk"])[1:nread]
    nllk_m = read(fid["nllk_m"])[istart:iend]
    simulid_h5 = read(fid["simulid"])

    close(fid)

    # check that the two files belong to the same simulation
    @assert simulid_jld2==simulid_h5

    out = ExtMetropOutput(datadir=datadir,simname=simname,
                        startingmodel=mstart,
                        simulid = simulid_h5,
                        userprob = userprob,
                        mcparams = mcparams,
                        refmod = refmod,
                        iter = iter,
                        iter_m = iter_m,
                        mods = mods,
                        nacc = nacc,
                        nllk = nllk,
                        nllk_m = nllk_m )

    return out
end

############################################################
