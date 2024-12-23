
#############################################################

abstract type AbstractMCParams end

###############################################################

"""
$(TYPEDEF)

Structure representing the negative logarithm of the posterior pdf.
 It is split in the two fields 'likelihood' and 'prior'.

The NLogPost struct is also made callable for computing the negative logarithm of the posterior pdf 
 and its gradient with respect to model parametes.

# Fields 

$(TYPEDFIELDS)
"""
Base.@kwdef struct NLogPostPDF
    "likelihood functional"
    likelihood
    "prior model"
    prior
    "last value of likelihood and prior"
    curvals::Union{Vector{Union{Missing,Float64}},Vector{Float64}}

    function NLogPostPDF(likelihood,prior)
        if likelihood==nothing && prior==nothing
            error("usermodel::NLogPostPDF(): both likelihood and prior == nothing")
        end
        if prior==nothing
            curvals = [0.0,missing]
        elseif likelihood==nothing
            curvals = [missing,0.0]
        else
            curvals = zeros(2)
        end
        return new(likelihood,prior,curvals)
    end
end

##############################################

# """

# Make type NLogPostPDF callable for computing the negative logarithm of the posterior pdf 
#  and its gradient with respect to model parametes.    
# """
function (nlogppd::NLogPostPDF)(m::Vector{Float64},kind::Symbol)::Union{Vector{Float64},Float64}
    # must return a vector

    if nlogppd.likelihood==nothing && nlogppd.prior==nothing
        error("nlogppd::NLogPostPDF(): both likelihood and prior == nothing")
    end
    
    if kind==:nlogpdf
        #############################################
        ## compute the neglogdensity value         ##
        #############################################
        if nlogppd.prior==nothing
            nlogpdf = nlogppd.likelihood(m,:nlogpdf)
            nlogppd.curvals[1] = nlogpdf
            nlogppd.curvals[2] = missing
        elseif nlogppd.likelihood==nothing
            nlogpdf = nlogppd.prior(m,:nlogpdf)
            nlogppd.curvals[1] = missing
            nlogppd.curvals[2] = nlogpdf
        else
            llk = nlogppd.likelihood(m,:nlogpdf)
            pri = nlogppd.prior(m,:nlogpdf)
            nlogppd.curvals .= (llk,pri)
            nlogpdf = llk + pri
        end
        return nlogpdf

    elseif kind==:gradnlogpdf
        #################################################
        ## compute the gradient of the misfit function ##
        #################################################
        if nlogppd.prior==nothing
            gradnlogpdf = nlogppd.likelihood(m,:gradnlogpdf)
        elseif nlogppd.likelihood==nothing
            gradnlogpdf = nlogppd.prior(m,:gradnlogpdf)
        else
            gradnlogpdf = nlogppd.likelihood(m,:gradnlogpdf) .+ nlogppd.prior(m,:gradnlogpdf)
        end
        return gradnlogpdf
       
    else
        error("nlogppd::NLogPostPDF(): Wrong argument 'kind': $kind ...")
    end
end


###############################################


"""
$(TYPEDSIGNATURES)

Start a Monte Carlo (MC) simulation.

# Arguments

- `pars`: a dictionary of parameters for the MC run, namely
    * `pars["maxiter"]`: maximum number of iterations
    * `pars["saveevery"]`: every how many HMC iterations save models to file
    * `pars["simname"]`: name of the simulation, used also for the name of the output files
    * `pars["outdir"]`: name of output directory, where to save results of the simulation
    * `pars["infoevery"]`: every how many HMC iterations to print info
    * `pars["savecalcdata"]`: whether to save calculated data or not (default is false)
    * `pars["stdout"]`: where to print the info messages. Can be
       - `nothing` (default) to print to standard output in the terminal
       - "file" to print to a file
       - "oneline" to print to standard output updating only a single line (useful in Jupyter notebooks).

- `MCpar`: an AbstractMCParams, containig all MC-related parameters, such as the mass matrix,
    the constraints, time step and number of iterations for the Hamiltonian dynamics.
    See, e.g,, [`HMCParams`](@ref), [`NUTSParams`](@ref), or [`ExtMetropParams`](@ref).

- `mstart`: the starting model, a vector (Vector{Float64})

- `likelihood`: the user-defined struct/type representing the pdf for the likelihood functional

- `prior`: the user-defined struct/type representing the pdf of the prior model

"""
function runMC(pars::Dict, MCpar::AbstractMCParams, mstart::Vector{Float64} ;
               likelihood=nothing, prior=nothing )

    pars["continueold"] = false

    #---------------------------------------
    if likelihood==nothing && prior==nothing
        error("runMC: Neither likelihood nor prior model specified. ")
    else
        nlogppd = NLogPostPDF(likelihood,prior)
    end
    #---------------------------------------

    ## run the simulation
    if typeof(MCpar)<:AbstractHMCParams
        ## HMC simulation (plain or NUTS)
        if isa(MCpar,HMCParams) && MCpar.algo==:plainHMC
            runsimulhmc(pars,MCpar,nlogppd,mstart)
        elseif isa(MCpar,NUTSParams) && MCpar.algo==:NUTS
            runsimulNUTS(pars,MCpar,nlogppd,mstart)
        else
            error("runMC(): Wrong kind of algorithm.")
        end

    elseif typeof(MCpar)<:AbstractMeHaParams
        ## M-H simulation type
        if isa(MCpar,ExtMetropParams) && MCpar.algo==:ExtMetrop
            runsimulextmetrop(pars,MCpar,nlogppd,mstart)
        else
            error("runMC(): Wrong kind of algorithm.")
        end
        
    end
    
    return 
end


####################################################

"""
$(TYPEDSIGNATURES)

Continue an old run of MC simulation either for given `pars` only, using old `MCParams` 
  or providing both `pars` and new `MCParams`, i.e., new simulation input parameters.
  In the first case, the input `MCParams` are read directly from the .jld2 file.

# Arguments

- `pars`: a dictionary of parameters for the MC run, namely
    * `pars["maxiter"]` maximum number of iterations
    * `pars["saveevery"]` every how many HMC iterations save models to file
    * `pars["simname"]` name of the simulation, used also for the name of the output files
    * `pars["outdir"]` name of output directory, where to save results of the simulation
    * `pars["infoevery"]` every how many MC iterations to print info
    * `pars["savecalcdata"]` whether to save calculated data or not (default is false)

- `MCpar`: an AbstractMCParams, containig all MC-related parameters for the type of
    simulation requested. For instance, in case of using the Hamiltonian Monte Carlo
    algorithm, it includes the mass matrix, the constraints, time step and number of
    iterations for the Hamiltonian dynamics.
    See, e.g,, [`HMCParams`](@ref), [`NUTSParams`](@ref), or [`ExtMetropParams`](@ref).

"""
function contrunMC(pars::Dict, MCpar::Union{AbstractMCParams,Nothing}=nothing)

    pars["continueold"] = true
    
    ## get the input parameters for the simulation
    outfile_inp = joinpath(pars["outdir"],pars["simname"]*"_inp.jld2")
    outfile_out = joinpath(pars["outdir"],pars["simname"]*"_outp.h5")


    if MCpar==nothing
        ## determine whether it'an HMC or M-H simulation
        mcparkey = "mcparams"
        fjd = JLD2.jldopen(outfile_inp,"r")
        mcpartype = typeof(fjd[mcparkey])
        # @show mcpartype
        close(fjd)
    
    elseif MCpar!=nothing
        ## new HMCpar

        # used to set ϵ for NUTS in "saveprintstuff()"
        pars["new_"*mcparkey] = true
        
        ## https://github.com/JuliaIO/JLD2.jl/issues/38

        ## read current inp file
        fjd = JLD2.jldopen(outfile_inp,"r") 
        lstkey = keys(fjd)
        inpdata = Dict()
        for ke in lstkey
            obj=fjd[ke]
            if typeof(obj)<:JLD2.Group
                error("continuerunHMC(): $outfile_inp contains groups, feature not yet implemented. Aborting.")
            else
                inpdata[ke]=obj
            end
        end
        close(fjd)
      
        ## delete current *_inp.jld2 file
        rm(outfile_inp)

        ## create new file
        itmcparchange = Array(h5read(outfile_out,"iter"))[end]
        JLD2.jldopen(outfile_inp,"w") do fjd
            if !(mcparkey*"_changed_at_iter" in lstkey)
                push!(lstkey,mcparkey*"_changed_at_iter")
                inpdata[mcparkey*"_changed_at_iter"] = itmcparchange
            end
            for ke in lstkey
                if ke==mcparkey
                    # set the new HMCpar
                   fjd[ke] = HMCpar 
                else
                    # copy old data
                    fjd[ke] = inpdata[ke]
                end
            end 
        end
        ## set dict to nothing and run garbage collection
        ##    (inpdata can be a large struct in memory...)
        inpdata = nothing
        GC.gc()        
    end

    ## read the input file containing the HMCparams and nlogppd
    fjd = JLD2.jldopen(outfile_inp,"r+") 
    nlogppd = fjd["problemparams"]
    MCpar = fjd[mcparkey]
    if !(mcparkey*"_changed_at_iter" in keys(fjd))
        fjd[mcparkey*"_changed_at_iter"] = 0
    end
    pars["simid_jld2"] = string(fjd["simulid"]) 
    close(fjd)

    
    ## output file .h5
    fid_h5 = h5open(outfile_out,"r",swmr=true)
    pars["simid_h5"] = string(read(fid_h5["simulid"]))
    mods = fid_h5["mods"]
    lastmo = size(mods,2)
    mstart = convert.(Float64,vec(mods[:,lastmo]))
    close(fid_h5)   

    ## check that simulation id is the same for .h5 and .jld2
    if pars["simid_h5"] != pars["simid_jld2"]
        @show pars["simid_h5"],pars["simid_jld2"]
        error("The simulation id (UUID) is not the same for input (.jld2) and output (.h5) files.")
    end
    
    ## finally run the simulation
    if mcpartype<:AbstractHMCParams
        ## HMC simulation
        if MCpar.algo==:plainHMC
            runsimulhmc(pars,MCpar,nlogppd,mstart)
        elseif MCpar.algo==:NUTS
            runsimulNUTS(pars,MCpar,nlogppd,mstart)
        else
            error("runMC(): Wrong kind of algorithm.")
        end

    elseif mcpartype<:AbstractMeHaParams
        ## M-H simulation
        if MCpar.algo==:ExtMetrop
            runsimulextmetrop(pars,MCpar,nlogppd,mstart)
        else
            error("runMC(): Wrong kind of algorithm.")
        end
        
    else
        error("runMC(): Wrong kind of MC parameters")
    end

    return 
end

####################################################
