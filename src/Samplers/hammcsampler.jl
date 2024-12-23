    


##================================================================
# params:
# pars["maxiter"]
# pars["saveevery"]
# pars["simname"]
# pars["outdir"]
# pars["infoevery"]

##===============================================

#############################################################

"""
$(TYPEDEF)

Structure containing the parameters for the plain HMC algorithm.

# Fields 

$(TYPEDFIELDS)

"""
struct HMCParams <: AbstractHMCParams
    "perform parallel calculations of kinetic energy by first reading the mass matrix, etc. from an HDF5 file"
    parallelKinEn::Bool
    "mass matrix file name"
    massMfile::String
    "Inverse of the mass matrix"
    invmassM::Union{AbstractMatrix{Float64},Array{Future,1}}
    "*Lower* triangular matrix from Cholesky decomposition of mass matrix"
    LcholmassM::Union{LowerTriangular{Float64,Matrix{Float64}},Diagonal{Float64}, Array{Future,1}}
    "is this a constrained HMC simulation?"
    constrained::Bool
    "lower constraints"
    mlowerconstr::Union{Vector{Float64},Nothing}
    "upper constraints"
    mupperconstr::Union{Vector{Float64},Nothing}
    "time step for the leap-frog algorithm (Hamiltonian dynamics)"
    ϵ::Float64
    "number of iterations for the leap-frog algorithm (Hamiltonian dynamics)"
    L::Int64
    "number of workers (cores) for parallel calculations" 
    numwks::Int64
    "which algorithm"
    algo::Symbol

    ## multiple dispatch works only on positional arguments
    function HMCParams( massMfile,constrained,mlowerconstr,mupperconstr,ϵ,L)        
        parallelKinEn=true
        ## arrays of Future because computations will be parallel, so
        ##  a worker will hold only a chunck of these matrices
        numwks = nworkers() # available workers
        println("HMCParams: using $numwks workers for kinetic energy calculations")
        invmassM = Array{Future,1}(undef,numwks)
        LcholmassM = Array{Future,1}(undef,numwks)
        if !constrained
            mlowerconstr,mupperconstr = nothing,nothing
        end
        algo = :plainHMC
        return new(parallelKinEn,massMfile,invmassM,LcholmassM,constrained,
                   mlowerconstr,mupperconstr, ϵ,L,numwks,algo)
    end
    ## multiple dispatch works only on positional arguments
    function HMCParams( invmassM,LcholmassM,constrained,mlowerconstr,mupperconstr,ϵ,L)
        parallelKinEn=false
        if !constrained
            mlowerconstr,mupperconstr  = nothing,nothing
        end
        massMfile=""
        numwks = nworkers() # available workers
        algo = :plainHMC
        return new(parallelKinEn,massMfile,invmassM,LcholmassM,constrained,
                   mlowerconstr,mupperconstr,ϵ,L,numwks,algo)
    end

end
   

#############################################################
"""
$(TYPEDSIGNATURES)

Outer constructor for HMCParams composite type to make use of keywords 
  arguments while still having some dispatch.
"""
function HMCParams(; parallelKinEn::Bool=false, constrained::Bool, ϵ::Float64,L::Int64,
                   massMfile::Union{String,Nothing}=nothing,
                   invmassM::Union{Array{Float64,2},Diagonal{Float64},Nothing}=nothing,
                   LcholmassM::Union{Array{Float64,2},LowerTriangular{Float64,Array{Float64,2}},Diagonal{Float64},Nothing}=nothing,
                   mlowerconstr::Union{Vector{Float64},Nothing}=nothing,
                   mupperconstr::Union{Vector{Float64},Nothing}=nothing )
    ## outer constructor
    ## dispatch works only on positional arguments, so we define
    ##  outer constructor methods with keywords to match the positional
    ##  arguments of the inner constructors
    if constrained
        @assert typeof(mlowerconstr)==Vector{Float64}
        @assert typeof(mupperconstr)==Vector{Float64}
    else
        @assert typeof(mlowerconstr)==Nothing
        @assert typeof(mupperconstr)==Nothing
    end

    if parallelKinEn
        @assert typeof(massMfile)==String
        @assert typeof(invmassM)==Nothing
        @assert typeof(LcholmassM)==Nothing
            
        hmcpar = HMCParams( massMfile,constrained,mlowerconstr,mupperconstr,ϵ,L )
    else
        @assert typeof(massMfile)==Nothing
        @assert typeof(invmassM)==Array{Float64,2} || typeof(invmassM)==Diagonal{Float64,Array{Float64,1}}
        @assert (typeof(LcholmassM)==Array{Float64,2} || typeof(LcholmassM)==LowerTriangular{Float64,Array{Float64,2}} || typeof(LcholmassM)==Diagonal{Float64,Array{Float64,1}} )

        hmcpar = HMCParams( invmassM,LcholmassM,constrained,mlowerconstr,mupperconstr,ϵ,L )
    end
    return hmcpar
end


###############################################################
###############################################################

"""
$(TYPEDSIGNATURES)

The actual function starting a HMC simulation.
See [`runMC`](@ref) for arguments explanation.
"""
function runsimulhmc(pars::Dict,HMCpar::HMCParams,nlogppd::NLogPostPDF,
                     mstart::Vector{Float64})

    #---------------------------------------------
    printstyled("\n ====================================",color=:cyan)
    printstyled("\n      Hamiltonian Monte Carlo      ",bold=true,color=:normal)
    printstyled("\n ====================================\n",color=:cyan)
    if pars["continueold"]
        printstyled(" Continuing previous simulation $(pars["simname"]) \n",color=:yellow)
    else
        printstyled(" Running simulation $(pars["simname"]) \n",color=:yellow)
    end
    #---------------------------------------

    #---------------------------------------
    # Load in parallel to workers the mass matrix stuff
    if HMCpar.parallelKinEn
        # multiple workers
        grptask = paraloadmassM!(HMCpar)
    else
        # single worker, placeholder
        grptask = [0 0]
        # since it's all in memory, do a check of the inverse of mass matrix
        @assert isposdef(HMCpar.invmassM)
    end

    #---------------------------------------
    maxiter = pars["maxiter"]::Int64
    saveevery = pars["saveevery"]::Int64
    ntobesaved = div(maxiter,saveevery)+1 # integer division
    mcur = copy(mstart)

    ##=================================================##
    # Init output file and save input parameters
    fid_h5,nacc,nsaved,Ucur,previt = initsavestuff!(pars,nlogppd,HMCpar,mcur)

    ##=================================================##
    dofwdchecks = (isdefined(nlogppd.likelihood,:dofwdchecks) && nlogppd.likelihood.dofwdchecks)

    ##=========================##
    # Initial potential energy value
    println(" Initial Ucur: $(round(Ucur, digits=3))")
    
    ##=========================##
    starttime = time()
    naccvec = Vector{Integer}(undef,maxiter-previt)
    llkprivals = Vector{Union{Missing,Float64}}(undef,2)
    llkprivals .= nlogppd.curvals

    ## big loop
    for it=1:(maxiter-previt)
        # the "actual" iteration number (in case of continuing an old simulation)
        actualit = it + previt

        # iteration of the Hamiltonian MC
        if dofwdchecks
            # HMC iterations performing also checks specific for given forward problems
            mcur,accept,Ucur = oneiterhmc_polyg(nlogppd,HMCpar,mcur,Ucur,grptask)
        else
            # standard iteration
            mcur,accept,Ucur = oneiterhmc!(nlogppd,HMCpar,mcur,Ucur,grptask,llkprivals,
                                           curiter=actualit)
        end

        # if accepted
        if (accept==true) nacc=nacc+1 end
        # vector storing all history of accepted/rejected for current run,
        #  but taking into account the starting value of 'nacc' from previous runs
        naccvec[it] = nacc
        
        ## save models etc
        if dofwdchecks
            # save also bodyindices for :polygonalprob
            savestuff!(fid_h5,ntobesaved,pars,nsaved,actualit,mcur,nacc,Ucur,
                       llkprival=llkprival,
                       algo=HMCpar.algo,fwdmod=nlogppd.likelihood)
        else
            savestuff!(fid_h5,ntobesaved,pars,nsaved,actualit,mcur,nacc,Ucur,
                       llkprival=llkprivals,
                       algo=HMCpar.algo)
        end
        
        ##=========================##
        ## print some info
        if (it%pars["infoevery"]==0) | (it==1)
            printiterinfo(starttime,it,maxiter,naccvec,actualit,Ucur,pars)
        end

    end
    println(" ")
    
    ##=========================##
    println("  End of main loop. Saved $(nsaved[]) models.")
    printstyled(" ====================================\n",color=:cyan)
    return
end


##====================================================
"""
$(TYPEDSIGNATURES)

A single iteration of the HMC algorithm, where a given number of 
    leap-frog iterations are performed.
"""
function oneiterhmc!(nlogppd::NLogPostPDF,HMCpar::HMCParams,mcur::Vector{Float64},Ucur::Float64,
                    grptask::Array{Int,2}, llkprivals::Vector{Union{Missing,Float64}};
                    curiter::Integer)

    # init
    pcur = draw_p(HMCpar,length(mcur),grptask) 
    pnew = copy(pcur)
    mnew = copy(mcur)

    ##-------------------------------------------------
    # Make a HALF step for momentum at the beginning
    pnew .= pnew .- (HMCpar.ϵ ./ 2.0) .* gradU(nlogppd,mnew) 


    # Alternate FULL steps for position and momentum 
    for il=1:HMCpar.L 

        ##-------------------------------------------------
        # Make a FULL step for the position

        if HMCpar.constrained
            ##-------------------------------------------------
            # Constrained HMC
            ## q >= lowcon
            ## q <= upcon
            ## Neal 2011 Handbook [...], pag 149, sec. 5.5.2
            
            lowcon = HMCpar.mlowerconstr #.* ones(length(mnew))
            upcon  = HMCpar.mupperconstr #.* ones(length(mnew))

            ## update m
            pnew_temp = copy(pnew)
            # make a FULL step for the position
            mnew_temp = mnew .+ HMCpar.ϵ .* gradK(HMCpar,pnew_temp,grptask)

            ## loop on parameters to fullfil constraints
            fullfilhmcconstr!(mnew_temp,pnew_temp,lowcon,upcon)

            ##
            mnew = copy(mnew_temp)
            pnew = copy(pnew_temp)
            ##-----------------------------------

        else
            ##-------------------------------------------------
            # Unconstrained HMC
            # make a FULL step for the position
            mnew .= mnew .+ HMCpar.ϵ .* gradK(HMCpar,pnew,grptask)
            ##-------------------------------------------------

        end

        ##-------------------------------------------------
        # Make a FULL step for the momentum, except at end of trajectory
        ##   *** L-1 ***
        if (il<HMCpar.L)
            pnew .= pnew .- HMCpar.ϵ .* gradU(nlogppd,mnew)  
        end
        #jlf = write_posmom(jlf,mnew,pnew,il+1)
    end

    ##-------------------------------------------------        
    ## Make a HALF step for momentum at the end.
    pnew .= pnew .- (HMCpar.ϵ ./ 2.0) .* gradU(nlogppd,mnew) 

    # Negate momentum at end of trajectory to make the proposal symmetric
    pnew .= -pnew 

    ##-------------------------------------------------        
    # Evaluate potential and kinetic energies at start and end of trajectory
    ##Ucur = calcU(params,mcur)
    Kcur = calcK(HMCpar,pcur,grptask)
    Unew = calcU(nlogppd,mnew)
    Knew = calcK(HMCpar,pnew,grptask)

    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    rannum=rand()

    ##----------------------------------------
    if (exp(Ucur-Unew+Kcur-Knew) >= rannum) 
        # accept
        accept = true
        mcur = copy(mnew) 
        Ucur = copy(Unew)
        llkprivals .= nlogppd.curvals
    else
        # reject
        accept = false
    end

    return mcur,accept,Ucur,llkprivals
end

###################################################################################

"""
$(TYPEDSIGNATURES)

A single iteration of the HMC algorithm, where a given number of 
    leap-frog iterations are performed.
(This version: forward problem-related checks during HMC leapfrog iterations.)
"""
function oneiterhmc_polyg(nlogppd::NLogPostPDF,HMCpar::HMCParams,mcur::Vector{Float64},Ucur::Float64,
                        grptask::Array{Int,2})

    maxbreakcheck = 20

    #bodyindicesorg = nlogppd.likelihood()
    # mnew = nothing
    # pnew = nothing
    breakcounter = 0
    checksucceded = false
    pnew = nothing
    mnew = nothing
    lowcon = HMCpar.mlowerconstr
    upcon  = HMCpar.mupperconstr

    while checksucceded == false

        # init
        pcur = draw_p(HMCpar,length(mcur),grptask)
        
        # perturb only certain parameters
        # nlogppd.likelihood(pcur)
           
        pnew = copy(pcur)
        mnew = copy(mcur)
        
        # Starting checks on geometries (if any issues then aborting!)
        if isdefined(nlogppd.likelihood,:dofwdchecks) && nlogppd.likelihood.dofwdchecks && nlogppd.likelihood.firstcheck[]
            
            # checksucceded = fwdchecks!(nlogppd.likelihood,mcur,mnew,pnew,il,HMCpar.L)
            nlogppd.likelihood(mcur,mnew,pnew,lowcon,upcon,HMCpar.LcholmassM,"performchecks")
        end

        ##-------------------------------------------------
        # Make a HALF step for momentum at the beginning
        pnew .= pnew .- (HMCpar.ϵ ./ 2.0) .* gradU(nlogppd,mnew)
        
        # Alternate FULL steps for position and momentum 
        for il=1:HMCpar.L

            
            if HMCpar.constrained
                ##-------------------------------------------------
                # Constrained HMC
                ## q >= lowcon
                ## q <= upcon
                ## Neal 2011 Handbook [...], pag 149, sec. 5.5.2
                #lowcon = HMCpar.mlowerconstr 
                #upcon  = HMCpar.mupperconstr 

                ## update m
                pnew_temp = copy(pnew)

                ##======================================================
                # make a FULL step for the position
                ##======================================================
                mnew_temp = mnew .+ HMCpar.ϵ .* gradK(HMCpar,pnew_temp,grptask)
                
                ## loop on parameters to fullfil constraints
                fullfilhmcconstr!(mnew_temp,pnew_temp,lowcon,upcon)

                ## 
                mnew = copy(mnew_temp)
                pnew = copy(pnew_temp)
                ##-----------------------------------

            else
                ##-------------------------------------------------
                # Unconstrained HMC
                # make a FULL step for the position
                mnew .= mnew .+ HMCpar.ϵ .* gradK(HMCpar,pnew,grptask)

            end

            ##===========================================================
            ## CHECKS on the perturbed model (e.g. in case of polygons)
            ##===========================================================
            if isdefined(nlogppd.likelihood,:dofwdchecks) && nlogppd.likelihood.dofwdchecks
                
                # checksucceded = fwdchecks!(nlogppd.likelihood,mcur,mnew,pnew,il,HMCpar.L)
                checksucceded = nlogppd.likelihood(mcur,mnew,pnew,lowcon,upcon,HMCpar.LcholmassM,"performchecks")

                if checksucceded==false
                    breakcounter +=1
                    if breakcounter >= maxbreakcheck
                        println("\n")
                        error("Checks during leapfrog iterations keep failing ($breakcounter times)...")
                    end
                    ## go back to resampling of momentum
                    break
                end

            else

                # in case there are no checks
                checksucceded = true
            end
            ##===========================================================

            ##-------------------------------------------------
            # Make a FULL step for the momentum, except at end of trajectory
            ##   *** L-1 ***
            if (il<HMCpar.L)
                pnew .= pnew .- HMCpar.ϵ .* gradU(nlogppd,mnew)
            end

        end
        
        if checksucceded
            ##-------------------------------------------------        
            ## Make a HALF step for momentum at the end.
            pnew .= pnew .- (HMCpar.ϵ ./ 2.0) .* gradU(nlogppd,mnew) 
            
            # Negate momentum at end of trajectory to make the proposal symmetric
            pnew .= -pnew
        end

    end  


    ##-------------------------------------------------        
    # Evaluate potential and kinetic energies at start and end of trajectory
    ##Ucur = calcU(params,mcur)
    Kcur = calcK(HMCpar,pcur,grptask)
    Unew = calcU(nlogppd,mnew)
    Knew = calcK(HMCpar,pnew,grptask)

    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    rannum=rand()
    if (exp(Ucur-Unew+Kcur-Knew) >= rannum) 
        # accept
        accept = true

        ##====================================
        ### UPDATE bodyindicesorg to mnew being accepted!!!
        nlogppd.likelihood("newinds")
        ##====================================
        
        mcur = copy(mnew) 
        pcur = copy(pnew)
        Ucur = copy(Unew)
        Kcur = copy(Knew)
    else
        # reject
        accept = false

        ##====================================
        ### RESET bodyindices to old mcur if not accepted!!!
        nlogppd.likelihood("backinds")
        ##====================================
    end

    return mcur,accept,Ucur,nlogppd.curvals
end

######################################################################################

























##================================================================

##========================================================================
