
###############################################################

"""
$(TYPEDEF)

Structure containing the parameters for the Extended Metropolis algorithm.

# Fields 

$(TYPEDFIELDS)

"""
struct ExtMetropParams <: AbstractMeHaParams
    "Which algorithm"
    algo::Symbol ## = :ExtMetrop

    function ExtMetropParams()
        algo = :ExtMetrop
        return new(algo)
    end
end

###############################################################

"""
$(TYPEDSIGNATURES)

The actual function starting an Extended Metropolis simulation.
See [`runMC`](@ref) for arguments explanation.
"""
function runsimulextmetrop(pars::Dict,MHpar::ExtMetropParams,nlogppd::NLogPostPDF,
                           mstart::Vector{Float64})


    #---------------------------------------------
    printstyled("\n ====================================",color=:cyan)
    printstyled("\n     Extended Metropolis algo    ",bold=true,color=:normal)
    printstyled("\n ====================================\n",color=:cyan)
    if pars["continueold"]
        printstyled(" Continuing previous simulation $(pars["simname"]) \n",color=:yellow)
    else
        printstyled(" Running simulation $(pars["simname"]) \n",color=:yellow)
    end
    #---------------------------------------

    maxiter = pars["maxiter"]::Int64
    saveevery = pars["saveevery"]::Int64
    ntobesaved = div(maxiter,saveevery)+1 # integer division
    mcur = copy(mstart)

    ##=================================================##
    # Init output file and save input parameters
    fid_h5,nacc,nsaved,nllkcur,previt = initsavestuff!(pars,nlogppd,MHpar,mcur)

    ##=================================================##
    dofwdchecks = (isdefined(nlogppd.likelihood,:dofwdchecks) && nlogppd.likelihood.dofwdchecks)

    ##=========================##
    # Initial potential energy value
    println(" Initial -log(likelihood): $(round(nllkcur, digits=3))")
    
    ##=========================##
    starttime = time()
    naccvec = Vector{Integer}(undef,maxiter-previt)

    ## big loop
    for it=1:(maxiter-previt)
        # the "actual" iteration number (in case of continuing an old simulation)
        actualit = it + previt

        # iteration of the Hamiltonian MC
        if dofwdchecks
            # HMC iterations performing also checks specific for given forward problems
            mcur,accept,nllkcur = oneiterextmetrop(nlogppd,mcur,nllkcur)
        else
            # standard iteration
            mcur,accept,nllkcur = oneiterextmetrop(nlogppd,mcur,nllkcur)
        end

        # if accepted
        if (accept==true) nacc=nacc+1 end
        # vector storing all history of accepted/rejected for current run,
        #  but taking into account the starting value of 'nacc' from previous runs
        naccvec[it] = nacc
        
        ## save models etc
        if dofwdchecks
            # save also bodyindices for :polygonalprob
            savestuff!(fid_h5,ntobesaved,pars,nsaved,actualit,mcur,nacc,nllkcur,
                       algo=MHpar.algo,fwdmod=nlogppd.likelihood)
        else
            savestuff!(fid_h5,ntobesaved,pars,nsaved,actualit,mcur,nacc,nllkcur,
                       algo=MHpar.algo)
        end
        
        ##=========================##
        ## print some info
        if (it%pars["infoevery"]==0) | (it==1)
            printiterinfo(starttime,it,maxiter,naccvec,actualit,nllkcur,pars)
        end

    end
    println(" ")
    
    ##=========================##
    println("  End of main loop. Saved $(nsaved[]) models.")
    printstyled(" ====================================\n",color=:cyan)
    return
end


#########################################################
"""
$(TYPEDSIGNATURES)

A single iteration of the HMC algorithm, where a given number of 
    leap-frog iterations are performed.
"""
function oneiterextmetrop(nlogppd::NLogPostPDF,mcur::Vector{Float64},
                          nllkcur::Float64)

    # Draw a sample from the prior
    mnew = nlogppd.prior(mcur,:drawsample)

    # Evaluate the negative log-likelihood
    nllknew = calcnllk(nlogppd,mnew)
   
    # Accept or reject the state 
    rannum=rand()
    if (exp(nllkcur-nllknew) >= rannum) 
        # accept
        accept = true
        mcur = copy(mnew) 
        nllkcur = copy(nllknew)
    else
        # reject
        accept = false
    end
  
    return mcur,accept,nllkcur  
end

#########################################################

function calcnllk(nlogppd::NLogPostPDF,mcur::Vector{Float64})
    mnew = nlogppd.likelihood(mcur,:nlogpdf)
end

#########################################################
