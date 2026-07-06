######################################################################

"""
$(TYPEDEF)

Structure containing the some EXTRA parameters for the NUTS algorithm.

# Fields 

$(TYPEDFIELDS)

"""
Base.@kwdef mutable struct NUTSExtraParams
    γ::Real
    t0::Real
    κ::Real
    μ::Real
    Hbar::Real
    ϵbar::Real
end

"""
$(TYPEDEF)

Structure containing the statistics for some parameters.

# Fields 

$(TYPEDFIELDS)

"""
Base.@kwdef mutable struct NUTSstats
    ϵ::Float64
    treeheight::Int64
    ϵbar::Float64
    lfsteps::Int64
    accatlfstep::Int64
end

######################################################################
    

"""
$(TYPEDEF)

Structure containing the parameters for the NUTS algorithm.

# Fields 

$(TYPEDFIELDS)

"""
struct NUTSParams <: AbstractHMCParams
    "perform parallel calculations of kinetic energy by first reading the mass matrix, etc. from an HDF5 file"
    parallelKinEn::Bool
    "Mass matrix file name"
    massMfile::String
    "Inverse of the mass matrix"
    invmassM::Union{AbstractMatrix{Float64},Array{Future,1}}
    "*Lower* triangular matrix from Cholesky decomposition of mass matrix"
    LcholmassM::Union{LowerTriangular{Float64,Matrix{Float64}},Diagonal{Float64}, Array{Future,1}}
    "Is this a constrained HMC simulation?"
    constrained::Bool
    "Lower constraints"
    mlowerconstr::Union{Vector{Float64},Nothing}
    "Upper constraints"
    mupperconstr::Union{Vector{Float64},Nothing}
    "Requested average acceptance rate"
    δ::Float64
    "Maximum acceptable error for the drift of the Hamiltonian"
    Δmax::Float64
    "Number of iterations for the adaptation of ϵ"
    niteradapt::Int64
    "Maximum value allowed for ϵ"
    ϵmax::Float64
    "Number of workers (cores) for parallel calculations" 
    numwks::Int64
    "Which algorithm"
    algo::Symbol
    "Which leap-frog algorithm to use, either :velocity or :position"
    whichleapfrog::Symbol
    "Maximum height of the balanced tree [j>=0], defaults to 4 (=31 leapfrog iterations). \
     Number of leap-frog iterations = 2 * 2^j -1."
    maxtreeheight::Int64
    "Specifies the way to perform the adaptation of the step-size ϵ for the leap-frog integration.\
    Can be either of type \n
          - AvEpsFromTree, which requires no parameters (e.g., `adapteepslf=AvEpsFromTree()`) and implements the dual average scheme of NUTS, or
          - AvEpsFromHMCIter, which requires the number of iterations to average on as a parameter \
             (e.g., `adaptepslf=AvEpsFromHMCIter(50)`) and implement a custom scheme based on the iterations of the chain.\n
    Defaults to AvEpsFromHMCIter(20)"
    adaptepslf::AdaptEpsLF

    
    ## multiple dispatch works only on positional arguments
    function NUTSParams( massMfile::String,constrained,mlowerconstr,mupperconstr,δ,Δmax,
                         niteradapt,ϵmax,whichleapfrog,maxtreeheight,
                         adaptepslf=AvEpsFromHMCIter(20) ) 
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
        algo = :NUTS
        @assert niteradapt>=0
        @assert Δmax>=0.0
        @assert maxtreeheight>=0
        @assert isa(adaptepslf,AdaptEpsLF)
        return new(parallelKinEn,massMfile,invmassM,LcholmassM,constrained,
                   mlowerconstr,mupperconstr,δ,Δmax,niteradapt,ϵmax,numwks,
                   algo,whichleapfrog,maxtreeheight,adaptepslf)
    end
    ## multiple dispatch works only on positional arguments
    function NUTSParams( invmassM,LcholmassM,constrained,mlowerconstr,mupperconstr,δ,Δmax,
                         niteradapt,ϵmax,whichleapfrog,maxtreeheight,
                         adaptepslf=AvEpsFromHMCIter(20) )    
        parallelKinEn=false
        if !constrained
            mlowerconstr,mupperconstr  = nothing,nothing
        end
        massMfile=""
        numwks = nworkers() # available workers
        algo = :NUTS
        @assert Δmax>=0.0
        @assert maxtreeheight>=0
        @assert isa(adaptepslf,AdaptEpsLF)
        return new(parallelKinEn,massMfile,invmassM,LcholmassM,constrained,
                   mlowerconstr,mupperconstr,δ,Δmax,niteradapt,ϵmax,numwks,algo,
                   whichleapfrog,maxtreeheight,adaptepslf)
    end

end

#############################################################

"""
$(TYPEDSIGNATURES)

Outer constructor for NUTSParams composite type to make use of keywords 
  arguments while still having some dispatch.
"""
function NUTSParams(; parallelKinEn::Bool=false, constrained::Bool, δ::Float64,Δmax::Float64,
                    niteradapt::Int64,ϵmax,
                    massMfile::Union{String,Nothing}=nothing,
                    invmassM::Union{AbstractMatrix{Float64},Nothing}=nothing,
                    LcholmassM::Union{Array{Float64,2},LowerTriangular{Float64,Array{Float64,2}},Diagonal{Float64},Nothing}=nothing,
                    mlowerconstr::Union{Vector{Float64},Nothing}=nothing,
                    mupperconstr::Union{Vector{Float64},Nothing}=nothing,
                    whichleapfrog::Symbol=:velocity,maxtreeheight::Integer=4,
                    adaptepslf::AdaptEpsLF=AvEpsFromHMCIter(20) )
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
            
        hmcpar = NUTSParams( massMfile,constrained,mlowerconstr,mupperconstr,δ,niteradapt,maxtreeheight )
    else
        @assert typeof(massMfile)==Nothing
        @assert typeof(invmassM) <: AbstractMatrix{Float64} 
        @assert (typeof(LcholmassM)==Array{Float64,2} || typeof(LcholmassM)==LowerTriangular{Float64,Array{Float64,2}} || typeof(LcholmassM)==Diagonal{Float64,Array{Float64,1}} )

        hmcpar = NUTSParams( invmassM,LcholmassM,constrained,mlowerconstr,mupperconstr,δ,Δmax,
                             niteradapt,ϵmax,whichleapfrog,maxtreeheight,adaptepslf )
    end
    return hmcpar
end


###############################################################
##############################################################################################

"""
$(TYPEDSIGNATURES)

The actual function starting a NUTS simulation.
See [`runMC`](@ref) for arguments explanation.
"""
function runsimulNUTS(pars::Dict,HMCpar::NUTSParams,nlogppd::NLogPostPDF,
                      mstart::Vector{Float64})

    #---------------------------------------------
    printstyled("\n ====================================",color=:cyan)
    printstyled("\n      NUTS (dynamic HMC)             ",bold=true,color=:normal)
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
    if pars["continueold"]==false
        fid_h5,nacc,nsaved,Ucur,previt = initsavestuff!(pars,nlogppd,HMCpar,mcur)

        if HMCpar.niteradapt>0
            if typeof(HMCpar.adaptepslf) ==  AvEpsFromHMCIter
                ϵ = 0.5 * HMCpar.ϵmax
            elseif typeof(HMCpar.adaptepslf) == AvEpsFromTree
                ϵ = find_reasonable_ϵ(mstart,nlogppd,HMCpar,grptask)
            end
        else
            ϵ = HMCpar.ϵmax
        end
        println(" Initial 'reasonable' ϵ: $ϵ")

    elseif pars["continueold"]
        fid_h5,nacc,nsaved,Ucur,previt,ϵ = initsavestuff!(pars,nlogppd,HMCpar,mcur)
        println(" current ϵ: $ϵ")

    end
    ##=================================================##

    ##=================================================##
    dofwdchecks = (isdefined(nlogppd.likelihood,:dofwdchecks) && nlogppd.likelihood.dofwdchecks)

    ##=================================================##
    ## Init some NUTS params
    γ = 0.05 # 0.05
    t0 = 10.0 # 10.0
    κ = 0.75 # 0.75
    μ = log(10*ϵ) # log(10*ϵ) 

    Hbar = 0.0 # 0.0
    # if pars["continueold"]==false
    #     ϵbar = 1.0
    # elseif pars["continueold"]
        ## !!! if continueold==true set ϵbar to ϵ  !!!
        ϵbar = ϵ
    # end
    NUTSpar = NUTSExtraParams(γ=γ,t0=t0,κ=κ,μ=μ,Hbar=Hbar,ϵbar=ϵbar)

    ##=========================##
    nustats = NUTSstats(ϵ=ϵ,treeheight=0,ϵbar=ϵbar,accatlfstep=0,lfsteps=0)
    
    ##=========================##
    # Initial potential energy value
    println(" Initial Ucur: $(round(Ucur, digits=3))")
    
    ##=========================##
    starttime = time()
    naccvec = zeros(Int64,maxiter-previt) #Vector{Integer}(undef,maxiter-previt)

    ## big loop
    for it=1:(maxiter-previt)
        # the "actual" iteration number (in case of continuing an old simulation)
        actualit = it + previt

        # iteration of NUTS
        if dofwdchecks
            # NUTS iterations performing also checks specific for polygonal bodies
            mcur,accept,Ucur,llkprival = oneiterNUTS_poly!(nlogppd,HMCpar,NUTSpar,mcur,grptask,
                                                           actualit,nustats,naccvec)
        else
            #stardard iteration
            mcur,accept,Ucur,llkprival = oneiterNUTS!(nlogppd,HMCpar,NUTSpar,mcur,grptask,
                                                      actualit,nustats,naccvec)
        end

        # if accepted
        if (accept==true) nacc=nacc+1 end
        # vector storing all history of accepted/rejected for current run,
        #  but taking into account the starting value of 'nacc' from previous runs
        naccvec[it] = nacc
        
        ## save models etc
        if dofwdchecks
            # save also bodyindices for :polyprob
            savestuff!(fid_h5,ntobesaved,pars,nsaved,actualit,mcur,nacc,Ucur,
                       llkprival=llkprival,
                       nustats=nustats,algo=HMCpar.algo,fwdmod=nlogppd.likelihood)
        else
            savestuff!(fid_h5,ntobesaved,pars,nsaved,actualit,mcur,nacc,Ucur,
                       llkprival=llkprival,
                       nustats=nustats,algo=HMCpar.algo)
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


######################################################################################

"""
$(TYPEDSIGNATURES)

A single iteration of the NUTS algorithm, where a given number of 
    leap-frog iterations are performed.
"""
function oneiterNUTS!(nlogppd::NLogPostPDF,HMCpar::NUTSParams,NUTSextrapar::NUTSExtraParams,
                     mcur::Vector{Float64},grptask::Array{Int,2},curiter::Integer,
                     nustats::NUTSstats,naccvec::Vector{<:Integer})

    # resample momentum
    pcur = draw_p(HMCpar,length(mcur),grptask)
    # sample slice variable in the interval [0, H(m,p)]
    #H = calcU(nlogppd,mcur) + calcK(HMCpar,pcur,grptask)
    hamilt_zero = calcU(nlogppd,mcur) + calcK(HMCpar,pcur,grptask)

    # u = rand() * exp(-H)
    ##  https://gist.github.com/bicycle1885/77d7ad79fbd3fa195ff432be167812d6
    ## This is mathematically equivalent to the next expressions, but
    ## numerically much safer.
    ##   u = rand(Uniform(0.0, exp(loglik(θ[:,m], r₀))))
    ##   logu = log(u)
    #logu = -H + log(rand())
    logu = -hamilt_zero + log(rand())

    
    # init stuff
    # Note: θ^{m-1} in the paper corresponds to `θs[m]` in the code
    θm = copy(mcur) #θs[m]
    θp = copy(mcur) #θs[m]
    pm = copy(pcur)
    pp = copy(pcur)
    mnew = copy(mcur) #θs[m+1]
   
    ##=====================================================
    ##  NUTS, self tuning of the number of leapfrog steps
    ##=====================================================
    j::Int64 = 0  # tree height
    n = 1.0
    s::Int64 = 1
    α, n_α = nothing,nothing
    accept = false
    acceptedatheight::Int64 = -1 # start from -1 so in case of nothing accepted we get 0 l-f steps
    
    ## 
    #hamilt_zero = calcU(nlogppd,mcur)+calcK(HMCpar,pcur,grptask)
    ϵ = nustats.ϵ

    while s == 1
        # draw a direction
        vj = rand([-1, 1])
        if vj == -1
            ## backward in time
            #θm,pm,_,_,θ1,n1,s1,α,n_α = buildtree(θm,pm,u,vj,j,ϵ,mcur,pcur,nlogppd,HMCpar,grptask)
            ## using log(u) instead of u
            θm,pm,_,_,θ1,n1,s1,α,n_α = buildtree(θm,pm,logu,vj,j,ϵ,hamilt_zero,nlogppd,HMCpar,grptask,
                                                 whichleapfrog=HMCpar.whichleapfrog)
        elseif vj == 1
            ## forward in time
            # _,_,θp,pp,θ1,n1,s1,α,n_α = buildtree(θp,pp,u,vj,j,ϵ,mcur,pcur,nlogppd,HMCpar,grptask)
            ## using log(u) instead of u
            _,_,θp,pp,θ1,n1,s1,α,n_α = buildtree(θp,pp,logu,vj,j,ϵ,hamilt_zero,nlogppd,HMCpar,grptask,
                                                 whichleapfrog=HMCpar.whichleapfrog)

        else
            error("onediterNUTS(): Direction in Hamiltonian trajectory is neither 1 nor -1.")
        end

        if s1 == 1
            rannum = rand()
            if rannum <= min(1.0,n1/n)
                # set mnew to the accepted model
                mnew .= θ1
                accept = true
                acceptedatheight = j
            end
        end
        n += n1
        # is it doubling back?
        dot1 = (dot(θp-θm,pm) >= 0)
        dot2 = (dot(θp-θm,pp) >= 0)
        s = s1 & dot1 & dot2
        j += 1

        ## The following means:
        ##  if the next iteration will perform more leapfrog
        ##  steps than allowed, then stop here
        # j starts from 0
        if j > HMCpar.maxtreeheight
            s=0
        end
        #println("height $(j-1), acc. at height $acceptedatheight, step $(2^(j-1+1)-1), acc. at step $(2^(acceptedatheight+1)-1) ")
    end


    ##========================================================
    ##  Set ϵ
    ##============================
    if isa(HMCpar.adaptepslf,AvEpsFromTree)
        # using dual averaging
        NUTSextrapar.ϵbar = ϵ
        ϵ = dual_averaging!(HMCpar,NUTSextrapar,α,n_α,curiter,naccvec,accept)

    elseif isa(HMCpar.adaptepslf,AvEpsFromHMCIter)
        # using custom averaging looking at a certain number of previous iterations
        ϵ = adhoc_averaging!(HMCpar,NUTSextrapar,curiter,naccvec,accept)

    end
    
    ##========================================================
    ## save some stats
    nustats.ϵ = ϵ
    ## 
    ## j is updated at the end of the while loop, so last does not count (j-1).
    jlast = j-1
    nustats.treeheight = jlast
    # number of leap-frog iterations (regardless of forward or backward in time)
    nustats.lfsteps = 2 * 2^jlast #-1
    nustats.ϵbar = NUTSextrapar.ϵbar
    if acceptedatheight==-1
        nustats.accatlfstep = 1 #0
    else
        nustats.accatlfstep = 2 * 2^(acceptedatheight) #-1
    end
    
    Ucur = calcU(nlogppd,mnew)
    if isnan(Ucur)
        error("oneiterNUTS!(): Ucur is NaN.")
    end
    ## above  mnew = copy(mcur), so return mnew directly
    return mnew,accept,Ucur,nlogppd.curvals
end


######################################################################################

function find_reasonable_ϵ(m,nlogppd,HMCpar,grptask)
    ϵ = 1.0
    if ϵ > HMCpar.ϵmax
        ϵ = HMCpar.ϵmax
    end

    p = draw_p(HMCpar,length(m),grptask)
    ## run the "position Verlep" leapfrog algo
    # m1,p1 = leapfrog_pos(m, p, ϵ, nlogppd,HMCpar,grptask)
    ## run the "velocity Verlep" leapfrog algo
    m1,p1 = leapfrog_mom(m, p, ϵ, nlogppd,HMCpar,grptask)

    # closing over some variables...
    hamiltonian(m,p,nlogppd) = calcU(nlogppd,m)+calcK(HMCpar,p,grptask)

    H1 = hamiltonian(m1,p1,nlogppd)
    H = hamiltonian(m,p,nlogppd)

    # ## if Hamiltonian ratio is >0.5 a will be 1, otherwise -1
    # H1overH = exp(-H1+H)
    # a = 2 * (H1overH > 0.5) - 1
    ## log-scale the above:
    a = 2 * (-H1+H > log(0.5)) - 1
 
    # while (H1overH)^float(a) > 2^float(-a)
    ## log-scale the above:
    while a*(-H1+H) > -a*log(2)
        ϵ = 2.0^a * ϵ
        if ϵ > HMCpar.ϵmax
            ϵ = HMCpar.ϵmax
            break
        end
        #@show a*(-H1+H), -a*log(2),ϵ
        ## run the "position Verlep" leapfrog algo
        # m1, p1 = leapfrog_pos(m, p, ϵ, nlogppd,HMCpar,grptask)
        ## run the "velocity Verlep" leapfrog algo
        m1, p1 = leapfrog_mom(m, p, ϵ, nlogppd,HMCpar,grptask)
        H1 = hamiltonian(m1,p1,nlogppd)
        #H1overH = exp(-H1+H)
    end

    if ϵ > HMCpar.ϵmax
        println(" ϵ > ϵmax, setting ϵ = ϵmax = $(HMCpar.ϵmax)")
        ϵ = HMCpar.ϵmax
    end
    return ϵ
end


######################################################################################

function buildtree(m, p, logu, v, treedepth, ϵ,  hamilt_zero, nlogppd,HMCpar,grptask ;
                   whichleapfrog=:velocity)
   # hamilt_zero instead of m0,p0
 
    # closures on nlogppd,HMCpar,grptask
    #exphamilt(m,p) = exp( -(calcU(nlogppd,m)+calcK(HMCpar,p,grptask)) )
    #hamiltonian(m,p) = calcU(nlogppd,m)+calcK(HMCpar,p,grptask)
    hamiltonian(m,p,usmod) = calcU(usmod,m)+calcK(HMCpar,p,grptask)
  

    if treedepth == 0
        # # store initial Hamiltonian 
        # hamilt_zero = hamiltonian(m0,p0,nlogppd)

        # Base case - take one leapfrog step in the direction v.
        if whichleapfrog==:position
            ## run the "position Verlep" leapfrog algo
            m1, p1 = leapfrog_pos(m,p,v*ϵ,nlogppd,HMCpar,grptask)
        elseif whichleapfrog==:velocity
            ## run the "velocity Verlep" leapfrog algo
            m1, p1 = leapfrog_mom(m,p,v*ϵ,nlogppd,HMCpar,grptask)
        end

        # n1 = u <= exp( - hamiltonian(m1,p1) )
        ## using log(u) instead of u
        ham1 = hamiltonian(m1,p1,nlogppd)
        n1 = Int(logu <= -ham1 )

        # s1 = u <= exp( HMCpar.Δmax - (hamiltonian(m1,p1) ) )
        ## <  or  <= ????
        ## using log(u) instead of u
        s1 = Int(logu <= (HMCpar.Δmax - ham1) )

        tmp = ham1 - hamilt_zero
        α1 = min(1,exp(-tmp))
        n1_α = 1
        #@show ham1,hamilt_zero,tmp
        return m1,p1, m1,p1, m1,n1, s1, α1,n1_α

    else
        # Recursion - build the left and right subtrees.
        mminus,pminus,mplus,pplus,m1,n1,s1,α1,n1_α = buildtree(m,p,logu,v,treedepth-1,ϵ,hamilt_zero, #m0,p0,
                                                               nlogppd,HMCpar,grptask,
                                                               whichleapfrog=whichleapfrog)
        
        if s1 == 1
            if v == -1
                mminus,pminus,_,_,m2,n2,s2,α2,n2_α = buildtree(mminus,pminus,logu,v,treedepth-1,ϵ,hamilt_zero, #m0,p0,
                                                               nlogppd,HMCpar,grptask,
                                                               whichleapfrog=whichleapfrog)
            elseif v == 1 
                _,_,mplus,pplus,m2,n2,s2,α2,n2_α = buildtree(mplus,pplus,logu,v,treedepth-1,ϵ,hamilt_zero, #m0,p0,
                                                             nlogppd,HMCpar,grptask,
                                                             whichleapfrog=whichleapfrog)
            end
            if rand() < n2 / (n1 + n2)
                m1 = m2
            end
            α1 = α1 + α2
            n1_α = n1_α + n2_α
            dot1 = dot(mplus-mminus, pminus)
            dot2 = dot(mplus-mminus, pplus)
            s1 = s2 & (dot1 >= 0) & (dot2 >= 0)
            n1 = n1 + n2
        end
        return mminus,pminus, mplus,pplus, m1,n1, s1, α1,n1_α
    end
end

######################################################################################




