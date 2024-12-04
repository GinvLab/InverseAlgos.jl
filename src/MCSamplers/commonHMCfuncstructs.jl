

#############################################################

abstract type AbstractHMCParams <: AbstractMCParams end

######################################################################

abstract type AdaptEpsLF end

struct AvEpsFromTree <: AdaptEpsLF
end

struct AvEpsFromHMCIter <: AdaptEpsLF
    "Number of HMC iterations to average on"
    N::Int64
end

######################################################################################

"""
 Leapfrog integration
  "position Verlet": half step for position, full for momentum, half for position
"""
function leapfrog_pos(m, p, ϵ, nlogppd,HMCpar,grptask)
    
    mtilde = copy(m)
    ptilde = copy(p)

    ##-----------------------------------
    # make a HALF step for momentum
    ptilde .= ptilde .- (ϵ./2.0) .* gradU(nlogppd,mtilde)
    
    if HMCpar.constrained
        ##-----------------------------------
        # Constrained HMC
        ## q >= lowcon
        ## q <= upcon
        ## Neal 2011 Handbook [...], pag 149, sec. 5.5.2
        
        lowcon = HMCpar.mlowerconstr 
        upcon  = HMCpar.mupperconstr 

        ## update m
        # make a FULL step for the position
        mtilde .= mtilde .+ ϵ .* gradK(HMCpar,ptilde,grptask)

        ## loop on parameters to fullfil constraints
        fullfilhmcconstr!(mtilde,ptilde,lowcon,upcon)

    else
        # NOT constrained

        ##-----------------------------------
        # make a FULL step for position
        mtilde .= mtilde .+ ϵ .* gradK(HMCpar,ptilde,grptask)

    end
    
    ##-----------------------------------
    # make a HALF step for momentum
    ptilde .= ptilde .- (ϵ./2.0) .* gradU(nlogppd,mtilde)    


    # ###################
    # ###   TEST
    # ###################
    # mtilde_vel,ptilde_vel = leapfrog_mom(m, p, ϵ, nlogppd,HMCpar,grptask)
    # @show mtilde_vel≈mtilde
    # @show ptilde_vel≈ptilde

    return mtilde,ptilde
end


######################################################################################

"""
 Leapfrog integration
  "velocity Verlet": half step for momentum, full for position, half for momentum
"""
function leapfrog_mom(m, p, ϵ, nlogppd,HMCpar,grptask)
    
    mtilde = copy(m)
    ptilde = copy(p)

    lowcon = HMCpar.mlowerconstr 
    upcon  = HMCpar.mupperconstr

    if HMCpar.constrained
        ##-----------------------------------
        # Constrained HMC
        ## q >= lowcon
        ## q <= upcon
        ## Neal 2011 Handbook [...], pag 149, sec. 5.5.2

        ## update m
        # pnew_temp = copy(p)
        # make a HALF step for the position
        mtilde .= mtilde .+ (ϵ./2.0) .* gradK(HMCpar,ptilde,grptask)

        ## loop on parameters to fullfil constraints
        fullfilhmcconstr!(mtilde,ptilde,lowcon,upcon)

    else
        # NOT constrained

        ##-----------------------------------
        # make a HALF step for position
        mtilde .= mtilde .+  (ϵ./2.0) .* gradK(HMCpar,p,grptask)

    end

    ##==========================================
    ## make a FULL step for the momentum
    ptilde .= ptilde .- ϵ .* gradU(nlogppd,mtilde)
    ##==========================================

    if HMCpar.constrained
        ##-----------------------------------
        # Constrained HMC
        ## q >= lowcon
        ## q <= upcon
        ## Neal 2011 Handbook [...], pag 149, sec. 5.5.2
        
        ## update m
        # make a HALF step for the position
        mtilde .= mtilde .+ (ϵ./2.0) .* gradK(HMCpar,ptilde,grptask)

        ## loop on parameters to fullfil constraints
        fullfilhmcconstr!(mtilde,ptilde,lowcon,upcon)

    else
        # NOT constrained

        ##-----------------------------------
        # make a FULL step for position
        mtilde .= mtilde .+  (ϵ./2.0) .* gradK(HMCpar,ptilde,grptask)

    end

    return mtilde,ptilde
end

     
######################################################################################
         
# function leapfrog(m, p, ϵ, nlogppd,HMCpar,grptask)
#     # make a HALF step for momentum
#     ptilde = p .- (ϵ ./ 2.0) .* gradU(nlogppd,m) 
#     # make a FULL step for position
#     mtilde = m .+  ϵ .* gradK(HMCpar,ptilde,grptask)
#     # make a HALF step for momentum
#     ptilde .= ptilde .- (ϵ ./ 2.0) .* gradU(nlogppd,mtilde)    
#     return mtilde,ptilde
# end

###############################################################

"""
$(TYPEDSIGNATURES)

Calculate the value of the kinetic energy.
"""
function calcK(HMCpar::AbstractHMCParams,p::Vector{Float64},grptask::Array{Int,2})::Float64
    ## use  x.T * C^-1 * x  = ||L^-1 * x ||^2 ?
    # D = length(p)
    #? + 0.5 * log((2*π)^D*(det(M)))
 
    if HMCpar.parallelKinEn
        # invMp = (HMCpar.invmassM * p)
        invMp = paramatvec(HMCpar.invmassM,p,grptask)
        KK = 1/2 .* dot(p, invMp ) 
    else
        KK = 1/2 .* dot(p,HMCpar.invmassM,p)

    end
    return KK
end

###############################################################################

"""
$(TYPEDSIGNATURES)

Compute the gradient of the kinetic energy.
"""
function gradK(HMCpar::AbstractHMCParams,p::Vector{Float64},grptask::Array{Int,2})::Vector{Float64}
    #println("gradK(): ")
    #@time begin
        if HMCpar.parallelKinEn
            grK = paramatvec(HMCpar.invmassM,p,grptask)       
        else
            grK = HMCpar.invmassM * p 
        end
    #end
    return grK
end

###############################################################################

"""
$(TYPEDSIGNATURES)

Draw values for momentum from its Gaussian distribution.
"""
function draw_p(HMCpar::AbstractHMCParams,npts::Int,grptask::Array{Int,2})::Vector{Float64}
    zdev = randn(npts)
    if HMCpar.parallelKinEn
        pp = paramatvec(HMCpar.invmassM,zdev,grptask)
    else
        pp = HMCpar.LcholmassM * zdev # zero mean
    end
    return pp
end

###############################################################################

"""
$(TYPEDSIGNATURES)

Calculate the value of the potential energy (-log(PPD)).
"""    
function calcU(nlogppd::NLogPostPDF,m::Vector{Float64})::Float64
    ## must return a Float64   
    poten = nlogppd(m,:nlogpdf)
    return poten
end

###############################################################################

"""
$(TYPEDSIGNATURES)

Compute the gradient of the potential energy.
"""    
function gradU(nlogppd::NLogPostPDF,m::Vector{Float64})::Vector{Float64}
    #@time begin
    ## must return a Vector
    gradpoten = nlogppd(m,:gradnlogpdf)
    #end
    return gradpoten
end

###############################################################################

function fullfilhmcconstr!(mnew_temp,pnew_temp,lowcon,upcon)

    ## loop on parameters to fullfil constraints
    for i=1:length(mnew_temp)
        if (mnew_temp[i] <= upcon[i]) & (mnew_temp[i] >= lowcon[i]) 
            constraintssatisfied = true
        else
            constraintssatisfied = false
        end

        counter = 0
        while (constraintssatisfied==false) 
            if mnew_temp[i] > upcon[i]
                mnew_temp[i] = upcon[i] - (mnew_temp[i]-upcon[i])
                pnew_temp[i] = -pnew_temp[i]
            end
            if mnew_temp[i] < lowcon[i]
                mnew_temp[i] = lowcon[i] + (lowcon[i]-mnew_temp[i])
                pnew_temp[i] = -pnew_temp[i]
            end
            if (mnew_temp[i] <= upcon[i]) & (mnew_temp[i] >= lowcon[i]) 
                constraintssatisfied = true
            end
            counter += 1
            if counter>=maxitconstr
                println("leapfrog(): index: $i  $(lowcon[i]) <= $(mnew_temp[i]) <= $(upcon[i]) ")
                error("leapfrog(): Reached limit of iterations in constraints loop: $(maxitconstr).")
            end
        end            
    end
    return
end     

######################################################################################

function adhoc_averaging!(HMCpar,NUTSextrapar,curiter,naccvec,accept)

    ϵ = NUTSextrapar.ϵbar

    if curiter <= HMCpar.niteradapt
    
        # compute acceptance ratio for previous N iterations
        N = HMCpar.adaptepslf.N

        if curiter<=1
            αave = accept ? (αave=1.0) : (αave=0.0)
        else
            iteracc = curiter-1 
            totnacc = naccvec[iteracc]
            totnacc = accept ? (totnacc+=1) : totnacc
            if curiter>N+1
                αave = (totnacc-naccvec[iteracc-N]-1)/N 
            else
                αave = (totnacc-1)/iteracc
            end
        end

        # custom update to adapt ϵ...
        ϵ = abs( ϵ + 1/N * ϵ * (αave-HMCpar.δ) )

        ## if ϵ is bigger than the allowed ϵmax, reset it to ϵmax
        ##  needed because some gradient calculations for certain problems break otherwise
        if ϵ > HMCpar.ϵmax
            ϵ = HMCpar.ϵmax
        end

        ## save stats
        NUTSextrapar.ϵbar = ϵ
    end
        
    return ϵ
end

######################################################################################

function dual_averaging!(HMCpar,NUTSextrapar,α,n_α,curiter,naccvec,accept)

    if curiter <= HMCpar.niteradapt
        ##============================
        ##  Dual averaging to set ϵ
        ##============================
        ## HMCpar.δ : target acceptance prob.
        ## αave : current acceptance prob.
        αave = α/n_α
            
        # NOTE: Hbar goes to negative when δ - α/n_α < 0
        tmp1 = 1.0/(curiter + NUTSextrapar.t0)
        NUTSextrapar.Hbar = (1.0 - tmp1) * NUTSextrapar.Hbar + tmp1 * (HMCpar.δ - αave)

        ϵ = exp(NUTSextrapar.μ - sqrt(curiter) / NUTSextrapar.γ * NUTSextrapar.Hbar)

        ## if ϵ is bigger than the allowed ϵmax, reset it to ϵmax
        ##  needed because some gradient calculations for certain problems break otherwise
        if ϵ > HMCpar.ϵmax
            ϵ = HMCpar.ϵmax
        end

        tmp2 = curiter^float(-NUTSextrapar.κ)
        NUTSextrapar.ϵbar = exp( tmp2*log(ϵ) + (1.0-tmp2) * log(NUTSextrapar.ϵbar) )

    else
        ϵ = NUTSextrapar.ϵbar
        
        ## if ϵ is bigger than the allowed ϵmax, reset it to ϵmax
        ##  needed because some gradient calculations for certain problems break otherwise
        if ϵ > HMCpar.ϵmax
            ϵ = HMCpar.ϵmax
        end
    end
    
    return ϵ
end

###############################################################################
