
"""
    $(TYPEDSIGNATURES)

    A single iteration of the NUTS algorithm specific for potential fields problems with polygonal parameterization,
      where a given number of leap-frog iterations are performed.
    """
function oneiterNUTS_polyg!(usermodel::NLogPostPDF,HMCpar::NUTSParams,NUTSextrapar::NUTSExtraParams,
                            mcur::Vector{Float64},grptask::Array{Int,2},curiter::Integer,
                            nustats::NUTSstats,naccvec::Vector{<:Integer})

    maxbreakcheck = 20
    # checksucceded = false
    breakcounter = 0
    α, n_α = nothing,nothing
    # acceptedatheight = -1 
    # mnew = copy(mcur)
    # accept = false
    acceptbodyindices = usermodel.likelihood("getbodyindices")
    # while checksucceded == false


    # resample momentum
    pcur = draw_p(HMCpar,length(mcur),grptask)
    # sample slice variable in the interval [0, H(m,p)]
    H = calcU(usermodel,mcur) + calcK(HMCpar,pcur,grptask)

    # u = rand() * exp(-H)
    ##  https://gist.github.com/bicycle1885/77d7ad79fbd3fa195ff432be167812d6
    ## This is mathematically equivalent to the next expressions, but
    ## numerically much safer.
    ##   u = rand(Uniform(0.0, exp(loglik(θ[:,m], r₀))))
    ##   logu = log(u)
    logu = -H + log(rand())
    
    # init stuff
    # Note: θ^{m-1} in the paper corresponds to `θs[m]` in the code
    θm = copy(mcur) #θs[m]
    θp = copy(mcur) #θs[m]
    pm = copy(pcur)
    pp = copy(pcur)
    mnew = copy(mcur) #θs[m+1]
    
  
    # Starting checks on geometries (if any issues then aborting!)
    if isdefined(usermodel.likelihood,:dofwdchecks) && usermodel.likelihood.dofwdchecks && usermodel.likelihood.firstcheck[]
        pnew = copy(pcur)
        # checksucceded = fwdchecks!(usermodel.likelihood,mcur,mnew,pnew,il,HMCpar.L)
        checksucceded = usermodel.likelihood(mcur,mnew,pnew,HMCpar.mlowerconstr,HMCpar.mupperconstr,HMCpar.LcholmassM,"performchecks")
        if checksucceded==false
            error("Initial checks on polygons failed. Aborting.")
        end
    end

    ##=====================================================
    ##  NUTS, self tuning of the number of leapfrog steps
    ##=====================================================
    j::Int64 = 0
    n = 1.0
    s::Int64 = 1       
    accept = false
    acceptedatheight::Int64 = -1 # start from -1 so in case of nothing accepted we get 0 l-f steps

    ##=====================================================
    ## save current bodyindices
    backwbodyind = usermodel.likelihood("getbodyindices")
    forwbodyind  = usermodel.likelihood("getbodyindices")

    ## 
    hamilt_zero = calcU(usermodel,mcur)+calcK(HMCpar,pcur,grptask)
    ϵ = nustats.ϵ
    
    while s == 1
        # draw a direction +1 or -1
        vj = rand([-1, 1])

        if vj == -1
            ## backward in time
            #θm,pm,_,_,θ1,n1,s1,α,n_α = buildtree(θm,pm,u,vj,j,ϵ,mcur,pcur,usermodel,HMCpar,grptask)
            ##----------------------------------------------------
            ## set proper bodyindices from last point (forw or back)
            usermodel.likelihood("newinds",newbodyindices=backwbodyind)
            ## using log(u) instead of u
            θm,pm,_,_,θ1,n1,s1,α,n_α,checksucceded = buildtree_polyg(θm,pm,logu,vj,j,ϵ,hamilt_zero,usermodel,HMCpar,grptask,
                                                                     whichleapfrog=HMCpar.whichleapfrog)
            ##----------------------------------------------------
            ## save current bodyindices
            backwbodyind .= usermodel.likelihood("getbodyindices")

        elseif vj == 1
            ## forward in time
            # _,_,θp,pp,θ1,n1,s1,α,n_α = buildtree(θp,pp,u,vj,j,ϵ,mcur,pcur,usermodel,HMCpar,grptask)
            ##----------------------------------------------------
            ## set proper bodyindices from last point (forw or back)
            usermodel.likelihood("newinds",newbodyindices=forwbodyind)
            ## using log(u) instead of u
            _,_,θp,pp,θ1,n1,s1,α,n_α,checksucceded = buildtree_polyg(θp,pp,logu,vj,j,ϵ,hamilt_zero,usermodel,HMCpar,grptask,
                                                                     whichleapfrog=HMCpar.whichleapfrog)
            ##----------------------------------------------------
            ## save current bodyindices
            forwbodyind .= usermodel.likelihood("getbodyindices")

        else
            error("onediterNUTS(): Direction in Hamiltonian trajectory is neither 1 nor -1.")
        end

   
        #################################
        ## Slightly modified NUTS:
        ##  when a check fails, then the iterations are stopped instead of jumping to momentum resampling
        #################################        
        if checksucceded==false
            
            breakcounter +=1
            if breakcounter >= maxbreakcheck
                println("\n")
                error("Checks during leapfrog iterations keep failing ($breakcounter times)...")
            end
            ##----------------------
            ## stop iterating
            #println("\n## Stopping iterations j+1=$(j+1) ###\n")
            s=0 # this causes the algo to stop iterating, i.e. stops the trajectory
            j += 1
            
        else
            ##=============================
            if s1 == 1
                rannum = rand()
                if rannum <= min(1.0,n1/n)
                    # set mnew to the accepted model
                    mnew .= θ1
                    accept = true
                    acceptedatheight = j
                    # get the bodyindices relative to the accepted model!
                    if vj == -1
                        acceptbodyindices .= backwbodyind
                    elseif vj == 1
                        acceptbodyindices .= forwbodyind
                    end
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
        end

        #println("$j, height $(j-1), acc. at height $acceptedatheight, step $(2^(j-1+1)-1), acc. at step $(2^(acceptedatheight+1)-1) ")
    end ## while s == 1

    ##==========================
    if accept == true           
        ##====================================
        ### UPDATE bodyindicesorg to mnew being accepted!!!
        usermodel.likelihood("newinds",newbodyindices=acceptbodyindices)
        ##====================================
        
    else
        ##====================================
        ### RESET bodyindices to old mcur if not accepted!!!
        usermodel.likelihood("backinds")
        ##====================================
    end

    ##========================================================
    ##  Dual averaging to set ϵ
    ##============================
    if isa(HMCpar.adaptepslf,AvEpsFromTree)
        NUTSextrapar.ϵbar = ϵ
        ϵ = dual_averaging!(HMCpar,NUTSextrapar,α,n_α,curiter,naccvec,accept)

    elseif isa(HMCpar.adaptepslf,AvEpsFromHMCIter)
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
    nustats.lfsteps = 2 * 2^jlast - 1
    nustats.ϵbar = NUTSextrapar.ϵbar
    if acceptedatheight==-1
        nustats.accatlfstep = 0
    else
        nustats.accatlfstep = 2 * 2^(acceptedatheight) - 1
    end

    ## above  mnew = copy(mcur), so return mnew directly
    Ucur = calcU(usermodel,mnew)
    if isnan(Ucur)
        println()
        @show any(isnan.(mnew))
        error("Ucur is NaN")
    end
    ## above  mnew = copy(mcur), so return mnew directly
    return mnew,accept,Ucur,usermodel.curvals
end


######################################################################################

"""
     Leapfrog integration
      "position Verlet": half step for position, full for momentum, half for position
"""
function leapfrog_pos_polyg(m, p, ϵ, usermodel,HMCpar,grptask)
    
    mtilde = copy(m)
    ptilde = copy(p)

    lowcon = HMCpar.mlowerconstr 
    upcon  = HMCpar.mupperconstr

    ##-----------------------------------
    # make a HALF step for momentum
    ptilde .= ptilde .- (ϵ ./ 2.0) .* gradU(usermodel,mtilde)
    
    if HMCpar.constrained
        ##-----------------------------------
        # Constrained HMC
        ## q >= lowcon
        ## q <= upcon
        ## Neal 2011 Handbook [...], pag 149, sec. 5.5.2

        ## update m
        # make a FULL step for the position
        mtilde .= mtilde .+ ϵ .* gradK(HMCpar,ptilde,grptask)

        ## loop on parameters to fullfil constraints
        fullfilhmcconstr!(mtilde,ptilde,lowcon,upcon)

    else
        # NOT constrained

        ##-----------------------------------
        # make a FULL step for position
        mtilde .= mtilde .+  ϵ .* gradK(HMCpar,ptilde,grptask)

    end
    
    ##===========================================================
    ## CHECKS on the perturbed model (e.g. in case of polygons)
    ##===========================================================
    if isdefined(usermodel.likelihood,:dofwdchecks) && usermodel.likelihood.dofwdchecks
        # checksucceded = fwdchecks!(usermodel.likelihood,mcur,mnew,pnew,il,HMCpar.L)
        checksucceded = usermodel.likelihood(m,mtilde,ptilde,lowcon,upcon,HMCpar.LcholmassM,"performchecks")
        if checksucceded==false
            # get out
            return mtilde,ptilde,checksucceded
        end
    else
        # in case there are no checks
        checksucceded = true
    end
    ##===========================================================

    ##-----------------------------------
    # make a HALF step for momentum
    ptilde .= ptilde .- (ϵ ./ 2.0) .* gradU(usermodel,mtilde)    

    # ###################
    # ###   TEST
    # ###################
    # mtilde_vel,ptilde_vel = leapfrog_mom(m, p, ϵ, usermodel,HMCpar,grptask)
    # @show mtilde_vel≈mtilde
    # @show ptilde_vel≈ptilde

    return mtilde,ptilde,checksucceded
end


######################################################################################

"""
   Leapfrog integration
   "velocity Verlet": half step for momentum, full for position, half for momentum
"""
function leapfrog_mom_polyg(m, p, ϵ, usermodel,HMCpar,grptask)

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
        # make a HALF step for the position
        mtilde .= m .+ (ϵ/2.0) .* gradK(HMCpar,ptilde,grptask)

        ## loop on parameters to fullfil constraints
        fullfilhmcconstr!(mtilde,ptilde,lowcon,upcon)

    else
        # NOT constrained

        ##-----------------------------------
        # make a HALF step for position
        mtilde .= mtilde .+  (ϵ/2.0) .* gradK(HMCpar,ptilde,grptask)
    end

    ##===========================================================
    ## CHECKS on the perturbed model (e.g. in case of polygons)
    ##===========================================================
    if isdefined(usermodel.likelihood,:dofwdchecks) && usermodel.likelihood.dofwdchecks
        # checksucceded = fwdchecks!(usermodel.likelihood,mcur,mnew,pnew,il,HMCpar.L)
        checksucceded = usermodel.likelihood(m,mtilde,ptilde,lowcon,upcon,HMCpar.LcholmassM,"performchecks")
        if checksucceded==false
            # get out
            return mtilde,ptilde,checksucceded
        end
    else
        # in case there are no checks
        checksucceded = true
    end
    ##===========================================================

    ##==========================================
    ## make a FULL step for the momentum
    ptilde .= ptilde .- ϵ .* gradU(usermodel,mtilde)
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
        mtilde .= mtilde .+  (ϵ/2.0) .* gradK(HMCpar,ptilde,grptask)

    end

    ##===========================================================
    ## CHECKS on the perturbed model (e.g. in case of polygons)
    ##===========================================================
    if isdefined(usermodel.likelihood,:dofwdchecks) && usermodel.likelihood.dofwdchecks
        # checksucceded = fwdchecks!(usermodel.likelihood,mcur,mnew,pnew,il,HMCpar.L)
        checksucceded = usermodel.likelihood(m,mtilde,ptilde,lowcon,upcon,HMCpar.LcholmassM,"performchecks")
        if checksucceded==false
            # get out
            return mtilde,ptilde,checksucceded
        end
    else
        # in case there are no checks
        checksucceded = true
    end
    ##===========================================================

    return mtilde,ptilde,checksucceded
end


######################################################################################

function buildtree_polyg(m, p, logu, v, j, ϵ, hamilt_zero, usermodel,HMCpar,grptask ;
                         whichleapfrog=:velocity)
    # hamilt_zero instead of m0,p0

    #   - θ   : model parameter
    #   - r   : momentum variable
    #   - u   : slice variable
    #   - v   : direction ∈ {-1, 1}
    #   - j   : depth of tree
    #   - ϵ   : leapfrog step size
    #   - θ0  : initial model parameter
    #   - r0  : initial mometum variable

    n1::Int64 = 0
    n2::Int64 = 0

    ## to keep original bodyindices for "hamiltonian()"
    # usermodel_zero = copy(usermodel

    # closures on usermodel,HMCpar,grptask
    #exphamilt(m,p) = exp( -(calcU(usermodel,m)+calcK(HMCpar,p,grptask)) )
    hamiltonian(m,p,usmod) = calcU(usmod,m)+calcK(HMCpar,p,grptask)
    # store initial Hamiltonian otherwise could have wrong bodyindices later on
    #hamilt_zero = hamiltonian(m0,p0,usermodel)

    if j == 0
        # Base case - take one leapfrog step in the direction v.
        if whichleapfrog==:position
            ## run the "position Verlep" leapfrog algo
            m1, p1, checksucceded = leapfrog_pos_polyg(m,p,v*ϵ,usermodel,HMCpar,grptask)
        elseif whichleapfrog==:velocity
            ## run the "velocity Verlep" leapfrog algo
            m1, p1, checksucceded = leapfrog_mom_polyg(m,p,v*ϵ,usermodel,HMCpar,grptask)
        end
        
        ##===========================================================
        ## If checks fail get out 
        ##===========================================================
        #bi = usermodel.likelihood("getbodyindices"); println(">> 1 buildtree(): $bi ")
        if checksucceded == false
            #println(">> 1 buildtree(): checksucceded==false, j=$j")
            
            return m1,p1, m1,p1, m1,0, 0, 0.0,1,checksucceded
        end
        ##===========================================================

        # n1 = u <= exp( - hamiltonian(m1,p1) )
        ## using log(u) instead of u
        n1 = Int(logu <= -hamiltonian(m1,p1,usermodel) )

        #println("u <= hamΔmax  instead of u < hamΔmax") 
        # s1 = u <= exp( HMCpar.Δmax - (hamiltonian(m1,p1) ) )
        ## <  or  <= ????
        ## using log(u) instead of u
        s1 = Int(logu <= HMCpar.Δmax - hamiltonian(m1,p1,usermodel) )

        ## for m0 p0 use "usermodel_zero" otherwise could have wrong bosyindices
        ##   tmp = hamiltonian(m1,p1,usermodel) - hamiltonian(m0,p0,usermodel_zero)
        tmp = hamiltonian(m1,p1,usermodel) - hamilt_zero
        α1 = min(1.0,exp(-tmp))
        h1_α = 1
        return m1,p1, m1,p1, m1,n1, s1, α1,h1_α,checksucceded

    else

        # Recursion - build the left and right subtrees.
        mminus,pminus,mplus,pplus,m1,n1,s1,α1,n1_α,checksucceded = buildtree_polyg(m,p,logu,v,j-1,ϵ,hamilt_zero, #m0,p0,
                                                                                   usermodel,HMCpar,grptask,
                                                                                   whichleapfrog=whichleapfrog)
        ##----------------------------------------------------
        ## save current bodyindices
        m1bodyind = usermodel.likelihood("getbodyindices")

        ##===========================================================
        ## If checks fail get out 
        ##===========================================================
        #bi = usermodel.likelihood("getbodyindices"); println(">>  2 buildtree(): $bi ")
        if checksucceded == false
            #println(">> 2 buildtree(): checksucceded==false, j=$j")
            return mminus,pminus,mplus,pplus,m1,n1,s1,α1,n1_α,checksucceded
        end
        ##===========================================================

        if s1 == 1
            if v == -1
                mminus,pminus,_,_,m2,n2,s2,α2,n2_α,checksucceded = buildtree_polyg(mminus,pminus,logu,v,j-1,ϵ,hamilt_zero, #m0,p0,
                                                                                   usermodel,HMCpar,grptask,
                                                                                   whichleapfrog=whichleapfrog)
                ##===========================================================
                ## If checks fail get out 
                ##===========================================================
                #bi = usermodel.likelihood("getbodyindices"); println(">> 3 buildtree(): $bi ")
                if checksucceded == false
                    #println(">> 3 buildtree(): checksucceded==false, j=$j")
                    return mminus,pminus,mminus,pminus,m2,n2,s2,α2,n2_α,checksucceded
                end
                ##===========================================================
                
            else
                _,_,mplus,pplus,m2,n2,s2,α2,n2_α,checksucceded = buildtree_polyg(mplus,pplus,logu,v,j-1,ϵ,hamilt_zero, #m0,p0,
                                                                                 usermodel,HMCpar,grptask,
                                                                                 whichleapfrog=whichleapfrog)
                ##===========================================================
                ## If checks fail get out 
                ##===========================================================
                #bi = usermodel.likelihood("getbodyindices"); println(">> 4 buildtree(): $bi ")
                if checksucceded == false
                    #println(">> 4 buildtree(): checksucceded==false, j=$j")
                    return mplus,pplus,mplus,pplus,m2,n2,s2,α2,n2_α,checksucceded
                end
                ##===========================================================

            end
            ##----------------------------------------------------
            ## save current bodyindices
            m2bodyind = usermodel.likelihood("getbodyindices")

            ##-----------------------------
            if rand() < n2 / (n1 + n2)
                m1 = m2
                ##----------------------------------------------------
                ## set proper bodyindices from last point (forw or back)
                usermodel.likelihood("newinds",newbodyindices=m1bodyind)
            else
                ##----------------------------------------------------
                ## set proper bodyindices from last point (forw or back)
                usermodel.likelihood("newinds",newbodyindices=m2bodyind)
            end
            α1 = α1 + α2
            n1_α = n1_α + n2_α
            s1 = s2 & (dot(mplus-mminus, pminus) >= 0) & (dot(mplus-mminus, pplus) >= 0)
            n1 = n1 + n2
        end
        
        return mminus,pminus, mplus,pplus, m1,n1, s1, α1,n1_α,checksucceded
    end
end

######################################################################################




