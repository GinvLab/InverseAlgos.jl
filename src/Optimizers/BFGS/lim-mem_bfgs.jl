
################################################

struct MemoryVariable
    indices::Vector{Int64}
    vals::Vector{Union{Vector{Float64},Float64}}
    #maxind::Int64
    function MemoryVariable(N::Integer,typeel::String,
                            M::Union{Integer,Nothing}=nothing)
        
        if typeel=="float" && M==nothing
            vals = zeros(Float64,N)
        elseif typeel=="vector" || M!=nothing
            vals = [Vector{Float64}(undef,M) for i=1:N]
        else
            error("MemoryVariable: Wrong input 'typeel'.")
        end
        nel = length(vals)
        indices = zeros(Int64,nel) ## circshift during iterations
        return new(indices,vals)
    end
end

################################################

function updatememvar!(vmem::MemoryVariable,v::Vector{<:Union{Vector{Float64},Float64}},
                       newval::Union{Vector{Float64},Float64},Nmem::Int64)
    @assert Nmem>=0
    m = length(vmem.indices)

    if Nmem==0
        ## The history is being initialized with one single element
        vmem.indices[1] = 1
        if eltype(v)==Vector{Float64}
            vmem.vals[1] .= newval
        elseif eltype(v)==Float64
            vmem.vals[1] = newval
        else
            error("updatememvar!(): Wrong 'typeel'.")
        end
        v[1] = vmem.vals[1]

    elseif (Nmem)<m && Nmem>0
        ## The history in vmem.vals is incomplete, only some elements
        laidx = Nmem+1
        ## new values to oldest element
        if eltype(v)==Vector{Float64}
            vmem.vals[laidx] .= newval
        elseif eltype(v)==Float64
            vmem.vals[laidx] = newval
        else
            error("updatememvar!(): Wrong 'typeel'.")
        end
        ## re-bind indices after new item substitution
        vmem.indices[laidx] = 0
        vmem.indices[1:laidx] .+= 1

        ## re-bind the items to items of memory variable
        for i=1:laidx
            v[i] = vmem.vals[vmem.indices[i]]
        end

    else
        ## We already have a full vector of history in vmem.vals
        ## new values to oldest element
        idxmax_ind,_ = findmax(vmem.indices[m])
        idxmax_val = vmem.indices[idxmax_ind]

        # vmem.vals[idxmax_val] = newval
        if eltype(v)==Vector{Float64}
            vmem.vals[idxmax_ind] .= newval
        elseif eltype(v)==Float64
            vmem.vals[idxmax_ind] = newval
        else
            error("updatememvar!(): Wrong 'typeel'.")
        end

        ## re-bind indices after new item substitution
        vmem.indices[idxmax_ind] = 0
        vmem.indices[:] .+= 1

        ## re-bind the items to items of memory variable
        for i=1:m
            v[i] = vmem.vals[vmem.indices[i]]
        end

    end
    return
end

#######################################################################

function cubint(a::Real, b::Real, phia::Real, phib::Real, ga::Real, gb::Real)
    if a==b
        alpha = a
    else
        d1 = ga + gb -3*(phia - phib)/(a - b)
   
        d2 = sqrt( (d1^2) - ga*gb )

        alpha = b - (b - a)*(( gb + d2 - d1 )/( gb - ga + 2*d2))
    end
    return alpha
end

################################################

function project_model_and_1Dgrad!(x,p,bounds)

    p_bound = copy(p)

    bot_ind = x .< bounds[:,1]
    top_ind = x .> bounds[:,2]

    x[bot_ind] .= bounds[bot_ind,1]
    p_bound[bot_ind] .= 0

    x[top_ind] .= bounds[top_ind,2]
    p_bound[top_ind] .= 0

    return p_bound 
end


################################################

function twoloops(xk::Vector,g::Vector,H0::UniformScaling,ρ::Vector{Float64},
                  s::Vector{Vector{Float64}},
                  y::Vector{Vector{Float64}},Nmem::Integer)
    ## algo 7.4, Nocedal & Wright, 2006
   
    # gradient of the misfit
    q = copy(g)

    # on first iteration (Nmem==0) we don't have s, y and ρ, so...
    if Nmem==0
        return H0*q
    end

    #ni = length(xk)
    α = Vector{Float64}(undef,Nmem)
    for i=1:Nmem  # k-1,k-2,...,k-m
        α[i] = ρ[i] * dot(s[i],q)
        q .= q .- α[i].*y[i]
    end

    r = H0*q

    for i=Nmem:-1:1 ## k-m,k-m+1,...,k-1
        β = ρ[i] * dot(y[i],r)
        r .= r .+ s[i].*(α[i].-β)
    end

    Hgradf = r
    return Hgradf
end

################################################

# Line Search
# using Wolfe conditions
# see section 3.5 of Nocedal & Wright, 2006 (page 60)

function linesearchWolfe!(fh!::Function,x0αp::Vector{Float64},ϕ0::Float64,
                          g::Vector{Float64},x0::Vector,p::Vector{Float64} ;
                          bounds::Union{Nothing,Array{Float64,2}},
                          α0::Real=1.0,maxiterwolfe::Integer=10, maxiterzoom::Integer=10,
                          c1::Real = 0.0001, c2::Real = 0.9)

    # see section 3.5 of Nocedal & Wright, 2006 (page 60)
    ## algo 3.5 of Nocedal & Wright, 2006

    αold = 0.0
    @assert 0.0<=α0

    @assert 0.0<c1<c2<1.0
   
    dϕdα_0 = dot(g,p)
    dϕdαold = dϕdα_0

    ϕold = ϕ0
    # x0αp = similar(x0)

    α = α0
    ϕ = 0.0 # to allow to return it
    for i=1:maxiterwolfe

        x0αp .= x0 .+ α.*p
        if bounds!=nothing
            p_bound = project_model_and_1Dgrad!(x0αp,p,bounds)            
            ϕ = fh!(g, x0αp ) #x0αp)
            # grad with respect to α
            dϕdα = dot(g,p_bound)
        else
            ϕ = fh!(g, x0αp ) #x0αp)
            # grad with respect to α
            dϕdα = dot(g,p)
        end

        ## Wolfe 1 check
        if ϕ>(ϕ0+c1*α0*dϕdα_0) || (ϕ>=ϕold && i>1 )
            αout,ϕout,success = zoom!(g,x0αp,x0,p,c1,c2,fh!,
                                      ϕ0,dϕdα_0,
                                      αold,α,
                                      ϕold,ϕ,
                                      dϕdαold,dϕdα,
                                      bounds=bounds,
                                      maxiter=maxiterzoom)
            return αout,ϕout,success # ϕ was calculated with αout
        end

        ## Wolfe 2 check
        if abs(dϕdα) <= -c2*dϕdα_0
            αout = α
            success = true
            return αout,ϕ,success # ϕ was calculated with α0
        end

        ## Overshoot, zoom
        if dϕdα >= 0.0
            αout,ϕout,success = zoom!(g,x0αp,x0,p,c1,c2,fh!,
                                      ϕ0,dϕdα_0,
                                      α,αold, # swapped w.r.t. Wolfe check 1
                                      ϕ,ϕold, # swapped w.r.t. Wolfe check 1
                                      dϕdα,dϕdαold, # swapped w.r.t. Wolfe check 1
                                      bounds=bounds,
                                      maxiter=maxiterzoom)
            return αout,ϕout,success # ϕ was calculated with αout
        end

        # update α old and new
        αold = α
        ##  α[i]<=α0<=αmax
        grat = dϕdα_0/(dϕdα_0-dϕdα)
        if grat > 1
            α = grat*αold #Estimate as desired change in gradient over actual change
        else
            α = 2*αold #Don't do the above when it doesn't result in a step increase
        end
        # update ϕold
        ϕold = ϕ
        dϕdαold = dϕdα

    end

    # line search failed
    success = false
    return αold,ϕ,success # which is the last
end

#######################################################################

function zoom!(g_full::Vector,x0αp::Vector{Float64},
               x0::Vector{<:Real},
               p::Vector{<:Real},
               c1::Real,c2::Real,fh!::Function,
               ϕ0::Real,dϕdα_0::Real,
               αlo::Real,αhi::Real,ϕlo::Real,ϕhi::Real,
               glo::Real,ghi::Real;
               bounds::Union{Nothing,Array{Float64,2}}=nothing,
               maxiter::Integer=10)
    ## algo 3.6 of Nocedal & Wright, 2006
    
    α = 0.0 # to allow returning it outside the for loop
    ϕtrial = 0.0 # to allow returning it outside the for loop
    #x0αp = similar(x0)

    for i=1:maxiter

        if αlo < αhi
            α = cubint(αlo,αhi,ϕlo,ϕhi,glo,ghi)
        elseif αlo == αhi
            α = αlo
        else
            α = cubint(αhi,αlo,ϕhi,ϕlo,ghi,glo)
        end

        x0αp .= x0 .+ α.*p        
        if bounds!=nothing
            p_bound = project_model_and_1Dgrad!(x0αp,p,bounds)
            ϕtrial = fh!(g_full, x0αp) 
            gtrial = dot(g_full,p_bound)
        else
            ϕtrial = fh!(g_full, x0αp) 
            gtrial = dot(g_full,p)
        end
        
        if (ϕtrial > ϕ0.+c1*α*dϕdα_0) || (ϕtrial>=ϕlo)
            αhi = α

        else
            if abs(gtrial) <= -c2*dϕdα_0
                αout = α
                success = true
                return αout,ϕtrial,success # ϕtrial was calculated with αtrial
            end

            if gtrial*(αhi-αlo) >= 0.0
                αhi = αlo
            end

            αlo = α
            ϕlo = ϕtrial
        end

    end

    # line search failed
    success = false
    return α,ϕtrial,success
end

#########################################################


"""
  $(TYPEDSIGNATURES)

An implementation of the L-BFGS algorithm following Nocedal & Wright, 2006 with the addition of box constraints.

# Arguments
- `f`: a function returning the misfit (a ::Function)
- `∇f`: a function returning the gradient (a ::Function)
- `x0`: the starting model/initial guess
- `mem`: the length (number of iterations) used for memory variables
- `maxiter`: maximum number of iterations
- `bounds` (optional): a two-column array where the first column contains lower bounds (constraints) and the second upper bounds
- `target_update` (optional): initial step length for the line search
- `outfile` (optional): the name of the output file where to save the results
- `τgrad` (optional): minimum value of the gradient at which to stop the algorithm
- `overwriteoutput` (optional): if true overwrite the output file if already existing
- `maxiterwolfe` (optional): maximum number of iterations for the line search function
- `maxiterzoom` (optional): maximum number of iterations for the zoom function
- `c1` and `c2` (optional): strong Wolfe values
- `saveres` (optional): save results? Defaults to true

# Returns
- `x`: a vector containing the solution for each iterarion
- `misf`: a vector containing the misfit value for each iteration

"""
function lmbfgs(f::Function,∇f::Function, args... )

    function fh!(grad,x)
        grad .= ∇f(x)
        misf = f(x)
        return misf
    end

    return lmbfgs(fh!, args... )
end


"""
  $(TYPEDSIGNATURES)

An implementation of the L-BFGS algorithm following Nocedal & Wright, 2006 with the addition of box constraints.

# Arguments
- `fh!`: a function returning the misfit to be minimized and computing its gradient in place (a ::Function)
- `x0`: the starting model/initial guess
- `mem`: the length (number of iterations) used for memory variables
- `maxiter`: maximum number of iterations
- `bounds` (optional): a two-column array where the first column contains lower bounds (constraints) and the second upper bounds
- `target_update` (optional): initial step length for the line search
- `outfile` (optional): the name of the output file where to save the results
- `τgrad` (optional): minimum value of the gradient at which to stop the algorithm
- `overwriteoutput` (optional): if true overwrite the output file if already existing
- `maxiterwolfe` (optional): maximum number of iterations for the line search function
- `maxiterzoom` (optional): maximum number of iterations for the zoom function
- `c1` and `c2` (optional): strong Wolfe values
- `saveres` (optional): save results? Defaults to true

# Returns
- `x`: a vector containing the solution for each iterarion
- `misf`: a vector containing the misfit value for each iteration
 
"""
function lmbfgs(fh!::Function,
                x0::Vector{<:Real};
                bounds::Union{Nothing,Array{Float64,2}}=nothing,
                mem::Integer,
                maxiter::Integer,
                target_update::Real=1.0,
                outfile::String="results_lbgfs.h5",
                τgrad::Real=1e-8,
                overwriteoutput::Bool=false,
                maxiterwolfe::Integer=10,
                maxiterzoom::Integer=10,
                c1::Real=0.0001,
                c2::Real=0.9,
                saveres::Bool=true)
    ##
    if bounds==nothing
        @info "\n***  L-BFGS optimization (unconstrained) ***\n"
    else
        @info "\n***  L-BFGS optimization with box constraints  ***\n"
    end

    ## algo 7.5, Nocedal & Wright, 2006
    @assert mem>0
    @assert target_update >= 0.0
    if bounds!=nothing
        @assert all(bounds[:,1].<bounds[:,2])
    end

    npar = length(x0)
    x = [Vector{Float64}(undef,npar) for i=1:maxiter+1]  ##Vector{Vector{Float64}}(undef,maxiter+1)
    x[1] = x0
    misf = zeros(maxiter+1)

    ## s, y and ρ
    s = [Vector{Float64}(undef,npar) for i=1:mem]  #Vector{Vector{Float64}}(undef,mem)
    y = [Vector{Float64}(undef,npar) for i=1:mem]  #Vector{Vector{Float64}}(undef,mem)
    ρ = Vector{Float64}(undef,mem)
    ## set memory variables
    smem = MemoryVariable(mem,"vector",npar)
    ymem = MemoryVariable(mem,"vector",npar)
    ρmem = MemoryVariable(mem,"float")

    # descent direction
    p = zeros(npar)

    # initial misfit
    gradf_k = similar(x0)
    misf[1] = fh!(gradf_k,x[1])
    g0norm = norm(gradf_k)
    @info " Initial misfit: $(misf[1])"

    # pre-allocate stuff
    snew = similar(x0)
    ynew = similar(x0)
    x0αp = similar(x0)
    gradf_old = similar(gradf_k)


    if saveres
        ## init saving stuff
        if isfile(outfile)
            rm(outfile)
        end
        savestuff!(outfile,overwriteoutput,0,misf[1],x[1])
        @info " Saving results to $outfile"
    end

    ##------------------------
    ## Main loop
    Nmemk = -1
    for k=1:maxiter

        ##------------------------
        # choose Hk
        if k>1
            # last s and y, so s[1] and y[1]
            γ = dot(s[1],y[1])/dot(y[1],y[1])  #(7.20)
        else
            γ = 1.0
        end
        H0 = UniformScaling(γ) ## γ*I  # I->UniformScaling in Julia

        ##------------------------
        ## update length of memory variables (useful for k<m),
        ##  i.e. how many items are available up to now:
        ##  zero at iteration 1; 1 at iter. 2; ...
        if Nmemk < mem
            Nmemk += 1
        end

        ##------------------------
        ## Two loops
        # Hgradf = twoloops(x[k],gradf_k,H0,ρ,s,y,Nmemk)
        # p .= - Hgradf # negative
        p .= - twoloops(x[k],gradf_k,H0,ρ,s,y,Nmemk)

        ##------------------------
        # compute α by line search using strong Wolfe condition
        ϕ0 = misf[k]

        # compute initial step length based on prior knowledge
        if k==1
            # # first iteration only
            # if target_update == 0
            #     α0 = 0.1*ϕ0/(sum(p.^2)) # Assuming minimum objective is 0, tries to take a step 10% of the way there
            # else
            #     # α0 = target_update / sqrt(sum(p.^2))
                α0 = target_update
            # end 
        else
            α0 = 1
        end

        ## line search, Wolfe condition
        gradf_old .= gradf_k
        
        α,misf[k+1],success = linesearchWolfe!(fh!,x0αp,ϕ0,gradf_k,x[k],p,
                                               bounds=bounds,
                                               α0=α0,maxiterwolfe=maxiterwolfe,
                                               maxiterzoom=maxiterzoom,
                                               c1=c1,c2=c2)
        ##------------------------
        if !success
            @warn "The line search did not converge at iteration $k"
            # println("\n Line search failed, resetting history at iteration $k")
            # p = - γ .* gradf_k
            # Nmemk = 0 # Wipe history for ρ, s and y!
            
            # α,misf[k+1],success = linesearchWolfe!(fh!,x0αp,ϕ0,gradf_k,x[k],p,
            #                                        bounds=bounds,
            #                                        α0=α0,maxiterwolfe=maxiterwolfe,
            #                                        maxiterzoom=maxiterzoom,
            #                                        c1=c1,c2=c2)
        end
        # if !success
        #     println(" Line search failed AGAIN (iteration $k)   ")
        #     #println(" Current misfit $(misf[k]).\nAborting... \n")
        #     return x[1:k],misf[1:k] 
        # end
        ##------------------------

        ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)
        REPL.Terminals.clear_line(ter)
        #@info "Iteration $k of $maxiter, misfit: $(misf[k+1])         "
        @info "Iteration $k of $maxiter, misfit: $(misf[k+1])         "
        REPL.Terminals.cmove_line_up(ter)
        flush(stdout)
        sleep(2)
        ##------------------------
        # update the solution x
        x[k+1] .= x[k] .+ α.*p
        if bounds!=nothing
            # project x[k+1]
            project_model_and_1Dgrad!(x[k+1],p,bounds)
        end
        
        ##------------------------
        if saveres
            ## save stuff
            savestuff!(outfile,overwriteoutput,k,misf[k+1],x[k+1])
        end

        ##---------------------------------------------
        ## Update s, y and ρ, rearrange memory arrays

        # compute and save s and y
        snew .= x[k+1].-x[k]

        # if the solution does not change with the update...
        if all(snew.==0.0)
            @info "\n*** all(x[k+1].-x[k])==0.0, aborting ***"
            println()
            return x[1:k+1],misf[1:k+1] 
        end

        ynew .= gradf_k .- gradf_old
        # save also ρ
        ρnew = 1.0/dot(ynew,snew)
    
        # update the indices and values
        updatememvar!(smem,s,snew,Nmemk)
        updatememvar!(ymem,y,ynew,Nmemk)
        updatememvar!(ρmem,ρ,ρnew,Nmemk)
    
        ##------------------------
        ## check norm of the gradient
        gnorm = norm(gradf_k)
        if abs(gnorm/g0norm) < τgrad
            # println("gnorm is "*string(gnorm)*" and g0norm is "*string(g0norm))
            @info "\n***  norm(gradf) < τgrad, breaking iterations ***"
            return x[1:k+1],misf[1:k+1]
        end
    end

    #println("\n")
    return x[1:maxiter+1],misf[1:maxiter+1]
end


#############################################################################

function savestuff!(outfile::String,overwriteoutput::Bool,k::Integer,misfnew::Real,xnew::Vector{<:Real})

    nmpar = length(xnew)

    if k==0

        if overwriteoutput==true
            fid_h5 = h5open(outfile,"w") #,swmr=true)
        else
            fid_h5 = h5open(outfile,"cw") #,swmr=true)
        end
        misf_h5 = HDF5.create_dataset(fid_h5, "misfit", Float64, ((1,), (-1,)),chunk=(100,)  )
        mods_h5 = HDF5.create_dataset(fid_h5, "mods", Float64, ((nmpar,1), (nmpar,-1)),chunk=(nmpar,1) )

        HDF5.set_extent_dims(misf_h5,(k+1,))
        misf_h5[k+1] = misfnew

        HDF5.set_extent_dims(mods_h5,(nmpar,k+1))
        mods_h5[:,k+1] = xnew

        close(fid_h5)

    else

        fid_h5 = h5open(outfile,"r+")

        misf_h5 = HDF5.open_dataset(fid_h5,"misfit")
        HDF5.set_extent_dims(misf_h5,(k+1,))
        misf_h5[k+1] = misfnew

        mods_h5 = HDF5.open_dataset(fid_h5,"mods")
        HDF5.set_extent_dims(mods_h5,(nmpar,k+1))
        mods_h5[:,k+1] = xnew

        close(fid_h5)

    end

    return
end

#############################################################################
