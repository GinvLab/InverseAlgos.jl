
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
- `fh!`: a function (::Function) returning the misfit to be minimized and computing its gradient in place, e.g., misf=fh!(grad,x)
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
            # first iteration only
            α0 = target_update
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

        min_log_level_acc = Logging.min_enabled_level(current_logger())
        if min_log_level_acc <= LogLevel(Info)
            ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)
            if k>1
                REPL.Terminals.cmove_line_up(ter)
            end
            REPL.Terminals.clear_line(ter)
            @info "Iteration $k of $maxiter, misfit: $(misf[k+1])         "
            flush(stdout)
        end
        
        ##===============================
        # update the solution x
        x[k+1] .= x[k] .+ α.*p
        if bounds!=nothing
            # project x[k+1]
            bot_ind = x[k+1] .< bounds[:,1]
            top_ind = x[k+1] .> bounds[:,2]
            x[k+1][bot_ind] .= bounds[bot_ind,1]
            x[k+1][top_ind] .= bounds[top_ind,2]
            ## project_model_and_1Dgrad!(x[k+1],p,bounds)
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
            #println()
            @info "\n*** all(x[k+1].-x[k])==0.0, aborting ***"
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
            #println()
            # println("gnorm is "*string(gnorm)*" and g0norm is "*string(g0norm))
            @info "\n***  norm(gradf) < τgrad, breaking iterations ***"
            return x[1:k+1],misf[1:k+1]
        end
    end

    #println()
    return x[1:maxiter+1],misf[1:maxiter+1]
end


#############################################################################
