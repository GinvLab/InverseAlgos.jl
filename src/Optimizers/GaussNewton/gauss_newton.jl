

"""
  $(TYPEDSIGNATURES)

An implementation of the Gauss-Newton algorithm with the addition of box constraints to solve non-linear least squares problems. This algorithm assumes both the likelihood function and the prior to be Gaussian.

# Arguments
- `calcfwd!`: a function (::Function) solving the forward problem. It must have the following signature `calcfwd!(u,x)`, where `u` represents the vector of calculated data and `x` a vector of the input model parameters.
- `calcjac!`: a function (::Function) computing the Jacobian of the forward problem. It must have the following signature `o = calcjac!(jac,x)`, where `jac` is the Jacobian matrix of the forward (partial derivatives) and `x` a vector of the input model parameters.
- `obsdata`: the vector containing the observed data
- `invCd`: the inverse of the covariance of the observed data
- `invCm`: the inverse of the covariance of the prior model
- `xprior`: the prior model (a vector)
- `x0` (optional): the starting model/initial guess
- `maxiter`: maximum number of iterations
- `target_update` (optional): initial step length for the line search
- `bounds` (optional): a two-column array where the first column contains lower bounds (constraints) and the second upper bounds
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
function gaussnewton(calcfwd!::Function,
                     calcjac!::Function;
                     obsdata::Vector{T},
                     invCd::Matrix{T},
                     invCm::Matrix{T},
                     xprior::Vector{T},
                     x0::Union{Vector{T},Nothing}=nothing,
                     maxiter::Integer,
                     target_update::T=1.0,
                     bounds::Union{Nothing,Array{Float64,2}}=nothing,
                     outfile::String="results_gaussnewton.h5",
                     τgrad::T=1e-8,
                     overwriteoutput::Bool=false,
                     maxiterwolfe::Integer=10,
                     maxiterzoom::Integer=10,
                     c1::T=0.0001,
                     c2::T=0.9,
                     saveres::Bool=true) where T<:AbstractFloat

    ## checks
    @assert size(invCm,1)==length(xprior)
    @assert length(xprior)==length(xprior)
    @assert size(invCd,1)==length(obsdata)
    if x0==nothing
        x0 = copy(xprior)
    end
    @assert length(xprior)==length(x0)

    @info "\n*** Gauss-Newton optimization  *** "

    ## init
    N = size(invCd,1)
    M = length(x0)
    x = [Vector{Float64}(undef,M) for i=1:maxiter+1]
    misf = zeros(eltype(x0),maxiter+1)
    H = zeros(eltype(x0),M,M)
    u_calc = zeros(eltype(x0),N)
    invCd_J = zeros(eltype(x0),N,M)
    grad = zeros(eltype(x0),M)
    pkgn = zeros(eltype(x0),M)
    res_d = zeros(eltype(x0),N)
    res_m = zeros(eltype(x0),M)
    jac = zeros(eltype(x0),N,M) 
    x0αp = zeros(eltype(x0),M)
    tmpgr_d = zeros(eltype(x0),N)
    tmpgr_m = zeros(eltype(x0),M)

    x[1] .= x0
    jac = zeros(eltype(x0),N,M)


    function fh!(grad,xcur) 
        # Calculated data
        calcfwd!(u_calc,xcur)
        # Jacobian
        calcjac!(jac,xcur)
        ##=======================
        #   Value of objective function
        # misfit
        res_d .= u_calc - obsdata
        objval = 0.5 * dot(res_d,invCd,res_d)
        # prior term
        res_m .= xcur - xprior
        objval += 0.5 * dot(res_m,invCm,res_m)
        ##=======================
        #   Gradient
        #  grad = J^T * Cd^-1 * res_d + Cm^-1 * res_m
        mul!(tmpgr_d,invCd,res_d)
        mul!(grad,transpose(jac),tmpgr_d)
        mul!(tmpgr_m,invCm,res_m)
        grad .+= tmpgr_m
        return objval
    end


    if saveres
        ## init saving stuff
        if isfile(outfile)
            rm(outfile)
        end
        savestuff!(outfile,overwriteoutput,0,misf[1],x[1])
        @info " Saving results to $outfile"
    end

    ##===============================
    # Calculate initial gradient and objective function
    #   and get the in-place jacobian as an optional argument
    misf[1] = fh!(grad,x[1])
    g0norm = norm(grad)

    @info "Initial misfit $(misf[1])     "
    ## Loop
    for k=1:maxiter
     
        ##===============================
        # Jacobian
        calcjac!(jac,x[k])
        # Now (J^T*invCd*J + invCm) p_gn = - (J^T*invCd) *res
        # invCd * J
        mul!(invCd_J,invCd,jac)
        # temporary store invCm in the Hessian array
        H .= invCm
        # get approximation of the Hessian
        #  J^T * invCd * J + invCm
        mul!(H,transpose(jac),invCd_J,1.0,1.0)
        # solve linear system
        #faJtJ = factorize(Symmetric(H)) # this seems to get singular...
        @show H
        faH = lu(H)
        @show faH
        # ## (J^T*invCd*J + invCm) p_gn = - (J^T*invCd) *res
        # ##  pkgn is the descent direction
        ldiv!(pkgn,faH,-grad)
        ##===============================

        ##===============================
        ## Line search
        if k==1
            α0 = target_update
        else
            α0 = 1.0
        end
        α,misf[k+1],success = linesearchWolfe!(fh!,x0αp,misf[k],grad,
                                               x[k],pkgn,
                                               bounds=bounds,
                                               α0=α0,maxiterwolfe=maxiterwolfe,
                                               maxiterzoom=maxiterzoom,
                                               c1=c1,c2=c2)
        # Update the solution
        x[k+1] .= x0αp   ###x[k] .+ α.*pkgn
        ##===============================


        if bounds!=nothing
            # project x[k+1]
            bot_ind = x[k+1] .< bounds[:,1]
            top_ind = x[k+1] .> bounds[:,2]
            x[k+1][bot_ind] .= bounds[bot_ind,1]
            x[k+1][top_ind] .= bounds[top_ind,2]
            ## project_model_and_1Dgrad!(x[k+1],p,bounds)
        end

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
        
 
        ##------------------------
        if saveres
            ## save stuff
            savestuff!(outfile,overwriteoutput,k,misf[k+1],x[k+1])
        end
  
        ##------------------------
        ## check norm of the gradient
        gnorm = norm(grad)
        if abs(gnorm/g0norm) < τgrad
            @info "\n***  norm(gradf) < τgrad, breaking iterations ***"
            return x[1:k+1],misf[1:k+1]
        end
    end

    return x[1:maxiter+1],misf[1:maxiter+1]
end
