


function gaussnewton(calcfwd!::Function,
                     calcjac!::Function,
                     obsdata::Vector{<:Real},
                     invCd::Matrix{<:Real},
                     invCm::Matrix{<:Real},
                     xprior::::Vector{<:Real},
                     x0::Vector{<:Real};
                     bounds::Union{Nothing,Array{Float64,2}}=nothing,
                     mem::Integer,
                     maxiter::Integer,
                     target_update::Real=1.0,
                     outfile::String="results_gaussnewton.h5",
                     τgrad::Real=1e-8,
                     overwriteoutput::Bool=false,
                     maxiterwolfe::Integer=10,
                     maxiterzoom::Integer=10,
                     c1::Real=0.0001,
                     c2::Real=0.9,
                     saveres::Bool=true)

    ## checks
    @assert size(invCd)==size(invCm)
    @assert size(invCm,1)==length(x0)
    @assert size(invCd,1)==length(obsdata)


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
    tmpgr_d = zeros(eltype(x0),M)
    tmpgr_m = zeros(eltype(x0),M)

    x[1] .= x0
    jac = zeros(eltype(x0),N,M)
    
    
    function fh!(grad,x) 
        # Calculated data
        calcfwd!(u_calc,x)
        # Jacobian
        calcjac!(jac,x)
        ##=======================
        #   Value of objective function
        # misfit
        res_d .= u_calc - u_obs
        objval = 0.5 * dot(res_d,invCd,res_d)
        # prior term
        res_m .= x - xprior
        objval += 0.5 * dot(res_m,invCm,res_m)
        ##=======================
        #    Gradient
        # Jtres = J*res, i.e., the gradient
        mul!(tmpgr_d,invCd,res_d)
        mul!(grad,transpose(jac),tmpgr_d)
        mul!(tmpgr_m,invCm,res_m)
        grad .+= tmpgr_m
        return objval
    end

    ##===============================
    # Calculate initial gradient and objective function
    #   and get the in-place jacobian as an optional argument
    misf[1] = fh!(grad,x[k])

    
    ## Loop
    for k=1:maxiter
     
        ##===============================
        # Jacobian
        calcjac!(jac,x)
        # invCd * J
        mul!(invCd_J,invCd,jac)
        # temporary store invCm in the Hessian array
        H .= invCm
        # get approximation of the Hessian
        mul!(H,transpose(jac),invCd_J,1.0,1.0)
        # solve linear system
        faJtJ = factorize(Symmetric(H))
        # J^t*J*p_gn = -J^t*res
        #  pkgn is the descent direction
        ldiv!(pkgn,faJtJ,-grad)

        ##===============================
        # Line search
        α,misf[k+1],success = linesearchWolfe!(fh!,x0αp,misf[k],grad,
                                               x[k],pkgn,
                                               bounds=bounds,
                                               α0=α0,maxiterwolfe=maxiterwolfe,
                                               maxiterzoom=maxiterzoom,
                                               c1=c1,c2=c2)

        ##===============================
        # Update the solution
        x[k+1] .= x[k] .+ α.*pkgn

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
  
        ##------------------------
        ## check norm of the gradient
        gnorm = norm(grad_k)
        if abs(gnorm/g0norm) < τgrad
            @info "\n***  norm(gradf) < τgrad, breaking iterations ***"
            return x[1:k+1],misf[1:k+1]
        end
    end

    return x[1:maxiter+1],misf[1:maxiter+1]
end
