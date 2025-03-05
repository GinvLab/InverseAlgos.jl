

using LinearAlgebra
## using GLMakie


function test_gaussnewton1()

    tval = −2.903534
    tvec = [tval,tval] # location of the minimum
    stmin = -39.16617*length(tvec) # value at the minimum
  
    function calcfwd!(u,x)
        u .= styblinskitang(x)
        return 
    end

    function calcjac!(g,x)
        g[1,:] .= ∇styblinskitang(x)
        return 
    end
    
    xprior = [-1.5,-1.5]

    obsdata = [stmin]
    invCd = [1/10.0;;]
    invCm = [1/5.0   0;
             0   1/5.0]

    maxiter = 100

    xout,misfout = gaussnewton(calcfwd!,calcjac!,
                               obsdata=obsdata,
                               invCd=invCd,
                               invCm=invCm,
                               xprior=xprior,
                               maxiter=maxiter)
    @show tvec
    @show xout[end]

    
    # N = size(invCd,1)
    # M = length(xprior)
    # u_calc = zeros(eltype(xprior),N)
    # grad = zeros(eltype(xprior),M)
    # jac = zeros(eltype(xprior),N,M) 
    
    # function fh_bfgs!(grad,xcur)
    #     # Calculated data
    #     calcfwd!(u_calc,xcur)
    #     # Jacobian
    #     calcjac!(jac,xcur)
    #     ##=======================
    #     #   Value of objective function
    #     # misfit
    #     res_d = u_calc - obsdata
    #     objval = 0.5 * dot(res_d,invCd,res_d)
    #     # prior term
    #     res_m = xcur - xprior
    #     objval += 0.5 * dot(res_m,invCm,res_m)
    #     grad .= transpose(jac) * invCd * res_d + invCm * res_m
    #     return objval
    # end
    
    # xout_bfgs,misfout_bfgs = lmbfgs(fh_bfgs!,
    #                                 xprior,
    #                                 mem=20,
    #                                 maxiter=maxiter)
    # @show xout_bfgs[end]
    
    ce1 = isapprox(xout[end],tvec,atol=0.3)
    return ce1 
end


function test_gaussnewton2()

    # generate function
    function fwd(a::Real,b::Real,x::Vector{Float64})
        return a.*x./(b.+x)
    end

    function calcfwd!(u,m)
        u .= fwd(m[1],m[2],x)
        return nothing
    end

    function calcjac!(jac,m)
        aj = m[1]
        bj = m[2]
        grad_a = x ./ (bj.+x) 
        grad_b = - ((aj.*x) ./ ((bj .+ x).^2))
        jac .= [grad_a grad_b]
        return
    end
    
    a,b = 2,3
    x = collect(range(0,5,length=25))
    y = fwd(a,b,x) 

    ab0 = [1.0,5.0]
    xprior = copy(ab0)
    obsdata = y
    invCd = diagm(1.0 .* ones(length(obsdata)))
    invCm = diagm(0.0 .* ones(length(ab0)))
    
    maxiter = 30

    xout,misfout = gaussnewton(calcfwd!,calcjac!,
                               obsdata=obsdata,
                               invCd=invCd,invCm=invCm,
                               xprior=xprior,maxiter=maxiter,
                               target_update=3.0)

    ce1 = xout[end] ≈ [a,b]
    return ce1
end
    

function test_gaussnewton3()
  
    function calcfwd!(u,x)
        u .= sum(x.^2)
        return 
    end

    function calcjac!(J,x)
        J[1,:] .= 2 .* x
        return 
    end
    
    xprior = [1.5,1.5]

    obsdata = [0.0]
    invCd = [1.0;;]
    invCm = [1/5.0   0;
             0   1/5.0]

    maxiter = 50

    xout,misfout = gaussnewton(calcfwd!,calcjac!,
                               obsdata=obsdata,
                               invCd=invCd,
                               invCm=invCm,
                               xprior=xprior,
                               maxiter=maxiter)

    # N = size(invCd,1)
    # M = length(xprior)
    # u_calc = zeros(eltype(xprior),N)
    # grad = zeros(eltype(xprior),M)
    # jac = zeros(eltype(xprior),N,M) 
    
    # function fh_bfgs!(grad,xcur)
    #     # Calculated data
    #     calcfwd!(u_calc,xcur)
    #     # Jacobian
    #     calcjac!(jac,xcur)
    #     ##=======================
    #     #   Value of objective function
    #     # misfit
    #     res_d = u_calc - obsdata
    #     objval = 0.5 * dot(res_d,invCd,res_d)
    #     # prior term
    #     res_m = xcur - xprior
    #     objval += 0.5 * dot(res_m,invCm,res_m)
    #     grad .= transpose(jac) * invCd * res_d + invCm * res_m
    #     return objval
    # end
    
    # xout_bfgs,misfout_bfgs = lmbfgs(fh_bfgs!,
    #                                 xprior,
    #                                 mem=20,
    #                                 maxiter=maxiter)
    # @show xout_bfgs[end]

    @show xout[end]
    ce1 = isapprox(xout[end],[0.0,0.0],atol=0.6)
    return ce1 
end
