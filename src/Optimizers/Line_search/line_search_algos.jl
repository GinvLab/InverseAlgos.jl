


################################################

# Line Search
# using Wolfe conditions
# see section 3.5 of Nocedal & Wright, 2006 (page 60)

function linesearchWolfe!(fh!::Function,x0αp::Vector{Float64},ϕ0::Float64,
                          grad::Vector{Float64},x0::Vector,p::Vector{Float64} ;
                          bounds::Union{Nothing,Array{Float64,2}},
                          α0::Real=1.0,maxiterwolfe::Integer=10, maxiterzoom::Integer=10,
                          c1::Real = 0.0001, c2::Real = 0.9)

    # see section 3.5 of Nocedal & Wright, 2006 (page 60)
    ## algo 3.5 of Nocedal & Wright, 2006

    αold = 0.0
    @assert 0.0<=α0

    @assert 0.0<c1<c2<1.0
   
    dϕdα_0 = dot(grad,p)
    dϕdαold = dϕdα_0

    ϕold = ϕ0
    # x0αp = similar(x0)

    α = α0
    ϕ = 0.0 # to allow to return it
    for i=1:maxiterwolfe

        x0αp .= x0 .+ α.*p
        if bounds!=nothing
            p_bound = project_model_and_1Dgrad!(x0αp,p,bounds)            
            ϕ = fh!(grad, x0αp ) #x0αp)
            # grad with respect to α
            dϕdα = dot(grad,p_bound)
        else
            ϕ = fh!(grad, x0αp ) #x0αp)
            # grad with respect to α
            dϕdα = dot(grad,p)
        end 

        ## Wolfe 1 check
        if ϕ>(ϕ0+c1*α0*dϕdα_0) || (ϕ>=ϕold && i>1 )
            αout,ϕout,success = zoom!(grad,x0αp,x0,p,c1,c2,fh!,
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
            αout,ϕout,success = zoom!(grad,x0αp,x0,p,c1,c2,fh!,
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
        gradat = dϕdα_0/(dϕdα_0-dϕdα)
        if gradat > 1
            # Estimate as desired change in gradient over actual change
            α = gradat*αold 
        else
            # Don't do the above when it doesn't result in a step increase
            α = 2*αold 
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
