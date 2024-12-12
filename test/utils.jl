


function styblinskitang(x)
    N = length(x)
    z=0.0
    for i=1:N
        z += 0.5*(x[i]^4-16*x[i]^2+5*x[i])
    end
    return z
end


function ∇styblinskitang(x)
    N = length(x)
    gradz = zeros(N)
    for i=1:N
        gradz[i] = 0.5*( 4*x[i]^3 - 32*x[i] + 5 )
    end
    return gradz
end
