


function test_bfgs1()

    A = [
        0.0690656   0.993496   0.445903   0.572853   0.947117  0.720711    0.696925  0.965087   0.353282  0.315425;
        0.611602    0.053809   0.669161   0.464807   0.924365  0.871622    0.847269  0.0360395  0.141117  0.0159493;
        0.0395514   0.142501   0.711801   0.233039   0.505166  0.0690927   0.965322  0.725682   0.495679  0.890202;
        0.916267    0.30668    0.343496   0.510578   0.902848  0.00737472  0.29374   0.538768   0.214043  0.0489696;
        0.00138366  0.33564    0.89326    0.623868   0.384869  0.30316     0.229153  0.204849   0.140656  0.0369546;
        0.948038    0.125859   0.880326   0.894388   0.658991  0.167686    0.555454  0.507087   0.186623  0.185145;
        0.150909    0.0126957  0.495094   0.0892106  0.878634  0.948445    0.829939  0.0195551  0.974903  0.733281;
        0.476652    0.421556   0.0572526  0.827577   0.12473   0.522138    0.505297  0.189092   0.490643  0.601307;
        0.772615    0.171942   0.418479   0.891878   0.386754  0.187241    0.993221  0.866472   0.259396  0.718616;
        0.447949    0.383255   0.843822   0.978838   0.990403  0.821275    0.687248  0.802845   0.11191   0.110758
    ]

    A = A'*A

    N = size(A,1)
    b = A*(1:N)

    function fh!(g,x)
        phi = 0.5 * dot(x,A,x)-dot(x,b)
        g .= A*x .- b
        return phi
    end

    x0 = zeros(N)

    xout,misfout = lmbfgs(fh!,x0,mem=20,maxiter=10,
                          target_update=1.0,
                          τgrad=1e-8,c2=0.1,
                          saveres=false)

    closeenough = xout[end] ≈ collect(1:N)

    return closeenough
end




function test_bfgs2()

    f = styblinskitang
    ∇f = ∇styblinskitang

    tval = −2.903534
    tvec = [tval,tval] # location of the minimum
    stmin = -39.16617*length(tvec) # value at the minimum

    function fh!(g,x)
        phi = styblinskitang(x)
        g .= ∇styblinskitang(x)
        return phi
    end

    x0 = [-0.5,4.5]

    xout,misfout = lmbfgs(fh!,x0,mem=10,maxiter=10,
                          saveres=false)

    ce1 = xout[end] ≈ tvec
    ce2 = isapprox(misfout[end],stmin,rtol=1e-4)

    @show ce1,ce2
    @show xout[end],tvec

    return ce1 && ce2
end
