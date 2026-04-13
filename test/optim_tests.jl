

using InverseAlgos.Optimizers
using NonlinearOptimizationTestFunctions


function test_bfgs_testfunc()

    N = 10
    test_funcs = ["rosenbrock","himmelblau","styblinski_tang","sphere",
                  "ackley"] # Failing on this one...

    allpass = true
    for tf in test_funcs
        testfun = fixed(TEST_FUNCTIONS[tf],n=N)
        pass = optimtestfun(testfun,N)
        allpass = allpass && pass
    end

    @show allpass
    return allpass
end


function optimtestfun(testfun,N)

    println("\n##===>>> Testing $(testfun.name) function, $N dimensions ")
    minval = min_value(testfun)
    minpos = min_position(testfun)

    function fh!(g,x)
        phi = testfun.f(x)
        testfun.gradient!(g,x)
        return phi
    end

    x0 = start(testfun)
    @show x0
    @show minpos
    
    xout,misfout = lmbfgs(fh!,x0,
        mem=20,
        maxiter=50,
        saveres=false)

    ce1 = isapprox(xout[end],minpos,atol=1e-8)
    ce2 = isapprox(misfout[end],minval,atol=1e-8)

    # @show ce1,ce2
    # @show minval,misfout[end]
    # @show xout[end].-minpos
    out = ce1 && ce2
    return out
end
