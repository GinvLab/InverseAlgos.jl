

using Test
using LinearAlgebra
using InverseAlgos.Optimizers
using InverseAlgos.KronLinInv


include("utils.jl")
include("test_lmbfgs.jl")
include("test_kronlininv.jl")


# Optimizers
@testset "Test L-BFGS algo" begin

    @test test_bfgs1()
    @test test_bfgs2()

end


# KronLinInv
@testset "Test KronLinInv" begin

    printstyled("Testing 2D example \n", bold=true,color=:cyan)
    @test test2D()

    printstyled("Testing 3D example \n", bold=true,color=:cyan)
    @test test3D()

    println()
end

