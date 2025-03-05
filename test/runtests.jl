

using Test
using LinearAlgebra
using InverseAlgos.Optimizers
using InverseAlgos.KronLinInv
using Logging

include("utils.jl")
include("test_lmbfgs.jl")
include("test_kronlininv.jl")
include("test_gaussnewton.jl")


# suppress all messages from algos
logger = NullLogger()
#logger = ConsoleLogger()

with_logger(logger) do
    # Optimizers
    @testset "Test L-BFGS algo" begin

        printstyled("L-BFGS: Test 1\n", bold=true,color=:cyan)
        @test test_bfgs1()
        
        printstyled("L-BFGS: Test 2\n", bold=true,color=:cyan)
        @test test_bfgs2()

    end

    @testset "Test Gauss-Newton algo" begin

        printstyled("Gauss-Newton: Test 1\n", bold=true,color=:cyan)
        @test test_gaussnewton1()
        
        printstyled("Gauss-Newton: Test 2\n", bold=true,color=:cyan)
        @test test_gaussnewton2()

        printstyled("Gauss-Newton: Test 3\n", bold=true,color=:cyan)
        @test test_gaussnewton3()
    end
end


# KronLinInv
@testset "Test KronLinInv" begin

    printstyled("KronLinInv: Testing 2D example \n", bold=true,color=:cyan)
    @test test_KLI2D()

    printstyled("KronLinInv: Testing 3D example \n", bold=true,color=:cyan)
    @test test_KLI3D()

    println()
end


