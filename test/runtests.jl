

using Test
using LinearAlgebra
using InverseAlgos.Optimizers
using InverseAlgos.KronLinInv
using Logging

include("utils.jl")
include("test_lmbfgs.jl")
include("test_kronlininv.jl")




# suppress all messages from algos
logger = NullLogger()
with_logger(logger) do

    println()

    # Optimizers
    @testset "Test L-BFGS algo" begin

        printstyled("L-BFGS: Test 1\n", bold=true,color=:cyan)
        @test test_bfgs1()
        printstyled("L-BFGS: Test 2\n", bold=true,color=:cyan)
        @test test_bfgs2()

    end


    # KronLinInv
    @testset "Test KronLinInv" begin

        printstyled("KronLinInv: Testing 2D example \n", bold=true,color=:cyan)
        @test test2D()

        printstyled("KronLinInv: Testing 3D example \n", bold=true,color=:cyan)
        @test test3D()

        println()
    end

end
