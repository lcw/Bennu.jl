@testset "cells" begin
    TAs = ((Float64,  Array),
           (Float32,  Array),
           (BigFloat, Array))
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end

    for (T, A) in TAs
        cell = LobattoCell{T, A}(3, 3)
        @test floattype(typeof(cell)) == T
        @test arraytype(typeof(cell)) <: A
        @test Base.ndims(typeof(cell)) == 2
        @test size(typeof(cell)) == (3, 3)
        @test length(typeof(cell)) == 9
        @test floattype(cell) == T
        @test arraytype(cell) <: A
        @test Base.ndims(cell) == 2
        @test size(cell) == (3, 3)
        @test length(cell) == 9
        @test sum(mass(cell)) .≈ 4
        @test mass(cell) isa Diagonal
        D = derivatives(cell)
        @test Array(D[1] * points(cell)) ≈ fill(SVector(one(T), zero(T)), 9)
        @test Array(D[2] * points(cell)) ≈ fill(SVector(zero(T), one(T)), 9)

        @test adapt(Array, cell) isa LobattoCell{T, Array}

        s = (3, 4, 2)
        cell = LobattoCell{T, A}(s...)
        @test floattype(cell) == T
        @test arraytype(cell) <: A
        @test Base.ndims(cell) == 3
        @test size(cell) == s
        @test length(cell) == prod(s)
        @test sum(mass(cell)) .≈ 8
        @test mass(cell) isa Diagonal
        D = derivatives(cell)
        @test Array(D[1] * points(cell)) ≈ fill(SVector(one(T),  zero(T), zero(T)), prod(s))
        @test Array(D[2] * points(cell)) ≈ fill(SVector(zero(T),  one(T), zero(T)), prod(s))
        @test Array(D[3] * points(cell)) ≈ fill(SVector(zero(T), zero(T),  one(T)), prod(s))

        cell = LobattoCell{T, A}(5)
        @test floattype(cell) == T
        @test arraytype(cell) <: A
        @test Base.ndims(cell) == 1
        @test size(cell) == (5,)
        @test length(cell) == 5
        @test sum(mass(cell)) .≈ 2
        @test mass(cell) isa Diagonal
        D = derivatives(cell)
        @test Array(D[1] * points(cell)) ≈ fill(SVector(one(T)), 5)
    end

    cell = LobattoCell{BigFloat}(3)
    @test floattype(cell) == BigFloat
    @test arraytype(cell) <: Array
end
