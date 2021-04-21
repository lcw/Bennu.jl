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
        @test number_of_faces(cell) == (1, 4, 4)
        @test number_of_faces(cell) == size.(Bennu.materializefaces(cell), 2)
        @test connectivity(cell) ==  adapt(A, (([1 4 7; 2 5 8; 3 6 9],),
                                               ([1, 4, 7], [3, 6, 9],
                                                [1, 2, 3], [7, 8, 9]),
                                               (1, 3, 7, 9)))
        @test Bennu.connectivityoffsets(cell, Val(1)) == (0, 9)
        @test Bennu.connectivityoffsets(cell, Val(2)) == (0, 3, 6, 9, 12)
        @test Bennu.connectivityoffsets(cell, Val(3)) == (0, 1, 2, 3, 4)
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
        @test number_of_faces(cell) == (1, 6, 12, 8)
        @test number_of_faces(cell) == size.(Bennu.materializefaces(cell), 2)
        @test connectivity(cell)[1] == (adapt(A, reshape(collect(1:24),3,4,2)),)
        @test connectivity(cell)[2:end] ==
            adapt(A, (([1 13; 4 16; 7 19; 10 22], [3 15; 6 18; 9 21; 12 24],
                       [1 13; 2 14; 3 15], [10 22; 11 23; 12 24],
                       [1 4 7 10; 2 5 8 11; 3 6 9 12],
                       [13 16 19 22; 14 17 20 23; 15 18 21 24]),
                      ([1, 13], [3, 15], [10, 22], [12, 24], [1, 4, 7, 10],
                       [3, 6, 9, 12], [13, 16, 19, 22], [15, 18, 21, 24],
                       [1, 2, 3], [10, 11, 12], [13, 14, 15], [22, 23, 24]),
                      (1, 3, 10, 12, 13, 15, 22, 24)))
        @test Bennu.connectivityoffsets(cell, Val(1)) == (0, 24)
        @test Bennu.connectivityoffsets(cell, Val(2)) ==
            (0, 8, 16, 22, 28, 40, 52)
        @test Bennu.connectivityoffsets(cell, Val(3)) ==
            (0, 2, 4, 6, 8, 12, 16, 20, 24, 27, 30, 33, 36)
        @test Bennu.connectivityoffsets(cell, Val(4)) ==
            (0, 1, 2, 3, 4, 5, 6, 7, 8)

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
        @test number_of_faces(cell) == (1, 2)
        @test number_of_faces(cell) == size.(Bennu.materializefaces(cell), 2)
        @test connectivity(cell) == adapt(A, (([1, 2, 3, 4, 5],), (1, 5)))
        @test Bennu.connectivityoffsets(cell, Val(1)) == (0, 5)
        @test Bennu.connectivityoffsets(cell, Val(2)) == (0, 1, 2)
    end

    cell = LobattoCell{BigFloat}(3)
    @test floattype(cell) == BigFloat
    @test arraytype(cell) <: Array
end
