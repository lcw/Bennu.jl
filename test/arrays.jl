@testset "arrays" begin
    @test arraytype(ones(3, 5)) === Array
    if CUDA.has_cuda_gpu()
        @test arraytype(CuArray(ones(3, 5))) === CuArray
    end

    @test [1, 3, 1, 2, 1, 2] == Bennu.numbercontiguous([10, 29, 10, 23, 10, 23])
    TAs = ((Float64,  Array), (BigFloat, Array))
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end
    for (T, A) in TAs
        a = fieldarray(undef, T, A, ())
        @test size(a) == ()
        @test typeof(a) == typeof(fieldarray(a))
        @test Bennu.isfieldarray(a)

        b = fieldarray(a, (1,2,1,3))
        @test size(b) == (1,2,1,3)
        @test Bennu.isfieldarray(b)

        a = fieldarray(undef, T, A, (3,))
        @test size(a) == (3,)
        @test eltype(a) == T
        @test Tullio.storage_type(a) <: A
        @test typeof(a) == typeof(fieldarray(a))
        @test Bennu.isfieldarray(a)

        b = fieldarray(a, (1,2,1,3))
        @test size(b) == (1,2,1,3)
        @test eltype(b) == T
        @test Tullio.storage_type(b) <: A
        @test Bennu.isfieldarray(b)

        a = fieldarray(undef, T, A, (3,4))
        @test size(a) == (3,4)
        @test eltype(a) == T
        @test Tullio.storage_type(a) <: A
        @test typeof(a) == typeof(fieldarray(a))
        @test Bennu.isfieldarray(a)

        b = fieldarray(a, (1,2,1,3))
        @test size(b) == (1,2,1,3)
        @test eltype(b) == T
        @test Tullio.storage_type(b) <: A
        @test Bennu.isfieldarray(b)

        a = fieldarray(undef, (a=SVector{2, T}, b=T), A, (3,4))
        @test size(a) == (3,4)
        @test eltype(a) == NamedTuple{(:a, :b), Tuple{SVector{2, T}, T}}
        @test Tullio.storage_type(a) <: A
        @test typeof(a) == typeof(fieldarray(a))
        @test Bennu.isfieldarray(a)
        @test !Bennu.isfieldarray(similar(a))

        b = fieldarray(a, (1,2,1,3))
        @test size(b) == (1,2,1,3)
        @test eltype(b) == NamedTuple{(:a, :b), Tuple{SVector{2, T}, T}}
        @test Tullio.storage_type(b) <: A
        @test Bennu.isfieldarray(b)
    end
end
