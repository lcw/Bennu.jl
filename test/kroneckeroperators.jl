@testset "Kronecker operators" begin
    TAs = ((Float64,  Array),
           (BigFloat, Array))
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end

    for (T, A) in TAs
        rng = MersenneTwister(37)
        a = adapt(A, rand(rng, T, 3, 2))
        b = adapt(A, rand(rng, T, 4, 5))
        c = adapt(A, rand(rng, T, 1, 7))

        for args in ((a, Eye{T}(5)),
                     (Eye{T}(2), b),
                     (a, b),
                     (Eye{T}(3), Eye{T}(2), c),
                     (Eye{T}(2), b, Eye{T}(7)),
                     (a, Eye{T}(4), Eye{T}(7)),
                     (a, b, c))

            K = adapt(A, collect(Bennu.Kron(adapt(Array, args))))
            d = adapt(A, rand(SVector{2, T}, size(K, 2), 6))
            e = adapt(A, rand(SVector{2, T}, size(K, 2)))
            @test Array(Bennu.Kron(args) * e) ≈ Array(K * e)
            @test Array(Bennu.Kron(args) * d) ≈ Array(K * d)

            if isbits(T)
                f = rand(rng, T, size(K, 2), 3, 2)
                f = adapt(A, reinterpret(reshape, SVector{2, T},
                                         PermutedDimsArray(f, (3, 1, 2))))
                @test Array(Bennu.Kron(args) * f) ≈ Array(K * f)
            end

            @test adapt(Array, Bennu.Kron(args)) == Bennu.Kron(adapt.(Array, args))
        end
    end
end
