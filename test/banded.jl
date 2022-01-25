@testset "Banded Solvers" begin
    n = 100
    Nqh, Neh = 32, 100

    kuls = ((3, 3), (3, 10), (10, 3), (10, 10))
    kuls = ((10, 10),)

    TAs = ((Float64,  Array), (Float32,  Array))
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end

    for (T, AT) in TAs, (ku, kl) in kuls
        rng = MersenneTwister(777)

        width = ku + kl + 1

        h_A = zeros(T, Nqh, width, n, Neh)
        h_D = zeros(T, Nqh, width, n, Neh)
        for eh = 1:Neh, ij = 1:Nqh
            # Create some random factors
            U = diagm(0=>ones(T, n),
                      ntuple(k -> k-1=>rand(rng, T, n+1-k) / k, ku + 1)...)
            L = diagm(0=>ones(T, n),
                      ntuple(k -> -k=>rand(rng, T, n-k) / k, kl)...)
            C = L * U
            L, U
            D = L + U - I
            for k = -ku:kl
                h_A[ij, k + ku + 1, max(1, 1-k):min(n-k, n), eh] = diag(C, -k)
                h_D[ij, k + ku + 1, max(1, 1-k):min(n-k, n), eh] = diag(D, -k)
            end
        end
        d_A = AT(h_A)
        Bennu.bandedlu!(d_A, kl)
        @test Array(d_A) ≈ h_D
        if kl == ku
            d_A = AT(h_A)
            Bennu.bandedlu!(d_A)
            @test Array(d_A) ≈ h_D
        end
    end
end
