@testset "Banded Solvers" begin
    Nfields = 5
    Nev = 10
    Nqv = 4

    Nqh, Neh = 32, 10

    kuls = ((Nqv * Nfields, 2Nqv * Nfields),
            (2Nqv * Nfields, Nqv * Nfields),
            (2Nqv * Nfields, 2Nqv * Nfields))

    n = Nfields * Nev * Nqv

    TAs = ((Float64,  Array), (Float32,  Array))
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end

    for (T, AT) in TAs, (ku, kl) in kuls
        rng = MersenneTwister(777)

        width = ku + kl + 1

        # Banded matrices
        h_A = zeros(T, Nqh, width, n, Neh)
        # Banded matrix factors
        h_D = zeros(T, Nqh, width, n, Neh)
        # RHS and solution vector
        h_x = zeros(T, Nqh, n, Neh)
        h_b = zeros(T, Nqh, n, Neh)
        # Loop through and set columns
        for eh = 1:Neh, ij = 1:Nqh
            # Create some random factors
            U = diagm(0=>ones(T, n),
                      ntuple(k -> k-1=>rand(rng, T, n+1-k) / k, ku + 1)...)
            L = diagm(0=>ones(T, n),
                      ntuple(k -> -k=>rand(rng, T, n-k) / k, kl)...)
            C = L * U
            D = L + U - I

            # Store matrices in banded form
            for k = -ku:kl
                h_A[ij, k + ku + 1, max(1, 1-k):min(n-k, n), eh] = diag(C, -k)
                h_D[ij, k + ku + 1, max(1, 1-k):min(n-k, n), eh] = diag(D, -k)
            end

            # Create column solution and RHS vectors
            h_x[ij, :, eh] = rand(rng, T, n)
            h_b[ij, :, eh] = C * h_x[ij, :, eh]
        end

        # Check the factorization
        d_A = AT(h_A)
        d_LU = batchedbandedlu!(d_A, kl)
        @test Array(parent(d_LU)) ≈ h_D

        if kl == ku
            d_A = AT(h_A)
            d_LU = batchedbandedlu!(d_A)
            @test Array(parent(d_LU)) ≈ h_D
        end

        # copy b to a device field array
        d_b = AT(h_b)
        d_x = similar(d_b)

        ldiv!(d_x, d_LU, d_b)
        @test Array(d_x) ≈ h_x
    end
end
