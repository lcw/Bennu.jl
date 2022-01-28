@testset "Banded Solvers" begin
    Nfields = 5
    Nev = 10

    Nq = (5, 6, 4)

    Nqv = Nq[end]

    Nqh, Neh = prod(Nq[1:end-1]), 10

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

        # make sure this all works with fieldarray types too
        fld_b = fieldarray(undef, SVector{Nfields, T}, AT, (Nq..., Nev * Neh))
        parent(components(fld_b)[1])[:] .= d_b[:]
        fld_x = fieldarray(undef, SVector{Nfields, T}, AT, (Nq..., Nev * Neh))
        ldiv!(fld_x, d_LU, fld_b)

        @test Array(parent(components(fld_x)[1])) ≈
           reshape(h_x, Nq..., Nfields, Nev * Neh)

        # Check the that the banded matvec works
        # NOTE: This NOT high-performance code!
        A = Bennu.batchedbandedmatrix!(AT(h_A), kl)
        d_x = AT(h_x)
        d_y = similar(d_x)
        mul!(d_y, A, d_x)

        @test Array(d_y) ≈ h_b

        # Check that we can form a banded matrix
        rhs!(y, x, event) = mul!(y, A, x, event)

        B = Bennu.batchedbandedmatrix(rhs!, d_y, d_x, kl, ku, Nqh+max(ku, kl))
        @test all(parent(A) ≈ parent(B))

        Bennu.batchedbandedmatrix!(rhs!, B, d_y, d_x)
        @test all(parent(A) ≈ parent(B))
    end

    for (T, AT) in TAs
        cellsandcoordinates =
        (
         (LobattoCell{T, AT}(5), (zero(T):1//4:one(T),)),
         (LobattoCell{T, AT}(3, 5),
          (-one(T):1//2:one(T), zero(T):1//4:one(T))),
         (LobattoCell{T, AT}(3, 5, 2),
          (-one(T):1//2:one(T), zero(T):1//4:one(T), zero(T):1//8:one(T),))
        )

        for (cell, coord) in cellsandcoordinates
            rng = MersenneTwister(777)

            Nq = size(cell)

            # Setup the grid
            grid = brickgrid(cell, coord,
                             ordering = StackedOrdering{CartesianOrdering}())

            # Get a derivative matrix
            # Total hack to make the GPU happy!
            D = AT(Array(derivatives(LobattoCell{T, Array}(Nq...))[end]))

            # Set up the field storage
            q = fieldarray(undef, SVector{Nfields, T}, grid)
            dq = fieldarray(undef, SVector{Nfields, T}, grid)
            A = fieldarray(undef, SMatrix{Nfields, Nfields, T}, grid)

            # Some random coefficients to keep things interesting!
            Np = prod(Nq)
            Ne = length(grid)
            Nev = Bennu.stacksize(grid)
            Neh = Bennu.horizontalsize(grid)
            Nqv = Nq[end]
            Nqh = div(prod(Nq), Nqv)
            parent(components(A)[1]) .= adapt(AT, rand(rng, T, Np, Nfields^2, Ne))

            # extract the data for just the top and bottom faces
            Nfp = div.(Np, Nq)
            frange = ndims(cell) > 1 ? (2sum(Nfp[1:end-1])+1:2sum(Nfp)) : (1:2sum(Nfp))
            faceixm, faceixp = faceindices(grid)
            fdata = (faceixm[frange, :], faceixp[frange, :]);

            # Define a RHS with a DG-like connectivity
            @kernel function face_kernel!(dq, q, findm, findp, A)
                I = @index(Global, Linear)
                if I ≤ length(findm)
                    fm = findm[I]
                    fp = findp[I]
                    dq[fm] += (A[fm] * q[fm] + A[fp] * q[fp])
                end
            end

            function matvec(dq, q, event)
                wait(event)

                # volume term
                @tullio dq[i, e] += D[i, j] * (A[j, e] * q[j, e])
                event = Event(Bennu.device(q))
                knl = face_kernel!(Bennu.device(A), 256)
                event = knl(dq, q, fdata..., A; ndrange = length(fdata[1]),
                            dependencies = (event,))

                return event
            end

            # Create the banded matrix from the matvec function
            eb = 1
            mat = Bennu.batchedbandedmatrix(matvec, grid, dq, q, eb)

            # Fill q with some random data and then compute the matvec using the
            # matvec function and then the banded matrix
            parent(components(q)[1]) .= AT(rand(rng, prod(Nq), Nfields, length(grid)))

            fill!(parent(components(dq)[1]), 0)
            matvec(dq, q, Event(Bennu.device(q)))

            # To use the banded-matrix we need to reshape the data arrays
            n = Nqv * Nev * Nfields
            q_array = reshape(parent(components(q)[1]), Nqh, n, Neh)
            dq_array = similar(q_array)
            mul!(dq_array, mat, q_array)

            @test Array(dq_array) ≈ Array(reshape(parent(components(dq)[1]), Nqh, n, Neh))
        end
    end
end
