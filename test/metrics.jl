# Someb of this is a translation of Canary.jl's metric tests written by Jeremy
# Kozdon.

@testset "metrics" begin
    TAs = ((Float64,  Array),
           (Float32,  Array),
           # (BigFloat, Array)
          )
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end

    for (T, A) in TAs

        @testset "1D" begin
            cell = LobattoCell{T, A}(4);
            grid = brickgrid(cell, ([-1, 0, 10],))
            g, J = adapt(Array, components(metrics(grid)))
            g, = components(g)

            @test all(g[:, 1] .≈ 2)
            @test all(g[:, 2] .≈ 1//5)

            @test all(J[:, 1] .≈ 1//2)
            @test all(J[:, 2] .≈ 5)

            n, sJ = adapt(Array, components(facemetrics(grid)))
            n, = components(n)

            @test all(n[1, :] .≈ -1)
            @test all(n[2, :] .≈ 1)
            @test all(sJ[:] .≈ 1)
        end

        @testset "2D" begin
            f(x) = SA[9 * x[1] - (1 + x[1]) * x[2]^2 +
                      (x[1] - 1)^2 * (1 - x[2]^2 + x[2]^3),
                      10 * x[2] + x[1] * x[1]^3 * (1 - x[2]) +
                      x[1]^2 * x[2] * (1 + x[2])]
            f₁₁(x) = 7 + x[2]^2 - 2 * x[2]^3 + 2 * x[1] * (1 - x[2]^2 + x[2]^3)
            f₁₂(x) = -2 * (1 + x[1]) * x[2] +
                     (-1 + x[1])^2 * x[2] * (-2 + 3 * x[2])
            f₂₁(x) = -4 * x[1]^3 * (-1 + x[2]) + 2 * x[1] * x[2] * (1 + x[2])
            f₂₂(x) = 10 - x[1]*x[1]^3 + x[1]^2 * (1 + 2 * x[2])

            L = 10
            M = 12
            cell = LobattoCell{T, A}(L, M)
            Δx₁ = 1//5
            Δx₂ = 1//4

            unwarpedgrid = brickgrid(cell, (-1:Δx₁:1, -1:Δx₂:1))
            x̂ = points(unwarpedgrid)
            ĥ = fieldarray(SMatrix{2, 2, T, 4},
                           (f₁₁.(x̂).*(Δx₁/2), f₂₁.(x̂).*(Δx₁/2),
                            f₁₂.(x̂).*(Δx₂/2), f₂₂.(x̂).*(Δx₂/2)))
            Ĵ = adapt(Array, det.(ĥ))
            ĝ = adapt(Array, inv.(ĥ))

            grid = brickgrid(f, cell, (-1:Δx₁:1, -1:Δx₂:1))
            g, J = map(x->adapt(Array, x), components(metrics(grid)))

            @test J ≈ Ĵ
            @test all(g .≈ ĝ)

            n, sJ = map(x->adapt(Array, x), components(facemetrics(grid)))

            @test all(norm.(n) .≈ 1)

            a = n .* sJ
            a₁, a₂ = components(a)
            ĥ₁₁, ĥ₂₁, ĥ₁₂, ĥ₂₂ = map(x->adapt(Array, x),
                                     components(reshape(ĥ, size(cell)..., :)))

            @test a₁[1:M, :] ≈ -ĥ₂₂[1, :, :]
            @test a₂[1:M, :] ≈  ĥ₁₂[1, :, :]

            @test a₁[M+1:2M, :] ≈  ĥ₂₂[end, :, :]
            @test a₂[M+1:2M, :] ≈ -ĥ₁₂[end, :, :]

            @test a₁[2M+1:2M+L, :] ≈  ĥ₂₁[:, 1, :]
            @test a₂[2M+1:2M+L, :] ≈ -ĥ₁₁[:, 1, :]

            @test a₁[2M+L+1:2M+2L, :] ≈ -ĥ₂₁[:, end, :]
            @test a₂[2M+L+1:2M+2L, :] ≈  ĥ₁₁[:, end, :]
        end

        @testset "2D Constant Preserving" begin
            f(x) = SA[9 * x[1] - (1 + x[1]) * x[2]^2 +
                      (x[1] - 1)^2 * (1 - x[2]^2 + x[2]^3),
                      10 * x[2] + x[1] * x[1]^3 * (1 - x[2]) +
                      x[1]^2 * x[2] * (1 + x[2])]

            L = 3
            M = 4
            cell = LobattoCell{T, A}(L, M)
            Δx₁ = 1//1
            Δx₂ = 1//2

            grid = brickgrid(f, cell, (-1:Δx₁:1, -1:Δx₂:1))
            g, J = map(x->adapt(Array, x), components(metrics(grid)))
            D₁, D₂ = adapt(Array, derivatives(cell))

            @test norm((D₁*(J .* g.:1) + D₂*(J .* g.:2)), Inf) < 100*eps(T)
            @test norm((D₁*(J .* g.:3) + D₂*(J .* g.:4)), Inf) < 100*eps(T)
        end

        @testset "3D" begin
            f(x) = SA[3x[1] + x[2]/5 + x[3]/10 + x[1]*x[2]^2*x[3]^3/3,
                      4x[2] + x[1]^3*x[2]^2*x[3]/4,
                      2x[3] + x[1]^2*x[2]*x[3]^3/2]
            f₁₁(x) = 3*oneunit(eltype(x)) + x[2]^2*x[3]^3/3
            f₁₂(x) = oneunit(eltype(x))/5 + 2*x[1]*x[2]*x[3]^3/3
            f₁₃(x) = oneunit(eltype(x))/10 + 3*x[1]*x[2]^2*x[3]^2/3
            f₂₁(x) = 3*x[1]^2*x[2]^2*x[3]/4
            f₂₂(x) = 4*oneunit(eltype(x)) + 2*x[1]^3*x[2]*x[3]/4
            f₂₃(x) = x[1]^3*x[2]^2/4
            f₃₁(x) = 2*x[1]*x[2]*x[3]^3/2
            f₃₂(x) = x[1]^2*x[3]^3/2
            f₃₃(x) = 2*oneunit(eltype(x)) + 3*x[1]^2*x[2]*x[3]^2/2

            L = 6
            M = 8
            N = 7
            cell = LobattoCell{T, A}(L, M, N)
            Δx₁ = 1//1
            Δx₂ = 2//1
            Δx₃ = 2//3

            unwarpedgrid = brickgrid(cell, (-1:Δx₁:1, -1:Δx₂:1, -1:Δx₃:1))
            x̂ = points(unwarpedgrid)
            ĥ = fieldarray(SMatrix{3, 3, T, 9},
                           (
                            f₁₁.(x̂).*(Δx₁/2),
                            f₂₁.(x̂).*(Δx₁/2),
                            f₃₁.(x̂).*(Δx₁/2),
                            f₁₂.(x̂).*(Δx₂/2),
                            f₂₂.(x̂).*(Δx₂/2),
                            f₃₂.(x̂).*(Δx₂/2),
                            f₁₃.(x̂).*(Δx₃/2),
                            f₂₃.(x̂).*(Δx₃/2),
                            f₃₃.(x̂).*(Δx₃/2),
                           ))
            Ĵ = adapt(Array, det.(ĥ))
            ĝ = adapt(Array, inv.(ĥ))

            grid = brickgrid(f, cell, (-1:Δx₁:1, -1:Δx₂:1, -1:Δx₃:1))
            g, J = map(x->adapt(Array, x), components(metrics(grid)))

            @test J ≈ Ĵ
            @test all(g .≈ ĝ)

            n, sJ = map(x->adapt(Array, x), components(facemetrics(grid)))

            @test all(norm.(n) .≈ 1)

            a = n .* sJ
            a₁, a₂, a₃ = components(a)
            ĝ₁₁, ĝ₂₁, ĝ₃₁, ĝ₁₂, ĝ₂₂, ĝ₃₂, ĝ₁₃, ĝ₂₃, ĝ₃₃ =
                map(x->adapt(Array, x),
                    components(reshape(ĝ, size(cell)..., :)))
            Ĵ = reshape(Ĵ, size(cell)..., :)

            @test reshape(a₁[      1:M*N, :], M, N, :) ≈ -Ĵ[  1, :, :, :] .* ĝ₁₁[  1, :, :, :]
            @test reshape(a₁[M*N+1:2*M*N, :], M, N, :) ≈  Ĵ[end, :, :, :] .* ĝ₁₁[end, :, :, :]
            @test reshape(a₁[ 2*M*N        .+ (1:L*N), :], L, N, :) ≈ -Ĵ[:,   1, :, :] .* ĝ₂₁[:,   1, :, :]
            @test reshape(a₁[(2*M*N + L*N) .+ (1:L*N), :], L, N, :) ≈  Ĵ[:, end, :, :] .* ĝ₂₁[:, end, :, :]
            @test reshape(a₁[(2*M*N + 2*L*N      ) .+ (1:L*M), :], L, M, :) ≈ -Ĵ[:, :,   1, :] .* ĝ₃₁[:, :,   1, :]
            @test reshape(a₁[(2*M*N + 2*L*N + L*M) .+ (1:L*M), :], L, M, :) ≈  Ĵ[:, :, end, :] .* ĝ₃₁[:, :, end, :]

            @test reshape(a₂[      1:M*N, :], M, N, :) ≈ -Ĵ[  1, :, :, :] .* ĝ₁₂[  1, :, :, :]
            @test reshape(a₂[M*N+1:2*M*N, :], M, N, :) ≈  Ĵ[end, :, :, :] .* ĝ₁₂[end, :, :, :]
            @test reshape(a₂[ 2*M*N        .+ (1:L*N), :], L, N, :) ≈ -Ĵ[:,   1, :, :] .* ĝ₂₂[:,   1, :, :]
            @test reshape(a₂[(2*M*N + L*N) .+ (1:L*N), :], L, N, :) ≈  Ĵ[:, end, :, :] .* ĝ₂₂[:, end, :, :]
            @test reshape(a₂[(2*M*N + 2*L*N      ) .+ (1:L*M), :], L, M, :) ≈ -Ĵ[:, :,   1, :] .* ĝ₃₂[:, :,   1, :]
            @test reshape(a₂[(2*M*N + 2*L*N + L*M) .+ (1:L*M), :], L, M, :) ≈  Ĵ[:, :, end, :] .* ĝ₃₂[:, :, end, :]

            @test reshape(a₃[      1:M*N, :], M, N, :) ≈ -Ĵ[  1, :, :, :] .* ĝ₁₃[  1, :, :, :]
            @test reshape(a₃[M*N+1:2*M*N, :], M, N, :) ≈  Ĵ[end, :, :, :] .* ĝ₁₃[end, :, :, :]
            @test reshape(a₃[ 2*M*N        .+ (1:L*N), :], L, N, :) ≈ -Ĵ[:,   1, :, :] .* ĝ₂₃[:,   1, :, :]
            @test reshape(a₃[(2*M*N + L*N) .+ (1:L*N), :], L, N, :) ≈  Ĵ[:, end, :, :] .* ĝ₂₃[:, end, :, :]
            @test reshape(a₃[(2*M*N + 2*L*N      ) .+ (1:L*M), :], L, M, :) ≈ -Ĵ[:, :,   1, :] .* ĝ₃₃[:, :,   1, :]
            @test reshape(a₃[(2*M*N + 2*L*N + L*M) .+ (1:L*M), :], L, M, :) ≈  Ĵ[:, :, end, :] .* ĝ₃₃[:, :, end, :]
        end

        @testset "3D Constant Preserving" begin
            f(x) = SA[3x[1] + x[2]/5 + x[3]/10 + x[1]*x[2]^2*x[3]^3/3,
                      4x[2] + x[1]^3*x[2]^2*x[3]/4,
                      2x[3] + x[1]^2*x[2]*x[3]^3/2]

            L = 3
            M = 4
            N = 2
            cell = LobattoCell{T, A}(L, M, N)
            Δx₁ = 1//1
            Δx₂ = 1//2
            Δx₃ = 2//3

            grid = brickgrid(f, cell, (-1:Δx₁:1, -1:Δx₂:1, -1:Δx₃:1))
            g, J = map(x->adapt(Array, x), components(metrics(grid)))
            D₁, D₂, D₃ = adapt(Array, derivatives(cell))

            @test norm((D₁*(J .* g.:1) + D₂*(J .* g.:2) + D₃*(J .* g.:3)),
                       Inf) < 100*eps(T)
            @test norm((D₁*(J .* g.:4) + D₂*(J .* g.:5) + D₃*(J .* g.:6)),
                       Inf) < 100*eps(T)
            @test norm((D₁*(J .* g.:7) + D₂*(J .* g.:8) + D₃*(J .* g.:9)),
                       Inf) < 100*eps(T)
        end
    end
end
