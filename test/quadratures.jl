@testset "quadratures" begin
    for T in (Float32, Float64, BigFloat)
        x, w = legendregauss(T, 1)
        @test iszero(x)
        @test w ≈ [2one(T)]

        x, w = legendregauss(T, 1, Bennu.LeftEndPoint())
        @test x ≈ [-one(T)]
        @test w ≈ [2one(T)]

        x, w = legendregauss(T, 1, Bennu.RightEndPoint())
        @test x ≈ [one(T)]
        @test w ≈ [2one(T)]

        x, w = legendregauss(T, 2, Bennu.LeftEndPoint())
        @test x ≈ [-one(T); T(1 // 3)]
        @test w ≈ [T(1 // 2); T(3 // 2)]

        x, w = legendregauss(T, 2, Bennu.RightEndPoint())
        @test x ≈ [T(-1 // 3); one(T)]
        @test w ≈ [T(3 // 2); T(1 // 2)]

        err = ErrorException("Must have at least two points for both ends.")
        @test_throws err legendregauss(T, 1, Bennu.BothEndPoint())

        a, b = Bennu.legendrecoefficients(T, 100)
        err = ErrorException("No convergence after 1 iterations " *
                             "(try increasing maxiterations)")
        @test_throws err Bennu.gaussrule(-one(T), one(T), a, b, Bennu.BothEndPoint(), 1)
    end

    x1, w1 = legendregauss(23)
    x2, w2 = FastGaussQuadrature.gausslegendre(23)

    @test x1 ≈ x2
    @test w1 ≈ w2

    x1, w1 = legendregausslobatto(33)
    x2, w2 = FastGaussQuadrature.gausslobatto(33)

    @test x1 ≈ x2
    @test w1 ≈ w2

    x1, w1 = legendregauss(13, Bennu.LeftEndPoint())
    x2, w2 = FastGaussQuadrature.gaussradau(13)

    @test x1 ≈ x2
    @test w1 ≈ w2
end
