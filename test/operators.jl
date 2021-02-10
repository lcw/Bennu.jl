@testset "spectral operators" begin
    function P5(r::AbstractVector{T}) where {T}
        return T(1) / T(8) * (T(15) * r - T(70) * r .^ 3 + T(63) * r .^ 5)
    end
    function P6(r::AbstractVector{T}) where {T}
        return T(1) / T(16) * (-T(5) .+ T(105) * r .^ 2 - T(315) * r .^ 4 + T(231) * r .^ 6)
    end
    function DP6(r::AbstractVector{T}) where {T}
        return T(1) / T(16) * (T(2 * 105) * r - T(4 * 315) * r .^ 3 + T(6 * 231) * r .^ 5)
    end

    IPN(::Type{T}, N) where {T} = T(2) / T(2 * N + 1)

    for T in (Float32, Float64, BigFloat)
        r, w = legendregausslobatto(T, 7)
        x = LinRange{T}(-1, 1, 101)
        D = spectralderivative(r)
        P = spectralinterpolation(r, x)

        @test sum(P5(r) .^ 2 .* w) ≈ IPN(T, 5)
        @test D * P6(r) ≈ DP6(r)
        @test P * P6(r) ≈ P6(x)
    end

    for T in (Float32, Float64, BigFloat)
        r, w = legendregauss(T, 7)
        D = spectralderivative(r)

        @test sum(P5(r) .^ 2 .* w) ≈ IPN(T, 5)
        @test sum(P6(r) .^ 2 .* w) ≈ IPN(T, 6)
        @test D * P6(r) ≈ DP6(r)
    end
end
