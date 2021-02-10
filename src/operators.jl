"""
    barycentricweights(r)

Create an array analogous to `r` of the barycentric weights associated with the
array of points `r`. See also `spectralderivative` and
`spectralinterpolation`.

# Examples
Create barycentric weights for equally spaced points
```jldoctest
julia> Bennu.barycentricweights(collect(-1.0:1.0))
3-element Vector{Float64}:
  0.5
 -1.0
  0.5
```
or create barycentric weights for Legendre--Gauss--Lobatto points
```jldoctest
julia> x, _ = legendregausslobatto(4);
julia> Bennu.barycentricweights(x)
4-element Vector{Float64}:
 -0.6250000000000002
  1.3975424859373686
 -1.3975424859373689
  0.6250000000000006
```

# References
  Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation",
  SIAM Review 46 (2004), pp. 501-517.
  <https://doi.org/10.1137/S0036144502417715>
"""
function barycentricweights(r)
    T = eltype(r)
    N = length(r)
    w = similar(r)
    w .= one(T)

    for j in 1:N
        for i in 1:N
            if i != j
                w[j] = w[j] * (r[j] - r[i])
            end
        end
        w[j] = one(T) / w[j]
    end

    return w
end

"""
    spectralderivative(r, w=barycentricweights(r))

Create the differentiation matrix, type analogous to `r`, for polynomials defined
on the points `r` with associated barycentric weights `w`. See also
`barycentricweights` and `spectralinterpolation`.

# Examples
Create the differentiation matrix for Legendre--Gauss--Lobatto points
```jldoctest
julia> x, _ = legendregausslobatto(4);
julia> spectralderivative(x)
4×4 Matrix{Float64}:
 -3.0        4.04508      -1.54508       0.5
 -0.809017   7.77156e-16   1.11803      -0.309017
  0.309017  -1.11803      -1.55431e-15   0.809017
 -0.5        1.54508      -4.04508       3.0
```

# References
  Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation",
  SIAM Review 46 (2004), pp. 501-517.
  <https://doi.org/10.1137/S0036144502417715>
"""
function spectralderivative(r, w=barycentricweights(r))
    T = promote_type(eltype(r), eltype(w))
    M = length(r)
    @assert M == length(w)
    D = zeros(T, M, M)

    for k in 1:M
        for j in 1:M
            if k == j
                for l in 1:M
                    if l != k
                        D[j, k] = D[j, k] + one(T) / (r[k] - r[l])
                    end
                end
            else
                D[j, k] = (w[k] / w[j]) / (r[j] - r[k])
            end
        end
    end

    return D
end

"""
    spectralinterpolation(r, s, w=barycentricweights(r))

Create the interpolation matrix, type analogous to `r`, for polynomials defined
on the points `r` with associated barycentric weights `w` evaulated at the
points `s`. See also `barycentricweights` and `spectralderivative`.

# Examples
Create the interpolation matrix which evaluates polynomials on
Legendre--Gauss--Lobatto points at an equally spaced grid.
```jldoctest
julia> x, _ = legendregausslobatto(4);
julia> y = collect(-1:0.4:1);
julia> spectralinterpolation(x,y)
6×4 Matrix{Float64}:
  1.0           0.0           0.0           0.0
  0.16          0.936656     -0.136656      0.04
 -0.12          0.868328      0.331672     -0.08
 -0.08          0.331672      0.868328     -0.12
  0.04         -0.136656      0.936656      0.16
 -1.11022e-16   3.43078e-16  -8.98189e-16   1.0
```

# References
  Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation",
  SIAM Review 46 (2004), pp. 501-517.
  <https://doi.org/10.1137/S0036144502417715>
"""
function spectralinterpolation(r, s, w=barycentricweights(r))
    T = promote_type(eltype(r), eltype(s), eltype(w))
    M = length(s)
    N = length(r)
    @assert N == length(w)
    P = zeros(T, M, N)

    for k in 1:M
        for j in 1:N
            P[k, j] = w[j] / (s[k] - r[j])
            if !isfinite(P[k, j])
                P[k, :] .= T(0)
                P[k, j] = T(1)
                break
            end
        end
        d = sum(P[k, :])
        P[k, :] = P[k, :] / d
    end

    return P
end
