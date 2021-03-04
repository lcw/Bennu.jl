function Base.copy(M::Mul{LazyArrays.ApplyLayout{typeof(kron)},
                          L, Kron{T, 2, Tuple{D}}, F}) where
        {T, L, D <: AbstractMatrix{T}, F <: AbstractVecOrMat}
    K, f = M.A, M.B
    (d,) = LazyArrays.arguments(K)

    g = reshape(f, size(d, 2), :)
    r = similar(f, size(d, 1), size(g, 2))

    @tullio r[i, e] = d[i, l] * g[l, e]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function Base.copy(M::Mul{LazyArrays.ApplyLayout{typeof(kron)},
                          L, Kron{T, 2, Tuple{E, D}}, F}) where
        {T, L, D <: AbstractMatrix{T}, E <: Eye{T}, F <: AbstractVecOrMat}
    K, f = M.A, M.B
    e, d = LazyArrays.arguments(K)

    g = reshape(f, size(d, 2), size(e, 1), :)
    r = similar(f, size(d, 1), size(e, 1), size(g, 3))

    @tullio r[i, j, k] = d[i, l] * g[l, j, k]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function Base.copy(M::Mul{LazyArrays.ApplyLayout{typeof(kron)},
                          L, Kron{T, 2, Tuple{D, E}}, F}) where
        {T, L, D <: AbstractMatrix{T}, E <: Eye{T}, F <: AbstractVecOrMat}
    K, f = M.A, M.B
    d, e = LazyArrays.arguments(K)

    g = reshape(f, size(e, 1), size(d, 2),  :)
    r = similar(f, size(e, 1), size(d, 1), size(g, 3))

    @tullio r[i, j, k] = d[j, l] * g[i, l, k]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function Base.copy(M::Mul{LazyArrays.ApplyLayout{typeof(kron)},
                          L, Kron{T, 2, Tuple{B, A}}, F}) where
        {T, L, A <: AbstractMatrix{T}, B <: AbstractMatrix{T},
         F <: AbstractVecOrMat}
    K, f = M.A, M.B
    b, a = LazyArrays.arguments(K)

    g = reshape(f, size(a, 2), size(b, 2),  :)
    r = similar(f, size(a, 1), size(b, 1), size(g, 3))

    @tullio r[i, j, k] = b[j, m] * a[i, l] * g[l, m, k]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function Base.copy(M::Mul{LazyArrays.ApplyLayout{typeof(kron)},
                   L, Kron{T, 2, Tuple{E₃, E₂, D}}, F}) where
        {T, L, D <: AbstractMatrix{T}, E₃ <: Eye{T},  E₂ <: Eye{T},
         F <: AbstractVecOrMat}
    K, f = M.A, M.B
    e₃, e₂, d = LazyArrays.arguments(K)

    g = reshape(f, size(d, 2), size(e₂, 1), size(e₃, 1),  :)
    r = similar(f, size(d, 1), size(e₂, 1), size(e₃, 1), size(g, 4))

    @tullio r[i, j, k, e] = d[i, l] * g[l, j, k, e]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function Base.copy(M::Mul{LazyArrays.ApplyLayout{typeof(kron)},
                          L, Kron{T, 2, Tuple{E₃, D, E₁}}, F}) where
        {T, L, D <: AbstractMatrix{T}, E₁ <: Eye{T}, E₃ <: Eye{T},
         F <: AbstractVecOrMat}
    K, f = M.A, M.B
    e₃, d, e₁ = LazyArrays.arguments(K)

    g = reshape(f, size(e₁, 1), size(d, 2), size(e₃, 1),  :)
    r = similar(f, size(e₁, 1), size(d, 1), size(e₃, 1), size(g, 4))

    @tullio r[i, j, k, e] = d[j, l] * g[i, l, k, e]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function Base.copy(M::Mul{LazyArrays.ApplyLayout{typeof(kron)},
                          L, Kron{T, 2, Tuple{D, E₂, E₁}}, F}) where
        {T, L, D <: AbstractMatrix{T}, E₁ <: Eye{T},  E₂ <: Eye{T},
         F <: AbstractVecOrMat}
    K, f = M.A, M.B
    d, e₂, e₁ = LazyArrays.arguments(K)

    g = reshape(f, size(e₁, 1), size(e₂, 1), size(d, 2),  :)
    r = similar(f, size(e₁, 1), size(e₂, 1), size(d, 1), size(g, 4))

    @tullio r[i, j, k, e] = d[k, l] * g[i, j, l, e]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function Base.copy(M::Mul{LazyArrays.ApplyLayout{typeof(kron)},
                          L, Kron{T, 2, Tuple{C, B, A}}, F}) where
        {T, L, A <: AbstractMatrix{T}, B <: AbstractMatrix{T},
         C <: AbstractMatrix{T}, F <: AbstractVecOrMat}
    K, f = M.A, M.B
    c, b, a = LazyArrays.arguments(K)

    g = reshape(f, size(a, 2), size(b, 2), size(c, 2),  :)
    r = similar(f, size(a, 1), size(b, 1), size(c, 1), size(g, 4))

    @tullio r[i, j, k, e] = c[k, n] * b[j, m] * a[i, l] * g[l, m, n, e]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end
