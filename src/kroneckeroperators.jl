struct Kron{T}
    args::T
    Kron(args::Tuple) = new{typeof(args)}(args)
end

Adapt.adapt_structure(to, K::Kron) = Kron(map(x->Adapt.adapt(to, x), K.args))
components(K::Kron) = K.args
Base.collect(K::Kron) = collect(kron(K.args...))
Base.size(K::Kron, j::Int) = prod(size.(K.args, j))

import Base.==
==(J::Kron, K::Kron) = all(J.args .== K.args)

import Base.*

function (*)(K::Kron{Tuple{D}}, f::F) where {D <: AbstractMatrix,
                                           F <: AbstractVecOrMat}
    (d,) = components(K)

    g = reshape(f, size(d, 2), :)
    r = similar(f, size(d, 1), size(g, 2))

    @tullio r[i, e] = d[i, l] * g[l, e]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{E, D}}, f::F) where
        {D <: AbstractMatrix, E <: Eye, F <: AbstractVecOrMat}
    e, d = components(K)

    g = reshape(f, size(d, 2), size(e, 1), :)
    r = similar(f, size(d, 1), size(e, 1), size(g, 3))

    @tullio r[i, j, k] = d[i, l] * g[l, j, k]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{D, E}}, f::F) where
        {D <: AbstractMatrix, E <: Eye, F <: AbstractVecOrMat}
    d, e = components(K)

    g = reshape(f, size(e, 1), size(d, 2),  :)
    r = similar(f, size(e, 1), size(d, 1), size(g, 3))

    @tullio r[i, j, k] = d[j, l] * g[i, l, k]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{B, A}}, f::F) where
        {A <: AbstractMatrix, B <: AbstractMatrix, F <: AbstractVecOrMat}
    b, a = components(K)

    g = reshape(f, size(a, 2), size(b, 2),  :)
    r = similar(f, size(a, 1), size(b, 1), size(g, 3))

    @tullio r[i, j, k] = b[j, m] * a[i, l] * g[l, m, k]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{E₃, E₂, D}}, f::F) where
        {D <: AbstractMatrix, E₃ <: Eye,  E₂ <: Eye, F <: AbstractVecOrMat}
    e₃, e₂, d = components(K)

    g = reshape(f, size(d, 2), size(e₂, 1), size(e₃, 1),  :)
    r = similar(f, size(d, 1), size(e₂, 1), size(e₃, 1), size(g, 4))

    @tullio r[i, j, k, e] = d[i, l] * g[l, j, k, e]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{E₃, D, E₁}}, f::F) where
        {D <: AbstractMatrix, E₁ <: Eye, E₃ <: Eye, F <: AbstractVecOrMat}
    e₃, d, e₁ = components(K)

    g = reshape(f, size(e₁, 1), size(d, 2), size(e₃, 1),  :)
    r = similar(f, size(e₁, 1), size(d, 1), size(e₃, 1), size(g, 4))

    @tullio r[i, j, k, e] = d[j, l] * g[i, l, k, e]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{D, E₂, E₁}}, f::F) where
        {D <: AbstractMatrix, E₁ <: Eye,  E₂ <: Eye, F <: AbstractVecOrMat}
    d, e₂, e₁ = components(K)

    g = reshape(f, size(e₁, 1), size(e₂, 1), size(d, 2),  :)
    r = similar(f, size(e₁, 1), size(e₂, 1), size(d, 1), size(g, 4))

    @tullio r[i, j, k, e] = d[k, l] * g[i, j, l, e]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end

function (*)(K::Kron{Tuple{C, B, A}}, f::F) where
        {A <: AbstractMatrix, B <: AbstractMatrix,
         C <: AbstractMatrix, F <: AbstractVecOrMat}
    c, b, a = components(K)

    g = reshape(f, size(a, 2), size(b, 2), size(c, 2),  :)
    r = similar(f, size(a, 1), size(b, 1), size(c, 1), size(g, 4))

    @tullio r[i, j, k, e] = c[k, n] * b[j, m] * a[i, l] * g[l, m, n, e]

    return F <: AbstractVector ? vec(r) : reshape(r, size(K, 1), size(f, 2))
end
