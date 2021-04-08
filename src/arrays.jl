arraytype(A) = Tullio.storage_type(A) <: CuArray ? CuArray : Array
arraytype(::Type{T}) where {T} = Array
arraytype(::Type{<:CuArray}) = CuArray
arraytype(::Type{<:CUDA.Adaptor}) = CuArray

components(a::AbstractArray) = (a,)

function numbercontiguous(A; by=identity)
    p = sortperm(A; by=by)
    notequalprevious = fill!(similar(p, Bool), false)
    @tullio notequalprevious[i] =
        @inbounds(begin by(A[p[i]]) != by(A[p[i-1]]) end) (i in 2:length(p))

    B = similar(p)
    B[p] .= cumsum(notequalprevious) .+ 1

    return B
end
