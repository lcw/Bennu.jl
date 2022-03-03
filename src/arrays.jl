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

_numfields(::Type) = 1
_numfields(::Type{S}) where {S <: SArray} = length(S)
_numfields(T::Tuple) = sum(map(_numfields, T))
_numfields(N::NamedTuple) = sum(map(_numfields, N))

_fieldtype(::Type{S}) where {S} = eltype(S)
_fieldtype(T::Tuple) = promote_type(map(_fieldtype, T)...)
_fieldtype(N::NamedTuple) = promote_type(map(_fieldtype, N)...)

function fieldarray(::UndefInitializer, S, ::Type{A}, dims::Dims) where {A}
    N = _numfields(S)
    T = _fieldtype(S)

    if N == 1 && S isa Type && S <: Number
        return A{S}(undef, dims)
    end

    if length(dims) == 0
        d = (N, )
    elseif length(dims) == 1
        d = (dims[1], N)
    else
        d = (dims[1:end-1]..., N, dims[end])
    end

    data = A{T}(undef, d)
    dataviews = ntuple(N) do i
        offset = length(dims) > 1 ? 0 : 1
        viewtuple = ntuple(j->(j==length(dims)+offset ? i : :), length(d))
        return view(data, viewtuple...)
    end

    return fieldarray(S, dataviews)
end

function _ckfieldargs(S, data::Tuple)
    if _numfields(S) != length(data)
        throw(ArgumentError("Number of data fields is incorrect."))
    end
    T = _fieldtype(S)
    dims = size(first(data))
    for d in data
        if eltype(d) != T
            throw(ArgumentError("Data arrays do not have the same eltype."))
        end
        if size(d) != dims
            throw(ArgumentError("Data arrays do not have the same size."))
        end
    end
end

function fieldarray(::Type{S}, data::Tuple) where {S}
    d = only(data)
    if S != eltype(d)
        throw(ArgumentError("Data array does not have the correct eltype."))
    end

    return d
end

function fieldarray(::Type{S}, data::Tuple) where {S <: SArray}
    _ckfieldargs(S, data)
    return StructArray{S}(data)
end

function fieldarray(S::NamedTuple, data::Tuple)
    _ckfieldargs(S, data)

    offsets = cumsum((1, map(_numfields, S)...))
    fields = ntuple(length(S)) do i
        fieldarray(S[i], data[offsets[i]:offsets[i+1]-1])
    end

    return StructArray(NamedTuple{keys(S)}(fields))
end

function fieldarray(a::AbstractArray, dims::Dims=size(a))
    similar(a, dims)
end

_namedtuple(::Type{S}) where {S} = S
function _namedtuple(::Type{S}) where {S <: NamedTuple}
    k = S.parameters[1]
    v = S.parameters[2].parameters
    return NamedTuple{k}(_namedtuple.(v))
end
function fieldarray(a::StructArray{S}, dims::Dims=size(a)) where {S}
    @assert isfieldarray(a)
    fieldarray(undef, _namedtuple(S), arraytype(a), dims)
end
