Base.similar(::Type{A}, ::Type{T}, dims...) where {A <: AbstractArray, T} =
    similar(similar(A, 0), T, dims...)

Tullio.storage_type(s::StructArray) =
    promote_type(map(Tullio.storage_type, StructArrays.components(s))...)

function components(s::StructArray{S,
                                   N,
                                   NamedTuple{(:data,),
                                              Tuple{StructArray{T, N, C, I}}},
                                   I}) where {S <: StaticArray, T, N, C, I}
    return StructArrays.components(StructArrays.components(s).data)
end

function Base.similar(s::StructArray{S, N,
                                     NamedTuple{(:data,),
                                                Tuple{StructArray{T, N, C, I}}},
                                     I},
                      element_type::Type{E}=S,
                      dims::Tuple{Vararg{Int64, M}}=size(s)) where
                          {S<:StaticArray, T, N, C, I, E, M}
    cs = ntuple(i->similar(first(components(s)), dims), length(element_type))
    return structarray(element_type, cs)
end

function structarray(components::Tuple)
    T = promote_type(eltype.(components)...)
    N = length(components)
    return structarray(SVector{N,T}, components)
end

function structarray(::Type{T}, components::Tuple) where {T}
    return StructArray{T}(data = StructArray(components))
end
