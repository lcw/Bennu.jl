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
