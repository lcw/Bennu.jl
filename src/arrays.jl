Base.similar(::Type{A}, ::Type{T}, dims...) where {A <: AbstractArray, T} =
    similar(similar(A, 0), T, dims...)

Tullio.storage_type(s::StructArray) =
    promote_type(map(Tullio.storage_type, StructArrays.components(s))...)
