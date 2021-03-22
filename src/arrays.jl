arraytype(A) = Tullio.storage_type(A) <: CuArray ? CuArray : Array
arraytype(::Type{T}) where {T} = Array
arraytype(::Type{<:CuArray}) = CuArray
arraytype(::Type{<:CUDA.Adaptor}) = CuArray
