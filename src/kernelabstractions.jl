device(::Type{T}) where {T} = T <: Array ? CPU() : CUDADevice()
device(A) = device(ArrayInterface.parent_type(A))
