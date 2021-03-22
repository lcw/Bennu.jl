device(::Type{T}) where {T} = T <: Array ? CPU() : CUDADevice()
device(A) = device(typeof(A))
