module Bennu

using Adapt
using CUDA
using FillArrays
using KernelAbstractions
using LazyArrays
using LinearAlgebra
using LoopVectorization
using StaticArrays
using StaticArrays: tuple_prod, tuple_length, size_to_tuple
using Tullio
using WriteVTK

export spectralderivative, spectralinterpolation, legendregauss,
       legendregausslobatto, partition, hilbertcode, quantize

include("operators.jl")
include("partitions.jl")
include("quadratures.jl")

end
