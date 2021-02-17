module Bennu

using StructArrays
using Tullio

export spectralderivative, spectralinterpolation, legendregauss,
       legendregausslobatto, partition, hilbertcode, quantize

include("arrays.jl")
include("operators.jl")
include("partitions.jl")
include("quadratures.jl")

end
