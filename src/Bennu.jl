module Bennu

export spectralderivative, spectralinterpolation, legendregauss,
       legendregausslobatto, partition, hilbertcode, quantize

include("operators.jl")
include("partitions.jl")
include("quadratures.jl")

end
