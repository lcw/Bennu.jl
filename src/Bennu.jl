module Bennu

export spectralderivative, spectralinterpolation, legendregauss,
       legendregausslobatto, partition, hilbertcode

include("operators.jl")
include("partitions.jl")
include("quadratures.jl")

end
