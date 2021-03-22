module Bennu

using Adapt
using CUDA
using FillArrays
using KernelAbstractions
using LazyArrays
using LinearAlgebra
using LoopVectorization
using SparseArrays
using StaticArrays
using StaticArrays: tuple_prod, tuple_length, size_to_tuple
using Tullio
using WriteVTK

export LobattoCell, NodalGrid, CartesianOrdering, HilbertOrdering

export spectralderivative, spectralinterpolation, legendregauss,
       legendregausslobatto, partition, quantize,
       hilbertcode, hilbertindices, hilbertperm,
       floattype, arraytype, points, mass, derivatives, degrees,
       celltype, referencecell, vertices, connectivity, points, brickgrid

export number_of_faces

include("permutations.jl")

include("arrays.jl")
include("cells.jl")
include("gridgenerators.jl")
include("grids.jl")
include("kernelabstractions.jl")
include("kroneckeroperators.jl")
include("operators.jl")
include("partitions.jl")
include("quadratures.jl")
include("sparsearrays.jl")
include("tuples.jl")

end
