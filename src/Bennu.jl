module Bennu

using Adapt
using ArrayInterface
using CUDA
using CUDAKernels
using FillArrays
using GPUArrays
using KernelAbstractions
using LazyArrays
using LinearAlgebra
using LoopVectorization
using SparseArrays
using StaticArrays
using StaticArrays: tuple_prod, tuple_length, size_to_tuple
using StructArrays
using Tullio
using WriteVTK

export LobattoCell, NodalGrid, CartesianOrdering, HilbertOrdering,
       StackedOrdering

export spectralderivative, spectralinterpolation, legendregauss,
       legendregausslobatto, partition, quantize,
       hilbertcode, hilbertindices, hilbertperm,
       floattype, arraytype, points, mass, facemass, derivatives,
       derivatives_1d, degrees, celltype, referencecell, vertices,
       connectivity, points, brickgrid, toequallyspaced,
       cubedspheregrid, min_node_distance

export fieldarray

export components

export metrics, facemetrics

export faces, faceindices, boundaryfaces, number_of_faces, faceviews

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
include("structarrays.jl")
include("tuples.jl")
include("banded.jl")

end
