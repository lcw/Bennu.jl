using Bennu
using Test
using Pkg: Pkg

using Adapt
using Combinatorics
using CUDA
using CUDAKernels
using EzXML: EzXML
using FastGaussQuadrature: FastGaussQuadrature
using FillArrays
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays
using StructArrays
using Tullio
using WriteVTK

CUDA.allowscalar(false)

include("arrays.jl")
include("cells.jl")
include("gridgenerators.jl")
include("grids.jl")
include("stackedgrids.jl")
include("kernelabstractions.jl")
include("kroneckeroperators.jl")
include("metrics.jl")
include("operators.jl")
include("partitions.jl")
include("permutations.jl")
include("quadratures.jl")
include("sparsearrays.jl")
include("structarrays.jl")
include("tuples.jl")
include("banded.jl")

@testset "examples" begin
    julia = Base.julia_cmd()
    base_dir = joinpath(@__DIR__, "..")

    for example_dir in readdir(joinpath(base_dir, "examples"), join=true)
        @testset "$example_dir" begin
            mktempdir() do tmp_dir
                # Change to temporary directory so that any files created by the
                # example get cleaned up after execution.
                cd(tmp_dir)
                example_project = Pkg.Types.projectfile_path(example_dir)
                tmp_project = Pkg.Types.projectfile_path(tmp_dir)
                cp(example_project, tmp_project)

                for script in filter!(s->endswith(s, ".jl"),
                                      readdir(example_dir, join=true))
                    cmd = `$julia --project=$tmp_project -e "import Pkg; Pkg.develop(path=raw\"$base_dir\"); Pkg.instantiate(); include(raw\"$script\")"`
                    @test success(pipeline(cmd, stderr=stderr, stdout=stdout))
                end
            end
        end
    end
end
