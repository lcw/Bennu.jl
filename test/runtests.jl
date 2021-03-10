using Bennu
using Test
using Pkg: Pkg

using Adapt
using Combinatorics
using CUDA
using EzXML: EzXML
using FastGaussQuadrature: FastGaussQuadrature
using FillArrays
using LazyArrays
using LinearAlgebra
using Random
using StaticArrays
using Tullio
using WriteVTK

CUDA.allowscalar(false)

include("cells.jl")
include("gridgenerators.jl")
include("grids.jl")
include("kroneckeroperators.jl")
include("operators.jl")
include("partitions.jl")
include("quadratures.jl")
include("tuples.jl")

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
                    cmd = `$julia --project=$tmp_project -e "import Pkg; Pkg.develop(path=raw\"$base_dir\"); Pkg.resolve(); Pkg.instantiate(); include(raw\"$script\")"`
                    @test success(pipeline(cmd, stderr=stderr, stdout=stdout))
                end
            end
        end
    end
end
