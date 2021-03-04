using Bennu
using Test
using Pkg: Pkg

using Adapt
using CUDA
using EzXML: EzXML
using FastGaussQuadrature: FastGaussQuadrature
using LinearAlgebra
using StaticArrays
using Tullio
using WriteVTK

CUDA.allowscalar(false)

include("operators.jl")
include("partitions.jl")
include("quadratures.jl")

@testset "examples" begin
    julia = Base.julia_cmd()
    base_dir = joinpath(@__DIR__, "..")

    for example_dir in readdir(joinpath(base_dir, "examples"), join=true)
        @testset "$example_dir" begin
            mktempdir() do tmp_dir
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
