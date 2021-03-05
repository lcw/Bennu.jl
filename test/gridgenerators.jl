@testset "brickgrids" begin
    TAs = ((Float64,  Array), (Float32,  Array), (BigFloat, Array))
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end

    for (T, A) in TAs
        cellsandcoordinates =
            (
             (LobattoCell{T, A}(5), (-one(T):1//2:one(T),)),
             (LobattoCell{T, A}(4), ((1, 2, 5, 8),)),
             (LobattoCell{T, A}(3, 5),
              (-one(T):1//2:one(T), zero(T):1//4:one(T))),
             (LobattoCell{T, A}(3, 5, 2),
              (-one(T):1//2:one(T), zero(T):1//4:one(T), 1:3))
            )
        for (cell, coordinates) in cellsandcoordinates
            grid = brickgrid(cell, coordinates)

            @test floattype(grid) == T
            @test arraytype(grid) <: A
            @test celltype(grid) == typeof(cell)
            @test ndims(grid) == length(coordinates)
            @test size(grid) == length.(coordinates) .- 1
            @test length(grid) == prod(length.(coordinates) .- 1)

            @test floattype(typeof(grid)) == T
            @test arraytype(typeof(grid)) <: A
            @test celltype(typeof(grid)) == typeof(cell)
            @test ndims(typeof(grid)) == length(coordinates)
            @test size(typeof(grid)) == length.(coordinates) .- 1
            @test length(typeof(grid)) == prod(length.(coordinates) .-1)

            cell = referencecell(grid)
            @test size(points(grid)) == (length(cell), length(grid))
            x = points(grid)
            Ds = derivatives(cell)

            for (i, D) in enumerate(Ds)
                otherdims = filter(!isequal(i), 1:length(Ds))
                dx = reshape(Array(D * x), (length(cell), size(grid)...))
                pdx = vec(minimum(dx, dims=(1, (otherdims .+ 1)...)))
                qdx = vec(maximum(dx, dims=(1, (otherdims .+ 1)...)))
                cdx = diff(collect(coordinates[i]))./2

                @test isapprox(pdx, qdx, atol=100eps(T))
                @test isapprox(getindex.(pdx, i), cdx, atol=100eps(T))
                for j in otherdims
                    @test isapprox(getindex.(pdx, j), zeros(T, length(pdx)),
                                   atol=100eps(T))
                end
            end

            if T != BigFloat
                mktempdir() do tmp
                    outfiles = vtk_grid(joinpath(tmp, "grid_$(T)_$(A)"), grid;
                                        append=false, ascii=true) do vtk
                        vtk["Time"] = 37.0
                    end
                    for file in outfiles
                        @test EzXML.readxml(file) isa EzXML.Document
                    end
                end
            end
        end
    end
end
