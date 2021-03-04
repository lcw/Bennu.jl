@testset "grids" begin
    TAs = ((Float64,  Array),
           (Float32,  Array),
           (BigFloat, Array))
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end

    for (T, A) in TAs
        verts = adapt(A, [SVector{2,T}(0, 0),
                          SVector{2,T}(1, 0),
                          SVector{2,T}(0, 1),
                          SVector{2,T}(1, 1),
                          SVector{2,T}(2, 0),
                          SVector{2,T}(2, 1)])
        conn = adapt(A, [SVector(1, 2, 3, 4),
                         SVector(2, 5, 4, 6)])
        cell = LobattoCell{T, A}(3, 4)
        grid = NodalGrid(cell, verts, conn)

        @test floattype(grid) == T
        @test arraytype(grid) <: A
        @test celltype(grid) == typeof(cell)
        @test Base.ndims(grid) == Base.ndims(conn)
        @test size(grid) == size(conn)
        @test length(grid) == length(conn)

        @test floattype(typeof(grid)) == T
        @test arraytype(typeof(grid)) <: A
        @test celltype(typeof(grid)) == typeof(cell)
        @test Base.ndims(typeof(grid)) == Base.ndims(conn)
        @test size(typeof(grid)) == size(conn)
        @test length(typeof(grid)) == length(conn)

        @test referencecell(grid) === cell
        @test vertices(grid) === verts
        @test connectivity(grid) === conn
        @test size(points(grid)) == (length(cell), length(conn))
        D = derivatives(referencecell(grid))
        @test Array(D[1] * points(grid)) ≈
            fill(SVector(one(T)/2, zero(T)), size(points(grid)))
        @test Array(D[2] * points(grid)) ≈
            fill(SVector(zero(T), one(T)/2), size(points(grid)))

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
