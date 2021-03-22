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

        faceindices⁻, faceindices⁺ = faceindices(grid)
        x = points(grid)
        @test isapprox(adapt(Array, x[faceindices⁻]),
                       adapt(Array, x[faceindices⁺]), atol=10eps(T))

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

        let
            verts = adapt(A, [
                              SVector{3,T}(0, 0, 0),
                              SVector{3,T}(0, 1, 0),
                              SVector{3,T}(1, 0, 0),
                              SVector{3,T}(1, 1, 0),
                              SVector{3,T}(2, 0, 0),
                              SVector{3,T}(2, 1, 0),
                              SVector{3,T}(3, 0, 0),
                              SVector{3,T}(3, 1, 0),
                              SVector{3,T}(4, 0, 0),
                              SVector{3,T}(4, 1, 0),
                              SVector{3,T}(5, 0, 0),
                              SVector{3,T}(5, 1, 0),
                              SVector{3,T}(6, 0, 0),
                              SVector{3,T}(6, 1, 0),
                              SVector{3,T}(7, 0, 0),
                              SVector{3,T}(7, 1, 0),
                              SVector{3,T}(8, 0, 0),
                              SVector{3,T}(8, 1, 0),
                              SVector{3,T}(9, 0, 0),
                              SVector{3,T}(9, 1, 0),
                              SVector{3,T}(0, 0, 1),
                              SVector{3,T}(0, 1, 1),
                              SVector{3,T}(1, 0, 1),
                              SVector{3,T}(1, 1, 1),
                              SVector{3,T}(2, 0, 1),
                              SVector{3,T}(2, 1, 1),
                              SVector{3,T}(3, 0, 1),
                              SVector{3,T}(3, 1, 1),
                              SVector{3,T}(4, 0, 1),
                              SVector{3,T}(4, 1, 1),
                              SVector{3,T}(5, 0, 1),
                              SVector{3,T}(5, 1, 1),
                              SVector{3,T}(6, 0, 1),
                              SVector{3,T}(6, 1, 1),
                              SVector{3,T}(7, 0, 1),
                              SVector{3,T}(7, 1, 1),
                              SVector{3,T}(8, 0, 1),
                              SVector{3,T}(8, 1, 1),
                              SVector{3,T}(9, 0, 1),
                              SVector{3,T}(9, 1, 1)
                             ])
            conn = adapt(A, [
                             SA[ 1,  2, 21, 22,  3,  4, 23, 24], # 0
                             SA[ 3,  4, 23, 24,  5,  6, 25, 26], # 1
                             SA[ 6, 26,  5, 25,  8, 28,  7, 27], # 2
                             SA[27,  7, 28,  8, 29,  9, 30, 10], # 3
                             SA[30, 29, 10,  9, 32, 31, 12, 11], # 4
                             SA[34, 14, 33, 13, 32, 12, 31, 11], # 5
                             SA[33, 13, 34, 14, 35, 15, 36, 16], # 6
                             SA[18, 17, 38, 37, 16, 15, 36, 35], # 7
                             SA[17, 18, 37, 38, 19, 20, 39, 40], # 8
                            ])
            cell = LobattoCell{T,A}(4,4,5)
            grid = NodalGrid(cell, verts, conn)
            faceindices⁻, faceindices⁺ = faceindices(grid)
            x = points(grid)
            @test isapprox(adapt(Array, x[faceindices⁻]),
                           adapt(Array, x[faceindices⁺]), atol=100eps(T))
            @test size(faces(grid)[1]) == (54, 46)
            @test size(faces(grid)[2]) == (108, 76)
            @test size(faces(grid)[3]) == (72, 40)
            @test boundaryfaces(grid) == adapt(A, [1  1  1  1  1  1  1  1  1
                                                   1  1  1  1  1  1  1  1  1
                                                   1  1  1  1  1  1  1  1  1
                                                   1  1  1  1  1  1  1  1  1
                                                   1  0  0  0  0  0  0  0  0
                                                   0  0  0  0  0  0  0  0  1])
        end

        let
            verts = adapt(A, [SVector{2,T}(1, 1),
                              SVector{2,T}(2, 1),
                              SVector{2,T}(3, 1),
                              SVector{2,T}(1, 2),
                              SVector{2,T}(2, 2),
                              SVector{2,T}(3, 2),
                              SVector{2,T}(1, 3),
                              SVector{2,T}(2, 3),
                              SVector{2,T}(3, 3)])
            conn = adapt(A, [
                             SA[1, 2, 4, 5],
                             SA[5, 2, 6, 3],
                             SA[5, 8, 4, 7],
                             SA[9, 8, 6, 5],
                            ])
            cell = LobattoCell{T,A}(4,4)
            grid = NodalGrid(cell, verts, conn)
            faceindices⁻, faceindices⁺ = faceindices(grid)
            x = points(grid)
            @test isapprox(adapt(Array, x[faceindices⁻]),
                           adapt(Array, x[faceindices⁺]), atol=10eps(T))
            @test size(faces(grid)[1]) == (16, 12)
            @test size(faces(grid)[2]) == (16, 9)
            @test boundaryfaces(grid) == adapt(A, [1  0  0  1
                                                   0  1  1  0
                                                   1  0  0  1
                                                   0  1  1  0])
        end

        let
            verts = adapt(A, [SVector{1,T}(1),
                              SVector{1,T}(2),
                              SVector{1,T}(3)])
            conn = adapt(A, [
                             SA[1, 2],
                             SA[3, 2],
                            ])
            cell = LobattoCell{T,A}(6)
            grid = NodalGrid(cell, verts, conn)
            faceindices⁻, faceindices⁺ = faceindices(grid)
            x = points(grid)
            @test isapprox(adapt(Array, x[faceindices⁻]),
                           adapt(Array, x[faceindices⁺]), atol=10eps(T))
            @test size(faces(grid)[1]) == (4, 3)
            @test boundaryfaces(grid) == adapt(A, [1  1
                                                   0  0])
        end
    end
end
