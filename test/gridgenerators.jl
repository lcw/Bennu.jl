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

            x = adapt(Array, points(grid))
            faceindices⁻, faceindices⁺ = adapt.(Array, faceindices(grid))
            @test isapprox(x[faceindices⁻], x[faceindices⁺], atol=100eps(T))
            matfaces = Bennu.materializefaces(cell, connectivity(grid))
            for (f, g) in zip(matfaces, faces(grid))
                @test size(f) == size(g)
            end
            for (b, d) in zip(adapt(Array,
                                    Bennu.materializeboundaryfaces(cell,
                                                                   matfaces)),
                              adapt(Array, boundaryfaces(grid)))
                @test (b ≠ 0 && d ≠ 0) || (b == d == 0)
            end

            Ds = derivatives(cell)

            for (i, D) in enumerate(Ds)
                otherdims = filter(!isequal(i), 1:length(Ds))
                D = adapt(Array, D)
                dx = reshape(D * x, (length(cell), size(grid)...))
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

    grid = brickgrid(LobattoCell(3), (-1:1,); periodic=(true,))
    @test faceindices(grid) == ([1 4; 3 6], [6 3; 4 1])
    @test boundaryfaces(grid) == [0  0;  0  0]

    grid = brickgrid(LobattoCell(3, 4), (-1:1, -1:1); periodic=(true, false))
    @test size(faces(grid)[1]) == (16, 10)
    @test size(faces(grid)[2]) == (16, 6)
    @test boundaryfaces(grid) == [0  0  0  0
                                  0  0  0  0
                                  3  3  0  0
                                  0  0  4  4]


    grid = brickgrid(LobattoCell(3, 4, 2), (-1:1, -1:1, -1:1);
                     periodic=(true, false, true))
    @test size(faces(grid)[1]) == (48, 28)
    @test size(faces(grid)[2]) == (96, 32)
    @test size(faces(grid)[3]) == (64, 12)
    @test boundaryfaces(grid) == [0  0  0  0  0  0  0  0
                                  0  0  0  0  0  0  0  0
                                  3  3  0  0  3  3  0  0
                                  0  0  4  4  0  0  4  4
                                  0  0  0  0  0  0  0  0
                                  0  0  0  0  0  0  0  0]
end

@testset "cubedsphere" begin
    TAs = ((Float64,  Array), (Float32,  Array), (BigFloat, Array))
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end

    for (T, A) in TAs
        cell = LobattoCell{T, A}(3, 3, 4)
        vert_coord = one(T):1//2:4one(T)
        ncells_h_panel = 2

        grid = cubedspheregrid(cell, vert_coord, ncells_h_panel)

        # Check the first level connectivity
        h_offset = 6 * (ncells_h_panel + 1)^2

        connectivity = adapt(Array, grid.connectivity)
        @test connectivity[1, 1, 1, 1][1:4] == connectivity[1, 1, 1, 1][5:8] .- h_offset == (1, 2, 4, 5)
        @test connectivity[1, 2, 1, 1][1:4] == connectivity[1, 2, 1, 1][5:8] .- h_offset == (2, 3, 5, 6)
        @test connectivity[1, 1, 2, 1][1:4] == connectivity[1, 1, 2, 1][5:8] .- h_offset == (4, 5, 7, 8)
        @test connectivity[1, 2, 2, 1][1:4] == connectivity[1, 2, 2, 1][5:8] .- h_offset == (5, 6, 8, 9)

        @test connectivity[1, 1, 1, 2][1:4] == connectivity[1, 1, 1, 2][5:8] .- h_offset == (10, 11, 13, 14)
        @test connectivity[1, 2, 1, 2][1:4] == connectivity[1, 2, 1, 2][5:8] .- h_offset == (11, 12, 14, 15)
        @test connectivity[1, 1, 2, 2][1:4] == connectivity[1, 1, 2, 2][5:8] .- h_offset == (13, 14, 16, 17)
        @test connectivity[1, 2, 2, 2][1:4] == connectivity[1, 2, 2, 2][5:8] .- h_offset == (14, 15, 17, 18)

        @test connectivity[1, 1, 1, 3][1:4] == connectivity[1, 1, 1, 3][5:8] .- h_offset == (3, 20, 6, 23)
        @test connectivity[1, 2, 1, 3][1:4] == connectivity[1, 2, 1, 3][5:8] .- h_offset == (20, 10, 23, 13)
        @test connectivity[1, 1, 2, 3][1:4] == connectivity[1, 1, 2, 3][5:8] .- h_offset == (6, 23, 9, 26)
        @test connectivity[1, 2, 2, 3][1:4] == connectivity[1, 2, 2, 3][5:8] .- h_offset == (23, 13, 26, 16)

        @test connectivity[1, 1, 1, 4][1:4] == connectivity[1, 1, 1, 4][5:8] .- h_offset == (12, 29, 15, 32)
        @test connectivity[1, 2, 1, 4][1:4] == connectivity[1, 2, 1, 4][5:8] .- h_offset == (29, 1, 32, 4)
        @test connectivity[1, 1, 2, 4][1:4] == connectivity[1, 1, 2, 4][5:8] .- h_offset == (15, 32, 18, 35)
        @test connectivity[1, 2, 2, 4][1:4] == connectivity[1, 2, 2, 4][5:8] .- h_offset == (32, 4, 35, 7)

        @test connectivity[1, 1, 1, 5][1:4] == connectivity[1, 1, 1, 5][5:8] .- h_offset == (1, 29, 2, 41)
        @test connectivity[1, 2, 1, 5][1:4] == connectivity[1, 2, 1, 5][5:8] .- h_offset == (29, 12, 41, 11)
        @test connectivity[1, 1, 2, 5][1:4] == connectivity[1, 1, 2, 5][5:8] .- h_offset == (2, 41, 3, 20)
        @test connectivity[1, 2, 2, 5][1:4] == connectivity[1, 2, 2, 5][5:8] .- h_offset == (41, 11, 20, 10)

        @test connectivity[1, 1, 1, 6][1:4] == connectivity[1, 1, 1, 6][5:8] .- h_offset == (9, 26, 8, 50)
        @test connectivity[1, 2, 1, 6][1:4] == connectivity[1, 2, 1, 6][5:8] .- h_offset == (26, 16, 50, 17)
        @test connectivity[1, 1, 2, 6][1:4] == connectivity[1, 1, 2, 6][5:8] .- h_offset == (8, 50, 7, 35)
        @test connectivity[1, 2, 2, 6][1:4] == connectivity[1, 2, 2, 6][5:8] .- h_offset == (50, 17, 35, 18)

        # Check the higher levels
        for z = 1:length(vert_coord) - 2
            for (a, b) in zip(connectivity[z, :, :, :],  connectivity[z + 1, :, :, :])
                @test a .+ h_offset == b
            end
        end

        # Check the vertices are at the right levels
        vertices = adapt(Array, grid.vertices)
        for z = 1:length(vert_coord)
            for p in vertices[:, :, :, z]
                @test norm(Bennu.cubespherewarp(p)) ≈ vert_coord[z]
            end
        end
    end
end
