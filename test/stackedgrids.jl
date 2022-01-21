@testset "brick stackedgrids" begin
    TAs = ((Float64,  Array),)
    BaseOrderings = (CartesianOrdering, HilbertOrdering)
    for (T, A) in TAs
        cellsandcoordinates =
        (
         (LobattoCell{T, A}(5), (zero(T):1//4:one(T),)),
         (LobattoCell{T, A}(3, 5),
          (-one(T):1//2:one(T), zero(T):1//4:one(T))),
         (LobattoCell{T, A}(3, 5, 2),
          (-one(T):1//2:one(T), zero(T):1//4:one(T), zero(T):1//8:one(T),))
        )
        for (cell, coord) in cellsandcoordinates, BaseOrdering in BaseOrderings
            stacksize = length(coord[end]) - 1

            brick = brickgrid(cell, coord;
                              ordering = BaseOrdering())
            @test !Bennu.isstacked(brick)
            @test Bennu.stacksize(brick) == 0

            brick = brickgrid(cell, coord;
                              ordering = StackedOrdering{BaseOrdering}())
            @test Bennu.isstacked(brick)
            @test Bennu.stacksize(brick) == stacksize
            @test Bennu.horizontalsize(brick) * stacksize == length(brick)

            # Check the elements really have stacked vertical elements
            horz_size = Bennu.horizontalsize(brick)
            points = reshape(Bennu.points(brick), :, stacksize, horz_size)

            # Get the vertical offsets
            vpnts = points[1:1, :, 1:1] .- points[1:1, 1:1, 1:1]

            # Check that the points are purely vertical
            dim = ndims(cell)
            for (i, p) in enumerate(vpnts)
                @test all(p[1:dim-1] .== 0)
                @test all(p[end] .== coord[end][i])
            end

            # Make sure the points are just the vertical plus the vertical
            # offsets
            @test all(points[:, 1:1, :] .+ vpnts .≈ points)
        end
    end
end

@testset "sphere stackedgrids" begin
    TAs = ((Float64,  Array),)
    ncells_h_panel = 2
    for (T, A) in TAs
        cell = LobattoCell{T, A}(3, 3, 4)
        vert_coord = one(T):1//2:4one(T)
        stacksize = length(vert_coord) - 1
        sphere = cubedspheregrid(cell, vert_coord, ncells_h_panel)

        @test Bennu.isstacked(sphere)
        @test Bennu.stacksize(sphere) == stacksize
        @test Bennu.horizontalsize(sphere) * stacksize == length(sphere)

        # Check the elements really have stacked vertical elements
        horz_size = Bennu.horizontalsize(sphere)
        points = reshape(Bennu.points(sphere), :, stacksize, horz_size)

        # Get the vertical offsets
        R = reshape(vert_coord[1:end-1], 1, stacksize, 1) .- vert_coord[1]

        # Check that the points are purely vertical
        base = points[:, 1:1, :]
        Rbase = norm.(base)
        vbase = Rbase .+ R
        # Idea here is the all the points are the spherical locations with
        # different R values, so we normalize the base and ass a shift for the
        # vertical R
        @test all(vbase .* base ./ Rbase .≈ points)
    end
end
