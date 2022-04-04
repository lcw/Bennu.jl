abstract type AbstractOrdering end
struct CartesianOrdering <: AbstractOrdering end
struct HilbertOrdering <: AbstractOrdering end
struct StackedOrdering{T <: AbstractOrdering} <: AbstractOrdering end

function brickgrid(referencecell::LobattoCell, coordinates::Tuple;
                   ordering::AbstractOrdering=CartesianOrdering(),
                   periodic=nothing)
    return brickgrid(identity, referencecell, coordinates;
                   ordering=ordering, periodic=periodic)
end

function brickgrid(warp::Function, referencecell::LobattoCell,
                   coordinates::Tuple;
                   ordering::AbstractOrdering=CartesianOrdering(),
                   periodic=nothing)

    M = ndims(referencecell)
    if M != length(coordinates)
        throw(ArgumentError("The number of dimensions of the reference cell and number of coordinates must match."))
    end

    dims = length.(coordinates) .- 1
    if !all(dims .>= 1)
        throw(ArgumentError("Each coordinate needs to be of atleast length 2."))
    end

    A = arraytype(referencecell)
    coordinates = adapt.(A, collect.(coordinates))
    coordinates = ntuple(M) do i
        coorddims = ntuple(j->ifelse(i==j, length(coordinates[i]), 1), M)
        return reshape(coordinates[i], coorddims)
    end
    vertices = SVector.(coordinates...)

    linear = adapt(A, collect(LinearIndices(dims.+1)))
    connectivity = similar(vertices, NTuple{2^M, Int}, dims)

    if M == 1
        @tullio connectivity[i] = (linear[i], linear[i + 1])
    elseif M == 2
        @tullio connectivity[i, j] = (
                                      linear[i,     j    ],
                                      linear[i + 1, j    ],
                                      linear[i,     j + 1],
                                      linear[i + 1, j + 1],
                                     )
    elseif M == 3
        @tullio connectivity[i, j, k] = (
                                         linear[i,     j,     k    ],
                                         linear[i + 1, j,     k    ],
                                         linear[i,     j + 1, k    ],
                                         linear[i + 1, j + 1, k    ],
                                         linear[i,     j,     k + 1],
                                         linear[i + 1, j,     k + 1],
                                         linear[i,     j + 1, k + 1],
                                         linear[i + 1, j + 1, k + 1],
                                        )
    else
        throw(ArgumentError("Reference cells with number of dimensions greater than 3 not implemented."))
    end

    if isnothing(periodic)
        periodic = ntuple(i->false, M)
    end

    boundaryfaces = fill!(A{Int, M+1}(undef, 2*M, size(connectivity)...), 0)
    if M == 1
        if !periodic[1]
            boundaryfaces[1,     1:1] .= 1
            boundaryfaces[2, end:end] .= 2
        end
    elseif M == 2
        if !periodic[1]
            boundaryfaces[1,   1,   :] .= 1
            boundaryfaces[2, end,   :] .= 2
        end
        if !periodic[2]
            boundaryfaces[3,   :,   1] .= 3
            boundaryfaces[4,   :, end] .= 4
        end
    elseif M == 3
        if !periodic[1]
            boundaryfaces[1,   1,   :,   :] .= 1
            boundaryfaces[2, end,   :,   :] .= 2
        end
        if !periodic[2]
            boundaryfaces[3,   :,   1,   :] .= 3
            boundaryfaces[4,   :, end,   :] .= 4
        end
        if !periodic[3]
            boundaryfaces[5,   :,   :,   1] .= 5
            boundaryfaces[6,   :,   :, end] .= 6
        end
    else
        throw(ArgumentError("Reference cells with number of dimensions greater than 3 not implemented."))
    end
    boundaryfaces = reshape(boundaryfaces, 2*M, length(connectivity))

    if M == 1
        corners = A{Int, M+1}(undef, 2, size(connectivity)...)
        corners[:] .= 1:length(corners)

        corners[2, 1:end-1] .= corners[1, 2:end]
        if periodic[1]
          corners[2, end:end] .= corners[1, 1:1]
        end

        facenumbers = (corners,)
    elseif M == 2
        corners = A{Int, M+1}(undef, 4, size(connectivity)...)
        corners[:] .= 1:length(corners)

        corners[[2,4], 1:end-1, :] .= corners[[1,3], 2:end, :]
        if periodic[1]
          corners[[2,4], end, :] .= corners[[1,3], 1, :]
        end

        corners[[3,4], :, 1:end-1] .= corners[[1,2], :, 2:end]
        if periodic[2]
          corners[[3,4], :, end] .= corners[[1,2], :, 1]
        end

        edges = A{Int, M+1}(undef, 4, size(connectivity)...)
        edges[:] .= 1:length(edges)

        edges[2, 1:end-1, :] .= edges[1, 2:end, :]
        if periodic[1]
          edges[2, end, :] .= edges[1, 1, :]
        end

        edges[4, :, 1:end-1] .= edges[3, :, 2:end]
        if periodic[2]
          edges[4, :, end] .= edges[3, :, 1]
        end

        facenumbers = (edges, corners)
    elseif M == 3
        corners = A{Int, M+1}(undef, 8, size(connectivity)...)
        corners[:] .= 1:length(corners)

        corners[[2,4,6,8], 1:end-1, :, :] .= corners[[1,3,5,7], 2:end, :, :]
        if periodic[1]
            corners[[2,4,6,8], end, :, :] .= corners[[1,3,5,7], 1, :, :]
        end

        corners[[3,4,7,8], :, 1:end-1, :] .= corners[[1,2,5,6], :, 2:end, :]
        if periodic[2]
            corners[[3,4,7,8], :, end, :] .= corners[[1,2,5,6], :, 1, :]
        end

        corners[[5,6,7,8], :, :, 1:end-1] .= corners[[1,2,3,4], :, :, 2:end]
        if periodic[3]
            corners[[5,6,7,8], :, :, end] .= corners[[1,2,3,4], :, :, 1]
        end

        edges = A{Int, M+1}(undef, 12, size(connectivity)...)
        edges[:] .= 1:length(edges)

        edges[[2,4], :, 1:end-1, :] .= edges[[1,3], :, 2:end, :]
        if periodic[2]
          edges[[2,4], :, end, :] .= edges[[1,3], :, 1, :]
        end
        edges[[3,4], :, :, 1:end-1] .= edges[[1,2], :, :, 2:end]
        if periodic[3]
          edges[[3,4], :, :, end] .= edges[[1,2], :, :, 1]
        end

        edges[[6,8], 1:end-1, :, :] .= edges[[5,7], 2:end, :, :]
        if periodic[1]
          edges[[6,8], end, :, :] .= edges[[5,7], 1, :, :]
        end
        edges[[7,8], :, :, 1:end-1] .= edges[[5,6], :, :, 2:end]
        if periodic[3]
          edges[[7,8], :, :, end] .= edges[[5,6], :, :, 1]
        end

        edges[[10,12], 1:end-1, :, :] .= edges[[9,11], 2:end, :, :]
        if periodic[1]
          edges[[10,12], end, :, :] .= edges[[9,11], 1, :, :]
        end
        edges[[11,12], :, 1:end-1, :] .= edges[[9,10], :, 2:end, :]
        if periodic[2]
          edges[[11,12], :, end, :] .= edges[[9,10], :, 1, :]
        end

        faces = A{Int, M+1}(undef, 6, size(connectivity)...)
        faces[:] .= 1:length(faces)

        faces[2, 1:end-1, :, :] .= faces[1, 2:end, :, :]
        if periodic[1]
          faces[2, end, :, :] .= faces[1, 1, :, :]
        end

        faces[4, :, 1:end-1, :] .= faces[3, :, 2:end, :]
        if periodic[2]
          faces[4, :, end, :] .= faces[3, :, 1, :]
        end

        faces[6, :, :, 1:end-1] .= faces[5, :, :, 2:end]
        if periodic[3]
          faces[6, :, :, end] .= faces[5, :, :, 1]
        end

        facenumbers = (faces, edges, corners,)
    else
        throw(ArgumentError("Reference cells with number of dimensions greater than 3 not implemented."))
    end
    facenumbers = ntuple(length(facenumbers)) do i
        return reshape(facenumbers[i], size(facenumbers[i], 1), :)
    end

    if ordering isa CartesianOrdering
        stacksize = 0
    elseif ordering isa HilbertOrdering
        stacksize = 0
        perm = hilbertperm(dims)
        connectivity = connectivity[perm]
        boundaryfaces = boundaryfaces[:, perm]
        facenumbers = ntuple(i->facenumbers[i][:, perm], length(facenumbers))
    elseif ordering isa StackedOrdering
        stacksize = dims[end]
        if ordering isa StackedOrdering{CartesianOrdering}
            base_perm = Vector(1:prod(dims[1:end-1]))
        elseif ordering isa StackedOrdering{HilbertOrdering}
            base_perm = length(dims) == 1 ? [1] : hilbertperm(dims[1:end-1])
        else
            throw(ArgumentError("Unsuported stacked ordering $ordering."))
        end
        stack = length(base_perm) * (0:dims[end]-1)
        perm = reshape(stack .+ base_perm', prod(dims))
        connectivity = connectivity[perm]
        boundaryfaces = boundaryfaces[:, perm]
        facenumbers = ntuple(i->facenumbers[i][:, perm], length(facenumbers))
    else
        throw(ArgumentError("Unsuported ordering $ordering."))
    end

    faces = ntuple(length(facenumbers)) do i
        globalfacenumbers = vec(adapt(Array, facenumbers[i]))
        globalfacenumbers = numbercontiguous(globalfacenumbers)
        localfacenumbers = 1:length(globalfacenumbers)
        P = M == 3 ? 4 : M
        facepermutations = zeros(Permutation{P}, length(globalfacenumbers))

        S = sparse(localfacenumbers, globalfacenumbers, facepermutations)
        if A <: CuArray
            S = adapt(A, GeneralSparseMatrixCSC(S))
        end
        return S
    end

    return NodalGrid(warp, referencecell, vertices, connectivity, :brick;
                     faces=faces, boundaryfaces=boundaryfaces, stacksize=stacksize)
end

function cubespherewarp(point)
    # Put the points in reverse magnitude order
    p = sortperm(abs.(point))
    point = point[p]

    # Convert to angles
    ξ = π * point[2] / 4point[3]
    η = π * point[1] / 4point[3]

    # Compute the ratios
    y_x = tan(ξ)
    z_x = tan(η)

    # Compute the new points
    x = point[3] / hypot(1, y_x, z_x)
    y = x * y_x
    z = x * z_x

    # Compute the new points and unpermute
    point = SVector(z, y, x)[sortperm(p)]

    return point
end

function cubesphereunwarp(point)
    # Put the points in reverse magnitude order
    p = sortperm(abs.(point))
    point = point[p]

    # Convert to angles
    ξ = 4atan(point[2]/point[3])/π
    η = 4atan(point[1]/point[3])/π
    R = sign(point[3])*hypot(point...)

    x = R
    y = R * ξ
    z = R * η

    # Compute the new points and unpermute
    point = SVector(z, y, x)[sortperm(p)]

    return point
end

function cubedshellconnectivity(referencecell::LobattoCell,
                                R::Real,
                                ncells::Integer;
                                ordering::AbstractOrdering=CartesianOrdering())

    if ndims(referencecell) != 2
        throw(ArgumentError("cubed sphere shell requires 2-D reference cell"))
    end

    # Get the array and float type from the reference cell
    A = arraytype(referencecell)
    T = floattype(referencecell)

    # Create the 1-D grid along an edge of the cube
    coord1d = adapt(A, collect(range(-T(R), length = ncells + 1, stop = R)))

    # Create the vertices and cell connectivity on each cube face
    vertices = similar(coord1d, SVector{3, T}, (ncells+1, ncells+1, 6))

    # face 1: ξ1 = -1
    @tullio vertices[i, j, 1] = SVector(-R, -coord1d[i], +coord1d[j])

    # face 2: ξ1 =  1
    @tullio vertices[i, j, 2] = SVector(+R, +coord1d[i], +coord1d[j])

    # face 3: ξ2 = -1
    @tullio vertices[i, j, 3] = SVector(+coord1d[i], -R, +coord1d[j])

    # face 4: ξ2 =  1
    @tullio vertices[i, j, 4] = SVector(-coord1d[i], +R, +coord1d[j])

    # face 5: ξ3 = -1
    @tullio vertices[i, j, 5] = SVector(+coord1d[i], -coord1d[j], -R)

    # face 6: ξ3 =  1
    @tullio vertices[i, j, 6] = SVector(+coord1d[i], +coord1d[j], +R)

    # Create a connectivity map for each element
    linear = adapt(A, collect(LinearIndices((ncells + 1, ncells + 1, 6))))
    conn = similar(vertices, Int, (2, 2, ncells, ncells, 6))
    @tullio conn[a, b, i, j, f] = linear[i + a - 1, j + b - 1, f]

    # Remove the vertex references that are actually for duplicate points

    # face 3/4, edge 1 -> face 1/2, edge 2
    conn[1, :,   1, :, [3, 4]] .= conn[2, :, end, :, [1, 2]]
    # face 3/4, edge 2 -> face 2/1, edge 1
    conn[2, :, end, :, [3, 4]] .= conn[1, :,   1, :, [2, 1]]

    # face 5, edge 1 -> face 1, edge 3
    conn[1, :,   1, :, 5] .= conn[:,      1, :,        1, 1]
    # face 5, edge 2 -> face 2, edge -3
    conn[2, :, end, :, 5] .= conn[2:-1:1, 1, end:-1:1, 1, 2]
    # face 5, edge 3 -> face 3, edge -3
    conn[:, 1, :,   1, 5] .= conn[2:-1:1, 1, end:-1:1, 1, 4]
    # face 5, edge 4 -> face 3, edge 3
    conn[:, 2, :, end, 5] .= conn[:,      1, :,        1, 3]

    # face 6, edge 1 -> face 1, edge -4
    conn[1, :,   1, :, 6] .= conn[2:-1:1, 2, end:-1:1, end, 1]
    # face 6, edge 2 -> face 2, edge 4
    conn[2, :, end, :, 6] .= conn[:,      2, :,        end, 2]
    # face 6, edge 3 -> face 3, edge 4
    conn[:, 1, :,   1, 6] .= conn[:,      2, :,        end, 3]
    # face 6, edge 4 -> face 4, edge -4
    conn[:, 2, :, end, 6] .= conn[2:-1:1, 2, end:-1:1, end, 4]

    connectivity = similar(vertices, NTuple{4, Int}, (ncells, ncells, 6))

    @tullio connectivity[i, j, f] = (
                                     conn[1, 1, i, j, f],
                                     conn[2, 1, i, j, f],
                                     conn[1, 2, i, j, f],
                                     conn[2, 2, i, j, f],
                                    )

    if ordering isa CartesianOrdering
    else
        throw(ArgumentError("Unsuported ordering $ordering."))
    end

    return (vertices = vertices, connectivity = connectivity)
end

function cubedspheregrid(referencecell::LobattoCell,
                         vert_coordinate::AbstractArray,
                         ncells_h_panel::Integer;
                         horz_ordering::AbstractOrdering=CartesianOrdering())

    if ndims(referencecell) != 3
        throw(ArgumentError("cubed sphere shell requires 3-D reference cell"))
    end

    if length(vert_coordinate) < 2
        throw(ArgumentError("Vertical coordinate needs to be of atleast length 2."))
    end

    Nq = size(referencecell)
    if Nq[1] != Nq[2]
        throw(ArgumentError("`referencecell` should have first two dims the same"))
    end

    h_refcell = similar(referencecell, Nq[1], Nq[2])

    # Get the array and float type from the reference cell
    A = arraytype(referencecell)
    T = floattype(referencecell)

    # Create a unit shell
    (h_vert, h_conn) = cubedshellconnectivity(h_refcell,
                                              1,
                                              ncells_h_panel;
                                              ordering = horz_ordering)

    # Blow out to a stacked cubed shell
    vert_coordinate = adapt(A, collect(vert_coordinate))
    vertices = similar(
                       h_vert,
                       size(h_vert)...,
                       length(vert_coordinate),
                       )
    @tullio vertices[x, y, f, z] = h_vert[x, y, f] * vert_coordinate[z]

    # Add the vertical connectivity
    connectivity = similar(
                           h_conn,
                           NTuple{8, Int},
                           length(vert_coordinate) - 1,
                           size(h_conn)...,
                          )
    offset = prod(size(h_vert))
    @tullio connectivity[z, x, y, f] = begin
        c = h_conn[x, y, f]
        l = (z - 1) * offset .+ c
        u = z * offset .+ c
        (l..., u...)
    end
    stacksize = length(vert_coordinate) - 1

    return NodalGrid(cubespherewarp, referencecell, vertices, connectivity, :cubedsphere; stacksize=stacksize)
end
