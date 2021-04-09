abstract type AbstractOrdering end
struct CartesianOrdering <: AbstractOrdering end
struct HilbertOrdering <: AbstractOrdering end

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
    elseif ordering isa HilbertOrdering
        perm = hilbertperm(dims)
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

    return NodalGrid(warp, referencecell, vertices, connectivity;
                     faces=faces, boundaryfaces=boundaryfaces)
end
