abstract type AbstractOrdering end
struct CartesianOrdering <: AbstractOrdering end
struct HilbertOrdering <: AbstractOrdering end

function brickgrid(referencecell::LobattoCell, coordinates::Tuple;
                   ordering::AbstractOrdering=CartesianOrdering(),
                   periodic=nothing)

    M = ndims(referencecell)
    if M != length(coordinates)
        throw(ArgumentError("The number of dimensions of the reference cell and number of coordinates must match."))
    end

    dims = length.(coordinates) .- 1
    if !all(dims .> 1)
        throw(ArgumentError("Each coordinate needs to be of atleast length 2."))
    end

    A = arraytype(referencecell)
    coordinates = adapt.(A, collect.(coordinates))
    coordinates = ntuple(i->reshape(coordinates[i],
                                    ntuple(j->ifelse(i==j,
                                                     length(coordinates[i]), 1),
                                                     M)), M)
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

    if periodic != nothing
        throw(ArgumentError("Periodic meshes not implemented."))
    end

    if ordering isa CartesianOrdering
    elseif ordering isa HilbertOrdering
        connectivity = connectivity[hilbertperm(dims)]
    else
        throw(ArgumentError("Unsuported ordering $ordering."))
    end

    return NodalGrid(referencecell, vertices, connectivity)
end
