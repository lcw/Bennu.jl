abstract type AbstractCell{T, A <: AbstractArray, S <: Tuple, N} end

floattype(::Type{<:AbstractCell{T}}) where {T} = T
arraytype(::Type{<:AbstractCell{T, A}}) where {T, A} = A
Base.ndims(::Type{<:AbstractCell{T, A, S, N}}) where {T, A, S, N} = N
Base.size(::Type{<:AbstractCell{T, A, S}}) where {T, A, S} = size_to_tuple(S)
Base.length(::Type{<:AbstractCell{T, A, S}}) where {T, A, S} = tuple_prod(S)

floattype(cell::AbstractCell) = floattype(typeof(cell))
arraytype(cell::AbstractCell) = arraytype(typeof(cell))
Base.ndims(cell::AbstractCell) = Base.ndims(typeof(cell))
Base.size(cell::AbstractCell) = Base.size(typeof(cell))
Base.length(cell::AbstractCell) = Base.length(typeof(cell))

function lobattooperators_1d(::Type{T}, M) where {T}
    points, weights  = legendregausslobatto(BigFloat, M)
    derivative = spectralderivative(points)
    equallyspacedpoints = range(-one(BigFloat), stop=one(BigFloat), length=M)
    toequallyspaced = spectralinterpolation(points, equallyspacedpoints)

    return map(Array{T}, (points=points, weights=weights, derivative=derivative,
                          toequallyspaced=toequallyspaced))
end

struct LobattoCell{T, A, S, N, O, P, D, M, E, C} <: AbstractCell{T, A, S, N}
    points_1d::O
    weights_1d::O
    points::P
    derivatives::D
    mass::M
    toequallyspaced::E
    connectivity::C
end

function LobattoCell{T, A}(dims...) where {T, A}
    N = length(dims)
    if all(dims[1] .== dims)
        oall = adapt(A, lobattooperators_1d(T, first(dims)))
        o = ntuple(i->oall, N)
    else
        o = ntuple(i->adapt(A, lobattooperators_1d(T, dims[i])), N)
    end

    points_1d = ntuple(N) do i
        return reshape(o[i].points, ntuple(j->ifelse(i==j, dims[i], 1), N))
    end
    weights_1d = ntuple(N) do i
        return reshape(o[i].weights, ntuple(j->ifelse(i==j, dims[i], 1), N))
    end

    points = vec(SVector.(points_1d...))
    # TODO Should we use a struct of arrays style layout?
    # if isbitstype(T) && N > 1
    #     # Setup struct of arrays style layout of the points
    #     points = reinterpret(reshape, T, points)
    #     points = permutedims(points, (2, 1))
    #     points = PermutedDimsArray(points, (2, 1))
    #     points = reinterpret(reshape, SVector{N, T}, points)
    # end

    derivatives = ntuple(N) do i
        tup = ntuple(j->ifelse(i==j, o[i].derivative, Eye{T}(dims[j])), N)
        return Kron(reverse(tup)...)
    end

    mass = Diagonal(vec(.*(weights_1d...)))

    toequallyspaced = Kron(reverse(ntuple(i->o[i].toequallyspaced, N))...)

    connectivity = adapt(A, materializeconnectivity(LobattoCell, dims...))

    args = (points_1d, weights_1d, points, derivatives, mass, toequallyspaced,
            connectivity)
    LobattoCell{T, A, Tuple{dims...}, N, typeof.(args[2:end])...}(args...)
end

function Adapt.adapt_structure(to,
                               cell::LobattoCell{T, A, S, N}) where {T, A, S, N}
    names = fieldnames(LobattoCell)
    args = ntuple(j->adapt(to, getfield(cell, names[j])), length(names))
    B = arraytype(to)

    LobattoCell{T, B, S, N, typeof.(args[2:end])...}(args...)
end

LobattoCell{T}(dims...) where {T} = LobattoCell{T, Array}(dims...)
LobattoCell(dims...) = LobattoCell{Float64}(dims...)

const LobattoLine{T, A} = LobattoCell{T, A, Tuple{B}} where {B}
const LobattoQuad{T, A} = LobattoCell{T, A, Tuple{B, C}} where {B, C}
const LobattoHex{T, A} = LobattoCell{T, A, Tuple{B, C, D}} where {B, C, D}

points_1d(cell::LobattoCell) = cell.points_1d
weights_1d(cell::LobattoCell) = cell.weights_1d
points(cell::LobattoCell) = cell.points
derivatives(cell::LobattoCell) = cell.derivatives
mass(cell::LobattoCell) = cell.mass
toequallyspaced(cell::LobattoCell) = cell.toequallyspaced
connectivity(cell::LobattoCell) = cell.connectivity
degrees(cell::LobattoCell) = size(cell) .- 1

number_of_faces(cell::LobattoCell) = number_of_faces(typeof(cell))
number_of_faces(::Type{<:LobattoLine}) = (1, 2)
number_of_faces(::Type{<:LobattoQuad}) = (1, 4, 4)
number_of_faces(::Type{<:LobattoHex}) = (1, 6, 12, 8)

celltype_vtk(::LobattoLine) = VTKCellTypes.VTK_LAGRANGE_CURVE
celltype_vtk(::LobattoQuad) = VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL
celltype_vtk(::LobattoHex)  = VTKCellTypes.VTK_LAGRANGE_HEXAHEDRON

function connectivity_vtk(cell::LobattoLine)
    L = LinearIndices(size(cell))
    return [
            L[1],      # corners
            L[end],
            L[2:end-1]..., # interior
           ]
end

function connectivity_vtk(cell::LobattoQuad)
    L = LinearIndices(size(cell))
    return [
            L[1,     1], # corners
            L[end,   1],
            L[end, end],
            L[1,   end],
            L[2:end-1,       1]..., # edges
            L[end,     2:end-1]...,
            L[2:end-1,     end]...,
            L[1,       2:end-1]...,
            L[2:end-1, 2:end-1]..., # interior
           ]
end

function connectivity_vtk(cell::LobattoHex)
    L = LinearIndices(size(cell))
    return [
            L[  1,   1,   1], # corners
            L[end,   1,   1],
            L[end, end,   1],
            L[  1, end,   1],
            L[  1,   1, end],
            L[end,   1, end],
            L[end, end, end],
            L[  1, end, end],
            L[2:end-1,       1,       1]..., # edges
            L[    end, 2:end-1,       1]...,
            L[2:end-1,     end,       1]...,
            L[      1, 2:end-1,       1]...,
            L[2:end-1,       1,     end]...,
            L[    end, 2:end-1,     end]...,
            L[2:end-1,     end,     end]...,
            L[      1, 2:end-1,     end]...,
            L[      1,       1, 2:end-1]...,
            L[    end,       1, 2:end-1]...,
            L[      1,     end, 2:end-1]...,
            L[    end,     end, 2:end-1]...,
            L[      1, 2:end-1, 2:end-1]..., # faces
            L[    end, 2:end-1, 2:end-1]...,
            L[2:end-1,       1, 2:end-1]...,
            L[2:end-1,     end, 2:end-1]...,
            L[2:end-1, 2:end-1,       1]...,
            L[2:end-1, 2:end-1,     end]...,
            L[2:end-1, 2:end-1, 2:end-1]..., # interior
           ]
end

function materializepoints(referencecell::LobattoLine, vertices, connectivity)
    T = floattype(referencecell)
    A = arraytype(referencecell)
    r = points_1d(referencecell)
    p = fieldarray(undef, SVector{1, T}, A,
                   (size(referencecell)..., length(connectivity)))
    connectivity = vec(connectivity)
    vertices = vec(vertices)

    @tullio p[i, e] =
        @inbounds(begin
                      c1, c2 = connectivity[e]
                      ri = $(r[1])[i]

                      ((1 - ri) * vertices[c1] +
                       (1 + ri) * vertices[c2]) / 2
                  end) (i in axes(p, 1), e in axes(p, 2))

    return reshape(p, (length(referencecell), length(connectivity)))
end

function materializepoints(referencecell::LobattoQuad, vertices, connectivity)
    T = floattype(referencecell)
    A = arraytype(referencecell)
    r = vec.(points_1d(referencecell))
    p = fieldarray(undef, SVector{2, T}, A,
                   (size(referencecell)..., length(connectivity)))

    connectivity = vec(connectivity)
    vertices = vec(vertices)

    @tullio p[i, j, e] =
        @inbounds(begin
                      c1, c2, c3, c4 = connectivity[e]
                      ri, rj = $(r[1])[i], $(r[2])[j]

                      ((1 - ri) * (1 - rj) * vertices[c1] +
                       (1 + ri) * (1 - rj) * vertices[c2] +
                       (1 - ri) * (1 + rj) * vertices[c3] +
                       (1 + ri) * (1 + rj) * vertices[c4]) / 4
                  end) (i in axes(p, 1), j in axes(p, 2), e in axes(p, 3))

    return reshape(p, (length(referencecell), length(connectivity)))
end

function materializepoints(referencecell::LobattoHex, vertices, connectivity)
    T = floattype(referencecell)
    A = arraytype(referencecell)
    r = vec.(points_1d(referencecell))
    p = fieldarray(undef, SVector{3, T}, A,
                   (size(referencecell)..., length(connectivity)))

    connectivity = vec(connectivity)
    vertices = vec(vertices)

    @tullio p[i, j, k, e] =
        @inbounds(begin
                      c1, c2, c3, c4, c5, c6, c7, c8 = connectivity[e]
                      ri, rj, rk = $(r[1])[i], $(r[2])[j], $(r[3])[k]

                      ((1 - ri) * (1 - rj) * (1 - rk) * vertices[c1] +
                       (1 + ri) * (1 - rj) * (1 - rk) * vertices[c2] +
                       (1 - ri) * (1 + rj) * (1 - rk) * vertices[c3] +
                       (1 + ri) * (1 + rj) * (1 - rk) * vertices[c4] +
                       (1 - ri) * (1 - rj) * (1 + rk) * vertices[c5] +
                       (1 + ri) * (1 - rj) * (1 + rk) * vertices[c6] +
                       (1 - ri) * (1 + rj) * (1 + rk) * vertices[c7] +
                       (1 + ri) * (1 + rj) * (1 + rk) * vertices[c8]) / 8
                  end) (i in axes(p, 1), j in axes(p, 2), k in axes(p, 3),
                        e in axes(p, 4))

    return reshape(p, (length(referencecell), length(connectivity)))
end

materializefaces(cell::AbstractCell) = materializefaces(typeof(cell))
function materializefaces(::Type{<:LobattoLine})
    return (
            SA[1; 2], # edge
            SA[1 2], # corners
           )
end

function materializefaces(::Type{<:LobattoQuad})
    return (
            SA[1; 2; 3; 4; 5; 6; 7; 8], # face
            SA[1 2 1 3;
               3 4 2 4], # edges
            SA[1 2 3 4] # corners
           )
end

function materializefaces(::Type{<:LobattoHex})
    return (
            SA[1; 2; 3; 4; 5; 6; 7; 8], # volume
            SA[1 2 1 3 1 5; 3 4 2 4 2 6; 5 6 5 7 3 7; 7 8 6 8 4 8], # faces
            SA[1 3 5 7 1 2 5 6 1 2 3 4;
               2 4 6 8 3 4 7 8 5 6 7 8], # edges
            SA[1 2 3 4 5 6 7 8] # corners
           )
end

function connectivityoffsets(cell::LobattoCell, ::Val{N}) where {N}
    connectivityoffsets(typeof(cell), Val(N))
end
connectivityoffsets(::Type{C}, ::Val{1}) where {C<:LobattoLine} = (0,)
connectivityoffsets(::Type{C}, ::Val{2}) where {C<:LobattoLine} = (0, 1)

connectivityoffsets(::Type{C}, ::Val{1}) where {C<:LobattoQuad} = (0,)
function connectivityoffsets(::Type{C}, ::Val{2}) where {C<:LobattoQuad}
    L, M = size(C)
    return (0, M, 2M, 2M+L)
end
connectivityoffsets(::Type{C}, ::Val{3}) where {C<:LobattoQuad} = (0, 1, 2, 3)

connectivityoffsets(::Type{C}, ::Val{1}) where {C<:LobattoHex} = (0,)
function connectivityoffsets(::Type{C}, ::Val{2}) where {C<:LobattoHex}
    L, M, N = size(C)
    return cumsum((0, M*N, M*N, L*N, L*N, L*M))
end
function connectivityoffsets(::Type{C}, ::Val{3}) where {C<:LobattoHex}
    L, M, N = size(C)
    return cumsum((0, N, N, N, N, M, M, M, M, L, L, L))
end
function connectivityoffsets(::Type{C}, ::Val{4}) where {C<:LobattoHex}
    return (0, 1, 2, 3, 4, 5, 6, 7, 8)
end

function materializeconnectivity(::Type{<:LobattoCell}, L::Integer)
    indices = collect(LinearIndices((L,)))

    conn = (
            (indices,), # edge
            ( # corners
             indices[1],
             indices[end]
            )
           )

    return conn
end

function materializeconnectivity(::Type{<:LobattoCell}, L::Integer, M::Integer)
    indices = collect(LinearIndices((L, M)))

    conn = (
            (indices,), # face
            ( # edges
             (indices[1,     1:end]),
             (indices[end,   1:end]),
             (indices[1:end,     1]),
             (indices[1:end,   end])
            ),
            ( # corners
             (indices[  1,   1]),
             (indices[end,   1]),
             (indices[  1, end]),
             (indices[end, end])
            )
           )

    return conn
end

function materializeconnectivity(::Type{<:LobattoCell}, L::Integer, M::Integer,
                                 N::Integer)
    indices = collect(LinearIndices((L, M, N)))

    conn = (
            (indices,), # volume
            ( # faces
             (indices[    1, 1:end, 1:end]),
             (indices[  end, 1:end, 1:end]),
             (indices[1:end,     1, 1:end]),
             (indices[1:end,   end, 1:end]),
             (indices[1:end, 1:end,     1]),
             (indices[1:end, 1:end,   end])
            ),
            ( # edges
             (indices[    1,     1, 1:end]),
             (indices[  end,     1, 1:end]),
             (indices[    1,   end, 1:end]),
             (indices[  end,   end, 1:end]),
             (indices[    1, 1:end,     1]),
             (indices[  end, 1:end,     1]),
             (indices[    1, 1:end,   end]),
             (indices[  end, 1:end,   end]),
             (indices[1:end,     1,     1]),
             (indices[1:end,   end,     1]),
             (indices[1:end,     1,   end]),
             (indices[1:end,   end,   end])
            ),
            ( # corners
             (indices[  1,   1,   1]),
             (indices[end,   1,   1]),
             (indices[  1, end,   1]),
             (indices[end, end,   1]),
             (indices[  1,   1, end]),
             (indices[end,   1, end]),
             (indices[  1, end, end]),
             (indices[end, end, end])
            )
           )

    return conn
end
