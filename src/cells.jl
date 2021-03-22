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

function lobattooperators_1d(::Type{T}, ::Type{A}, M) where {T, A}
    points, weights  = legendregausslobatto(BigFloat, M)
    derivative = spectralderivative(points)
    equallyspacedpoints = range(-one(BigFloat), stop=one(BigFloat), length=M)
    toequallyspaced = spectralinterpolation(points, equallyspacedpoints)

    points = adapt(A, Array{T}(points))
    weights = adapt(A, Array{T}(weights))
    derivative = adapt(A, Array{T}(derivative))
    toequallyspaced = adapt(A, Array{T}(toequallyspaced))

    return (points=points, weights=weights, derivative=derivative,
            toequallyspaced=toequallyspaced)
end

struct LobattoCell{T, A, S, N, O, P, D, M, E} <: AbstractCell{T, A, S, N}
    points_1d::O
    weights_1d::O
    points::P
    derivatives::D
    mass::M
    toequallyspaced::E
end

function LobattoCell{T, A}(dims...) where {T, A}
    N = length(dims)
    if all(dims[1] .== dims)
        oall = lobattooperators_1d(T, A, first(dims))
        o = ntuple(i->oall, N)
    else
        o = ntuple(i->lobattooperators_1d(T, A, dims[i]), N)
    end

    points_1d = ntuple(i->reshape(o[i].points,
                                  ntuple(j->ifelse(i==j, dims[i], 1), N)), N)
    weights_1d = ntuple(i->reshape(o[i].weights,
                                   ntuple(j->ifelse(i==j, dims[i], 1), N)), N)

    points = vec(SVector.(points_1d...))
    # TODO Should we use a struct of arrays style layout?
    # if isbitstype(T) && N > 1
    #     # Setup struct of arrays style layout of the points
    #     points = reinterpret(reshape, T, points)
    #     points = permutedims(points, (2, 1))
    #     points = PermutedDimsArray(points, (2, 1))
    #     points = reinterpret(reshape, SVector{N, T}, points)
    # end

    derivatives = ntuple(i->Kron(reverse(ntuple(j->ifelse(i==j,
                                                          o[i].derivative,
                                                          Eye{T}(dims[j])),
                                                N))...), N)

    mass = Diagonal(vec(.*(weights_1d...)))

    toequallyspaced = Kron(reverse(ntuple(i->o[i].toequallyspaced, N))...)

    LobattoCell{T, A, Tuple{dims...}, N,
                typeof.((points_1d, points, derivatives, mass,
                         toequallyspaced))...}(points_1d, weights_1d, points,
                                               derivatives, mass,
                                               toequallyspaced)
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
    r = points_1d(referencecell)
    p = similar(vertices, SVector{1, T},
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
    r = vec.(points_1d(referencecell))
    p = similar(vertices, SVector{2, T},
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
    r = vec.(points_1d(referencecell))
    p = similar(vertices, SVector{3, T},
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
