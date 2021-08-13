abstract type AbstractGrid{C <: AbstractCell, N <: Tuple} end

floattype(::Type{<:AbstractGrid{C}}) where {C} = floattype(C)
arraytype(::Type{<:AbstractGrid{C}}) where {C} = arraytype(C)
celltype(::Type{<:AbstractGrid{C}}) where {C} = C
Base.ndims(::Type{<:AbstractGrid{C, N}}) where {C, N} = tuple_length(N)
Base.size(::Type{<:AbstractGrid{C, N}}) where {C, N} = size_to_tuple(N)
Base.length(::Type{<:AbstractGrid{C, N}}) where {C, N} = tuple_prod(N)

floattype(grid::AbstractGrid) = floattype(typeof(grid))
arraytype(grid::AbstractGrid) = arraytype(typeof(grid))
celltype(grid::AbstractGrid) = celltype(typeof(grid))
Base.ndims(grid::AbstractGrid)= Base.ndims(typeof(grid))
Base.size(grid::AbstractGrid) = Base.size(typeof(grid))
Base.length(grid::AbstractGrid) = Base.length(typeof(grid))

function WriteVTK.vtk_grid(filename::AbstractString, grid::AbstractGrid,
                           args...; kwargs...)
    vtk = vtk_grid(filename, points_vtk(grid), cells_vtk(grid), args...;
                   kwargs...)
    data_vtk!(vtk, grid)

    return vtk
end

struct NodalGrid{C <: AbstractCell, N <: Tuple, V, Y, P, Q, F, G, H, B} <: AbstractGrid{C, N}
    referencecell::C
    vertices::V
    connectivity::Y
    points::P
    metrics::Q
    faces::F
    faceindices::G
    facemetrics::H
    boundaryfaces::B
end

function NodalGrid(warp::Function, referencecell, vertices, connectivity;
                   faces=nothing, boundaryfaces=nothing)
    C = typeof(referencecell)
    N = size(connectivity)
    V = typeof(vertices)
    Y = typeof(connectivity)

    points = materializepoints(referencecell, vertices, connectivity)
    points = warp.(points)
    P = typeof(points)

    metrics, facemetrics = materializemetrics(referencecell, points)
    Q = typeof(metrics)
    H = typeof(facemetrics)

    if isnothing(faces)
        faces = materializefaces(referencecell, connectivity)
    end
    F = typeof(faces)

    faceindices = materializefaceindices(referencecell, faces)
    G = typeof(faceindices)

    if isnothing(boundaryfaces)
        boundaryfaces = materializeboundaryfaces(referencecell, faces)
    end
    B = typeof(boundaryfaces)

    types = (V, Y, P, Q, F, G, H, B)
    return NodalGrid{C, Tuple{N...}, types...}(referencecell, vertices,
                                               connectivity, points, metrics,
                                               faces, faceindices, facemetrics,
                                               boundaryfaces)
end

function NodalGrid(referencecell, vertices, connectivity;
                   faces=nothing, boundaryfaces=nothing)
    return NodalGrid(identity, referencecell, vertices, connectivity;
                     faces=faces, boundaryfaces=boundaryfaces)
end

referencecell(grid::NodalGrid) = grid.referencecell
vertices(grid::NodalGrid) = grid.vertices
connectivity(grid::NodalGrid) = grid.connectivity
points(grid::NodalGrid) = grid.points
metrics(grid::NodalGrid) = grid.metrics
faces(grid::NodalGrid) = grid.faces
faceindices(grid::NodalGrid) = grid.faceindices
facemetrics(grid::NodalGrid) = grid.facemetrics
boundaryfaces(grid::NodalGrid) = grid.boundaryfaces

function points_vtk(grid::NodalGrid)
    P = toequallyspaced(referencecell(grid))
    x = P * points(grid)

    return reinterpret(reshape, floattype(grid), vec(adapt(Array, x)))
end

function cells_vtk(grid::NodalGrid)
    type = celltype_vtk(referencecell(grid))
    connectivity = connectivity_vtk(referencecell(grid))

    cells = [MeshCell(type, e * length(connectivity) .+ connectivity)
             for e = 0:length(grid)-1]

    return cells
end

function data_vtk!(vtk, grid::NodalGrid)
    higherorderdegrees = zeros(Int, 3, length(grid))
    ds = [degrees(referencecell(grid))...]
    higherorderdegrees[1:length(ds), :] .= repeat(ds, 1, length(grid))

    vtk["HigherOrderDegrees", VTKCellData()] = higherorderdegrees

    return
end

function materializefaces(referencecell::AbstractCell, connectivity)
    cellfaces = materializefaces(referencecell)[2:end]
    return ntuple(i->connect(cellfaces[i], connectivity), length(cellfaces))
end

function connect(cellfaces, connectivity)
    A = arraytype(connectivity)
    # TODO Should we move this calculation to the GPU?
    connectivity = adapt(Array, connectivity)
    connectivity = reinterpret(reshape, Int, vec(connectivity))

    faces = Array{Int}(undef, size(cellfaces)..., size(connectivity, 2))
    @tullio faces[i, j, k] = connectivity[cellfaces[i, j], k]
    faces = vec(reinterpret(reshape, SVector{size(cellfaces, 1), Int}, faces))

    localfacenumbers = 1:length(faces)
    globalfacenumbers = numbercontiguous(faces; by=sort)
    facepermutations = tuplesortpermutation.(Tuple.(faces))

    M = sparse(localfacenumbers, globalfacenumbers, facepermutations)
    if A <: CuArray
        M = adapt(A, GeneralSparseMatrixCSC(M))
    end
    return M
end

struct FaceConnectionException <: Exception end

@kernel function materializefaceindices!(referencecell::AbstractCell,
                                         faceindices⁺, faceindices⁻,
                                         @Const(globaltolocalfaces))
    @uniform begin
        localfaces = rowvals(globaltolocalfaces)
        vertexpermutations = nonzeros(globaltolocalfaces)
        num_faces = number_of_faces(referencecell)[2]
        num_dof_per_cell = length(referencecell)
        conn = connectivity(referencecell)[2]
        offsets = connectivityoffsets(referencecell, Val(2))
    end

    i, j = @index(Global, NTuple)
    r = nzrange(globaltolocalfaces, j)

    r₁ = first(r)
    r₂ = last(r)

    if r₂ - r₁ ≥ 2 || r₂ - r₁ < 0
        throw(FaceConnectionException())
    end

    e₁, f₁ = divrem(localfaces[r₁] - 1, num_faces) .+ 1
    e₂, f₂ = divrem(localfaces[r₂] - 1, num_faces) .+ 1

    p₂₁ = vertexpermutations[r₁] ∘ inv(vertexpermutations[r₂])
    p₁₂ = vertexpermutations[r₂] ∘ inv(vertexpermutations[r₁])

    if i ≤ length(conn[f₁])
        i₁ = i + offsets[f₁]
        i₂ = i + offsets[f₂]

        j₁ = conn[f₁][i] + (e₁ - 1) * num_dof_per_cell
        j₂ = conn[f₂][i] + (e₂ - 1) * num_dof_per_cell

        j₂₁ = getpermutedindex(conn[f₁], p₂₁, i) .+ (e₁ - 1) * num_dof_per_cell
        j₁₂ = getpermutedindex(conn[f₂], p₁₂, i) .+ (e₂ - 1) * num_dof_per_cell

        faceindices⁻[i₁, e₁] = j₁
        faceindices⁻[i₂, e₂] = j₂

        faceindices⁺[i₁, e₁] = j₁₂
        faceindices⁺[i₂, e₂] = j₂₁
    end
end

function materializefaceindices(referencecell::AbstractCell, faces)
    C = typeof(referencecell)
    A = arraytype(C)
    globaltolocalfaces = first(faces)
    num_localfaces, num_globalfaces = size(globaltolocalfaces)
    num_cells = div(num_localfaces, number_of_faces(C)[2])
    faceconn = connectivity(referencecell)[2]
    max_num_indices_per_face = maximum(length.(faceconn))
    num_facesindices = sum(length.(faceconn))

    faceindices⁺ = fill!(A{Int, 2}(undef, num_facesindices, num_cells), 0)
    faceindices⁻ = fill!(A{Int, 2}(undef, num_facesindices, num_cells), 0)

    kernel = materializefaceindices!(device(A), (max_num_indices_per_face, 5))
    event = Event(device(A))
    event = kernel(referencecell, faceindices⁺, faceindices⁻,
                   globaltolocalfaces;
                   ndrange = (max_num_indices_per_face, num_globalfaces),
                   dependencies = (event,))
    wait(event)

    inidices = (faceindices⁻, faceindices⁺)
    return inidices
end

@kernel function materializeboundaryfaces!(referencecell::AbstractCell,
                                           boundaryfaces,
                                           @Const(globaltolocalfaces))
    @uniform begin
        localfaces = rowvals(globaltolocalfaces)
        vertexpermutations = nonzeros(globaltolocalfaces)
        num_faces = number_of_faces(referencecell)[2]
        num_dof_per_cell = length(referencecell)
        conn = connectivity(referencecell)[2]
        offsets = connectivityoffsets(referencecell, Val(2))
    end

    j = @index(Global)
    r = nzrange(globaltolocalfaces, j)

    r₁ = first(r)
    r₂ = last(r)

    if r₂ - r₁ ≥ 2 || r₂ - r₁ < 0
        throw(FaceConnectionException())
    end

    e₁, f₁ = divrem(localfaces[r₁] - 1, num_faces) .+ 1
    e₂, f₂ = divrem(localfaces[r₂] - 1, num_faces) .+ 1

    if r₁ == r₂
        boundaryfaces[f₁, e₁] = 1
    else
        boundaryfaces[f₁, e₁] = 0
        boundaryfaces[f₂, e₂] = 0
    end
end

function materializeboundaryfaces(referencecell, faces)
    A = arraytype(referencecell)
    globaltolocalfaces = first(faces)
    num_localfaces, num_globalfaces = size(globaltolocalfaces)
    num_cells = div(num_localfaces, number_of_faces(referencecell)[2])
    num_faces = number_of_faces(referencecell)[2]

    boundaryfaces = fill!(A{Int, 2}(undef, num_faces, num_cells), 0)

    kernel = materializeboundaryfaces!(device(A), (256,))
    event = Event(device(A))
    event = kernel(referencecell, boundaryfaces, globaltolocalfaces;
                   ndrange = (num_globalfaces,), dependencies = (event,))
    wait(event)

    return boundaryfaces
end

function faceviews(referencecell, A::AbstractMatrix)
    num_faces = number_of_faces(referencecell)[2]
    offsets = connectivityoffsets(referencecell, Val(2))
    facesizes = size.(connectivity(referencecell)[2])

    if last(offsets) != size(A, 1)
        throw(ArgumentError("The first dimension of A needs to contain the face degrees of freedom."))
    end

    return ntuple(Val(num_faces)) do f
        reshape(view(A, (1+offsets[f]):offsets[f+1], :), facesizes[f]..., :)
    end
end

function min_node_distance(grid::NodalGrid; dims = 1:ndims(grid))
    @assert maximum(dims) <= ndims(grid)

    A = arraytype(grid)
    T = floattype(grid)
    cell = referencecell(grid)

    min_neighbour_distance = similar(points(grid), T)

    Np = length(cell)
    event = min_neighbour_distance_kernel(device(A), 256)(
        min_neighbour_distance, points(grid), Val(strides(cell)), Val(Np), Val(dims);
        ndrange = length(grid) * length(cell))
    wait(event)

    minimum(min_neighbour_distance)
end

@kernel function min_neighbour_distance_kernel(min_neighbour_distance, points,
                                               ::Val{S}, ::Val{Np}, ::Val{dims}) where {S, Np, dims}
    I = @index(Global, Linear)

    @inbounds begin
        e = (I - 1) ÷ Np + 1
        ijk = (I - 1) % Np + 1

        md = typemax(eltype(min_neighbour_distance))
        x⃗ = points[ijk, e]
        for d in dims
            for m in (-1, 1)
                ijknb = ijk + S[d] * m
                    if 1 <= ijknb <= Np
                        x⃗nb = points[ijknb, e]
                        md = min(norm(x⃗ - x⃗nb), md)
                    end
            end
        end
        min_neighbour_distance[ijk, e] = md
    end
end
