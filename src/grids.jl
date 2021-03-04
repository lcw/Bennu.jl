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

struct NodalGrid{C <: AbstractCell, N <: Tuple, V, Y, P} <: AbstractGrid{C,  N}
    referencecell::C
    vertices::V
    connectivity::Y
    points::P
end

function NodalGrid(referencecell, vertices, connectivity)
    C = typeof(referencecell)
    N = size(connectivity)
    V = typeof(vertices)
    Y = typeof(connectivity)

    points = pointsfromvertices(referencecell, vertices, connectivity)
    P = typeof(points)

    return NodalGrid{C, Tuple{N...}, V, Y, P}(referencecell, vertices,
                                              connectivity, points)
end

referencecell(grid::NodalGrid) = grid.referencecell
vertices(grid::NodalGrid) = grid.vertices
connectivity(grid::NodalGrid) = grid.connectivity
points(grid::NodalGrid) = grid.points

function points_vtk(grid::NodalGrid)
    P = toequallyspaced(referencecell(grid))
    x = P * points(grid)

    return reinterpret(reshape, floattype(grid), vec(adapt(Array, x)))
end

function cells_vtk(grid::NodalGrid)
    celltype = celltype_vtk(referencecell(grid))
    cellconnectivity = connectivity_vtk(referencecell(grid))

    cells = MeshCell[]
    offset = 0
    for e = 1:length(grid)
        push!(cells,  MeshCell(celltype, offset .+ cellconnectivity))
        offset += length(cellconnectivity)
    end

    return cells
end

function data_vtk!(vtk, grid::NodalGrid)
    higherorderdegrees = zeros(Int, 3, length(grid))
    ds = [degrees(referencecell(grid))...]
    higherorderdegrees[1:length(ds), :] .= repeat(ds, 1, length(grid))

    vtk["HigherOrderDegrees", VTKCellData()] = higherorderdegrees

    return
end
