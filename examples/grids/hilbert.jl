using Bennu
using WriteVTK

grid = brickgrid(LobattoCell(5, 4), (-1:1//8:1, -1:1//8:1);
                 ordering=HilbertOrdering())

vtk_grid("grid", grid) do vtk
    vtk["CellNumber"] = 1:length(grid)
end
