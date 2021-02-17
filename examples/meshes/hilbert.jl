using Bennu
using Meshes: Meshes
using Plots
unicodeplots()

mesh = Meshes.CartesianGrid((4, 4), (10., 10.), (0.1, 0.1))

u = Meshes.coordinates(minimum(mesh))
v = Meshes.coordinates(maximum(mesh))
integercoordinates(c) = quantize.((c .- u) ./ (v .- u))

centroids = Meshes.coordinates.(Meshes.centroid.(Meshes.elements(mesh)))
centroids = centroids[sortperm(hilbertcode.(integercoordinates.(centroids)))]

p = plot(mesh)
for r in partition(1:length(centroids), 5)
  plot!(p, Tuple.(centroids[r]))
end
display(p)
println()
