using Bennu
using Meshes
using Plots
unicodeplots()

mesh = CartesianGrid((4, 4), (10., 10.), (0.1, 0.1))

u = coordinates(minimum(mesh))
v = coordinates(maximum(mesh))
integercoordinates(c) = quantize.((c .- u) ./ (v .- u))

centroids = coordinates.(Meshes.centroid.(elements(mesh)))
centroids = centroids[sortperm(hilbertcode.(integercoordinates.(centroids)))]

p = plot(mesh)
for r in partition(1:length(centroids), 5)
  plot!(p, Tuple.(centroids[r]))
end
display(p)
println()
