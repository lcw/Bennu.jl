using Bennu
using StaticArrays
using CUDA
using Printf
using LinearAlgebra
using WriteVTK
CUDA.allowscalar(false)

# advection velocity
const v = SVector(1, 1)

# Gaussian initial condition
function gaussian(x, t)
  FT = eltype(x)
  xp = mod1.(x .- v .* t, FT(2π))
  xc = SVector{2, FT}(π, π)
  r² = norm(xp .- xc) ^ 2
  exp(-2r²)
end

# sine initial condition
sineproduct(x, t) = prod(sin.(x .- v .* t))

function run(solution, FT, A, N, K; outputvtk=false, vtkdir="output")
  Nq = N + 1
  
  cell = LobattoCell{FT, A}(Nq, Nq)

  vert1d = range(FT(0), stop=FT(2π), length=K+1)

  function meshwarp(x)
    x₁, x₂ = x
    x̃₁ = x₁ + sin(x₁ / 2) * sin(x₂)
    x̃₂ = x₂ - sin(x₁) * sin(x₂ / 2) / 2
    SVector(x̃₁, x̃₂)
  end

  grid = brickgrid(meshwarp, cell, (vert1d, vert1d); periodic=(true, true))

  g, J = components(metrics(grid))

  g₁₁, g₁₂, g₂₁, g₂₂ = components(g)

  M = mass(cell)
  D₁, D₂ = derivatives(cell)
  MJ = M * J
  
  facenormal, faceJ = components(facemetrics(grid))
  faceM = facemass(cell)
  faceMJ = faceM * faceJ
  faceix⁻, faceix⁺ = faceindices(grid)

  facedata = (faceix⁻, faceix⁺, faceMJ, facenormal) 

  rhs! = function(dq, q)
    # volume term
    dq .-= (v[1] .* g₁₁ .+ v[2] .* g₂₁) .* (D₁ * q) +
           (v[2] .* g₂₂ .+ v[1] .* g₁₂) .* (D₂ * q)
    # surface term
    for (f⁻, f⁺, fMJ, n) in zip(faceviews.(Ref(cell), facedata)...)
      dq[f⁻] .-= fMJ .* dot.(n, Ref(v)) .* (q[f⁺] .- q[f⁻]) ./ 2 ./ MJ[f⁻]
    end
  end
  
  timeend = FT(2π)
 
  # crude dt estimate
  cfl = 1 // 10
  dx = Base.step(vert1d)
  dt = cfl * dx / N

  numberofsteps = ceil(Int, timeend / dt)
  dt = timeend / numberofsteps

  RKA = (
         FT(0),
         FT(-567301805773 // 1357537059087),
         FT(-2404267990393 // 2016746695238),
         FT(-3550918686646 // 2091501179385),
         FT(-1275806237668 // 842570457699),
        )
  RKB = (
         FT(1432997174477 // 9575080441755),
         FT(5161836677717 // 13612068292357),
         FT(1720146321549 // 2090206949498),
         FT(3134564353537 // 4481467310338),
         FT(2277821191437 // 14882151754819),
        )
  
  if outputvtk
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end

  do_output = function(step, time, q)
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0 
      filename = "step$(lpad(step, 6, '0'))"
      vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
      P = Bennu.toequallyspaced(cell)
      vtkfile["q"] = vec(Array(P * q))
      vtk_save(vtkfile)
      pvd[time] = vtkfile
    end
  end
  
  # initialize state
  q = solution.(points(grid), FT(0))

  # storage for rhs
  dq = similar(q)
  dq .= 0
  
  # initial output
  step = 0
  time = FT(0)
  do_output(step, time, q)
 
  ### time integration
  for step = 1:numberofsteps
    if time + dt > timeend
      dt = timeend - time
    end
    for stage = 1:length(RKA)
      dq *= RKA[stage]
      rhs!(dq, q)
      q .+= RKB[stage] * dt * dq
    end
    time += dt
    do_output(step, time, q)
  end

  # final output
  do_output(numberofsteps, timeend, q)
  outputvtk && vtk_save(pvd)
 
  # compute error
  qexact = solution.(points(grid), timeend)
  errf = sqrt(sum(MJ .* (q .- qexact) .^ 2))
end

let
  FT = Float64
  N = 4

  # run on the GPU if possible
  A = CUDA.has_cuda_gpu() ? CuArray : Array
  
  @info """Configuration:
  precision        = $FT 
  polynomial order = $N
  array type       = $A
  """

  # visualize solution of advected gaussian
  
  K = 16
  vtkdir = "vtk_advection_K$(K)x$(K)"
  @info "Starting Gaussian advection with ($K, $K) elements"
  run(gaussian, FT, A, N, K; outputvtk=true, vtkdir)
  @info "Finished, vtk output written to $vtkdir"

  # run convergence study using a simple sine field
  
  @info "Starting convergence study"
  numlevels = 5
  err = zeros(FT, numlevels)
  for l in 1:numlevels
    K = 4 * 2 ^ (l - 1)
    err[l] = run(sineproduct, FT, A, N, K)
    @info @sprintf("Level %d, elements = (%2d, %2d), error = %.16e", l, K, K, err[l])
  end
  rates = log2.(err[1:numlevels-1] ./ err[2:numlevels])
  @info "Convergence rates:\n" *
    join(["rate for levels $l → $(l + 1) = $(rates[l])"
          for l in 1:(numlevels - 1)], "\n")
end
