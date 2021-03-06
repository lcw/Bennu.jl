"""
    partition(r, P)

Partition `r` into `P` pieces.

This will provide an equal partition when `N = length(r)` is divisible by `P`
and otherwise the ranges will have lengths of either `div(N, P)` or `cld(N, P)`.

# Examples

We can partition `10:20` into three pieces with
```jldoctest
julia> partition(10:20, 3)
3-element Vector{UnitRange{Int64}}:
 10:12
 13:16
 17:20
```

We can partition `10:2:20` into four pieces with
```jldoctest
julia> partition(10:2:20, 4)
4-element Vector{StepRange{Int64, Int64}}:
 10:2:10
 12:2:14
 16:2:16
 18:2:20
```
"""
partition(r, P) = [partition(r, p, P) for p in 1:P]

"""
    partition(r, p, P)

Partition `r` into `P` pieces and return the `p`th piece.

This will provide an equal partition when `N = length(r)` is divisible by `P`
and otherwise the ranges will have lengths of either `div(N, P)` or `cld(N, P)`.

# Examples

We can get the second piece of `10:20` partitioned into three pieces with
```jldoctest
julia> partition(10:20, 2, 3)
13:16
```
We can get the third piece of `10:2:20` partitioned into four pieces with
```jldoctest
julia> partition(10:2:20, 3, 4)
16:2:16
```
"""
partition(r, p, P) = r[(div((p - 1)*length(r), P) + 1):(div(p*length(r), P))]

"""
    hilbertcode(Y::AbstractArray{T}; bits=8sizeof(T)) where T

Given an array of axes coordinates `Y` stored as `bits`-bit integers
the function returns the associated Hilbert integer `H`.

The encoding of the Hilbert integer is best described by example.
If 5-bits are used from each of 3 coordinates then the function performs

     X[2]|                       H[0] = A B C D E
         | /X[1]       ------->  H[1] = F G H I J
    axes |/                      H[2] = K L M N O
         0------ X[0]                   high low

where the 15-bit Hilbert integer = `A B C D E F G H I J K L M N O` is stored
in `H`.

This function is based on public domain code from John Skilling which can be
found in <https://dx.doi.org/10.1063/1.1751381>.

# Examples
We can generate the two-dimensional Hilbert code for a 1-bit integers with
```jldoctest
julia> hilbertcode([0,0], bits=1)
2-element Vector{Int64}:
 0
 0

julia> hilbertcode([0,1], bits=1)
2-element Vector{Int64}:
 0
 1

julia> hilbertcode([1,1], bits=1)
2-element Vector{Int64}:
 1
 0

julia> hilbertcode([1,0], bits=1)
2-element Vector{Int64}:
 1
 1
```


# References
  John Skilling, "Programming the Hilbert curve",
  AIP Conference Proceedings 707, 381 (2004).
  <https://doi.org/10.1063/1.1751381>
"""
function hilbertcode(Y::AbstractArray{T}; bits=8sizeof(T)) where T
  # Below is Skilling's AxestoTranspose
  X = similar(Y)
  X .= Y
  N = length(X)
  M = one(T) << (bits-1)

  Q = M
  for j = 1:bits-1
    P = Q - one(T)
    for i = 1:N
      if X[i] & Q != zero(T)
        X[1] ⊻= P
      else
        t = (X[1] ⊻ X[i]) & P
        X[1] ⊻= t
        X[i] ⊻= t
      end
    end
    Q >>>= one(T)
  end

  for i = 2:N
    X[i] ⊻= X[i - 1]
  end

  t = zero(T)
  Q = M
  for j = 1:bits-1
    if X[N] & Q != zero(T)
      t ⊻= Q - one(T)
    end
    Q >>>= one(T)
  end

  for i = 1:N
    X[i] ⊻= t
  end

  # Below we transpose X and store it in H, i.e.:
  #
  #   X[0] = A D G J M               H[0] = A B C D E
  #   X[1] = B E H K N   <------->   H[1] = F G H I J
  #   X[2] = C F I L O               H[2] = K L M N O
  #
  # The 15-bit Hilbert integer is then = A B C D E F G H I J K L M N O
  H = zero(X)
  for i = 0:N-1, j = 0:bits-1
    k = i * bits + j
    bit = (X[N - mod(k, N)] >>> div(k, N)) & one(T)
    H[N - i] |= (bit << j)
  end

  return H
end

hilbertperm(indices) = hilbertperm(CartesianIndices(indices))
function hilbertperm(indices::CartesianIndices)
    return sortperm(vec(hilbertcode.(map(v->v.-1, SVector.(Tuple.(indices))))))
end

function hilbertindices(indices)
    CIs = CartesianIndices(indices)
    return reshape(invperm(hilbertperm(CIs)), size(CIs))
end

"""
    quantize([T=UInt64], x)

Quantize a number `x`, between `zero(x)` and `one(x)`, to an integer of type `T`
between `zero(T)` and `typemax(T)`.  If `x` is an array or tuple each element is
quantized.

# Examples
```jldoctest
julia> quantize(0.0)
0x0000000000000000

julia> quantize(UInt32, 0.5f0)
0x7fffffff

julia> quantize((1.0, 0.5))
(0xffffffffffffffff, 0x7fffffffffffffff)

```

"""
function quantize end

quantize(x) = quantize(UInt64, x)
quantize(::Type{T}, x::AbstractArray) where T = quantize.(T, x)
quantize(::Type{T}, x::Tuple) where T = quantize.(T, x)
quantize(::Type{T}, x::Number) where {T <: Integer} =
    convert(T, floor(typemax(T) * BigFloat(x; precision = 16sizeof(T))))
