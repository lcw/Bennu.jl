using KernelAbstractions.Extras: @unroll

# Simple wrapper struct that allows us to index a column-wise banded matrix like
# it is a non-banded matrix via views
struct BatchedView{A, I}
    mat::A
    ij::I
    eh::I
end
Base.@propagate_inbounds function Base.getindex(B::BatchedView, u, v)
    return B.mat[B.ij, u, v, B.eh]
end
Base.@propagate_inbounds function Base.setindex!(B::BatchedView, val, u, v)
    return B.mat[B.ij, u, v, B.eh] = val
end

abstract type AbstractBatchedBandedMatrix{Nqh, n, ku, kl, Neh, T, A <: AbstractArray{T, 4}} end
struct BatchedBandedMatrix{Nqh, n, ku, kl, Neh, T, A} <: AbstractBatchedBandedMatrix{Nqh, n, ku, kl, Neh, T, A}
    data::A
end
struct BatchedBandedLU{Nqh, n, ku, kl, Neh, T, A} <: AbstractBatchedBandedMatrix{Nqh, n, ku, kl, Neh, T, A}
    data::A
end
Base.parent(mat::AbstractBatchedBandedMatrix) = mat.data
Base.@propagate_inbounds function Base.getindex(
        B::AbstractBatchedBandedMatrix{Nqh, n, ku, kl}, ij, u, v, eh
    ) where {Nqh, n, ku, kl}
    return parent(B)[ij, ku + 1 + u - v, v, eh]
end
Base.@propagate_inbounds function Base.setindex!(
        B::AbstractBatchedBandedMatrix{Nqh, n, ku, kl}, val, ij, u, v, eh
    ) where {Nqh, n, ku, kl}
    return parent(B)[ij, ku + 1 + u - v, v, eh] = val
end
Base.view(B::AbstractBatchedBandedMatrix, ij, ::Colon, ::Colon, eh) = BatchedView(B, ij, eh)
function Adapt.adapt_structure(to,
        mat::BatchedBandedLU{Nqh, n, ku, kl, Neh, T, A}
    ) where{Nqh, n, ku, kl, Neh, T, A}
    mat = adapt(to, parent(mat))
    B = typeof(mat)
    return BatchedBandedLU{Nqh, n, ku, kl, Neh, T, B}(mat)
end
function Adapt.adapt_structure(to,
        mat::BatchedBandedMatrix{Nqh, n, ku, kl, Neh, T, A}
    ) where{Nqh, n, ku, kl, Neh, T, A}
    mat = adapt(to, parent(mat))
    B = typeof(mat)
    return BatchedBandedMatrix{Nqh, n, ku, kl, Neh, T, B}(mat)
end

"""
    batchedbandedlu!(A::AbstractArray{T, 4} [, kl = div(size(mat, 2) - 1, 2)])

Compute the banded LU factors of the batched matrix `A`. Uses the `A` as storage
for the factors. The optional argument `kl` is the lower bandwidth of the
matrix; upper bandwidth is calculated as `ku = size(mat, 2) - 1 - kl`.

The matrix `A` should have indexing `A[i, b, v, h]` where the indices `i` and
`h` are the batch indices with the size of the first index defining the kernel 
workgroup size the last index the number of workgroups. The index `b` is the
band index for column `v`; storage uses the [LAPACK
format](https://www.netlib.org/lapack/lug/node124.html).

For example the matrix `A` would be stored in banded storage `B` with lower
bandwidth `kl = 2`.

```reply
julia> A = [11 12  0  0  0  0  0
            21 22 23  0  0  0  0
            31 32 33 34  0  0  0
            41 42 43 44 45  0  0
             0 52 53 54 55 56  0
             0  0 63 64 65 66 67
             0  0  0 74 75 76 77]
7×7 Matrix{Int64}:
 11  12   0   0   0   0   0
 21  22  23   0   0   0   0
 31  32  33  34   0   0   0
 41  42  43  44  45   0   0
  0  52  53  54  55  56   0
  0   0  63  64  65  66  67
  0   0   0  74  75  76  77

julia> B = [ 0 12 23 34 45 56 66 67
            11 22 33 44 55 65 77  0
            21 32 43 54 64 76  0  0
            31 42 53 63 75  0  0  0
            41 52  0 74  0  0  0  0]
5×8 Matrix{Int64}:
  0  12  23  34  45  56  66  67
 11  22  33  44  55  65  77   0
 21  32  43  54  64  76   0   0
 31  42  53  63  75   0   0   0
 41  52   0  74   0   0   0   0
```
"""
function batchedbandedlu!(
        mat::A,
        kl = div(size(mat, 2) - 1, 2)
    ) where {T, A <: AbstractArray{T, 4}}
    (Nqh, width, n, Neh) = size(mat)
    ku = width - 1 - kl
    fac = BatchedBandedLU{Nqh, n, ku, kl, Neh, T, A}(mat)
    return batchedbandedlu!(fac)
end

function batchedbandedlu!(
        fac::BatchedBandedLU{Nqh, n, ku, kl, Neh, T, A}
    ) where {Nqh, n, ku, kl, Neh, T, A}
    event = Event(device(A))
    kernel! = bandedlu_kernel!(device(A), (Nqh,))
    event = kernel!(fac; ndrange = (Nqh * Neh,), dependencies = (event,))
    wait(event)
    return fac
end

@kernel function bandedlu_kernel!(
        fac::BatchedBandedLU{Nqh, n, ku, kl, Neh}
    ) where{Nqh, n, ku, kl, Neh}

    @uniform begin
        # center index of band
        c = ku + 1
    end

    # horizonal element number
    eh = @index(Group, Linear)

    # horizontal degree of freedom
    ij = @index(Local, Linear)

    # Create an object that indexes like a matrix for this thread
    U = L = view(fac, ij, :, :, eh)

    # matrix index: (u,v) -> banded index: (c + u - v, u)
    # v is column index
    @inbounds for v = 1:n
        # Get the pivot
        invUvv = 1/U[v, v]

        # Fill L
        @unroll for p = 1:kl
            L[v + p, v] *= invUvv
        end

        # Update U
        for q = 1:ku
            # u is row index
            u = v + q
            # If this row is part of the matrix update it
            if u ≤ n
                # Uvu = U[c - q, u]
                Uvu = U[v, u]
                @unroll for p = 1:kl
                    # U[v + p, u] -= L[v + p, v] * U[v, u]
                    # U[c + p - q, u] -= L[c + p, v] * Uvu
                    U[v + p, u] -= L[v + p, v] * Uvu
                end
            end
        end
    end
end

@kernel function banded_forward_kernel!(
        x_::AbstractArray{T, 3},
        fac::BatchedBandedLU{Nqh, n, ku, kl, Neh, T},
        b_::AbstractArray{T, 3},
    ) where{Nqh, n, ku, kl, Neh, T}

    # private storage for the part of b we are working on
    p_b = @private T (kl + 1)

    # horizonal element number
    eh = @index(Group, Linear)

    # horizontal degree of freedom
    ij = @index(Local, Linear)

    # Create an object that indexes like a matrix for this thread
    L = view(fac, ij, :, :, eh)
    b = view(b_, ij, :, eh)
    x = view(x_, ij, :, eh)

    # Fill the private storage of b
    @inbounds for v = 1:kl+1
        p_b[v] = v ≤ n ? b[v] : -zero(T)
    end

    # Loop over the columns
    @inbounds for v = 1:n
        # Pull out the b associated with v
        x[v] = bv = p_b[1]

        # Loop over the rows
        @unroll for p = 1:kl
            # compute row index from band index
            u = v + p

            # Update element and shift in the private array back one
            Luv = L[u, v]
            p_b[p] = p_b[p + 1] - Luv * bv
        end

        # If we have more elements, get the next value
        if v + kl < n
            p_b[kl + 1] = b[v + kl + 1]
        end
    end
end

@kernel function banded_backward_kernel!(
        x_::AbstractArray{T, 3},
        fac::BatchedBandedLU{Nqh, n, ku, kl, Neh, T},
        b_::AbstractArray{T, 3},
    ) where{Nqh, n, ku, kl, Neh, T}

    # private storage for the part of b we are working on
    p_b = @private T (ku + 1)

    # horizonal element number
    eh = @index(Group, Linear)

    # horizontal degree of freedom
    ij = @index(Local, Linear)

    # Create an object that indexes like a matrix for this thread
    U = view(fac, ij, :, :, eh)
    b = view(b_, ij, :, eh)
    x = view(x_, ij, :, eh)

    # Fill the private storage of b
    @inbounds for q = 1:ku + 1
        v = n + 1 - q
        p_b[q] = v > 0 ? b[v] : -zero(T)
    end

    # Loop over the columns
    @inbounds for v = n:-1:1
        # Scale and store the first element of b
        Uvv = U[v, v]
        x[v] = bv = p_b[1] / Uvv

        # Loop over the rows
        @unroll for q = 1:ku
            # compute row index from band index
            u = v - q

            # Update element and shift in the private array back one
            Uuv = U[u, v]
            p_b[q] = p_b[q + 1] - Uuv * bv
        end

        # If we have more elements, get the next value
        if v - ku > 1
            p_b[ku + 1] = b[v - ku - 1]
        end
    end
end

function LinearAlgebra.ldiv!(
        x::AbstractArray{T, 3},
        fac::BatchedBandedLU{Nqh, n, ku, kl, Neh, T, A},
        b::AbstractArray{T, 3} = x,
    ) where {Nqh, n, ku, kl, Neh, T, A}

    @assert (Nqh, n, Neh) == size(b)
    @assert (Nqh, n, Neh) == size(x)

    event = Event(device(A))
    kernel! = banded_forward_kernel!(device(A), (Nqh,))
    event = kernel!(x, fac, b;
                    ndrange = (Nqh * Neh,), dependencies = (event,))
    kernel! = banded_backward_kernel!(device(A), (Nqh,))
    event = kernel!(x, fac, x;
                    ndrange = (Nqh * Neh,), dependencies = (event,))
    wait(event)
end

function LinearAlgebra.ldiv!(
        x::StructArray,
        fac::BatchedBandedLU{Nqh, n, ku, kl, Neh, T, A},
        b::StructArray
    ) where {Nqh, n, ku, kl, Neh, T, A}

    x_array = reshape(parent(components(x)[1]), Nqh, n, Neh)
    b_array = reshape(parent(components(b)[1]), Nqh, n, Neh)

    # Make sure this is really a fieldarray!
    @assert all(map(y -> pointer(y) === pointer(x_array),
                    parent.(components(x))))
    @assert all(map(y -> pointer(y) === pointer(b_array),
                    parent.(components(b))))

    ldiv!(x_array, fac, b_array)
end
