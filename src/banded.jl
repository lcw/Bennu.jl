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
    get_batched_array(x::StructArray, Nqh, Neh)

Get the array that backs fieldarray `x` with horiztonal workgroup size `Nqh` and
horizonal number workgroups `Neh` in the format used by the batched routines.

The main functionality of this routine is to permuate the vertical dof and field
dimensions for more efficient banded solvers.
"""
function get_batched_array(x::StructArray, Nqh, Neh)
    Nfields = length(eltype(x))
    Nev = div(size(x)[end], Neh)
    Nqv = div(prod(size(x)[1:end-1]), Nqh)
    n = Nev * Nqv * Nfields
    xa = parent(components(x)[1])
    @assert isfieldarray(x)

    # Reshape so that fields come before vertical dofs (smaller bandwidth)
    xa = PermutedDimsArray(reshape(xa, Nqh, Nqv, Nfields, Nev, Neh),
                                (1, 3, 2, 4, 5))

    return reshape(xa, Nqh, n, Neh)
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
        fac::BatchedBandedLU{Nqh, n, ku, kl, Neh, T}
    ) where{Nqh, n, ku, kl, Neh, T}

    @uniform begin
        # center index of band
        c = ku + 1
    end

    # horizonal element number
    eh = @index(Group, Linear)

    # horizontal degree of freedom
    ij = @index(Local, Linear)

    # matrix index: (u,v) -> banded index: (c + u - v, u)
    # v is column index
    @inbounds for v = 1:n
        # Get the pivot
        invUvv = 1/fac.data[ij, ku + 1, v, eh]

        # Fill L
        for p = 1:kl
            fac.data[ij, ku + 1 + p, v, eh] *= invUvv
        end

        # Update U
        for q = 1:ku
            # u is row index
            u = v + q
            # If this row is part of the matrix update it
            if u ≤ n
                # Uvu = U[c - q, u]
                Uvu = fac.data[ij, ku + 1 - q, u, eh]
                for p = 1:kl
                    # U[v + p, u] -= L[v + p, v] * U[v, u]
                    # U[c + p - q, u] -= L[c + p, v] * Uvu
                    fac.data[ij, ku + 1 + p - q, u, eh] -=
                        fac.data[ij, ku + 1 + p, v, eh] * Uvu
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

    # Fill the private storage of b
    @inbounds for v = 1:kl+1
        p_b[v] = v ≤ n ? b_[ij, v, eh] : -zero(T)
    end

    # Loop over the columns
    @inbounds for v = 1:n
        # Loop over the rows
        @unroll for p = 1:kl
            # Update element
            Luv = fac.data[ij, ku + 1 + p, v, eh]
            p_b[p + 1] -= Luv * p_b[1]
        end

        # Pull out the b associated with v
        x_[ij, v, eh] = p_b[1]

        # Loop over the rows
        @unroll for p = 1:kl
            # shift the private array back one
            p_b[p] = p_b[p + 1]
        end

        # If we have more elements, get the next value
        if v + kl < n
            p_b[kl + 1] = b_[ij, v + kl + 1, eh]
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

    # Fill the private storage of b
    @inbounds for q = 1:ku + 1
        v = n + 1 - q
        p_b[q] = v > 0 ? b_[ij, v, eh] : -zero(T)
    end

    # Loop over the columns
    @inbounds for v = n:-1:1
        # Scale and store the first element of b
        Uvv = fac.data[ij, ku + 1, v, eh]
        p_b[1] /= Uvv

        # Loop over the rows
        @unroll for q = 1:ku
            # Update element
            Uuv = fac.data[ij, ku + 1 - q, v, eh]
            p_b[q + 1] -= Uuv * p_b[1]
        end

        x_[ij, v, eh] = p_b[1]

        # Loop over the rows
        @unroll for q = 1:ku
            # shift the private array back one
            p_b[q] = p_b[q + 1]
        end

        # If we have more elements, get the next value
        if v - ku > 1
            p_b[ku + 1] = b_[ij, v - ku - 1, eh]
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

    x_array = get_batched_array(x, Nqh, Neh)
    b_array = get_batched_array(b, Nqh, Neh)

    ldiv!(x_array, fac, b_array)
end

# NOTE: This kernel has not been optimized at all!
@kernel function banded_multiply_kernel!(
        y_::AbstractArray{T, 3},
        mat::BatchedBandedMatrix{Nqh, n, ku, kl, Neh, T},
        x_::AbstractArray{T, 3},
    ) where{Nqh, n, ku, kl, Neh, T}
    # private storage for the part of b we are working on
    # horizonal element number
    eh = @index(Group, Linear)

    # horizontal degree of freedom
    ij = @index(Local, Linear)

    # Create an object that indexes like a matrix for this thread
    A = view(mat, ij, :, :, eh)
    x = view(x_, ij, :, eh)
    y = view(y_, ij, :, eh)

    # Loop over the rows
    @inbounds for u = 1:n
        tmp = -zero(T)
        @unroll for p = 1:kl
            v = u - p
            v > 0 || break
            tmp += A[u, v] * x[v]
        end
        tmp += A[u, u] * x[u]
        @unroll for q = 1:ku
            v = u + q
            v ≤ n || break
            tmp += A[u, v] * x[v]
        end
        y[u] = tmp
    end
end

# NOTE: This not high-performance code!
function LinearAlgebra.mul!(
        y::AbstractArray{T, 3},
        mat::BatchedBandedMatrix{Nqh, n, ku, kl, Neh, T, A},
        x::AbstractArray{T, 3},
        in_event = nothing
    ) where {Nqh, n, ku, kl, Neh, T, A}

    @assert (Nqh, n, Neh) == size(y)
    @assert (Nqh, n, Neh) == size(x)

    event = in_event isa Event ? in_event : Event(device(A))
    kernel! = banded_multiply_kernel!(device(A), (Nqh,))
    event = kernel!(y, mat, x;
                    ndrange = (Nqh * Neh,), dependencies = (event,))
    in_event isa Event || wait(event)
    return in_event isa Event ? event : y
end

function batchedbandedmatrix!(
        mat::A,
        kl = div(size(mat, 2) - 1, 2)
    ) where {T, A <: AbstractArray{T, 4}}
    (Nqh, width, n, Neh) = size(mat)
    ku = width - 1 - kl
    return BatchedBandedMatrix{Nqh, n, ku, kl, Neh, T, A}(mat)
end

@kernel function banded_setvector_kernel!(
        y::AbstractArray{T, 3},
        x::AbstractArray{T, 3},
        w,
        ::Val{width}, 
        ::Val{Nqv},
        ::Val{n}
    ) where{T, width, Nqv, n}

    # horizonal element number
    eh, ev = @index(Group, NTuple)

    # horizontal degree of freedom
    ij, k = @index(Local, NTuple)

    # Loop over the "band" and set the values of `0` or `1`
    @inbounds for p = 0:Nqv:width-1
        v = k + p + (ev - 1) * width

        if v ≤ n
            x[ij, v, eh] = mod1(v, width) == w
            y[ij, v, eh] = 0
        end
    end
end

@kernel function banded_setmatrix_kernel!(
        A_::BatchedBandedMatrix{Nqh, n, ku, kl, Neh, T},
        x_::AbstractArray{T, 3},
        w,
        ::Val{Nqv}
    ) where{Nqh, n, ku, kl, Neh, T, Nqv}

    # horizonal element number
    eh, ev = @index(Group, NTuple)

    # horizontal degree of freedom
    ij, k = @index(Local, NTuple)

    v = w + (ev-1) * (kl + ku + 1)

    A = view(A_, ij, :, :, eh)
    # Using a view for x cause too much register pressure
    # x = view(x_, ij, :, eh)
    # if v is a valid column
    @inbounds if v ≤ n
        # Loop over the band and set the matrix values
        for b = -ku:Nqv:kl
            p = b + k - 1
            u = v + p
            # If inside the matrix copy from x otherwise set 0
            if 1 ≤ u ≤ n && p ≤ kl
               # A[u, v] = x[u]
                A[u, v] = x_[ij, u, eh]
            elseif p ≤ kl
                A[u, v] = 0
            end
        end
    end
end

"""
    batchedbandedmatrix(matvec!, y::AbstractArray{T, 3}, x::AbstractArray{T, 3},
                        kl, ku[, nthreads=1024])

Return the banded matrix with lower and upper bandwidths `kl` and `ku` defined
by the matrix-vector multiplication `matvec!` where `matvec!(y, x)` should set
`y = A * x` where `x` and `y` are batched vectors of with `size(x) == size(y) ==
[Nqh, n, Neh]`.

The optional argument `nthreads` defines the number of total threads to use per
workgroup in the kernel launch.
"""
function batchedbandedmatrix(
        matvec!::Function,
        y::A, x::A, kl, ku,
        nthreads = 1024
    ) where {T, A<:AbstractArray{T, 3}}

    (Nqh, n, Neh) = size(y)
    @assert (Nqh, n, Neh) == size(x)

    width = ku + kl + 1
    data = similar(y, Nqh, width, n, Neh)

    mat = batchedbandedmatrix!(data, kl)

    return batchedbandedmatrix!(matvec!, mat, y, x, nthreads)
end

"""
    batchedbandedmatrix!(matvec!, A::BatchedBandedMatrix, y, x[, nthreads=1024])

Update the banded matrix `A` using the `matvec!` function. See also
[`batchedbandedmatrix`](@ref)
"""
function batchedbandedmatrix!(
        matvec!::Function,
        mat::BatchedBandedMatrix{Nqh, n, ku, kl, Neh, T, AT},
        y::A, x::A,
        nthreads = 1024
    ) where {Nqh, n, ku, kl, Neh, T, A<:AbstractArray{T, 3}, AT}

    @assert (Nqh, n, Neh) == size(y)
    @assert (Nqh, n, Neh) == size(x)
    @assert nthreads >= Nqh

    width = ku + kl + 1
    Nqv = min(width, div(nthreads, Nqh))
    Nev = cld(n, width)

    event = Event(device(AT))
    setvec! = banded_setvector_kernel!(device(AT), (Nqh, Nqv))
    setmat! = banded_setmatrix_kernel!(device(AT), (Nqh, Nqv))
    # we can set every `width` columns in parallel, so we only need to do
    # `width` launches to set the entire matrix as we stride over `width`
    # columns with every work group
    for v = 1:width
        # set ones and zeros
        event = setvec!(y, x, v, Val(width), Val(Nqv), Val(n);
                        ndrange = (Nqh * Neh, Nqv * Nev),
                        dependencies = (event,))

        # multiply by these rows of the identity
        event = matvec!(y, x, event)

        # set the matrix values
        event = setmat!(mat, y, v, Val(Nqv);
                        ndrange = (Nqh * Neh, Nqv * Nev),
                        dependencies = (event,))
        wait(event)
    end
    return mat
end

function batchedbandedmatrix(
        rhs!::Function,
        grid::AbstractGrid,
        y::StructArray,
        x::StructArray,
        eb,
        nthreads = 1024
    ) where {T, S}

    @assert eltype(x) == eltype(y)
    @assert size(x) == size(y)
    @assert isstacked(grid)

    Nfields = _numfields(eltype(y))

    Nqv = size(celltype(grid))[end]
    Nqh = div(length(celltype(grid)), Nqv)
    Nev = stacksize(grid)
    Neh = horizontalsize(grid)

    # Matrix size
    n = Nqv * Nev * Nfields

    @assert length(x) == (Nqv * Nqh * Nev * Neh)

    x_array = get_batched_array(x, Nqh, Neh)
    y_array = get_batched_array(y, Nqh, Neh)

    matvec!(_, _, event) = rhs!(y, x, event)

    # Due to fields being second, we have to expand the bandwidth by one whole
    # element!
    kl = ku = Nqv * Nfields * eb

    return batchedbandedmatrix(matvec!, y_array, x_array, kl, ku, nthreads)
end
