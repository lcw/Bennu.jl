using KernelAbstractions.Extras: @unroll

# Simple wrapper struct that allows us to index a column-wise banded matrix like
# it is a non-banded matrix
struct BandIndexer{C, A, I, I}
    mat::A
    ij::I
    eh::I
    BandIndexer{C}(mat::A, ij::I, eh::I) where {A, I, C} = new{C, A, I, I}(mat, ij, eh)
end
Base.@propagate_inbounds function Base.getindex(B::BandIndexer{C}, u, v) where {C}
    return B.mat[B.ij, C + u - v, v, B.eh]
end
Base.@propagate_inbounds function Base.setindex!(B::BandIndexer{C}, val, u, v) where {C}
    return B.mat[B.ij, C + u - v, v, B.eh] = val
end

@kernel function bandedlu_kernel!(
        A, ::Val{Nqh}, ::Val{n}, ::Val{ku}, ::Val{kl}, ::Val{Neh}
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
    U = L = BandIndexer{c}(A, ij, eh)

    # matrix index: (u,v) -> banded index: (c + u - v, u)
    # v is column index
    @inbounds for v = 1:n
        # Get the pivot
        invUvv = 1/U[v, v]

        # Fill L
        for p = 1:kl
            # L[v + p, v] = U[v + p, v] / U[v, v]
            # L[c + p, v] *= invUvv
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

function bandedlu!(mat, kl = div(size(mat, 2) - 1, 2))
    @assert ndims(mat) == 4

    (Nqh, width, n, Neh) = size(mat)
    ku = width - 1 - kl
    A = arraytype(mat)

    event = Event(device(A))
    kernel! = bandedlu_kernel!(device(A), (Nqh,))
    event = kernel!(mat, Val(Nqh), Val(n), Val(ku), Val(kl), Val(Neh);
                    ndrange = (Nqh * Neh,), dependencies = (event,))
    wait(event)
end

@kernel function banded_forward_kernel!(
        x_::AbstractArray{T, 3},
        fac::AbstractArray{T, 4},
        b_::AbstractArray{T, 3},
        ::Val{Nqh}, ::Val{n}, ::Val{ku}, ::Val{kl}, ::Val{Neh}
    ) where{Nqh, n, ku, kl, Neh, T}

    # private storage for the part of b we are working on
    p_b = @private T (kl + 1)

    # horizonal element number
    eh = @index(Group, Linear)

    # horizontal degree of freedom
    ij = @index(Local, Linear)

    # Create an object that indexes like a matrix for this thread
    L = BandIndexer{ku + 1}(fac, ij, eh)
    b = view(b_, ij, :, eh)
    x = view(x_, ij, :, eh)

    # Fill the private storage of b
    for v = 1:kl+1
        p_b[v] = v ≤ n ? b[v] : -zero(T)
    end

    # Loop over the columns
    for v = 1:n
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
        fac::AbstractArray{T, 4},
        b_::AbstractArray{T, 3},
        ::Val{Nqh}, ::Val{n}, ::Val{ku}, ::Val{kl}, ::Val{Neh}
    ) where{Nqh, n, ku, kl, Neh, T}

    # private storage for the part of b we are working on
    p_b = @private T (ku + 1)

    # horizonal element number
    eh = @index(Group, Linear)

    # horizontal degree of freedom
    ij = @index(Local, Linear)

    # Create an object that indexes like a matrix for this thread
    U = BandIndexer{ku + 1}(fac, ij, eh)
    b = view(b_, ij, :, eh)
    x = view(x_, ij, :, eh)

    # Fill the private storage of b
    for q = 1:ku + 1
        v = n + 1 - q
        p_b[q] = v > 0 ? b[v] : -zero(T)
    end

    # Loop over the columns
    for v = n:-1:1
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

function bandedsolve!(x::AbstractArray{T, 3},
        fac::AbstractArray{T, 4},
        b::AbstractArray{T, 3},
        kl = div(size(fac, 2) - 1, 2)
    ) where {T}

    (Nqh, width, n, Neh) = size(fac)
    ku = width - 1 - kl
    @assert (Nqh, n, Neh) == size(b)
    @assert (Nqh, n, Neh) == size(x)

    event = Event(device(fac))
    kernel! = banded_forward_kernel!(device(fac), (Nqh,))
    event = kernel!(x, fac, b,
                    Val(Nqh), Val(n), Val(ku), Val(kl), Val(Neh);
                    ndrange = (Nqh * Neh,), dependencies = (event,))
    kernel! = banded_backward_kernel!(device(fac), (Nqh,))
    event = kernel!(x, fac, x,
                    Val(Nqh), Val(n), Val(ku), Val(kl), Val(Neh);
                    ndrange = (Nqh * Neh,), dependencies = (event,))
    wait(event)
end
