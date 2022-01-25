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
        A, ::Val{Nqh}, ::Val{n}, ::Val{ku}, ::Val{kl}, ::Val{Ne_h}
    ) where{Nqh, n, ku, kl, Ne_h}

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
                for p = 1:kl
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

    (Nqh, width, n, Ne_h) = size(mat)
    ku = width - 1 - kl
    A = arraytype(mat)

    event = Event(Bennu.device(A))
    kernel! = bandedlu_kernel!(Bennu.device(A), (Nqh,))
    event = kernel!(mat, Val(Nqh), Val(n), Val(ku), Val(kl), Val(Ne_h);
                    ndrange = (Nqh * Ne_h,), dependencies = (event,))
    wait(event)
end
