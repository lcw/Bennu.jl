# Based off of code from julia SparseArrays.jl
struct GeneralSparseMatrixCSC{Tv, Ti<:Integer, VTv<:AbstractVector{Tv}, VTi<:AbstractVector{Ti}} <: SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}
    m::Int       # Number of rows
    n::Int       # Number of columns
    colptr::VTi  # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval::VTi  # Row indices of stored values
    nzval::VTv   # Stored values, typically nonzeros

    function GeneralSparseMatrixCSC{Tv, Ti, VTv, VTi}(m::Integer, n::Integer,
                                                      colptr::AbstractVector{Ti},
                                                      rowval::AbstractVector{Ti},
                                                      nzval::AbstractVector{Tv}) where {Tv, Ti<:Integer,
                                                                                        VTv<:AbstractVector{Tv},
                                                                                        VTi<:AbstractVector{Ti}}
        new(Int(m), Int(n), colptr, rowval, nzval)
    end
end

function GeneralSparseMatrixCSC(m::Integer, n::Integer, colptr::AbstractVector,
                                rowval::AbstractVector, nzval::AbstractVector)
    Tv = eltype(nzval)
    VTv = typeof(nzval)
    Ti = promote_type(eltype(colptr), eltype(rowval))
    VTi = promote_type(typeof(colptr), typeof(rowval))

    GeneralSparseMatrixCSC{Tv, Ti, VTv, VTi}(m, n, colptr, rowval, nzval)
end

SparseArrays.getcolptr(S::GeneralSparseMatrixCSC) = getfield(S, :colptr)
SparseArrays.rowvals(S::GeneralSparseMatrixCSC) = getfield(S, :rowval)
SparseArrays.nonzeros(S::GeneralSparseMatrixCSC) = getfield(S, :nzval)
Base.size(S::GeneralSparseMatrixCSC) = (S.m, S.n)

function GeneralSparseMatrixCSC(S::SparseArrays.AbstractSparseMatrixCSC)
    m, n = size(S)
    colptr = SparseArrays.getcolptr(S)
    rowval = rowvals(S)
    nzval = nonzeros(S)

    return GeneralSparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function Adapt.adapt_structure(to, S::GeneralSparseMatrixCSC)
    m, n = size(S)
    colptr = adapt(to, SparseArrays.getcolptr(S))
    rowval = adapt(to, rowvals(S))
    nzval = adapt(to, nonzeros(S))

    return GeneralSparseMatrixCSC(m, n, colptr, rowval, nzval)
end
