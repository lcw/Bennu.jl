@testset "sparse arrays" begin
    S = sparse([1],[2],[3])
    G = Bennu.GeneralSparseMatrixCSC(S)
    H = adapt(Array, G)

    for A = (G, H)
        @test size(S) == size(A)
        @test SparseArrays.getcolptr(S) == SparseArrays.getcolptr(A)
        @test rowvals(S) == rowvals(A)
        @test nonzeros(S) == nonzeros(A)
    end
end
