@testset "arrays" begin
    tup = ntuple(i->SizedArray{Tuple{3,1,2}}(rand(3,1,2)), 4)
    A = StructArray{SVector{4,Float64}}(data = StructArray(tup))
    @test Tullio.storage_type(A) <: Array
    @test Tullio.storage_type(SizedArray{Tuple{3,1,2}}(A)) <: Array
end
