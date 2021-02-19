@testset "arrays" begin
    @test eltype(similar(Array{Bool}, Int32, (3,2))) == Int32
    @test similar(Array{Float64}, Float32, (1,2)) isa Array
    @test size(similar(Array{Int}, Int8, (4,2))) == (4,2)

    tup = ntuple(i->SizedArray{Tuple{3,1,2}}(rand(3,1,2)), 4)
    A = StructArray{SVector{4,Float64}}(data = StructArray(tup))
    @test Tullio.storage_type(A) <: Array
    @test Tullio.storage_type(SizedArray{Tuple{3,1,2}}(A)) <: Array
    @test components(A) == tup
end
