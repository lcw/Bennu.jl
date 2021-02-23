@testset "arrays" begin
    @test eltype(similar(Array{Bool}, Int32, (3,2))) == Int32
    @test similar(Array{Float64}, Float32, (1,2)) isa Array
    @test size(similar(Array{Int}, Int8, (4,2))) == (4,2)

    tup = ntuple(i->SizedArray{Tuple{3,1,2}}(rand(3,1,2)), 4)
    A = StructArray{SVector{4,Float64}}(data = StructArray(tup))
    @test Tullio.storage_type(A) <: Array
    @test Tullio.storage_type(SizedArray{Tuple{3,1,2}}(A)) <: Array
    @test components(A) == tup

    data = ntuple(i->rand(3,4), 4)

    A = Bennu.structarray(data)
    @test all(components(A) .=== data)
    @test A isa StructArray
    @test eltype(A) == SVector{4,Float64}
    @test size(A) == (3,4)

    A = Bennu.structarray(SMatrix{2,2,Float64}, data)
    @test all(components(A) .=== data)
    @test A isa StructArray
    @test eltype(A) == SMatrix{2,2,Float64}
    @test size(A) == (3,4)

    B = similar(A)
    @test B isa StructArray
    @test eltype(B) == SMatrix{2,2,Float64}
    @test size(B) == (3,4)

    B = similar(A, (1,1,3))
    @test B isa StructArray
    @test eltype(B) == SMatrix{2,2,Float64}
    @test size(B) == (1,1,3)

    B = similar(A, SVector{3,Float32})
    @test B isa StructArray
    @test eltype(B) == SVector{3,Float32}
    @test size(B) == (3,4)

    B = similar(A, SVector{3,Float32}, (2,3,1))
    @test B isa StructArray
    @test eltype(B) == SVector{3,Float32}
    @test size(B) == (2,3,1)
end
