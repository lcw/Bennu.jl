@testset "partition" begin
    @test partition(1:1,1) == [1:1]
    @test partition(1:1,1,1) == 1:1
    @test partition(3:3:37,1) == [3:3:37]
    @test partition(3:3:37,1,1) == 3:3:37
    @test vcat(partition(5:10:99, 7)...) == collect(5:10:99)

    let
        r = 5:10:99
        N = length(r)
        P = 7
        for p = 1:7
            @test length(partition(r, p, P)) in (div(N, P), cld(N, P))
        end
    end
end

@testset "Hilbert code" begin
    @test hilbertcode([0,0], bits=1) == [0, 0]
    @test hilbertcode([0,1], bits=1) == [0, 1]
    @test hilbertcode([1,1], bits=1) == [1, 0]
    @test hilbertcode([1,0], bits=1) == [1, 1]
    @test hilbertcode([0,0], bits=2) == [0, 0]
    @test hilbertcode([1,0], bits=2) == [0, 1]
    @test hilbertcode([1,1], bits=2) == [0, 2]
    @test hilbertcode([0,1], bits=2) == [0, 3]
    @test hilbertcode([0,2], bits=2) == [1, 0]
    @test hilbertcode([0,3], bits=2) == [1, 1]
    @test hilbertcode([1,3], bits=2) == [1, 2]
    @test hilbertcode([1,2], bits=2) == [1, 3]
    @test hilbertcode([2,2], bits=2) == [2, 0]
    @test hilbertcode([2,3], bits=2) == [2, 1]
    @test hilbertcode([3,3], bits=2) == [2, 2]
    @test hilbertcode([3,2], bits=2) == [2, 3]
    @test hilbertcode([3,1], bits=2) == [3, 0]
    @test hilbertcode([2,1], bits=2) == [3, 1]
    @test hilbertcode([2,0], bits=2) == [3, 2]
    @test hilbertcode([3,0], bits=2) == [3, 3]

    @test hilbertcode(UInt64.([14,3,4])) == UInt64.([0x0,0x0,0xe25])
end
