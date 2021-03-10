@testset "tuples" begin
    for n = 1:4
        for t in unique(combinations(repeat(1:n, n), n))
            @test all(Bennu.tuplesort(tuple(t...)) .== sort(t))
            @test all(t[collect(Bennu.tuplesortperm(tuple(t...)))] .== sort(t))
        end

        for t in permutations(1:n)
            @test all(Bennu.tuplesortperm(tuple(t...)) .== sortperm(t))
        end

        @test Bennu.permutationtuples(n) == Tuple(Tuple.(collect(permutations(1:n))))
    end

    @test (1, 2, false) == Bennu.minmaxflip(1, 2)
    @test (1, 1, false) == Bennu.minmaxflip(1, 1)
    @test (1, 2, true) == Bennu.minmaxflip(2, 1)
end
