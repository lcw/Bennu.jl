@testset "permutations" begin
    for N = 1:4
        ps = collect(permutations(1:N))
        perminversion = [findfirst(isequal(invperm(p)), ps) for p in ps]
        permcomposition = [findfirst(isequal(p[q]), ps) for p in ps, q in ps]

        for i = 1:length(ps), j = 1:length(ps)
            @test inv(Bennu.Permutation{N}(i)) ==
                      Bennu.Permutation{N}(perminversion[i])
            @test Bennu.Permutation{N}(i) ∘ Bennu.Permutation{N}(j) ==
                      Bennu.Permutation{N}(permcomposition[i, j])
        end
    end

    for js in ([1,2,3,4], [1,3,2,4], [2,1,4,3], [2,4,1,3], [3,1,4,2], [3,4,1,2],
           [4,2,3,1], [4,3,2,1])
        p = inv(Bennu.tuplesortpermutation(js...))
        @test js == Bennu.getpermutedindex([1 3; 2 4], p, 1:4)
    end

    is = collect(LinearIndices((1:3,1:2)))
    @test [1,2,3,4,5,6] == Bennu.getpermutedindex(is, Bennu.Permutation{4}(1), 1:6)
    @test [1,4,2,5,3,6] == Bennu.getpermutedindex(is, Bennu.Permutation{4}(3), 1:6)
    @test [3,2,1,6,5,4] == Bennu.getpermutedindex(is, Bennu.Permutation{4}(8), 1:6)
    @test [3,6,2,5,1,4] == Bennu.getpermutedindex(is, Bennu.Permutation{4}(11), 1:6)
    @test [4,1,5,2,6,3] == Bennu.getpermutedindex(is, Bennu.Permutation{4}(14), 1:6)
    @test [4,5,6,1,2,3] == Bennu.getpermutedindex(is, Bennu.Permutation{4}(17), 1:6)
    @test [6,3,5,2,4,1] == Bennu.getpermutedindex(is, Bennu.Permutation{4}(22), 1:6)
    @test [6,5,4,3,2,1] == Bennu.getpermutedindex(is, Bennu.Permutation{4}(24), 1:6)

    is = collect(LinearIndices((1:3,)))
    @test [1,2,3] == Bennu.getpermutedindex(is, Bennu.Permutation{2}(1), 1:3)
    @test [3,2,1] == Bennu.getpermutedindex(is, Bennu.Permutation{2}(2), 1:3)
end
