@kernel function dup_kernel!(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = B[I] + B[I]
end

@kernel function split_kernel!(A, B, @Const(AB))
    I = @index(Global)
    @inbounds A[I] = AB[I].a
    @inbounds B[I] = AB[I].b
    nothing
end

@kernel function join_kernel!(AB, @Const(A), @Const(B))
    I = @index(Global)
    @inbounds AB[I] = (a=A[I], b=B[I])
end

@testset "StructArrays" begin
    TAs = ((Float64,  Array), (BigFloat, Array))
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end

    for (T, A) in TAs
        N = 4
        C = 3
        M = 5

        X = adapt(A, collect(reshape(oneunit(T):N*C*M, N, C, M)))

        tup = ntuple(i->view(X, :, i, :), C)
        b = StructArray{SMatrix{C, 1, T, C}}(tup)
        @test b isa StructArray
        @test Bennu.device(b) == Bennu.device(A)

        aos_b = collect(adapt(Array, b))
        @test aos_b isa Array
        @test Bennu.device(aos_b) == Bennu.device(Array)

        a = similar(b)
        aos_a = copy(aos_b)
        fill!(a, zero(eltype(a)))
        @test all(iszero.(a))
        @test a isa StructArray
        @test Bennu.device(a) == Bennu.device(A)

        a .= b .+ b
        @test isapprox(collect(adapt(Array, a)), aos_b .+ aos_b)

        ab = StructArray((a=a, b=b))
        @test ab isa StructArray
        @test Bennu.device(ab) == Bennu.device(A)
        @test all(ab.a .== a)
        @test all(ab.b .== b)

        ab2 = similar(ab, Int)
        @test eltype(ab2) == Int
        @test ab2 isa A
        @test Bennu.device(ab2) == Bennu.device(A)

        e = b .* 3
        @test isapprox(collect(adapt(Array, e)), aos_b .* 3 )
        @test e isa StructArray
        @test Bennu.device(e) == Bennu.device(A)

        c = norm.(b)
        @test isapprox(collect(adapt(Array, c)), norm.(aos_b))
        if T ≠ BigFloat
            @test c isa A
        end

        for s in ((), (1,), (1,3), (1,3,2))
            d = similar(b, SVector{2, Float32}, s)
            @test size(d) == s
            @test d isa StructArray
            @test Bennu.device(d) == Bennu.device(A)
        end

        a = similar(b)
        fill!(a, zero(eltype(a)))
        event = Event(Bennu.device(A))
        event = dup_kernel!(Bennu.device(A), 256)(a, b, ndrange=length(a),
                                                  dependencies = (event, ))
        wait(event)
        @test isapprox(collect(adapt(Array, a)), aos_b .* 2 )

        fill!(a, zero(eltype(a)))
        fill!(b, zero(eltype(b)))
        event = Event(Bennu.device(A))
        event = split_kernel!(Bennu.device(A), 256)(a, b, ab, ndrange=length(a),
                                                    dependencies = (event, ))
        wait(event)
        @test all(ab.a .== a)
        @test all(ab.b .== b)

        fill!(ab, (a=zero(eltype(a)), b=zero(eltype(b))))
        event = Event(Bennu.device(A))
        event = join_kernel!(Bennu.device(A), 256)(ab, a, b, ndrange=length(a),
                                                   dependencies = (event, ))
        wait(event)
        @test all(ab.a .== a)
        @test all(ab.b .== b)
    end
end
