@testset "arrays" begin
    @test arraytype(ones(3, 5)) === Array
    if CUDA.has_cuda_gpu()
        @test arraytype(CuArray(ones(3, 5))) === CuArray
    end

    @test [1, 3, 1, 2, 1, 2] == Bennu.numbercontiguous([10, 29, 10, 23, 10, 23])
end
