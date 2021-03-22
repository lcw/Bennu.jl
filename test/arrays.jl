@testset "arrays" begin
    @test arraytype(ones(3, 5)) === Array
    if CUDA.has_cuda_gpu()
        @test arraytype(CuArray(ones(3, 5))) === CuArray
    end
end
