@testset "device" begin
    @test Bennu.device(ones(3, 5)) isa CPU
    @test Bennu.device(view(ones(3, 5), :)) isa CPU
    if CUDA.has_cuda_gpu()
        @test Bennu.device(CuArray(ones(3, 5))) isa CUDADevice
        @test Bennu.device(view(CuArray(ones(3, 5)), :)) isa CUDADevice
    end
end
