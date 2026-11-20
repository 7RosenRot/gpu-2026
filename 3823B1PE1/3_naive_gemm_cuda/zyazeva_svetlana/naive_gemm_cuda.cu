#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void GemmKernel(
    const float* A,
    const float* B,
    float* C,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (row >= n || col + 3 >= n) {
        return;
    }

    float4 sum = make_float4(0, 0, 0, 0);

    #pragma unroll 4
    for (int k = 0; k < n; k++) {

        float a = A[row * n + k];

        float4 b = *reinterpret_cast<const float4*>(
            &B[k * n + col]
        );

        sum.x += a * b.x;
        sum.y += a * b.y;
        sum.z += a * b.z;
        sum.w += a * b.w;
    }

    *reinterpret_cast<float4*>(
        &C[row * n + col]
    ) = sum;
}


std::vector<float> NaiveGemmCUDA(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n
) {
    if (n == 0) {
        return {};
    }

    size_t bytes = n * n * sizeof(float);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(32, 8);

    dim3 blocks(
        (n / 4 + threads.x - 1) / threads.x,
        (n + threads.y - 1) / threads.y
    );

    GemmKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    std::vector<float> result(n * n);

    cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}
