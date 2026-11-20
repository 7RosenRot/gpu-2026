#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kTile = 32;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ float as[kTile][kTile];
    __shared__ float bs[kTile][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    float sum = 0.0f;
    for (int t = 0; t < n; t += kTile) {
        as[ty][tx] = (row < n && t + tx < n) ? a[static_cast<size_t>(row) * n + t + tx] : 0.0f;
        bs[ty][tx] = (t + ty < n && col < n) ? b[static_cast<size_t>(t + ty) * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum = fmaf(as[ty][k], bs[k][tx], sum);
        }
        __syncthreads();
    }
    if (row < n && col < n) c[static_cast<size_t>(row) * n + col] = sum;
}

struct DeviceBuffers {
    float *a = nullptr, *b = nullptr, *c = nullptr;
    size_t capacity = 0;
    cudaStream_t stream = nullptr;

    void Reserve(size_t elems) {
        if (!stream) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (elems <= capacity) return;
        cudaFree(a); cudaFree(b); cudaFree(c);
        cudaMalloc(&a, elems * sizeof(float));
        cudaMalloc(&b, elems * sizeof(float));
        cudaMalloc(&c, elems * sizeof(float));
        capacity = elems;
    }
    ~DeviceBuffers() {
        cudaFree(a); cudaFree(b); cudaFree(c);
        if (stream) cudaStreamDestroy(stream);
    }
};

DeviceBuffers g_buf;

}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};
    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);
    g_buf.Reserve(elems);

    cudaMemcpyAsync(g_buf.a, a.data(), bytes, cudaMemcpyHostToDevice, g_buf.stream);
    cudaMemcpyAsync(g_buf.b, b.data(), bytes, cudaMemcpyHostToDevice, g_buf.stream);

    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    BlockGemmKernel<<<grid, block, 0, g_buf.stream>>>(g_buf.a, g_buf.b, g_buf.c, n);

    std::vector<float> c(elems);

    cudaMemcpyAsync(c.data(), g_buf.c, bytes, cudaMemcpyDeviceToHost, g_buf.stream);
    cudaStreamSynchronize(g_buf.stream);
    return c;
}
