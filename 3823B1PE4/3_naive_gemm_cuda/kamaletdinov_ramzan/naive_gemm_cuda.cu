#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;

    const float* a_row = a + static_cast<size_t>(row) * n;
    const float* b_col = b + col;
    float sum = 0.0f;
#pragma unroll 8
    for (int k = 0; k < n; ++k) {
        sum = fmaf(__ldg(a_row + k), __ldg(b_col + static_cast<size_t>(k) * n), sum);
    }
    c[static_cast<size_t>(row) * n + col] = sum;
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

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};
    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);
    g_buf.Reserve(elems);

    cudaMemcpyAsync(g_buf.a, a.data(), bytes, cudaMemcpyHostToDevice, g_buf.stream);
    cudaMemcpyAsync(g_buf.b, b.data(), bytes, cudaMemcpyHostToDevice, g_buf.stream);

    const dim3 block(32, 8);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    NaiveGemmKernel<<<grid, block, 0, g_buf.stream>>>(g_buf.a, g_buf.b, g_buf.c, n);

    std::vector<float> c(elems);

    cudaMemcpyAsync(c.data(), g_buf.c, bytes, cudaMemcpyDeviceToHost, g_buf.stream);
    cudaStreamSynchronize(g_buf.stream);
    return c;
}
