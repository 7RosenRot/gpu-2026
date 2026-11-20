#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

constexpr int tile_size = 16;

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[tile_size][tile_size];
    __shared__ float tile_b[tile_size][tile_size];
    __shared__ float tile_c[tile_size][tile_size];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int row = blockIdx.y * tile_size + y;
    int col = blockIdx.x * tile_size + x;
    const size_t row_start = static_cast<size_t>(row) * n;
    tile_c[y][x] = 0.0f;

    for (int offset = 0; offset < n; offset += tile_size) {
        tile_a[y][x] = row < n && offset + x < n ? a[row_start + offset + x] : 0.0f;
        tile_b[y][x] = offset + y < n && col < n ? b[static_cast<size_t>(offset + y) * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < tile_size; ++k) {
            tile_c[y][x] += tile_a[y][k] * tile_b[k][x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row_start + col] = tile_c[y][x];
    }
}

}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n == 0) {
        return {};
    }

    size_t size = static_cast<size_t>(n) * n;
    size_t bytes = size * sizeof(float);
    float* device;
    std::vector<float> output;
    check(cudaMalloc(&device, 3 * bytes));
    float* device_a = device;
    float* device_b = device + size;
    float* device_c = device + 2 * size;
    try {
        check(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block(tile_size, tile_size);
        dim3 grid((n + tile_size - 1) / tile_size, (n + tile_size - 1) / tile_size);
        BlockGemmKernel<<<grid, block>>>(device_a, device_b, device_c, n);
        check(cudaGetLastError());

        output.resize(size);
        check(cudaMemcpy(output.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(device);
        throw;
    }
    check(cudaFree(device));
    return output;
}
