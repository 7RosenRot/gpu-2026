#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr int tile_size = 16;

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[tile_size][tile_size];
    __shared__ float tile_b[tile_size][tile_size];
    __shared__ float tile_c[tile_size][tile_size];

    const auto x = static_cast<int>(threadIdx.x);
    const auto y = static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * tile_size + x;
    const int row = static_cast<int>(blockIdx.y) * tile_size + y;
    const auto row_offset = static_cast<std::size_t>(row) * n;
    tile_c[y][x] = 0.0f;

    for (int start = 0; start < n; start += tile_size) {
        const int a_col = start + x;
        const int b_row = start + y;
        tile_a[y][x] = row < n && a_col < n ? a[row_offset + a_col] : 0.0f;
        tile_b[y][x] = b_row < n && col < n ? b[static_cast<std::size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();

        float sum = tile_c[y][x];
        #pragma unroll
        for (int k = 0; k < tile_size; ++k) {
            sum = fmaf(tile_a[y][k], tile_b[k][x], sum);
        }
        tile_c[y][x] = sum;
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row_offset + col] = tile_c[y][x];
    }
}

}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size must be non-negative");
    }
    const std::size_t count = static_cast<std::size_t>(n) * n;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Both matrices must contain n * n elements");
    }
    if (count == 0) {
        return {};
    }

    const std::size_t bytes = count * sizeof(float);
    float* allocation = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&allocation), 3 * count * sizeof(float)),
              "Allocate CUDA matrices");
    const std::unique_ptr<float, decltype(&cudaFree)> storage(allocation, cudaFree);
    float* device_a = storage.get();
    float* device_b = device_a + count;
    float* device_c = device_b + count;

    CheckCuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), "Copy matrix A");
    CheckCuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), "Copy matrix B");

    const dim3 block(tile_size, tile_size);
    const unsigned int tiles = (n - 1) / tile_size + 1;
    BlockGemmKernel<<<dim3(tiles, tiles), block>>>(device_a, device_b, device_c, n);
    CheckCuda(cudaGetLastError(), "Launch tiled GEMM kernel");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost), "Copy matrix C");
    return result;
}
