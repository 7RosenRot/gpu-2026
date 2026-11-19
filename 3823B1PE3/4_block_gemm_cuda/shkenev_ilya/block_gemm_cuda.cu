#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kTileSize = 32;

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

struct DeviceBuffer {
    float* data = nullptr;

    explicit DeviceBuffer(std::size_t bytes) {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes),
                  "cudaMalloc");
    }

    ~DeviceBuffer() { cudaFree(data); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ float tile_a[kTileSize][kTileSize];
    __shared__ float tile_b[kTileSize][kTileSize];
    __shared__ float tile_c[kTileSize][kTileSize];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const std::size_t row = blockIdx.y * kTileSize + ty;
    const std::size_t col = blockIdx.x * kTileSize + tx;
    const std::size_t size = static_cast<std::size_t>(n);
    tile_c[ty][tx] = 0.0f;

    for (std::size_t base = 0; base < size; base += kTileSize) {
        tile_a[ty][tx] = (row < size && base + tx < size)
                            ? a[row * size + base + tx] : 0.0f;
        tile_b[ty][tx] = (base + ty < size && col < size)
                            ? b[(base + ty) * size + col] : 0.0f;
        __syncthreads();

        float sum = tile_c[ty][tx];
#pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            sum = fmaf(tile_a[ty][k], tile_b[k][tx], sum);
        }
        tile_c[ty][tx] = sum;

        __syncthreads();
    }

    if (row < size && col < size) {
        c[row * size + col] = tile_c[ty][tx];
    }
}

}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size must be nonnegative");
    }
    const std::size_t size = static_cast<std::size_t>(n);
    if (size != 0 && size > std::numeric_limits<std::size_t>::max() /
                                sizeof(float) / size) {
        throw std::length_error("Matrix size is too large");
    }
    const std::size_t count = size * size;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Input matrices must contain n * n elements");
    }
    if (count == 0) {
        return {};
    }

    const std::size_t bytes = count * sizeof(float);
    DeviceBuffer device_a(bytes);
    DeviceBuffer device_b(bytes);
    DeviceBuffer device_c(bytes);
    CheckCuda(cudaMemcpy(device_a.data, a.data(), bytes, cudaMemcpyHostToDevice),
              "Copy matrix A to device");
    CheckCuda(cudaMemcpy(device_b.data, b.data(), bytes, cudaMemcpyHostToDevice),
              "Copy matrix B to device");

    const unsigned int tiles = static_cast<unsigned int>(
        (size + kTileSize - 1) / kTileSize);
    const dim3 threads(kTileSize, kTileSize);
    const dim3 blocks(tiles, tiles);
    BlockGemmKernel<<<blocks, threads>>>(device_a.data, device_b.data,
                                        device_c.data, n);
    CheckCuda(cudaGetLastError(), "Launch BlockGemmKernel");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), device_c.data, bytes,
                         cudaMemcpyDeviceToHost), "Copy matrix C to host");
    return result;
}
