#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void NaiveGemmKernel(const float* a, const float* b, float* c, int n) {
    const auto col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= n || col >= n) {
        return;
    }

    const auto row_offset = static_cast<std::size_t>(row) * n;
    float sum = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a[row_offset + k], b[static_cast<std::size_t>(k) * n + col], sum);
    }
    c[row_offset + col] = sum;
}

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
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

    const dim3 block(32, 8);
    const dim3 grid((n - 1) / block.x + 1, (n - 1) / block.y + 1);
    NaiveGemmKernel<<<grid, block>>>(device_a, device_b, device_c, n);
    CheckCuda(cudaGetLastError(), "Launch naive GEMM kernel");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost), "Copy matrix C");
    return result;
}
