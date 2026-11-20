#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("NaiveGemmCUDA: ") + cudaGetErrorString(status));
    }
}

struct DeviceMemory {
    float* data = nullptr;

    explicit DeviceMemory(std::size_t bytes) {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes));
    }

    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    ~DeviceMemory() {
        if (data != nullptr) cudaFree(data);
    }
};

__global__ void GemmKernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y * 4 + threadIdx.y;
    const std::size_t size = static_cast<std::size_t>(n);
    if (col >= size || row >= size) return;

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < n; ++k) {
        const float value = b[static_cast<std::size_t>(k) * size + col];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const std::size_t current_row = row + r * blockDim.y;
            if (current_row < size) {
                sums[r] = fmaf(a[current_row * size + k], value, sums[r]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const std::size_t current_row = row + r * blockDim.y;
        if (current_row < size) c[current_row * size + col] = sums[r];
    }
}

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
    if (n < 0) throw std::invalid_argument("Matrix size must be nonnegative");
    const std::size_t size = static_cast<std::size_t>(n);
    if (size != 0 && size > std::numeric_limits<std::size_t>::max() / size) {
        throw std::overflow_error("Matrix size overflow");
    }
    const std::size_t count = size * size;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Input sizes must equal n*n");
    }
    if (count == 0) return {};
    if (count > std::numeric_limits<std::size_t>::max() / (3 * sizeof(float))) {
        throw std::overflow_error("Device allocation size overflow");
    }

    const std::size_t bytes = count * sizeof(float);
    DeviceMemory memory(3 * bytes);
    float* device_a = memory.data;
    float* device_b = device_a + count;
    float* device_c = device_b + count;
    CheckCuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 threads(32, 8);
    const dim3 blocks(static_cast<unsigned int>((size + 31) / 32),
                      static_cast<unsigned int>((size + 31) / 32));
    GemmKernel<<<blocks, threads>>>(device_a, device_b, device_c, n);
    CheckCuda(cudaGetLastError());

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    return result;
}
