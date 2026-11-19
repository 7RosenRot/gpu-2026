#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__global__ void NaiveGemmKernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        const size_t row_start = static_cast<size_t>(row) * n;
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row_start + k] * b[static_cast<size_t>(k) * n + col];
        }
        c[row_start + col] = sum;
    }
}

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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

        dim3 block(32, 8);
        dim3 grid((n + 31) / 32, (n + 7) / 8);
        NaiveGemmKernel<<<grid, block>>>(device_a, device_b, device_c, n);
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
