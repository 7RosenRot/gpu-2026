#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("GeluCUDA: ") + cudaGetErrorString(status));
    }
}

struct DeviceBuffer {
    float* data = nullptr;
    std::size_t capacity = 0;
    int device = -1;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data != nullptr) {
            int current = -1;
            if (cudaGetDevice(&current) == cudaSuccess) {
                if (cudaSetDevice(device) == cudaSuccess) cudaFree(data);
                cudaSetDevice(current);
            }
        }
    }

    void Reserve(std::size_t size) {
        int current = -1;
        CheckCuda(cudaGetDevice(&current));
        if (data != nullptr && device != current) {
            CheckCuda(cudaSetDevice(device));
            const cudaError_t status = cudaFree(data);
            const cudaError_t restore = cudaSetDevice(current);
            data = nullptr;
            capacity = 0;
            CheckCuda(restore);
            CheckCuda(status);
        }
        if (size > capacity) {
            if (data != nullptr) {
                CheckCuda(cudaFree(data));
                data = nullptr;
                capacity = 0;
            }
            CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(float)));
            capacity = size;
            device = current;
        }
    }
};

__global__ void GeluKernel(float* data, std::size_t size) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        const float x = data[i];
        constexpr float sqrt_two_over_pi = 0.7978845608028654f;
        const float z = sqrt_two_over_pi * (x + 0.044715f * x * x * x);
        const float e = expf(-2.0f * fabsf(z));
        data[i] = x >= 0.0f ? x / (1.0f + e) : x * e / (1.0f + e);
    }
}

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) return {};
    thread_local DeviceBuffer buffer;
    buffer.Reserve(input.size());
    const std::size_t bytes = input.size() * sizeof(float);
    CheckCuda(cudaMemcpy(buffer.data, input.data(), bytes, cudaMemcpyHostToDevice));
    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>(
        std::min<std::size_t>((input.size() - 1) / threads + 1, 65535));
    GeluKernel<<<blocks, threads>>>(buffer.data, input.size());
    CheckCuda(cudaGetLastError());

    std::vector<float> result(input.size());
    CheckCuda(cudaMemcpy(result.data(), buffer.data, bytes, cudaMemcpyDeviceToHost));
    return result;
}
