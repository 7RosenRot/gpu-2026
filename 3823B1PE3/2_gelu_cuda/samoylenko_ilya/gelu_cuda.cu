#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__global__ void GeluKernel(float* data, size_t size) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    size_t bytes = input.size() * sizeof(float);
    float* device;
    std::vector<float> output;
    check(cudaMalloc(&device, bytes));
    try {
        check(cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice));

        int blocks = static_cast<int>((input.size() + 255) / 256);
        GeluKernel<<<blocks, 256>>>(device, input.size());
        check(cudaGetLastError());

        output.resize(input.size());
        check(cudaMemcpy(output.data(), device, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(device);
        throw;
    }
    check(cudaFree(device));
    return output;
}
