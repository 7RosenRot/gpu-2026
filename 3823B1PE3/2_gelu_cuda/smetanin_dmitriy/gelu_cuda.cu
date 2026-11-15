#include "gelu_cuda.h"

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

__global__ void GeluKernel(float* values, std::size_t size) {
    constexpr float scale = 0.7978845608028654f;
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t i = first; i < size; i += stride) {
        const float x = values[i];
        const float z = scale * x * fmaf(0.044715f * x, x, 1.0f);
        const float e = expf(-2.0f * fabsf(z));
        const float weight = (x < 0.0f ? e : 1.0f) / (1.0f + e);
        values[i] = isinf(x) && x < 0.0f ? -0.0f : x * weight;
    }
}

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const std::size_t bytes = input.size() * sizeof(float);
    float* allocation = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&allocation), bytes), "Allocate CUDA GELU buffer");
    const std::unique_ptr<float, decltype(&cudaFree)> buffer(allocation, cudaFree);
    CheckCuda(cudaMemcpy(buffer.get(), input.data(), bytes, cudaMemcpyHostToDevice), "Copy GELU input");

    constexpr unsigned int threads = 256;
    constexpr unsigned int max_blocks = 4096;
    const std::size_t needed_blocks = (input.size() - 1) / threads + 1;
    const auto blocks = static_cast<unsigned int>(needed_blocks < max_blocks ? needed_blocks : max_blocks);
    GeluKernel<<<blocks, threads>>>(buffer.get(), input.size());
    CheckCuda(cudaGetLastError(), "Launch GELU kernel");

    std::vector<float> result(input.size());
    CheckCuda(cudaMemcpy(result.data(), buffer.get(), bytes, cudaMemcpyDeviceToHost), "Copy GELU output");
    return result;
}
