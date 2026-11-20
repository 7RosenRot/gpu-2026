#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#include <math_constants.h>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void SoftmaxKernel(float* values, std::size_t row_size, int row_count) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;

    for (std::size_t row = blockIdx.x; row < static_cast<std::size_t>(row_count); row += gridDim.x) {
        float* data = values + row * row_size;
        float largest = -CUDART_INF_F;
        for (std::size_t col = lane; col < row_size; col += blockDim.x) {
            largest = fmaxf(largest, data[col]);
        }
        partial[lane] = largest;
        __syncthreads();

        for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                partial[lane] = fmaxf(partial[lane], partial[lane + offset]);
            }
            __syncthreads();
        }
        const float row_max = partial[0];
        __syncthreads();

        float sum = 0.0f;
        for (std::size_t col = lane; col < row_size; col += blockDim.x) {
            const float value = expf(data[col] - row_max);
            data[col] = value;
            sum += value;
        }
        partial[lane] = sum;
        __syncthreads();

        for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                partial[lane] += partial[lane + offset];
            }
            __syncthreads();
        }
        const float inverse_sum = 1.0f / partial[0];
        for (std::size_t col = lane; col < row_size; col += blockDim.x) {
            data[col] *= inverse_sum;
        }
        __syncthreads();
    }
}

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() && row_count >= 0) {
        return {};
    }
    if (row_count <= 0 || input.size() % static_cast<std::size_t>(row_count) != 0) {
        throw std::invalid_argument("Input must contain a whole number of elements per row");
    }
    const std::size_t row_size = input.size() / row_count;

    const std::size_t bytes = input.size() * sizeof(float);
    float* allocation = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&allocation), bytes), "Allocate CUDA softmax buffer");
    const std::unique_ptr<float, decltype(&cudaFree)> buffer(allocation, cudaFree);
    CheckCuda(cudaMemcpy(buffer.get(), input.data(), bytes, cudaMemcpyHostToDevice), "Copy softmax input");

    unsigned int threads = 32;
    while (threads < row_size && threads < 256) {
        threads *= 2;
    }
    const int blocks = row_count < 65535 ? row_count : 65535;
    SoftmaxKernel<<<blocks, threads, threads * sizeof(float)>>>(buffer.get(), row_size, row_count);
    CheckCuda(cudaGetLastError(), "Launch softmax kernel");

    std::vector<float> result(input.size());
    CheckCuda(cudaMemcpy(result.data(), buffer.get(), bytes, cudaMemcpyDeviceToHost), "Copy softmax output");
    return result;
}
