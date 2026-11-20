#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;

void Check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

float* g_device = nullptr;
std::size_t g_capacity = 0;

float* GetDeviceBuffer(std::size_t size) {
    if (size > g_capacity) {
        if (g_device != nullptr) {
            cudaFree(g_device);
            g_device = nullptr;
        }
        Check(cudaMalloc(&g_device, size * sizeof(float)), "cudaMalloc");
        g_capacity = size;
    }
    return g_device;
}

template <bool IsMax>
__device__ float BlockReduce(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other = shared[threadIdx.x + stride];
            shared[threadIdx.x] = IsMax ? fmaxf(shared[threadIdx.x], other)
                                        : shared[threadIdx.x] + other;
        }
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

__global__ void SoftmaxKernel(float* data, int row_size) {
    __shared__ float shared[kThreads];
    float* row = data + static_cast<std::size_t>(blockIdx.x) * row_size;

    float local_max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < row_size; i += kThreads) {
        local_max = fmaxf(local_max, row[i]);
    }
    const float row_max = BlockReduce<true>(local_max, shared);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += kThreads) {
        const float e = __expf(row[i] - row_max);
        row[i] = e;
        local_sum += e;
    }
    const float inv_sum = 1.0f / BlockReduce<false>(local_sum, shared);

    for (int i = threadIdx.x; i < row_size; i += kThreads) {
        row[i] *= inv_sum;
    }
}

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const std::size_t size = input.size();
    std::vector<float> output(size);
    if (size == 0 || row_count <= 0) {
        return output;
    }

    const int row_size = static_cast<int>(size / row_count);
    const std::size_t bytes = size * sizeof(float);
    float* device = GetDeviceBuffer(size);

    Check(cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    SoftmaxKernel<<<row_count, kThreads>>>(device, row_size);
    Check(cudaGetLastError(), "SoftmaxKernel");

    Check(cudaMemcpy(output.data(), device, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    return output;
}
