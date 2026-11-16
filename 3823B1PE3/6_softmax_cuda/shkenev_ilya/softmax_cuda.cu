#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

struct Workspace {
    float* data = nullptr;
    std::size_t capacity = 0;
    int device = 0;

    Workspace() {
        CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    }

    ~Workspace() {
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cudaFree(data);
        if (previous != device) {
            cudaSetDevice(previous);
        }
    }

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity) {
            return;
        }
        float* next = nullptr;
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&next), bytes),
                  "cudaMalloc");
        const cudaError_t status = cudaFree(data);
        if (status != cudaSuccess) {
            cudaFree(next);
            CheckCuda(status, "cudaFree");
        }
        data = next;
        capacity = bytes;
    }
};

template <bool Maximum>
__device__ float Reduce(float value, float* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            if constexpr (Maximum) {
                scratch[tid] = fmaxf(scratch[tid], scratch[tid + offset]);
            } else {
                scratch[tid] += scratch[tid + offset];
            }
        }
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

template <int Items>
__global__ void SoftmaxKernel(const float* input, float* output,
                              std::size_t row_size) {
    __shared__ float scratch[kThreads];
    const int tid = threadIdx.x;
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * row_size;
    float values[Items > 0 ? Items : 1];
    float maximum = -CUDART_INF_F;

    if constexpr (Items > 0) {
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            const std::size_t col = tid + i * kThreads;
            values[i] = col < row_size ? input[start + col] : -CUDART_INF_F;
            maximum = fmaxf(maximum, values[i]);
        }
    } else {
        for (std::size_t col = tid; col < row_size; col += kThreads) {
            maximum = fmaxf(maximum, input[start + col]);
        }
    }
    maximum = Reduce<true>(maximum, scratch);

    float sum = 0.0f;
    if constexpr (Items > 0) {
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            values[i] = expf(values[i] - maximum);
            sum += values[i];
        }
    } else {
        for (std::size_t col = tid; col < row_size; col += kThreads) {
            sum += expf(input[start + col] - maximum);
        }
    }
    sum = Reduce<false>(sum, scratch);
    const float inverse = 1.0f / sum;

    if constexpr (Items > 0) {
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            const std::size_t col = tid + i * kThreads;
            if (col < row_size) {
                output[start + col] = values[i] * inverse;
            }
        }
    } else {
        for (std::size_t col = tid; col < row_size; col += kThreads) {
            output[start + col] = expf(input[start + col] - maximum) * inverse;
        }
    }
}

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0) {
        throw std::invalid_argument("Row count must be positive");
    }
    const std::size_t rows = static_cast<std::size_t>(row_count);
    if (input.size() % rows != 0) {
        throw std::invalid_argument("Input size must be divisible by row count");
    }
    if (input.empty()) {
        return {};
    }
    if (input.size() > std::numeric_limits<std::size_t>::max() /
                           (2 * sizeof(float))) {
        throw std::length_error("Input is too large");
    }

    int device = 0;
    CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    static thread_local std::map<int, Workspace> workspaces;
    Workspace& workspace = workspaces.try_emplace(device).first->second;
    const std::size_t bytes = input.size() * sizeof(float);
    const std::size_t row_size = input.size() / rows;
    workspace.Reserve(2 * bytes);
    float* device_input = workspace.data;
    float* device_output = device_input + input.size();
    CheckCuda(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice),
              "Copy input to device");

    if (row_size <= kThreads) {
        SoftmaxKernel<1><<<row_count, kThreads>>>(device_input, device_output, row_size);
    } else if (row_size <= 4 * kThreads) {
        SoftmaxKernel<4><<<row_count, kThreads>>>(device_input, device_output, row_size);
    } else if (row_size <= 16 * kThreads) {
        SoftmaxKernel<16><<<row_count, kThreads>>>(device_input, device_output, row_size);
    } else if (row_size <= 64 * kThreads) {
        SoftmaxKernel<64><<<row_count, kThreads>>>(device_input, device_output, row_size);
    } else {
        SoftmaxKernel<0><<<row_count, kThreads>>>(device_input, device_output, row_size);
    }
    CheckCuda(cudaGetLastError(), "Launch SoftmaxKernel");

    std::vector<float> result(input.size());
    CheckCuda(cudaMemcpy(result.data(), device_output, bytes, cudaMemcpyDeviceToHost),
              "Copy result to host");
    return result;
}
