#include "softmax_cuda.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdexcept>

namespace {

constexpr int max_threads = 256;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__global__ void SoftmaxKernel(float* data, size_t row_size) {
    __shared__ float shared_vals[max_threads];
    int thread = threadIdx.x;
    float* row = data + blockIdx.x * row_size;

    float maximum = -CUDART_INF_F;
    for (size_t col = thread; col < row_size; col += blockDim.x) {
        maximum = fmaxf(maximum, row[col]);
    }
    shared_vals[thread] = maximum;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (thread < step) {
            shared_vals[thread] = fmaxf(shared_vals[thread], shared_vals[thread + step]);
        }
        __syncthreads();
    }
    const float row_max = shared_vals[0];
    __syncthreads();

    float sum = 0.0f;
    for (size_t col = thread; col < row_size; col += blockDim.x) {
        row[col] = expf(row[col] - row_max);
        sum += row[col];
    }
    shared_vals[thread] = sum;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (thread < step) {
            shared_vals[thread] += shared_vals[thread + step];
        }
        __syncthreads();
    }

    const float inverse_sum = 1.0f / shared_vals[0];
    for (size_t col = thread; col < row_size; col += blockDim.x) {
        row[col] *= inverse_sum;
    }
}

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty()) {
        return {};
    }

    const size_t bytes = input.size() * sizeof(float);
    float* device;
    std::vector<float> output;
    check(cudaMalloc(&device, bytes));
    try {
        check(cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice));

        const size_t row_size = input.size() / row_count;
        const int threads = static_cast<int>(std::min<size_t>(row_size, max_threads));
        SoftmaxKernel<<<row_count, threads>>>(device, row_size);
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
