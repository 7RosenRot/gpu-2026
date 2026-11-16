#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>

#include <cstddef>
#include <stdexcept>

namespace {

void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

struct DeviceBuffer {
    float* data = nullptr;

    ~DeviceBuffer() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }
};

struct Max {
    __device__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

template <int BlockSize>
__global__ void SoftmaxRows(float* matrix, std::size_t row_size, bool cache_row) {
    using Reduce = cub::BlockReduce<float, BlockSize>;
    __shared__ typename Reduce::TempStorage reduction;
    __shared__ float row_max;
    __shared__ float inverse_sum;
    extern __shared__ float cached_row[];

    float* row = matrix + static_cast<std::size_t>(blockIdx.x) * row_size;
    float* values = cache_row ? cached_row : row;
    const unsigned int lane = threadIdx.x;

    float local_max = -CUDART_INF_F;
    for (std::size_t col = lane; col < row_size; col += BlockSize) {
        const float value = row[col];
        if (cache_row) {
            cached_row[col] = value;
        }
        local_max = fmaxf(local_max, value);
    }

    const float max_value = Reduce(reduction).Reduce(local_max, Max{});
    if (lane == 0) {
        row_max = max_value;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (std::size_t col = lane; col < row_size; col += BlockSize) {
        const float value = expf(values[col] - row_max);
        values[col] = value;
        local_sum += value;
    }

    const float sum = Reduce(reduction).Sum(local_sum);
    if (lane == 0) {
        inverse_sum = 1.0f / sum;
    }
    __syncthreads();

    for (std::size_t col = lane; col < row_size; col += BlockSize) {
        row[col] = values[col] * inverse_sum;
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.size() % row_count != 0) {
        throw std::invalid_argument("Invalid row count");
    }
    if (input.empty()) {
        return {};
    }

    const std::size_t row_size = input.size() / row_count;
    const std::size_t bytes = input.size() * sizeof(float);
    const bool cache_row = row_size <= 4096;
    const std::size_t shared_bytes = cache_row ? row_size * sizeof(float) : 0;

    DeviceBuffer matrix;
    CheckCuda(cudaMalloc(&matrix.data, bytes));
    CheckCuda(cudaMemcpy(matrix.data, input.data(), bytes, cudaMemcpyHostToDevice));

    if (row_size <= 128) {
        SoftmaxRows<32><<<row_count, 32, shared_bytes>>>(matrix.data, row_size, cache_row);
    } else {
        SoftmaxRows<128><<<row_count, 128, shared_bytes>>>(matrix.data, row_size, cache_row);
    }
    CheckCuda(cudaGetLastError());

    // Allocate host output while the kernel can still be running.
    std::vector<float> output(input.size());
    CheckCuda(cudaMemcpy(output.data(), matrix.data, bytes, cudaMemcpyDeviceToHost));
    return output;
}