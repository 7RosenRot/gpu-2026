#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <float.h>

__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int row_size, int row_count) {
    int row_idx = blockIdx.x;
    if (row_idx >= row_count) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const float* d_row_in = input + row_idx * row_size;
    float* d_row_out = output + row_idx * row_size;

    extern __shared__ float s_data[];

    float local_max = -FLT_MAX;
    for (int i = tid; i < row_size; i += bdim) {
        local_max = fmaxf(local_max, d_row_in[i]);
    }
    s_data[tid] = local_max;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = s_data[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += bdim) {
        float val = __expf(d_row_in[i] - row_max);
        d_row_out[i] = val;
        local_sum += val;
    }
    s_data[tid] = local_sum;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = s_data[0];
    __syncthreads();

    float inv_sum = 1.0f / (row_sum + 1e-20f);
    for (int i = tid; i < row_size; i += bdim) {
        d_row_out[i] *= inv_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }

    size_t total_elements = input.size();
    size_t row_size = total_elements / row_count;
    size_t size_bytes = total_elements * sizeof(float);

    std::vector<float> output(total_elements);

    float *d_input = nullptr;
    float *d_output = nullptr;

    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, size_bytes);

    cudaMemcpy(d_input, input.data(), size_bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    if (row_size < 256) {
        threads_per_block = 32;
        while (threads_per_block < row_size) {
            threads_per_block *= 2;
        }
    }

    size_t shared_mem_bytes = threads_per_block * sizeof(float);

    dim3 blocks(row_count);
    dim3 threads(threads_per_block);

    softmax_kernel<<<blocks, threads, shared_mem_bytes>>>(d_input, d_output, row_size, row_count);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}