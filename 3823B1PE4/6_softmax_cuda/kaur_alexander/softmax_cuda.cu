#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int row_count,
                               int row_size) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= row_count) return;

    const float* row_in = input + row * row_size;
    float* row_out = output + row * row_size;

    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_in[i]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_sum += expf(row_in[i] - row_max);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];

    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < row_size; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    int total_size = input.size();
    int row_size = total_size / row_count;
    std::vector<float> output(total_size);

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));

    cudaMemcpy(d_input, input.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = row_count;
    size_t shared_mem = threads * sizeof(float);

    softmax_kernel<<<blocks, threads, shared_mem>>>(d_input, d_output, row_count, row_size);

    cudaMemcpy(output.data(), d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}