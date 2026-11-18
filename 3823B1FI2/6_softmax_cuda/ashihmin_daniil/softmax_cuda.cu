#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

namespace {

constexpr int WARP_SIZE = 32;

__inline__ __device__ float WarpReduceMax(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float WarpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float BlockReduceMax(float val) {
    __shared__ float shared[WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;

    val = WarpReduceMax(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    const int num_warps = blockDim.x / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : -FLT_MAX;

    if (wid == 0) {
        val = WarpReduceMax(val);
        if (threadIdx.x == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    return shared[0];
}

__inline__ __device__ float BlockReduceSum(float val) {
    __shared__ float shared[WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;

    val = WarpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    const int num_warps = blockDim.x / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = WarpReduceSum(val);
        if (threadIdx.x == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    return shared[0];
}

__global__ void SoftmaxSharedKernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int row_size) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_in = input + static_cast<size_t>(row) * row_size;
    float* row_out = output + static_cast<size_t>(row) * row_size;

    float local_max = -FLT_MAX;

    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = row_in[i];
        sdata[i] = val;
        local_max = fmaxf(local_max, val);
    }

    float row_max = (blockDim.x <= WARP_SIZE) ? WarpReduceMax(local_max) : BlockReduceMax(local_max);
    if (blockDim.x <= WARP_SIZE) {
        row_max = __shfl_sync(0xffffffff, row_max, 0);
    }

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float exp_val = __expf(sdata[i] - row_max);
        sdata[i] = exp_val;
        local_sum += exp_val;
    }

    float row_sum = (blockDim.x <= WARP_SIZE) ? WarpReduceSum(local_sum) : BlockReduceSum(local_sum);
    if (blockDim.x <= WARP_SIZE) {
        row_sum = __shfl_sync(0xffffffff, row_sum, 0);
    }

    const float inv_sum = 1.0f / row_sum;

    for (int i = tid; i < row_size; i += blockDim.x) {
        row_out[i] = sdata[i] * inv_sum;
    }
}

__global__ void SoftmaxLargeRowKernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int row_size) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float4* row_in4 = reinterpret_cast<const float4*>(input + static_cast<size_t>(row) * row_size);
    float4* row_out4 = reinterpret_cast<float4*>(output + static_cast<size_t>(row) * row_size);
    const int vec_size = row_size / 4;

    float local_max = -FLT_MAX;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = row_in4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    const float row_max = BlockReduceMax(local_max);

    float local_sum = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = row_in4[i];
        local_sum += __expf(v.x - row_max) + __expf(v.y - row_max) +
                     __expf(v.z - row_max) + __expf(v.w - row_max);
    }
    const float row_sum = BlockReduceSum(local_sum);
    const float inv_sum = 1.0f / row_sum;

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = row_in4[i];
        float4 out;
        out.x = __expf(v.x - row_max) * inv_sum;
        out.y = __expf(v.y - row_max) * inv_sum;
        out.z = __expf(v.z - row_max) * inv_sum;
        out.w = __expf(v.w - row_max) * inv_sum;
        row_out4[i] = out;
    }
}

struct PersistentGpuMemory {
    float* raw_data = nullptr;
    size_t capacity_bytes = 0;

    float* Ensure(size_t required_bytes) {
        if (required_bytes > capacity_bytes) {
            if (raw_data != nullptr) {
                cudaFree(raw_data);
            }
            cudaMalloc(&raw_data, required_bytes);
            capacity_bytes = required_bytes;
        }
        return raw_data;
    }

    ~PersistentGpuMemory() {
        if (raw_data != nullptr) {
            cudaFree(raw_data);
        }
    }
};

PersistentGpuMemory g_mem_pool;

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }

    const size_t total_elements = input.size();
    const int row_size = static_cast<int>(total_elements / row_count);
    const size_t bytes = total_elements * sizeof(float);

    float* d_all = g_mem_pool.Ensure(2 * bytes);
    float* d_input = d_all;
    float* d_output = d_all + total_elements;

    cudaMemcpyAsync(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    if (row_size <= 4096) {
        int threads = std::min(512, std::max(32, row_size));
        size_t shared_bytes = row_size * sizeof(float);
        SoftmaxSharedKernel<<<row_count, threads, shared_bytes>>>(d_input, d_output, row_size);
    } else {
        int threads = 256;
        SoftmaxLargeRowKernel<<<row_count, threads>>>(d_input, d_output, row_size);
    }

    std::vector<float> output(total_elements);

    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    return output;
}