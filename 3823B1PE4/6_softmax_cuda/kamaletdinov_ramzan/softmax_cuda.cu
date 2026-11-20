#include "softmax_cuda.h"

#include <cfloat>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float WarpMax(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

template <bool kIsMax>
__device__ float BlockReduce(float v, float* smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = kIsMax ? WarpMax(v) : WarpSum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < kWarps ? smem[lane] : (kIsMax ? -FLT_MAX : 0.0f);
        v = kIsMax ? WarpMax(v) : WarpSum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    const float r = smem[0];
    __syncthreads();
    return r;
}

__global__ void SoftmaxKernel(const float* __restrict__ in, float* __restrict__ out,
                              int row_size, bool use_cache) {
    extern __shared__ float cache[];
    __shared__ float red[kWarps];

    const float* row_in = in + static_cast<size_t>(blockIdx.x) * row_size;
    float* row_out = out + static_cast<size_t>(blockIdx.x) * row_size;

    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float x = row_in[i];
        if (use_cache) cache[i] = x;
        m = fmaxf(m, x);
    }
    m = BlockReduce<true>(m, red);

    float s = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float e = __expf((use_cache ? cache[i] : row_in[i]) - m);
        if (use_cache) cache[i] = e;
        s += e;
    }
    s = BlockReduce<false>(s, red);
    const float inv = 1.0f / s;

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float e = use_cache ? cache[i] : __expf(row_in[i] - m);
        row_out[i] = e * inv;
    }
}

struct DeviceBuffers {
    float *in = nullptr, *out = nullptr;
    size_t capacity = 0;
    cudaStream_t stream = nullptr;
    int max_smem = 0;

    void Reserve(size_t n) {
        if (!stream) {
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            max_smem -= 1024;
            cudaFuncSetAttribute(SoftmaxKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
        }
        if (n <= capacity) return;
        cudaFree(in); cudaFree(out);
        cudaMalloc(&in, n * sizeof(float));
        cudaMalloc(&out, n * sizeof(float));
        capacity = n;
    }
    ~DeviceBuffers() {
        cudaFree(in); cudaFree(out);
        if (stream) cudaStreamDestroy(stream);
    }
};

DeviceBuffers g_buf;

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const size_t n = input.size();
    if (n == 0 || row_count <= 0) return std::vector<float>(n);
    const int row_size = static_cast<int>(n / row_count);
    const size_t bytes = n * sizeof(float);

    g_buf.Reserve(n);
    cudaMemcpyAsync(g_buf.in, input.data(), bytes, cudaMemcpyHostToDevice, g_buf.stream);

    const size_t smem = static_cast<size_t>(row_size) * sizeof(float);
    const bool use_cache = smem <= static_cast<size_t>(g_buf.max_smem);
    SoftmaxKernel<<<row_count, kThreads, use_cache ? smem : 0, g_buf.stream>>>(
        g_buf.in, g_buf.out, row_size, use_cache);

    std::vector<float> output(n);

    cudaMemcpyAsync(output.data(), g_buf.out, bytes, cudaMemcpyDeviceToHost, g_buf.stream);
    cudaStreamSynchronize(g_buf.stream);
    return output;
}
