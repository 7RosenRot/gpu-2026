#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float WarpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float WarpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float BlockReduceMax(float val) {
    __shared__ float buf[32];
    val = WarpReduceMax(val);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) buf[wid] = val;
    __syncthreads();
    float w = (lane < kWarps) ? buf[lane] : -1e30f;
    if (wid == 0) w = WarpReduceMax(w);
    if (threadIdx.x == 0) buf[0] = w;
    __syncthreads();
    return buf[0];
}

__device__ __forceinline__ float BlockReduceSum(float val) {
    __shared__ float buf[32];
    val = WarpReduceSum(val);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) buf[wid] = val;
    __syncthreads();
    float w = (lane < kWarps) ? buf[lane] : 0.0f;
    if (wid == 0) w = WarpReduceSum(w);
    if (threadIdx.x == 0) buf[0] = w;
    __syncthreads();
    return buf[0];
}

__global__ void SoftmaxKernel(const float* __restrict__ in,
                              float* __restrict__ out,
                              int row_size) {
    const int tid = threadIdx.x;
    const int n4 = row_size >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(
        in + static_cast<size_t>(blockIdx.x) * row_size);
    float4* out4 = reinterpret_cast<float4*>(
        out + static_cast<size_t>(blockIdx.x) * row_size);

    float local_max = -1e30f;
    for (int i = tid; i < n4; i += kThreads) {
        float4 v = in4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    float row_max = BlockReduceMax(local_max);

    float local_sum = 0.0f;
    for (int i = tid; i < n4; i += kThreads) {
        float4 v = in4[i];
        float4 e;
        e.x = __expf(v.x - row_max);
        e.y = __expf(v.y - row_max);
        e.z = __expf(v.z - row_max);
        e.w = __expf(v.w - row_max);
        out4[i] = e;
        local_sum += e.x + e.y + e.z + e.w;
    }
    float inv_sum = 1.0f / BlockReduceSum(local_sum);

    for (int i = tid; i < n4; i += kThreads) {
        float4 e = out4[i];
        e.x *= inv_sum;
        e.y *= inv_sum;
        e.z *= inv_sum;
        e.w *= inv_sum;
        out4[i] = e;
    }
}

struct SoftmaxWorkspace {
    cudaStream_t stream = nullptr;
    float* buffer = nullptr;
    size_t capacity = 0;

    SoftmaxWorkspace() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~SoftmaxWorkspace() {
        if (buffer) cudaFree(buffer);
        if (stream) cudaStreamDestroy(stream);
    }

    void Reserve(size_t elements) {
        if (elements <= capacity) return;
        if (buffer) cudaFree(buffer);
        cudaMalloc(&buffer, 2 * elements * sizeof(float));
        capacity = elements;
    }
};

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.empty()) return {};

    const int row_size = static_cast<int>(input.size() / static_cast<size_t>(row_count));
    const size_t total = input.size();
    const size_t bytes = total * sizeof(float);

    static thread_local SoftmaxWorkspace ws;
    ws.Reserve(total);

    float* d_in = ws.buffer;
    float* d_out = d_in + total;

    cudaMemcpyAsync(d_in, input.data(), bytes, cudaMemcpyHostToDevice, ws.stream);
    SoftmaxKernel<<<row_count, kThreads, 0, ws.stream>>>(d_in, d_out, row_size);

    std::vector<float> output(total);
    cudaMemcpyAsync(output.data(), d_out, bytes, cudaMemcpyDeviceToHost, ws.stream);
    cudaStreamSynchronize(ws.stream);
    return output;
}
