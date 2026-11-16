#include "gelu_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr float kK2 = 2.0f * 0.7978845608028654f;
constexpr float kK3 = kK2 * 0.044715f;

__device__ __forceinline__ float Gelu(float x) {
    const float t = x * (kK2 + kK3 * x * x);
    return __fdividef(x, 1.0f + __expf(-t));
}

__global__ void GeluKernel(const float* __restrict__ in,
                           float* __restrict__ out,
                           size_t n4, size_t n) {
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = in4[i];
        v.x = Gelu(v.x); v.y = Gelu(v.y); v.z = Gelu(v.z); v.w = Gelu(v.w);
        out4[i] = v;
    }
    const size_t tail = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x;
    if (tail < n) out[tail] = Gelu(in[tail]);
}

struct DeviceBuffers {
    float* in = nullptr;
    float* out = nullptr;
    size_t capacity = 0;
    cudaStream_t stream = nullptr;

    void Reserve(size_t n) {
        if (!stream) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (n <= capacity) return;
        cudaFree(in);
        cudaFree(out);
        cudaMalloc(&in, n * sizeof(float));
        cudaMalloc(&out, n * sizeof(float));
        capacity = n;
    }
    ~DeviceBuffers() {
        cudaFree(in);
        cudaFree(out);
        if (stream) cudaStreamDestroy(stream);
    }
};

DeviceBuffers g_buf;

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) return {};

    g_buf.Reserve(n);
    const size_t bytes = n * sizeof(float);

    cudaMemcpyAsync(g_buf.in, input.data(), bytes, cudaMemcpyHostToDevice, g_buf.stream);

    const size_t n4 = n / 4;
    const int threads = 256;
    int blocks = static_cast<int>((n4 + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535 * 8) blocks = 65535 * 8;
    GeluKernel<<<blocks, threads, 0, g_buf.stream>>>(
        g_buf.in, g_buf.out, n4, n);

    std::vector<float> output(n);

    cudaMemcpyAsync(output.data(), g_buf.out, bytes, cudaMemcpyDeviceToHost, g_buf.stream);
    cudaStreamSynchronize(g_buf.stream);
    return output;
}
