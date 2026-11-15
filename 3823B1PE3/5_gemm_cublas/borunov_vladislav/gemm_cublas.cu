#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace {

struct GemmWorkspace {
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    float* buffer = nullptr;
    std::size_t capacity = 0;

    GemmWorkspace() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
    }

    ~GemmWorkspace() {
        if (handle) cublasDestroy(handle);
        if (buffer) cudaFree(buffer);
        if (stream) cudaStreamDestroy(stream);
    }

    void Reserve(std::size_t elements) {
        if (elements <= capacity) return;
        if (buffer) cudaFree(buffer);
        cudaMalloc(&buffer, 3 * elements * sizeof(float));
        capacity = elements;
    }
};

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const std::size_t count = static_cast<std::size_t>(n) * n;
    if (count == 0) return {};

    static thread_local GemmWorkspace ws;
    ws.Reserve(count);

    const std::size_t bytes = count * sizeof(float);
    float* d_a = ws.buffer;
    float* d_b = d_a + count;
    float* d_c = d_b + count;

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, ws.stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, ws.stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(ws.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha, d_b, n, d_a, n, &beta, d_c, n);

    std::vector<float> c(count);
    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, ws.stream);
    cudaStreamSynchronize(ws.stream);
    return c;
}
