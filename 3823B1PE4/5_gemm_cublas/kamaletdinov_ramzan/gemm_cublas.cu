#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace {

struct CublasContext {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    float *a = nullptr, *b = nullptr, *c = nullptr;
    size_t capacity = 0;

    void Init() {
        if (handle) return;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
    }
    void Reserve(size_t elems) {
        if (elems <= capacity) return;
        cudaFree(a); cudaFree(b); cudaFree(c);
        cudaMalloc(&a, elems * sizeof(float));
        cudaMalloc(&b, elems * sizeof(float));
        cudaMalloc(&c, elems * sizeof(float));
        capacity = elems;
    }
    ~CublasContext() {
        cudaFree(a); cudaFree(b); cudaFree(c);
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }
};

CublasContext g_ctx;

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) return {};
    g_ctx.Init();
    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);
    g_ctx.Reserve(elems);

    cudaMemcpyAsync(g_ctx.a, a.data(), bytes, cudaMemcpyHostToDevice, g_ctx.stream);
    cudaMemcpyAsync(g_ctx.b, b.data(), bytes, cudaMemcpyHostToDevice, g_ctx.stream);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, g_ctx.b, n, g_ctx.a, n, &beta, g_ctx.c, n);

    std::vector<float> c(elems);

    cudaMemcpyAsync(c.data(), g_ctx.c, bytes, cudaMemcpyDeviceToHost, g_ctx.stream);
    cudaStreamSynchronize(g_ctx.stream);
    return c;
}
