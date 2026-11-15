#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(cublasGetStatusString(status));
    }
}

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n == 0) {
        return {};
    }

    const size_t size = static_cast<size_t>(n) * n;
    const size_t bytes = size * sizeof(float);
    std::vector<float> output;
    float* device_a = nullptr;
    cublasHandle_t handle;
    check(cublasCreate(&handle));
    try {
        check(cudaMalloc(&device_a, 3 * bytes));
        float* device_b = device_a + size;
        float* device_c = device_b + size;

        check(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

        const float alpha = 1.0f;
        const float beta = 0.0f;
        check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                          &alpha, device_b, n, device_a, n, &beta, device_c, n));

        output.resize(size);
        check(cudaMemcpy(output.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        if (device_a) {
            cudaFree(device_a);
        }
        cublasDestroy(handle);
        throw;
    }

    const cudaError_t stat_free = cudaFree(device_a);
    const cublasStatus_t stat_destroy = cublasDestroy(handle);
    check(stat_free);
    check(stat_destroy);
    return output;
}
