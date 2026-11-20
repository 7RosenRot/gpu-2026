#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void CheckBlas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS error " + std::to_string(status));
    }
}

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size must be non-negative");
    }
    const std::size_t count = static_cast<std::size_t>(n) * n;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Both matrices must contain n * n elements");
    }
    if (count == 0) {
        return {};
    }

    constexpr std::size_t alignment = 256 / sizeof(float);
    const std::size_t pitch = ((count - 1) / alignment + 1) * alignment;
    const std::size_t bytes = count * sizeof(float);
    float* allocation = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&allocation), 3 * pitch * sizeof(float)),
              "Allocate CUDA matrices");
    const std::unique_ptr<float, decltype(&cudaFree)> storage(allocation, cudaFree);
    float* device_a = storage.get();
    float* device_b = device_a + pitch;
    float* device_c = device_b + pitch;

    CheckCuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), "Copy matrix A");
    CheckCuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), "Copy matrix B");

    cublasHandle_t raw_handle = nullptr;
    CheckBlas(cublasCreate(&raw_handle), "Create cuBLAS handle");
    const std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, decltype(&cublasDestroy)>
        handle(raw_handle, cublasDestroy);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    CheckBlas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                         &alpha, device_b, n, device_a, n, &beta, device_c, n),
              "Run cuBLAS SGEMM");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost), "Copy matrix C");
    return result;
}
