#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void CheckCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        CheckCuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() const { return pointer_; }

private:
    float* pointer_ = nullptr;
};

class CublasHandle {
public:
    CublasHandle() {
        CheckCublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size n must be non-negative");
    }
    if (n == 0) {
        if (!a.empty() || !b.empty()) {
            throw std::invalid_argument("Matrices must be empty when n is zero");
        }
        return {};
    }

    const std::size_t size = static_cast<std::size_t>(n);
    if (size > std::numeric_limits<std::size_t>::max() / size) {
        throw std::length_error("Matrix dimensions are too large");
    }
    const std::size_t elementCount = size * size;
    if (elementCount > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::length_error("Matrix is too large");
    }
    if (a.size() != elementCount || b.size() != elementCount) {
        throw std::invalid_argument("Input matrices must both contain n*n elements");
    }
