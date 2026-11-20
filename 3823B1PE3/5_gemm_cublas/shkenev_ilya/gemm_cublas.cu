#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void CheckCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

struct Workspace {
    cublasHandle_t handle = nullptr;
    float* data = nullptr;
    std::size_t capacity = 0;
    int device = 0;

    Workspace() {
        CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
        CheckCublas(cublasCreate(&handle), "cublasCreate");
    }

    ~Workspace() {
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cublasDestroy(handle);
        cudaFree(data);
        if (previous != device) {
            cudaSetDevice(previous);
        }
    }

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity) {
            return;
        }
        float* next = nullptr;
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&next), bytes),
                  "cudaMalloc");
        const cudaError_t status = cudaFree(data);
        if (status != cudaSuccess) {
            cudaFree(next);
            CheckCuda(status, "cudaFree");
        }
        data = next;
        capacity = bytes;
    }
};

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                             const std::vector<float>& b,
                             int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size must be nonnegative");
    }
    const std::size_t size = static_cast<std::size_t>(n);
    if (size != 0 && size > std::numeric_limits<std::size_t>::max() /
                                (3 * sizeof(float)) / size) {
        throw std::length_error("Matrix size is too large");
    }
    const std::size_t count = size * size;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Input matrices must contain n * n elements");
    }
    if (count == 0) {
        return {};
    }

    int device = 0;
    CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    static thread_local std::map<int, Workspace> workspaces;
    Workspace& workspace = workspaces.try_emplace(device).first->second;
    const std::size_t bytes = count * sizeof(float);
    workspace.Reserve(3 * bytes);

    float* device_a = workspace.data;
    float* device_b = device_a + count;
    float* device_c = device_b + count;
    CheckCuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice),
              "Copy matrix A to device");
    CheckCuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice),
              "Copy matrix B to device");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CheckCublas(cublasSgemm(workspace.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           n, n, n, &alpha, device_b, n, device_a, n,
                           &beta, device_c, n), "cublasSgemm");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost),
              "Copy matrix C to host");
    return result;
}
