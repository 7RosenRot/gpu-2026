#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTileSize = 16;

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
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

__global__ void BlockGemmKernel(const float* a,
                                const float* b,
                                float* c,
                                int n) {
    __shared__ float tileA[kTileSize][kTileSize];
    __shared__ float tileB[kTileSize][kTileSize];

    const int row = static_cast<int>(blockIdx.y) * kTileSize + threadIdx.y;
    const int column = static_cast<int>(blockIdx.x) * kTileSize + threadIdx.x;
    float sum = 0.0f;

    const int tileCount = (n - 1) / kTileSize + 1;
    for (int tile = 0; tile < tileCount; ++tile) {
        const int aColumn = tile * kTileSize + threadIdx.x;
        const int bRow = tile * kTileSize + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < n && aColumn < n)
                ? a[static_cast<std::size_t>(row) * n + aColumn]
                : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < n && column < n)
                ? b[static_cast<std::size_t>(bRow) * n + column]
                : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && column < n) {
        c[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) + column] = sum;
    }
}

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

    std::vector<float> c(elementCount);
    const std::size_t bytes = elementCount * sizeof(float);
    DeviceBuffer deviceA(bytes);
    DeviceBuffer deviceB(bytes);
    DeviceBuffer deviceC(bytes);

    CheckCuda(cudaMemcpy(deviceA.get(), a.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy A host-to-device");
    CheckCuda(cudaMemcpy(deviceB.get(), b.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy B host-to-device");

    const dim3 threads(kTileSize, kTileSize);
    const unsigned int tilesPerDimension = static_cast<unsigned int>((n - 1) / kTileSize + 1);
    const dim3 blocks(tilesPerDimension, tilesPerDimension);
    BlockGemmKernel<<<blocks, threads>>>(deviceA.get(), deviceB.get(), deviceC.get(), n);
    CheckCuda(cudaGetLastError(), "BlockGemmKernel launch");

    // The synchronous copy also waits for the kernel to finish.
    CheckCuda(cudaMemcpy(c.data(), deviceC.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy C device-to-host");
    return c;
}
