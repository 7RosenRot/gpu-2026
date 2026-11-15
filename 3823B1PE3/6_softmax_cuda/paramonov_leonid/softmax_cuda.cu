#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kMaxThreadsPerBlock = 1024;

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

__global__ void SoftmaxRowsKernel(const float* input,
                                  float* output,
                                  std::size_t row_size) {
    __shared__ float reduction[kMaxThreadsPerBlock];

    const unsigned int tid = threadIdx.x;
    const std::size_t row_offset = static_cast<std::size_t>(blockIdx.x) * row_size;

    float local_max = -CUDART_INF_F;
    for (std::size_t column = tid; column < row_size; column += blockDim.x) {
        local_max = fmaxf(local_max, input[row_offset + column]);
    }
    reduction[tid] = local_max;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + offset]);
        }
        __syncthreads();
    }
    const float row_max = reduction[0];

    float local_sum = 0.0f;
    for (std::size_t column = tid; column < row_size; column += blockDim.x) {
        const float value = expf(input[row_offset + column] - row_max);
        output[row_offset + column] = value;
        local_sum += value;
    }
    reduction[tid] = local_sum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduction[tid] += reduction[tid + offset];
        }
        __syncthreads();
    }
    const float row_sum = reduction[0];

    for (std::size_t column = tid; column < row_size; column += blockDim.x) {
        output[row_offset + column] /= row_sum;
    }
}

unsigned int ChooseBlockSize(std::size_t row_size) {
    unsigned int block_size = 1;
    const std::size_t limit = row_size < kMaxThreadsPerBlock
                                  ? row_size
                                  : kMaxThreadsPerBlock;
    while (static_cast<std::size_t>(block_size) * 2 <= limit) {
        block_size *= 2;
    }
    return block_size;
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0) {
        throw std::invalid_argument("row_count must be positive");
    }
    if (input.size() % static_cast<std::size_t>(row_count) != 0) {
        throw std::invalid_argument("input size must be divisible by row_count");
    }
    if (input.empty()) {
        return {};
    }

    const std::size_t row_size = input.size() / static_cast<std::size_t>(row_count);
    std::vector<float> output(input.size());
    const std::size_t bytes = input.size() * sizeof(float);
    DeviceBuffer deviceInput(bytes);
    DeviceBuffer deviceOutput(bytes);

    CheckCuda(cudaMemcpy(deviceInput.get(), input.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy input host-to-device");

    const unsigned int block_size = ChooseBlockSize(row_size);
    SoftmaxRowsKernel<<<static_cast<unsigned int>(row_count), block_size>>>(
        deviceInput.get(), deviceOutput.get(), row_size);
    CheckCuda(cudaGetLastError(), "SoftmaxRowsKernel launch");

    // The synchronous copy also waits for the kernel to finish.
    CheckCuda(cudaMemcpy(output.data(), deviceOutput.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy output device-to-host");
    return output;
}
