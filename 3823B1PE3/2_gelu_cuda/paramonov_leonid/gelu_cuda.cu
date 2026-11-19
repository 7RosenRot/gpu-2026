#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBlocks = 65535;
constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
constexpr float kCubicCoefficient = 0.044715f;

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

__global__ void GeluKernel(const float* input, float* output, std::size_t size) {
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t i = first; i < size; i += stride) {
        const float x = input[i];
        const float x3 = x * x * x;
        const float argument = kSqrtTwoOverPi * (x + kCubicCoefficient * x3);
        output[i] = 0.5f * x * (1.0f + tanhf(argument));
    }
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    if (input.empty()) {
        return output;
    }

    const std::size_t bytes = input.size() * sizeof(float);
    DeviceBuffer deviceInput(bytes);
    DeviceBuffer deviceOutput(bytes);

    CheckCuda(cudaMemcpy(deviceInput.get(), input.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host-to-device");

    const std::size_t requiredBlocks = (input.size() - 1) / kThreadsPerBlock + 1;
    const unsigned int blocks = static_cast<unsigned int>(
        requiredBlocks < kMaxBlocks ? requiredBlocks : kMaxBlocks);
    GeluKernel<<<blocks, kThreadsPerBlock>>>(deviceInput.get(), deviceOutput.get(), input.size());
    CheckCuda(cudaGetLastError(), "GeluKernel launch");

    // The synchronous copy also waits for the kernel to finish.
    CheckCuda(cudaMemcpy(output.data(), deviceOutput.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy device-to-host");
    return output;
}
