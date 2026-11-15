
std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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

    const std::size_t requiredBlocks = (elementCount - 1) / kThreadsPerBlock + 1;
    const unsigned int blocks = static_cast<unsigned int>(
        requiredBlocks < kMaxBlocks ? requiredBlocks : kMaxBlocks);
    NaiveGemmKernel<<<blocks, kThreadsPerBlock>>>(
        deviceA.get(), deviceB.get(), deviceC.get(), size, elementCount);
    CheckCuda(cudaGetLastError(), "NaiveGemmKernel launch");

    // The synchronous copy also waits for the kernel to finish.
    CheckCuda(cudaMemcpy(c.data(), deviceC.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy C device-to-host");
    return c;
}
