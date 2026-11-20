#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kGeluKernelSource[] = R"CLC(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const ulong count) {
    const size_t i = get_global_id(0);
    if ((ulong)i >= count) {
        return;
    }

    const float x = input[i];
    const float x3 = x * x * x;
    const float argument = 0.7978845608028654f * (x + 0.044715f * x3);
    output[i] = 0.5f * x * (1.0f + tanh(argument));
}
)CLC";

void CheckOpenCL(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with OpenCL error " + std::to_string(error));
    }
}

class OpenCLRuntime {
public:
    ~OpenCLRuntime() {
        if (kernel_ != nullptr) clReleaseKernel(kernel_);
        if (program_ != nullptr) clReleaseProgram(program_);
        if (queue_ != nullptr) clReleaseCommandQueue(queue_);
        if (context_ != nullptr) clReleaseContext(context_);
    }

    OpenCLRuntime(const OpenCLRuntime&) = delete;
    OpenCLRuntime& operator=(const OpenCLRuntime&) = delete;
    OpenCLRuntime() = default;

    void Initialize(int platform_index) {
        cl_uint platform_count = 0;
        CheckOpenCL(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
        if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= platform_count) {
            throw std::out_of_range("OpenCL platform index is out of range");
        }

        std::vector<cl_platform_id> platforms(platform_count);
        CheckOpenCL(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");
        const cl_platform_id platform = platforms[static_cast<std::size_t>(platform_index)];

        cl_uint device_count = 0;
        const cl_int device_query = clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (device_query == CL_DEVICE_NOT_FOUND || device_count == 0) {
            throw std::runtime_error("The selected OpenCL platform has no GPU device");
        }
        CheckOpenCL(device_query, "clGetDeviceIDs(count)");

        std::vector<cl_device_id> devices(device_count);
        CheckOpenCL(clGetDeviceIDs(
                        platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr),
                    "clGetDeviceIDs");
        const cl_device_id device = devices[0];

        cl_int error = CL_SUCCESS;
        context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        CheckOpenCL(error, "clCreateContext");

        queue_ = clCreateCommandQueue(context_, device, 0, &error);
        CheckOpenCL(error, "clCreateCommandQueue");

        const char* source = kGeluKernelSource;
        const std::size_t source_length = sizeof(kGeluKernelSource) - 1;
        program_ = clCreateProgramWithSource(context_, 1, &source, &source_length, &error);
        CheckOpenCL(error, "clCreateProgramWithSource");

        error = clBuildProgram(program_, 1, &device, nullptr, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                  0, nullptr, &log_size);
            std::string build_log(log_size, '\0');
            if (log_size > 0) {
                clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                      log_size, build_log.data(), nullptr);
            }
            throw std::runtime_error("clBuildProgram failed: " + build_log);
        }

        kernel_ = clCreateKernel(program_, "gelu_kernel", &error);
        CheckOpenCL(error, "clCreateKernel");
    }

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_kernel kernel() const { return kernel_; }

private:
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
};

OpenCLRuntime& GetRuntime(int platform) {
    static std::map<int, std::unique_ptr<OpenCLRuntime>> runtimes;
    auto found = runtimes.find(platform);
    if (found == runtimes.end()) {
        auto runtime = std::make_unique<OpenCLRuntime>();
        runtime->Initialize(platform);
        found = runtimes.emplace(platform, std::move(runtime)).first;
    }
    return *found->second;
}

class MemObject {
public:
    explicit MemObject(cl_mem handle = nullptr) : handle_(handle) {}
    ~MemObject() { if (handle_ != nullptr) clReleaseMemObject(handle_); }
    MemObject(const MemObject&) = delete;
    MemObject& operator=(const MemObject&) = delete;
    cl_mem get() const { return handle_; }
private:
    cl_mem handle_;
};

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    OpenCLRuntime& runtime = GetRuntime(platform);
    const std::size_t bytes = input.size() * sizeof(float);
    cl_int error = CL_SUCCESS;

    MemObject device_input(clCreateBuffer(runtime.context(),
                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          bytes,
                                          const_cast<float*>(input.data()),
                                          &error));
    CheckOpenCL(error, "clCreateBuffer(input)");

    MemObject device_output(clCreateBuffer(runtime.context(), CL_MEM_WRITE_ONLY,
                                           bytes, nullptr, &error));
    CheckOpenCL(error, "clCreateBuffer(output)");

    const cl_mem input_arg = device_input.get();
    const cl_mem output_arg = device_output.get();
    const cl_ulong element_count = static_cast<cl_ulong>(input.size());
    CheckOpenCL(clSetKernelArg(runtime.kernel(), 0, sizeof(cl_mem), &input_arg),
                "clSetKernelArg(input)");
    CheckOpenCL(clSetKernelArg(runtime.kernel(), 1, sizeof(cl_mem), &output_arg),
                "clSetKernelArg(output)");
    CheckOpenCL(clSetKernelArg(runtime.kernel(), 2, sizeof(cl_ulong), &element_count),
                "clSetKernelArg(count)");

    const std::size_t global_size = input.size();
    CheckOpenCL(clEnqueueNDRangeKernel(runtime.queue(), runtime.kernel(), 1, nullptr,
                                       &global_size, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel");

    std::vector<float> output(input.size());
    CheckOpenCL(clEnqueueReadBuffer(runtime.queue(), device_output.get(), CL_TRUE,
                                    0, bytes, output.data(), 0, nullptr, nullptr),
                "clEnqueueReadBuffer");
    return output;
}