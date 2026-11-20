#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#include <CL/cl.h>

#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr char kSource[] = R"CLC(
__kernel void gelu(__global const float* input,
                   __global float* output, ulong count) {
    const size_t i = get_global_id(0);
    if (i < count) {
        const float x = input[i];
        const float z = 1.5957691216057308f * (x + 0.044715f * x * x * x);
        const float e = exp(-fabs(z));
        output[i] = z >= 0.0f ? x / (1.0f + e) : (x * e) / (1.0f + e);
    }
}
)CLC";

void CheckCL(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": OpenCL status " +
                                 std::to_string(status));
    }
}

template <typename T, auto Release>
struct Resource {
    T value = nullptr;

    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;

    ~Resource() {
        if (value) {
            Release(value);
        }
    }
};

using Buffer = Resource<cl_mem, clReleaseMemObject>;

struct Workspace {
    Resource<cl_context, clReleaseContext> context;
    Resource<cl_command_queue, clReleaseCommandQueue> queue;
    Resource<cl_program, clReleaseProgram> program;
    Resource<cl_kernel, clReleaseKernel> kernel;
    Buffer input;
    Buffer output;
    std::size_t capacity = 0;

    explicit Workspace(int platform) {
        cl_uint count = 0;
        CheckCL(clGetPlatformIDs(0, nullptr, &count), "Get platform count");
        if (static_cast<unsigned int>(platform) >= count) {
            throw std::invalid_argument("OpenCL platform index is out of range");
        }
        std::vector<cl_platform_id> platforms(count);
        CheckCL(clGetPlatformIDs(count, platforms.data(), nullptr), "Get platforms");
        cl_device_id device = nullptr;
        CheckCL(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU,
                               1, &device, nullptr), "Get first GPU device");
        const cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(platforms[platform]), 0
        };
        cl_int status = CL_SUCCESS;
        context.value = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
        CheckCL(status, "Create context");
        queue.value = clCreateCommandQueue(context.value, device, 0, &status);
        CheckCL(status, "Create command queue");
        const char* source = kSource;
        program.value = clCreateProgramWithSource(context.value, 1, &source, nullptr, &status);
        CheckCL(status, "Create program");
        status = clBuildProgram(program.value, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            std::size_t length = 0;
            clGetProgramBuildInfo(program.value, device, CL_PROGRAM_BUILD_LOG,
                                  0, nullptr, &length);
            std::string log(length, '\0');
            if (length != 0) {
                clGetProgramBuildInfo(program.value, device, CL_PROGRAM_BUILD_LOG,
                                      length, log.data(), nullptr);
            }
            throw std::runtime_error("Build OpenCL program: " +
                                     std::to_string(status) + "\n" + log);
        }
        kernel.value = clCreateKernel(program.value, "gelu", &status);
        CheckCL(status, "Create GELU kernel");
    }

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity) {
            return;
        }
        cl_int status = CL_SUCCESS;
        Buffer next_input;
        Buffer next_output;
        next_input.value = clCreateBuffer(context.value, CL_MEM_READ_ONLY, bytes,
                                          nullptr, &status);
        CheckCL(status, "Allocate input buffer");
        next_output.value = clCreateBuffer(context.value, CL_MEM_WRITE_ONLY, bytes,
                                           nullptr, &status);
        CheckCL(status, "Allocate output buffer");
        std::swap(input.value, next_input.value);
        std::swap(output.value, next_output.value);
        capacity = bytes;
    }
};

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (platform < 0) {
        throw std::invalid_argument("OpenCL platform index must be nonnegative");
    }
    if (input.empty()) {
        return {};
    }
    if (input.size() > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::length_error("Input is too large");
    }

    static thread_local std::map<int, std::unique_ptr<Workspace>> workspaces;
    auto& entry = workspaces[platform];
    if (!entry) {
        entry = std::make_unique<Workspace>(platform);
    }
    Workspace& workspace = *entry;
    const std::size_t bytes = input.size() * sizeof(float);
    workspace.Reserve(bytes);
    CheckCL(clEnqueueWriteBuffer(workspace.queue.value, workspace.input.value,
                                 CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr),
            "Upload input");
    const cl_ulong count = static_cast<cl_ulong>(input.size());
    CheckCL(clSetKernelArg(workspace.kernel.value, 0, sizeof(cl_mem),
                           &workspace.input.value), "Set input argument");
    CheckCL(clSetKernelArg(workspace.kernel.value, 1, sizeof(cl_mem),
                           &workspace.output.value), "Set output argument");
    CheckCL(clSetKernelArg(workspace.kernel.value, 2, sizeof(count), &count),
            "Set element count");
    const std::size_t global = input.size();
    CheckCL(clEnqueueNDRangeKernel(workspace.queue.value, workspace.kernel.value,
                                   1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "Launch GELU kernel");
    CheckCL(clFlush(workspace.queue.value), "Flush command queue");
    std::vector<float> output(input.size());
    CheckCL(clEnqueueReadBuffer(workspace.queue.value, workspace.output.value,
                                CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr),
            "Download output");
    return output;
}
