#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#include <CL/cl.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace {

constexpr char gelu_source[] = R"CLC(
__kernel void gelu(__global float* values) {
    const size_t i = get_global_id(0);
    const float x = values[i];
    const float z = 0.7978845608028654f * x * fma(0.044715f * x, x, 1.0f);
    const float e = exp(-2.0f * fabs(z));
    const float weight = (x < 0.0f ? e : 1.0f) / (1.0f + e);
    values[i] = isinf(x) && x < 0.0f ? -0.0f : x * weight;
}
)CLC";

template <typename Handle, auto Release>
using ClHandle = std::unique_ptr<std::remove_pointer_t<Handle>, decltype(Release)>;

void CheckCl(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": OpenCL error " + std::to_string(status));
    }
}

class GeluProgram {
public:
    explicit GeluProgram(int platform) {
        cl_uint platform_count = 0;
        CheckCl(clGetPlatformIDs(0, nullptr, &platform_count), "Count OpenCL platforms");
        if (platform < 0 || static_cast<cl_uint>(platform) >= platform_count) {
            throw std::out_of_range("OpenCL platform index is out of range");
        }

        std::vector<cl_platform_id> platforms(platform_count);
        CheckCl(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "List OpenCL platforms");
        const cl_platform_id selected = platforms[platform];

        cl_device_id device = nullptr;
        CheckCl(clGetDeviceIDs(selected, CL_DEVICE_TYPE_GPU, 1, &device, nullptr),
                "Select the first OpenCL GPU");

        const cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(selected), 0
        };
        cl_int status = CL_SUCCESS;
        context_.reset(clCreateContext(properties, 1, &device, nullptr, nullptr, &status));
        CheckCl(status, "Create OpenCL context");
        queue_.reset(clCreateCommandQueue(context_.get(), device, 0, &status));
        CheckCl(status, "Create OpenCL command queue");

        const char* source = gelu_source;
        program_.reset(clCreateProgramWithSource(context_.get(), 1, &source, nullptr, &status));
        CheckCl(status, "Create OpenCL GELU program");
        status = clBuildProgram(program_.get(), 1, &device, nullptr, nullptr, nullptr);
        if (status != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(program_.get(), device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            if (log_size != 0) {
                clGetProgramBuildInfo(program_.get(), device, CL_PROGRAM_BUILD_LOG,
                                      log_size, log.data(), nullptr);
                if (log.back() == '\0') {
                    log.pop_back();
                }
            }
            throw std::runtime_error("Build OpenCL GELU program (" + std::to_string(status) + "): " + log);
        }

        kernel_.reset(clCreateKernel(program_.get(), "gelu", &status));
        CheckCl(status, "Create OpenCL GELU kernel");
    }

    std::vector<float> Run(const std::vector<float>& input) {
        const std::size_t size = input.size();
        const std::size_t bytes = size * sizeof(float);
        cl_int status = CL_SUCCESS;
        const ClHandle<cl_mem, clReleaseMemObject> buffer(
            clCreateBuffer(context_.get(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           bytes, const_cast<float*>(input.data()), &status),
            clReleaseMemObject);
        CheckCl(status, "Create OpenCL GELU buffer");

        const cl_mem memory = buffer.get();
        CheckCl(clSetKernelArg(kernel_.get(), 0, sizeof(memory), &memory), "Set GELU argument");
        CheckCl(clEnqueueNDRangeKernel(queue_.get(), kernel_.get(), 1, nullptr, &size, nullptr,
                                      0, nullptr, nullptr),
                "Launch OpenCL GELU kernel");

        std::vector<float> result(size);
        CheckCl(clEnqueueReadBuffer(queue_.get(), memory, CL_TRUE, 0, bytes, result.data(),
                                    0, nullptr, nullptr),
                "Read OpenCL GELU output");
        return result;
    }

private:
    ClHandle<cl_context, clReleaseContext> context_{nullptr, clReleaseContext};
    ClHandle<cl_command_queue, clReleaseCommandQueue> queue_{nullptr, clReleaseCommandQueue};
    ClHandle<cl_program, clReleaseProgram> program_{nullptr, clReleaseProgram};
    ClHandle<cl_kernel, clReleaseKernel> kernel_{nullptr, clReleaseKernel};
};

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    thread_local std::unordered_map<int, std::unique_ptr<GeluProgram>> programs;
    auto program = programs.find(platform);
    if (program == programs.end()) {
        program = programs.emplace(platform, std::make_unique<GeluProgram>(platform)).first;
    }
    return program->second->Run(input);
}
