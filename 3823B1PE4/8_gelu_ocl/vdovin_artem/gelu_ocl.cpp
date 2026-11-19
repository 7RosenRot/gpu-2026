#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

const char* kKernelSource = R"CLC(
__kernel void gelu(__global const float* input,
                   __global float* output,
                   const uint size)
{
    const uint i = get_global_id(0);
    if (i < size) {
        const float k = 1.5957691216057308f;
        const float x = input[i];
        const float t = k * x * (1.0f + 0.044715f * x * x);
        output[i] = t < -80.0f ? 0.0f : x / (1.0f + exp(-t));
    }
}
)CLC";

void Check(cl_int status, const char* what) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": OpenCL error " + std::to_string(status));
    }
}

struct OclState {
    int platform_index = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem input = nullptr;
    cl_mem output = nullptr;
    std::size_t capacity = 0;

    void Init(int platform) {
        cl_uint platform_count = 0;
        Check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs");
        if (platform < 0 || static_cast<cl_uint>(platform) >= platform_count) {
            throw std::invalid_argument("Invalid OpenCL platform index");
        }
        std::vector<cl_platform_id> platforms(platform_count);
        Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");

        cl_device_id device = nullptr;
        Check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr),
              "clGetDeviceIDs");

        cl_int status = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        Check(status, "clCreateContext");
        queue = clCreateCommandQueue(context, device, 0, &status);
        Check(status, "clCreateCommandQueue");

        program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &status);
        Check(status, "clCreateProgramWithSource");
        status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (status != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            throw std::runtime_error("clBuildProgram failed: " + log);
        }
        kernel = clCreateKernel(program, "gelu", &status);
        Check(status, "clCreateKernel");

        platform_index = platform;
    }

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity) {
            return;
        }
        ReleaseBuffers();
        cl_int status = CL_SUCCESS;
        input = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &status);
        Check(status, "clCreateBuffer input");
        output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &status);
        Check(status, "clCreateBuffer output");
        capacity = bytes;
    }

    void ReleaseBuffers() {
        if (input) clReleaseMemObject(input);
        if (output) clReleaseMemObject(output);
        input = output = nullptr;
        capacity = 0;
    }

    void Release() {
        ReleaseBuffers();
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        kernel = nullptr;
        program = nullptr;
        queue = nullptr;
        context = nullptr;
        platform_index = -1;
    }

    ~OclState() { Release(); }
};

OclState g_state;

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const std::size_t size = input.size();
    std::vector<float> output(size);
    if (size == 0) {
        return output;
    }

    if (g_state.platform_index != platform) {
        g_state.Release();
        g_state.Init(platform);
    }

    const std::size_t bytes = size * sizeof(float);
    g_state.Reserve(bytes);

    Check(clEnqueueWriteBuffer(g_state.queue, g_state.input, CL_TRUE, 0, bytes,
                               input.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer");

    const cl_uint count = static_cast<cl_uint>(size);
    Check(clSetKernelArg(g_state.kernel, 0, sizeof(cl_mem), &g_state.input), "clSetKernelArg 0");
    Check(clSetKernelArg(g_state.kernel, 1, sizeof(cl_mem), &g_state.output), "clSetKernelArg 1");
    Check(clSetKernelArg(g_state.kernel, 2, sizeof(cl_uint), &count), "clSetKernelArg 2");

    const std::size_t local = 256;
    const std::size_t global = (size + local - 1) / local * local;
    Check(clEnqueueNDRangeKernel(g_state.queue, g_state.kernel, 1, nullptr, &global, &local,
                                 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel");

    Check(clEnqueueReadBuffer(g_state.queue, g_state.output, CL_TRUE, 0, bytes,
                              output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer");
    return output;
}
