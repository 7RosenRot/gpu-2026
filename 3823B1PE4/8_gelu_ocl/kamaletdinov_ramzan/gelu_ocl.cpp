#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

const char* kKernelSource = R"CLC(
__kernel void gelu(__global const float4* in4, __global float4* out4,
                   __global const float* in, __global float* out,
                   const uint n4, const uint n) {
    const float k2 = 2.0f * 0.7978845608028654f;
    const float k3 = k2 * 0.044715f;
    const uint i = get_global_id(0);
    if (i < n4) {
        const float4 x = in4[i];
        const float4 t = x * (k2 + k3 * x * x);
        out4[i] = native_divide(x, 1.0f + native_exp(-t));
    }
    const uint tail = n4 * 4 + i;
    if (tail < n) {
        const float x = in[tail];
        const float t = x * (k2 + k3 * x * x);
        out[tail] = native_divide(x, 1.0f + native_exp(-t));
    }
}
)CLC";

void Check(cl_int err, const char* what) {
    if (err != CL_SUCCESS)
        throw std::runtime_error(std::string("OpenCL error ") + std::to_string(err) + " at " + what);
}

struct OclState {
    int platform_index = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem in = nullptr;
    cl_mem out = nullptr;
    size_t capacity = 0;

    void Release() {
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        in = out = nullptr; kernel = nullptr; program = nullptr;
        queue = nullptr; context = nullptr; capacity = 0; platform_index = -1;
    }

    void Init(int platform) {
        if (platform_index == platform && kernel) return;
        Release();

        cl_uint count = 0;
        Check(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs");
        if (platform < 0 || static_cast<cl_uint>(platform) >= count)
            throw std::runtime_error("Invalid OpenCL platform index");
        std::vector<cl_platform_id> platforms(count);
        Check(clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");

        cl_uint dev_count = 0;
        Check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 0, nullptr, &dev_count),
              "clGetDeviceIDs");
        std::vector<cl_device_id> devices(dev_count);
        Check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, dev_count, devices.data(), nullptr),
              "clGetDeviceIDs");
        cl_device_id device = devices[0];

        cl_int err = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        Check(err, "clCreateContext");
        queue = clCreateCommandQueue(context, device, 0, &err);
        Check(err, "clCreateCommandQueue");

        program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
        Check(err, "clCreateProgramWithSource");
        err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            throw std::runtime_error("OpenCL build failed: " + log);
        }
        kernel = clCreateKernel(program, "gelu", &err);
        Check(err, "clCreateKernel");
        platform_index = platform;
    }

    void Reserve(size_t n) {
        if (n <= capacity) return;
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        cl_int err = CL_SUCCESS;
        in = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &err);
        Check(err, "clCreateBuffer(in)");
        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);
        Check(err, "clCreateBuffer(out)");
        capacity = n;
    }

    ~OclState() { Release(); }
};

OclState g_state;

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) return {};

    g_state.Init(platform);
    g_state.Reserve(n);
    const size_t bytes = n * sizeof(float);

    Check(clEnqueueWriteBuffer(g_state.queue, g_state.in, CL_FALSE, 0, bytes, input.data(),
                               0, nullptr, nullptr), "clEnqueueWriteBuffer");

    const cl_uint n4 = static_cast<cl_uint>(n / 4);
    const cl_uint nn = static_cast<cl_uint>(n);
    clSetKernelArg(g_state.kernel, 0, sizeof(cl_mem), &g_state.in);
    clSetKernelArg(g_state.kernel, 1, sizeof(cl_mem), &g_state.out);
    clSetKernelArg(g_state.kernel, 2, sizeof(cl_mem), &g_state.in);
    clSetKernelArg(g_state.kernel, 3, sizeof(cl_mem), &g_state.out);
    clSetKernelArg(g_state.kernel, 4, sizeof(cl_uint), &n4);
    clSetKernelArg(g_state.kernel, 5, sizeof(cl_uint), &nn);

    const size_t local = 256;
    size_t work = n4 > (n - n4 * 4) ? n4 : (n - n4 * 4);
    size_t global = (work + local - 1) / local * local;
    Check(clEnqueueNDRangeKernel(g_state.queue, g_state.kernel, 1, nullptr, &global, &local,
                                 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    clFlush(g_state.queue);

    std::vector<float> output(n);

    Check(clEnqueueReadBuffer(g_state.queue, g_state.out, CL_TRUE, 0, bytes, output.data(),
                              0, nullptr, nullptr), "clEnqueueReadBuffer");
    return output;
}
