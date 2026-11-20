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
float4 gelu4(float4 x) {
    const float4 e = -x * (1.5957691216057308f + 0.0713548162726009f * x * x);
    const float4 r = x / (1.0f + exp(fmin(e, 80.0f)));
    return select(r, (float4)(0.0f), isgreater(e, (float4)(80.0f)));
}

float gelu1(float x) {
    const float e = -x * (1.5957691216057308f + 0.0713548162726009f * x * x);
    return e > 80.0f ? 0.0f : x / (1.0f + exp(e));
}

__kernel void gelu(__global const float* in, __global float* out,
                   const uint n4, const uint n) {
    const uint i = get_global_id(0);
    if (i < n4) {
        vstore4(gelu4(vload4(i, in)), i, out);
    }
    if (i == 0) {
        for (uint j = n4 * 4; j < n; ++j) {
            out[j] = gelu1(in[j]);
        }
    }
}
)CLC";

void Check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("OpenCL error ") + std::to_string(err) + " at " + what);
    }
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

    void Init(int platform) {
        if (platform_index == platform && kernel) return;

        cl_uint count = 0;
        Check(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs");
        if (platform < 0 || static_cast<cl_uint>(platform) >= count) {
            throw std::runtime_error("Invalid OpenCL platform index");
        }
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
        cl_context new_context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        Check(err, "clCreateContext");
        cl_command_queue new_queue = clCreateCommandQueue(new_context, device, 0, &err);
        Check(err, "clCreateCommandQueue");

        cl_program new_program = clCreateProgramWithSource(new_context, 1, &kKernelSource, nullptr, &err);
        Check(err, "clCreateProgramWithSource");
        err = clBuildProgram(new_program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(new_program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(new_program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            throw std::runtime_error("OpenCL build failed: " + log);
        }
        cl_kernel new_kernel = clCreateKernel(new_program, "gelu", &err);
        Check(err, "clCreateKernel");

        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);

        context = new_context;
        queue = new_queue;
        program = new_program;
        kernel = new_kernel;
        in = out = nullptr;
        capacity = 0;
        platform_index = platform;
    }

    void Reserve(size_t n) {
        if (n <= capacity) return;
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        in = out = nullptr;
        capacity = 0;
        cl_int err = CL_SUCCESS;
        in = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &err);
        Check(err, "clCreateBuffer(in)");
        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);
        Check(err, "clCreateBuffer(out)");
        capacity = n;
    }
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
    Check(clSetKernelArg(g_state.kernel, 0, sizeof(cl_mem), &g_state.in), "clSetKernelArg");
    Check(clSetKernelArg(g_state.kernel, 1, sizeof(cl_mem), &g_state.out), "clSetKernelArg");
    Check(clSetKernelArg(g_state.kernel, 2, sizeof(cl_uint), &n4), "clSetKernelArg");
    Check(clSetKernelArg(g_state.kernel, 3, sizeof(cl_uint), &nn), "clSetKernelArg");

    const size_t local = 256;
    const size_t work = n4 > 0 ? n4 : 1;
    size_t global = (work + local - 1) / local * local;
    Check(clEnqueueNDRangeKernel(g_state.queue, g_state.kernel, 1, nullptr, &global, &local,
                                 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    clFlush(g_state.queue);

    std::vector<float> output(n);

    Check(clEnqueueReadBuffer(g_state.queue, g_state.out, CL_TRUE, 0, bytes, output.data(),
                              0, nullptr, nullptr), "clEnqueueReadBuffer");
    return output;
}
