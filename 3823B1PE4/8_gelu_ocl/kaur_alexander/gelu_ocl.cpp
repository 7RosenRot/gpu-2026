#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <cstring>
#include <iostream>

static const char* kernel_source = R"(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float t = -2.0f * 0.7978845608028654f * (x + 0.044715f * x * x * x);
        output[i] = x / (1.0f + exp(t));
    }
}
)";

static cl_context context = nullptr;
static cl_command_queue queue = nullptr;
static cl_program program = nullptr;
static cl_kernel kernel = nullptr;
static cl_device_id device = nullptr;
static bool initialized = false;

static void init_opencl(int platform) {
    if (initialized) return;

    cl_platform_id platform_id;
    clGetPlatformIDs(1, &platform_id, nullptr);

    cl_uint num_devices;
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, device, 0, nullptr);

    program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    kernel = clCreateKernel(program, "gelu_kernel", nullptr);

    initialized = true;
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    init_opencl(platform);

    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    std::vector<float> output(n);

    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)input.data(), nullptr);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t global_size = ((n + 255) / 256) * 256;
    size_t local_size = 256;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);

    return output;
}