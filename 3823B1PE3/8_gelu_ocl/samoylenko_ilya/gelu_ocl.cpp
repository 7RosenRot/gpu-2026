#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

const char* source = R"CLC(
__kernel void GeluKernel(__global float* data) {
    size_t i = get_global_id(0);
    float x = data[i];
    data[i] = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}
)CLC";

void check(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}

struct OpenCLResources {
    int platform;
    size_t capacity = 0;

    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;

    explicit OpenCLResources(int platform_index) : platform(platform_index) {}
    OpenCLResources(const OpenCLResources&) = delete;
    OpenCLResources& operator=(const OpenCLResources&) = delete;

    ~OpenCLResources() {
        if (buffer) clReleaseMemObject(buffer);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

std::unique_ptr<OpenCLResources> create_resources(int platform) {
    cl_uint platform_count;
    check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs");

    if (platform < 0 || static_cast<cl_uint>(platform) >= platform_count) {
        throw std::out_of_range("OpenCL platform index out of range");
    }

    std::vector<cl_platform_id> platforms(platform_count);
    check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");

    cl_device_id device;
    check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");

    auto cl = std::make_unique<OpenCLResources>(platform);

    cl_int status;
    cl->context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    check(status, "clCreateContext");

    cl->queue = clCreateCommandQueue(cl->context, device, 0, &status);
    check(status, "clCreateCommandQueue");

    cl->program = clCreateProgramWithSource(cl->context, 1, &source, nullptr, &status);
    check(status, "clCreateProgramWithSource");
    check(clBuildProgram(cl->program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram");

    cl->kernel = clCreateKernel(cl->program, "GeluKernel", &status);
    check(status, "clCreateKernel");
    return cl;
}

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    static thread_local std::unique_ptr<OpenCLResources> cl;
    if (!cl || cl->platform != platform) {
        cl = create_resources(platform);
    }

    const size_t size = input.size();
    const size_t bytes = size * sizeof(float);

    try {
        if (bytes > cl->capacity) {
            if (cl->buffer) {
                check(clReleaseMemObject(cl->buffer), "clReleaseMemObject");
                cl->buffer = nullptr;
            }

            cl_int status;
            cl->buffer = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, bytes, nullptr, &status);
            check(status, "clCreateBuffer");
            check(clSetKernelArg(cl->kernel, 0, sizeof(cl->buffer), &cl->buffer), "clSetKernelArg");

            cl->capacity = bytes;
        }

        check(clEnqueueWriteBuffer(cl->queue, cl->buffer, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr),
              "clEnqueueWriteBuffer");
        check(clEnqueueNDRangeKernel(cl->queue, cl->kernel, 1, nullptr, &size, nullptr, 0, nullptr, nullptr),
              "clEnqueueNDRangeKernel");
        check(clFlush(cl->queue), "clFlush");

        std::vector<float> output(size);
        check(clEnqueueReadBuffer(cl->queue, cl->buffer, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr),
              "clEnqueueReadBuffer");
        return output;
    } catch (...) {
        clFinish(cl->queue);
        cl.reset();
        throw;
    }
}
