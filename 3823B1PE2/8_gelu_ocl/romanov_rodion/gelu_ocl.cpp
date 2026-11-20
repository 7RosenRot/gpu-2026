#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "gelu_ocl.h"

#include <CL/cl.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// tanh(z) = 1 - 2 / (exp(2z) + 1)  =>  gelu(x) = x / (1 + exp(-2z))
// 2 * sqrt(2/pi)            = 1.5957691216
// 2 * sqrt(2/pi) * 0.044715 = 0.0713548163
const char* kKernelSource = R"CLC(
inline float gelu1(float x) {
    float t = x * (1.5957691216f + 0.0713548163f * x * x);
    return x / (1.0f + exp(-t));
}

__kernel void gelu_kernel(__global const float* in,
                          __global float* out,
                          const int n) {
    int i = get_global_id(0);
    int base = i * 4;
    if (base + 3 < n) {
        float4 v = vload4(i, in);
        float4 r;
        r.x = gelu1(v.x);
        r.y = gelu1(v.y);
        r.z = gelu1(v.z);
        r.w = gelu1(v.w);
        vstore4(r, i, out);
    } else {
        for (int k = base; k < n; ++k) {
            out[k] = gelu1(in[k]);
        }
    }
}
)CLC";

void Check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("OpenCL error ") + std::to_string(err) + " in " + what);
    }
}

struct OclState {
    int platform_index = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    size_t preferred_wg = 256;

    void Release() {
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

    void Init(int platform) {
        if (platform_index == platform && kernel) return;
        Release();

        cl_uint num_platforms = 0;
        Check(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(count)");
        if (platform < 0 || static_cast<cl_uint>(platform) >= num_platforms) {
            throw std::runtime_error("Invalid platform index");
        }
        std::vector<cl_platform_id> platforms(num_platforms);
        Check(clGetPlatformIDs(num_platforms, platforms.data(), nullptr), "clGetPlatformIDs");

        cl_device_id device = nullptr;
        Check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr),
              "clGetDeviceIDs");

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
            throw std::runtime_error("clBuildProgram failed: " + log);
        }

        kernel = clCreateKernel(program, "gelu_kernel", &err);
        Check(err, "clCreateKernel");

        size_t wg = 0;
        if (clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(wg), &wg, nullptr) == CL_SUCCESS && wg > 0) {
            preferred_wg = wg < 256 ? wg : 256;
        }
        platform_index = platform;
    }

    ~OclState() { Release(); }
};

OclState& State() {
    static OclState state;
    return state;
}

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) return {};

    OclState& s = State();
    s.Init(platform);

    const size_t bytes = n * sizeof(float);
    cl_int err = CL_SUCCESS;

    cl_mem d_in = clCreateBuffer(s.context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    Check(err, "clCreateBuffer(in)");
    cl_mem d_out = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    Check(err, "clCreateBuffer(out)");

    // Asynchronous upload + kernel launch.
    Check(clEnqueueWriteBuffer(s.queue, d_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer");

    cl_int n_int = static_cast<cl_int>(n);
    Check(clSetKernelArg(s.kernel, 0, sizeof(cl_mem), &d_in), "clSetKernelArg(0)");
    Check(clSetKernelArg(s.kernel, 1, sizeof(cl_mem), &d_out), "clSetKernelArg(1)");
    Check(clSetKernelArg(s.kernel, 2, sizeof(cl_int), &n_int), "clSetKernelArg(2)");

    const size_t items = (n + 3) / 4;  // each work-item handles 4 floats
    const size_t local = s.preferred_wg;
    const size_t global = ((items + local - 1) / local) * local;
    Check(clEnqueueNDRangeKernel(s.queue, s.kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel");

    // Allocate the host result while the GPU is busy.
    std::vector<float> result(n);

    Check(clEnqueueReadBuffer(s.queue, d_out, CL_TRUE, 0, bytes, result.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer");

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    return result;
}